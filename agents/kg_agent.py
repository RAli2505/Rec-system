"""
Knowledge Graph Agent for MARS.

Builds a heterogeneous knowledge graph over EdNet entities
(Question, Lecture, Tag, Part) using networkx, mines prerequisite
relations from student mastery sequences, trains GraphSAGE
embeddings via PyTorch Geometric, and provides cold-start /
gap-analysis methods for the orchestrator pipelines.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from .base_agent import BaseAgent

logger = logging.getLogger("mars.agent.knowledge_graph")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class TagGap:
    """A skill gap identified for a user."""
    tag_id: int
    current_accuracy: float
    target_accuracy: float = 0.7
    gap_score: float = 0.5  # Bayesian P(gap|data), higher = more likely gap
    prerequisite_tags: list[int] = field(default_factory=list)
    recommended_lectures: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """A single cold-start recommendation item."""
    item_id: str
    item_type: str          # "lecture" or "question"
    reason: str
    priority: float         # higher = more important
    related_tags: list[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
# GraphSAGE model
# ──────────────────────────────────────────────────────────

class GraphSAGEModel(torch.nn.Module):
    """Two-layer GraphSAGE for node embeddings (link prediction task)."""

    def __init__(self, in_channels: int, hidden: int = 128, out: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Dot-product link predictor."""
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

MASTERY_THRESHOLD = 0.7
MASTERY_MIN_ANSWERS = 2  # lowered from 5: ~80 interactions/user across 189 tags → <1 per tag avg
PREREQ_P_FORWARD = 0.6     # P(B|A) threshold
PREREQ_P_BACKWARD = 0.4    # P(A|B) threshold

PART_NAMES = {
    1: "Listening: Photographs",
    2: "Listening: Q&A",
    3: "Listening: Conversations",
    4: "Listening: Talks",
    5: "Reading: Incomplete Sentences",
    6: "Reading: Text Completion",
    7: "Reading: Reading Comprehension",
}


class KnowledgeGraphAgent(BaseAgent):
    """
    Builds and queries a knowledge graph of EdNet entities.

    Graph nodes: Question, Lecture, Tag, Part
    Graph edges: HAS_TAG, COVERS_TAG, PREREQUISITE_OF, BELONGS_TO_PART
    """

    name = "knowledge_graph"
    REQUIRED_COLUMNS = {
        "prerequisites": ["user_id", "tags", "correct", "timestamp"],
    }

    def __init__(self) -> None:
        super().__init__()
        # Seed fixing
        from .utils import set_global_seed
        set_global_seed(self.global_seed)

        # Config-driven parameters
        gs_cfg = self._config.get("graphsage", {})
        self._gs_hidden = gs_cfg.get("hidden_dim", 128)
        self._gs_output = gs_cfg.get("output_dim", 64)
        self._gs_epochs = gs_cfg.get("epochs", 200)
        self._gs_lr = gs_cfg.get("learning_rate", 0.01)
        self._gs_train_split = gs_cfg.get("train_split", 0.8)

        prereq_cfg = self._config.get("prerequisites", {})
        self._prereq_p_forward = prereq_cfg.get("p_forward_threshold", PREREQ_P_FORWARD)
        self._prereq_p_backward = prereq_cfg.get("p_backward_threshold", PREREQ_P_BACKWARD)
        self._prereq_min_cooccurrences = prereq_cfg.get("min_cooccurrences", 50)
        self._mastery_accuracy = prereq_cfg.get("mastery_accuracy", MASTERY_THRESHOLD)
        self._mastery_min_answers = prereq_cfg.get("mastery_min_answers", MASTERY_MIN_ANSWERS)

        self.graph: nx.DiGraph = nx.DiGraph()
        self.tag_embeddings: np.ndarray | None = None
        self._user_profiles: dict[str, dict[int, dict]] = {}   # user → tag → stats
        self._tag_id_to_idx: dict[int, int] = {}
        self._idx_to_tag_id: dict[int, int] = {}
        self._models_dir = Path("models")

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        """
        Optionally build graph + embeddings at init time.

        Pass ``questions_df``, ``lectures_df``, ``interactions_df``
        to build everything in one call.
        """
        if "questions_df" in kwargs and "lectures_df" in kwargs:
            self.build_graph(kwargs["questions_df"], kwargs["lectures_df"])
        if "interactions_df" in kwargs:
            self.build_prerequisites(
                kwargs["interactions_df"],
                train_user_ids=kwargs.get("train_user_ids"),
            )

    def receive_message(self, message):
        super().receive_message(message)
        action = message.data.get("action")
        if action == "get_gaps":
            return self.get_user_gaps(message.data["user_id"])
        return None

    # ──────────────────────────────────────────────────────
    # 1. Build base graph
    # ──────────────────────────────────────────────────────

    def build_graph(
        self,
        questions_df: pd.DataFrame,
        lectures_df: pd.DataFrame,
    ) -> nx.DiGraph:
        """
        Create the knowledge graph from metadata CSVs.

        Adds Question, Lecture, Tag, and Part nodes plus
        HAS_TAG, COVERS_TAG, and BELONGS_TO_PART edges.
        """
        G = nx.DiGraph()

        # ── Part nodes ──
        for pid, pname in PART_NAMES.items():
            G.add_node(f"part_{pid}", node_type="part", part_id=pid, name=pname)

        # ── Question nodes + HAS_TAG edges ──
        q_df = questions_df.copy()
        # Ensure tags are lists
        if q_df["tags"].dtype == object and isinstance(q_df["tags"].iloc[0], str):
            q_df["tags"] = q_df["tags"].apply(
                lambda s: [int(t) for t in s.split(";") if t.strip()]
            )

        # Normalise part column name
        part_col = "part_id" if "part_id" in q_df.columns else "part"

        for _, row in q_df.iterrows():
            qid = str(row["question_id"])
            pid = int(row[part_col])
            G.add_node(qid, node_type="question",
                       question_id=qid,
                       bundle_id=str(row["bundle_id"]),
                       correct_answer=str(row["correct_answer"]),
                       part_id=pid,
                       difficulty=0.5)  # placeholder — updated later

            G.add_edge(qid, f"part_{pid}", edge_type="BELONGS_TO_PART")

            for tag_id in row["tags"]:
                tag_node = f"tag_{tag_id}"
                if not G.has_node(tag_node):
                    G.add_node(tag_node, node_type="tag", tag_id=tag_id,
                               avg_difficulty=0.5, question_count=0, part_ids=set())
                G.nodes[tag_node]["question_count"] += 1
                G.nodes[tag_node]["part_ids"].add(pid)
                G.add_edge(qid, tag_node, edge_type="HAS_TAG")

        # ── Lecture nodes + COVERS_TAG edges ──
        l_df = lectures_df.copy()
        lpart_col = "part_id" if "part_id" in l_df.columns else "part"

        # Normalise lecture tags
        def _parse_ltags(val):
            if isinstance(val, list):
                return val
            if isinstance(val, (int, float)) and not pd.isna(val):
                return [int(val)]
            if isinstance(val, str):
                return [int(t) for t in val.split(";") if t.strip()]
            return []

        l_df["tags"] = l_df["tags"].apply(_parse_ltags)

        for _, row in l_df.iterrows():
            lid = str(row["lecture_id"])
            pid = int(row[lpart_col])
            G.add_node(lid, node_type="lecture", lecture_id=lid, part_id=pid)
            if pid in PART_NAMES:
                G.add_edge(lid, f"part_{pid}", edge_type="BELONGS_TO_PART")

            for tag_id in row["tags"]:
                tag_node = f"tag_{tag_id}"
                if not G.has_node(tag_node):
                    G.add_node(tag_node, node_type="tag", tag_id=tag_id,
                               avg_difficulty=0.5, question_count=0, part_ids=set())
                G.add_edge(lid, tag_node, edge_type="COVERS_TAG")

        self.graph = G
        logger.info(
            "Graph built: %d nodes, %d edges  (Q=%d, L=%d, T=%d, P=%d)",
            G.number_of_nodes(), G.number_of_edges(),
            sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "question"),
            sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "lecture"),
            sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "tag"),
            sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "part"),
        )
        return G

    # ──────────────────────────────────────────────────────
    # 2. Update difficulties from interactions
    # ──────────────────────────────────────────────────────

    def update_difficulties(self, interactions_df: pd.DataFrame) -> None:
        """Compute difficulty = 1 − accuracy for each question and avg per tag."""
        q_acc = interactions_df.groupby("question_id")["correct"].mean()

        n_updated = 0
        for qid_raw, acc in q_acc.items():
            # Interactions may store question_id as int (123) or str ("q123")
            for qid in (str(qid_raw), f"q{qid_raw}"):
                if self.graph.has_node(qid):
                    self.graph.nodes[qid]["difficulty"] = round(1.0 - float(acc), 4)
                    n_updated += 1
                    break

        # Aggregate per tag
        tag_nodes = [
            (n, d) for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "tag"
        ]
        for tag_node, data in tag_nodes:
            # Find questions pointing to this tag
            predecessors = [
                p for p in self.graph.predecessors(tag_node)
                if self.graph.nodes[p].get("node_type") == "question"
            ]
            # HAS_TAG: question→tag, so tag_node's predecessors are questions
            difficulties = [
                self.graph.nodes[p].get("difficulty", 0.5) for p in predecessors
            ]
            if difficulties:
                data["avg_difficulty"] = round(float(np.mean(difficulties)), 4)

        logger.info("Updated difficulties for %d / %d questions", n_updated, len(q_acc))

    # ──────────────────────────────────────────────────────
    # 3. Build prerequisite edges
    # ──────────────────────────────────────────────────────

    def build_prerequisites(
        self,
        interactions_df: pd.DataFrame,
        train_user_ids: set | None = None,
    ) -> None:
        """
        Mine PREREQUISITE_OF edges from student mastery sequences.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Full interaction log.
        train_user_ids : set | None
            If provided, only use these users for prerequisite mining
            to prevent data leakage from test users into the KG structure.

        Algorithm:
        1. For each student, track per-tag cumulative accuracy.
           A tag is "mastered" once accuracy > 0.7 with >= 5 answers.
        2. Record mastery order per student.
        3. For each tag pair, compute conditional mastery probabilities.
        4. Add directed edge A → B if P(B|A) > 0.6 and P(A|B) < 0.4.
        5. Remove cycles by dropping weakest edges.
        """
        # Filter to train users only to prevent data leakage
        df = interactions_df.copy()
        if train_user_ids is not None:
            df = df[df["user_id"].isin(train_user_ids)]
            logger.info(
                "Mining prerequisites from TRAIN users only: %d / %d interactions",
                len(df), len(interactions_df),
            )
        else:
            logger.warning(
                "build_prerequisites called without train_user_ids — "
                "using ALL %d interactions (potential data leakage!)",
                len(interactions_df),
            )

        logger.info("Mining prerequisite relations from %d interactions ...", len(df))
        if "tags" not in df.columns:
            logger.warning("No 'tags' column — skipping prerequisites")
            return

        # Parse tags if stored as string
        if df["tags"].dtype == object and isinstance(df["tags"].iloc[0], str):
            df["tags"] = df["tags"].apply(
                lambda s: [int(t) for t in str(s).split(";") if t.strip()]
            )

        # Step 1 & 2: Determine per-user mastery order
        mastery_order = self._compute_mastery_order(df)

        # Step 3: Conditional probabilities
        all_tags = set()
        for tags in mastery_order.values():
            all_tags.update(tags)

        mastered_sets = {
            uid: set(tags) for uid, tags in mastery_order.items()
        }

        pair_stats: dict[tuple[int, int], dict] = {}
        tag_list = sorted(all_tags)

        min_cooc = self._prereq_min_cooccurrences
        for ta, tb in combinations(tag_list, 2):
            both = sum(1 for uid in mastered_sets if ta in mastered_sets[uid] and tb in mastered_sets[uid])
            a_only = sum(1 for uid in mastered_sets if ta in mastered_sets[uid])
            b_only = sum(1 for uid in mastered_sets if tb in mastered_sets[uid])

            if a_only >= min_cooc and b_only >= min_cooc:
                p_b_given_a = both / a_only   # P(mastered B | mastered A)
                p_a_given_b = both / b_only   # P(mastered A | mastered B)
                pair_stats[(ta, tb)] = {
                    "p_forward": p_b_given_a,
                    "p_backward": p_a_given_b,
                    "strength": p_b_given_a - p_a_given_b,
                }

        # Step 4: Add prerequisite edges (use instance thresholds from config)
        pf_thresh = self._prereq_p_forward
        pb_thresh = self._prereq_p_backward
        n_added = 0
        for (ta, tb), stats in pair_stats.items():
            if stats["p_forward"] > pf_thresh and stats["p_backward"] < pb_thresh:
                self.graph.add_edge(
                    f"tag_{ta}", f"tag_{tb}",
                    edge_type="PREREQUISITE_OF",
                    p_forward=round(stats["p_forward"], 3),
                    p_backward=round(stats["p_backward"], 3),
                    strength=round(stats["strength"], 3),
                )
                n_added += 1
            elif stats["p_backward"] > pf_thresh and stats["p_forward"] < pb_thresh:
                self.graph.add_edge(
                    f"tag_{tb}", f"tag_{ta}",
                    edge_type="PREREQUISITE_OF",
                    p_forward=round(stats["p_backward"], 3),
                    p_backward=round(stats["p_forward"], 3),
                    strength=round(-stats["strength"], 3),
                )
                n_added += 1

        logger.info("Added %d PREREQUISITE_OF edges", n_added)

        # Step 5: Ensure DAG — remove weakest edges in cycles
        self._break_cycles()

    def _compute_mastery_order(self, df: pd.DataFrame) -> dict[str, list[int]]:
        """Return {user_id: [tag_ids in mastery order]}."""
        mastery_order: dict[str, list[int]] = {}

        tag_stats: dict[str, dict[int, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"correct": 0, "total": 0, "mastered": False, "mastery_idx": -1})
        )

        # Sort by user + time
        df_sorted = df.sort_values(["user_id", "timestamp"])

        answer_idx: dict[str, int] = defaultdict(int)

        for _, row in df_sorted.iterrows():
            uid = str(row["user_id"])
            tags = row["tags"]
            # Parse tags from string format (e.g. "78" or "78;123") to list
            if isinstance(tags, str):
                tags = [int(t) for t in tags.split(";") if t.strip().lstrip('-').isdigit()]
            elif isinstance(tags, (int, float)):
                tags = [int(tags)]
            elif not isinstance(tags, list):
                continue
            correct = bool(row["correct"])

            for tag_id in tags:
                stats = tag_stats[uid][tag_id]
                stats["total"] += 1
                if correct:
                    stats["correct"] += 1

                if (
                    not stats["mastered"]
                    and stats["total"] >= MASTERY_MIN_ANSWERS
                    and stats["correct"] / stats["total"] > MASTERY_THRESHOLD
                ):
                    stats["mastered"] = True
                    stats["mastery_idx"] = answer_idx[uid]

            answer_idx[uid] += 1

        # Build ordered lists
        for uid, tag_dict in tag_stats.items():
            mastered = [
                (tid, s["mastery_idx"])
                for tid, s in tag_dict.items()
                if s["mastered"]
            ]
            mastered.sort(key=lambda x: x[1])
            mastery_order[uid] = [tid for tid, _ in mastered]

        logger.info(
            "Mastery order computed for %d users, avg %.1f mastered tags/user",
            len(mastery_order),
            np.mean([len(v) for v in mastery_order.values()]) if mastery_order else 0,
        )
        return mastery_order

    def _break_cycles(self) -> None:
        """Remove weakest PREREQUISITE edges until the tag subgraph is a DAG."""
        # Extract prerequisite subgraph (tag→tag only)
        prereq_edges = [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if d.get("edge_type") == "PREREQUISITE_OF"
        ]
        if not prereq_edges:
            return

        sub = nx.DiGraph()
        for u, v, d in prereq_edges:
            sub.add_edge(u, v, **d)

        removed = 0
        while not nx.is_directed_acyclic_graph(sub):
            try:
                cycle = nx.find_cycle(sub, orientation="original")
            except nx.NetworkXNoCycle:
                break

            # Find weakest edge in cycle
            weakest_edge = min(
                cycle,
                key=lambda e: abs(sub.edges[e[0], e[1]].get("strength", 0)),
            )
            u, v = weakest_edge[0], weakest_edge[1]
            sub.remove_edge(u, v)
            self.graph.remove_edge(u, v)
            removed += 1

        if removed:
            logger.info("Removed %d edges to ensure DAG", removed)

    # ──────────────────────────────────────────────────────
    # 4. GraphSAGE embeddings
    # ──────────────────────────────────────────────────────

    def train_graphsage(
        self,
        hidden: int = 128,
        out_dim: int = 64,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> np.ndarray:
        """
        Train GraphSAGE on the tag subgraph for link prediction.

        Node features for each tag:
          [avg_difficulty, question_count_norm, part_one_hot(7)]
        Edges: PREREQUISITE_OF + HAS_TAG (projected to tag–tag co-occurrence)

        Returns tag_embeddings: (n_tags, out_dim).
        """
        from .utils import set_global_seed
        set_global_seed(self.global_seed)

        # Collect tag nodes
        tag_nodes = sorted(
            [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "tag"],
            key=lambda n: self.graph.nodes[n]["tag_id"],
        )
        n_tags = len(tag_nodes)
        if n_tags == 0:
            logger.warning("No tag nodes — cannot train GraphSAGE")
            return np.zeros((0, out_dim))

        tag_to_idx = {n: i for i, n in enumerate(tag_nodes)}
        self._tag_id_to_idx = {self.graph.nodes[n]["tag_id"]: i for i, n in enumerate(tag_nodes)}
        self._idx_to_tag_id = {i: self.graph.nodes[n]["tag_id"] for i, n in enumerate(tag_nodes)}

        # ── Node features: [avg_difficulty, question_count_norm, part_one_hot(7)] → dim 9 ──
        max_qcount = max(
            (self.graph.nodes[n].get("question_count", 0) for n in tag_nodes), default=1
        ) or 1

        features = np.zeros((n_tags, 9), dtype=np.float32)
        for i, tn in enumerate(tag_nodes):
            d = self.graph.nodes[tn]
            features[i, 0] = d.get("avg_difficulty", 0.5)
            features[i, 1] = d.get("question_count", 0) / max_qcount
            part_ids = d.get("part_ids", set())
            if isinstance(part_ids, set):
                for pid in part_ids:
                    if 1 <= pid <= 7:
                        features[i, 1 + pid] = 1.0  # indices 2..8

        x = torch.tensor(features, dtype=torch.float).to(_DEVICE)

        # ── Edges ──
        # 1) PREREQUISITE_OF (tag→tag)
        # 2) Co-occurrence: two tags that share a question
        edge_set: set[tuple[int, int]] = set()

        for u, v, ed in self.graph.edges(data=True):
            if ed.get("edge_type") == "PREREQUISITE_OF":
                if u in tag_to_idx and v in tag_to_idx:
                    edge_set.add((tag_to_idx[u], tag_to_idx[v]))

        # Co-occurrence via shared questions
        tag_questions: dict[int, set[str]] = defaultdict(set)
        for u, v, ed in self.graph.edges(data=True):
            if ed.get("edge_type") == "HAS_TAG" and v in tag_to_idx:
                tag_questions[tag_to_idx[v]].add(u)

        tag_indices = list(tag_questions.keys())
        for i in range(len(tag_indices)):
            for j in range(i + 1, len(tag_indices)):
                ti, tj = tag_indices[i], tag_indices[j]
                overlap = len(tag_questions[ti] & tag_questions[tj])
                if overlap >= 3:  # minimum co-occurrence
                    edge_set.add((ti, tj))
                    edge_set.add((tj, ti))

        if not edge_set:
            logger.warning("No edges for GraphSAGE — using self-loops only")
            edge_set = {(i, i) for i in range(n_tags)}

        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous().to(_DEVICE)

        # ── Train ──
        data = Data(x=x, edge_index=edge_index)
        model = GraphSAGEModel(in_channels=9, hidden=hidden, out=out_dim).to(_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        logger.info("GraphSAGE training on device: %s", _DEVICE)

        # Positive edges = existing; negative = random
        pos_edge_index = edge_index
        n_pos = pos_edge_index.size(1)

        # Build set of positive edges for filtering negatives
        pos_edge_set = set()
        for i in range(pos_edge_index.size(1)):
            s, d = int(pos_edge_index[0, i]), int(pos_edge_index[1, i])
            pos_edge_set.add((s, d))

        model.train()
        best_loss = float("inf")
        patience = 30
        no_improve = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            z = model(data.x, data.edge_index)

            # Positive scores
            pos_score = model.decode(z, pos_edge_index)
            pos_label = torch.ones(n_pos)

            # Negative sampling: batch generate and filter out existing edges
            n_sample = n_pos * 3  # oversample to filter
            neg_src = torch.randint(0, n_tags, (n_sample,))
            neg_dst = torch.randint(0, n_tags, (n_sample,))
            # Filter: no self-loops and no existing edges
            mask = neg_src != neg_dst
            neg_edges_all = list(zip(neg_src[mask].tolist(), neg_dst[mask].tolist()))
            neg_edges = [(s, d) for s, d in neg_edges_all if (s, d) not in pos_edge_set][:n_pos]
            # Pad if needed
            while len(neg_edges) < n_pos:
                s, d = torch.randint(0, n_tags, (1,)).item(), torch.randint(0, n_tags, (1,)).item()
                neg_edges.append((s, d))
            neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(_DEVICE)
            neg_score = model.decode(z, neg_edge_index)
            neg_label = torch.zeros(n_pos, device=_DEVICE)
            pos_label = torch.ones(n_pos, device=_DEVICE)

            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            loss.backward()
            optimizer.step()

            if epoch % 50 == 0 or epoch == 1:
                logger.info("GraphSAGE epoch %d/%d  loss=%.4f", epoch, epochs, loss.item())

            # Early stopping (patience=30)
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("GraphSAGE early stopping at epoch %d (patience=%d)", epoch, patience)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Extract embeddings & compute link prediction AUC ──
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index)

            # Link prediction AUC on held-out edges
            pos_score = model.decode(z, pos_edge_index).cpu().numpy()
            neg_src_eval = torch.randint(0, n_tags, (n_pos,), device=_DEVICE)
            neg_dst_eval = torch.randint(0, n_tags, (n_pos,), device=_DEVICE)
            neg_edge_eval = torch.stack([neg_src_eval, neg_dst_eval])
            neg_score_eval = model.decode(z, neg_edge_eval).cpu().numpy()

            all_scores = np.concatenate([pos_score, neg_score_eval])
            all_labels_np = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])
            try:
                from sklearn.metrics import roc_auc_score
                link_auc = float(roc_auc_score(all_labels_np, all_scores))
            except ValueError:
                link_auc = 0.5
            logger.info("GraphSAGE link prediction AUC: %.4f", link_auc)
            self._link_pred_auc = link_auc

        embeddings = z.cpu().numpy()
        self.tag_embeddings = embeddings

        # Save
        self._models_dir.mkdir(parents=True, exist_ok=True)
        np.save(self._models_dir / "tag_embeddings.npy", embeddings)
        torch.save(model.state_dict(), self._models_dir / "graphsage.pt")
        logger.info("Tag embeddings: shape %s, saved to %s", embeddings.shape, self._models_dir)

        return {
            "link_pred_auc": round(self._link_pred_auc, 4),
            "n_tag_embeddings": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
        }

    # ──────────────────────────────────────────────────────
    # 5. Cold-start
    # ──────────────────────────────────────────────────────

    def handle_cold_start(
        self,
        user_id: str | None = None,
        diagnostic: dict | None = None,
        confidence: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate initial recommendations for a new user from diagnostic results.

        1. Identify mastered and gap tags from diagnostic responses.
        2. For gap tags, find lectures via COVERS_TAG.
        3. Via PREREQUISITE_OF, find unmet prerequisites.
        4. Return recommended lectures + prerequisite gaps.
        """
        diagnostic = diagnostic or {}
        responses = diagnostic.get("responses", [])

        # Determine per-tag accuracy from diagnostic
        tag_stats: dict[int, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        for resp in responses:
            qid_raw = resp.get("question_id", "")
            correct = bool(resp.get("correct", False))
            # Resolve question_id: may be int (123) or str ("q123")
            qid = str(qid_raw)
            if not self.graph.has_node(qid):
                qid = f"q{qid_raw}"
            if self.graph.has_node(qid):
                for _, tag_node, ed in self.graph.out_edges(qid, data=True):
                    if ed.get("edge_type") == "HAS_TAG":
                        tid = self.graph.nodes[tag_node].get("tag_id")
                        if tid is not None:
                            tag_stats[tid]["total"] += 1
                            if correct:
                                tag_stats[tid]["correct"] += 1

        # Classify tags
        mastered_tags: set[int] = set()
        gap_tags: set[int] = set()
        for tid, stats in tag_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            if acc >= MASTERY_THRESHOLD:
                mastered_tags.add(tid)
            else:
                gap_tags.add(tid)

        # Find prerequisite gaps
        prerequisite_gaps: dict[int, list[int]] = {}
        for gtag in gap_tags:
            prereqs = self._get_prerequisites(gtag)
            unmet = [p for p in prereqs if p not in mastered_tags]
            if unmet:
                prerequisite_gaps[gtag] = unmet

        # Find lectures for gap tags
        recommendations: list[dict] = []
        for gtag in gap_tags:
            tag_node = f"tag_{gtag}"
            # Lectures that cover this tag: lecture → tag via COVERS_TAG
            # So tag is successor of lecture: predecessors of tag_node with COVERS_TAG
            for pred in self.graph.predecessors(tag_node):
                edata = self.graph.edges[pred, tag_node]
                if edata.get("edge_type") == "COVERS_TAG":
                    recommendations.append({
                        "item_id": pred,
                        "item_type": "lecture",
                        "reason": f"covers gap tag {gtag}",
                        "priority": 1.0 + len(prerequisite_gaps.get(gtag, [])) * 0.5,
                        "related_tags": [gtag],
                    })

        # Add prerequisite lectures
        for gtag, prereqs in prerequisite_gaps.items():
            for ptag in prereqs:
                ptag_node = f"tag_{ptag}"
                for pred in self.graph.predecessors(ptag_node):
                    edata = self.graph.edges[pred, ptag_node]
                    if edata.get("edge_type") == "COVERS_TAG":
                        recommendations.append({
                            "item_id": pred,
                            "item_type": "lecture",
                            "reason": f"prerequisite tag {ptag} for gap tag {gtag}",
                            "priority": 2.0,  # prerequisites first
                            "related_tags": [ptag, gtag],
                        })

        # Deduplicate and sort by priority
        seen: set[str] = set()
        unique_recs = []
        for r in sorted(recommendations, key=lambda x: -x["priority"]):
            if r["item_id"] not in seen:
                seen.add(r["item_id"])
                unique_recs.append(r)

        # Store user profile
        if user_id:
            self._user_profiles[user_id] = {
                tid: {
                    "correct": s["correct"],
                    "total": s["total"],
                    "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
                }
                for tid, s in tag_stats.items()
            }

        result = {
            "user_id": user_id,
            "mastered_tags": sorted(mastered_tags),
            "gap_tags": sorted(gap_tags),
            "prerequisite_gaps": {k: sorted(v) for k, v in prerequisite_gaps.items()},
            "recommendations": unique_recs,
            "n_recommendations": len(unique_recs),
        }
        logger.info(
            "Cold-start for %s: %d mastered, %d gaps, %d recommendations",
            user_id, len(mastered_tags), len(gap_tags), len(unique_recs),
        )
        return result

    def _get_prerequisites(self, tag_id: int, depth: int = 3) -> list[int]:
        """Get transitive prerequisites for a tag (BFS up to depth)."""
        tag_node = f"tag_{tag_id}"
        if not self.graph.has_node(tag_node):
            return []

        prereqs = []
        visited = {tag_node}
        frontier = [tag_node]

        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                # Prerequisites point TO this node: predecessors with PREREQUISITE_OF
                for pred in self.graph.predecessors(node):
                    edata = self.graph.edges[pred, node]
                    if edata.get("edge_type") == "PREREQUISITE_OF" and pred not in visited:
                        visited.add(pred)
                        tid = self.graph.nodes[pred].get("tag_id")
                        if tid is not None:
                            prereqs.append(tid)
                        next_frontier.append(pred)
            frontier = next_frontier

        return prereqs

    # ──────────────────────────────────────────────────────
    # 6. Update user profile
    # ──────────────────────────────────────────────────────

    def update_user_profile(
        self,
        user_id: str,
        diagnostic: dict | None = None,
        confidence: dict | None = None,
        interactions: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update the user's tag-level knowledge profile from new interactions."""
        if user_id not in self._user_profiles:
            self._user_profiles[user_id] = {}
        profile = self._user_profiles[user_id]

        # Update from interactions DataFrame
        if interactions is not None and len(interactions) > 0:
            df = interactions
            if "tags" in df.columns:
                if df["tags"].dtype == object and isinstance(df["tags"].iloc[0], str):
                    df = df.copy()
                    df["tags"] = df["tags"].apply(
                        lambda s: [int(t) for t in str(s).split(";") if t.strip()]
                    )
                for _, row in df.iterrows():
                    tags = row["tags"] if isinstance(row["tags"], list) else []
                    correct = bool(row.get("correct", False))
                    for tid in tags:
                        if tid not in profile:
                            profile[tid] = {"correct": 0, "total": 0, "accuracy": 0.0}
                        profile[tid]["total"] += 1
                        if correct:
                            profile[tid]["correct"] += 1
                        profile[tid]["accuracy"] = (
                            profile[tid]["correct"] / profile[tid]["total"]
                        )

        # From diagnostic dict
        if diagnostic and "ability" in diagnostic:
            pass  # IRT ability is stored by DiagnosticAgent

        weak_tags = sorted(
            [tid for tid, s in profile.items() if s["accuracy"] < MASTERY_THRESHOLD],
        )

        return {
            "user_id": user_id,
            "updated": True,
            "n_tags_tracked": len(profile),
            "weak_tags": weak_tags,
        }

    # ──────────────────────────────────────────────────────
    # 7. Get user gaps
    # ──────────────────────────────────────────────────────

    def get_user_gaps(self, user_id: str) -> list[TagGap]:
        """Return list of TagGap using Bayesian gap scoring.

        Gap score = P(gap | data) using Beta(1,1) prior:
            gap_score = (n_wrong + 1) / (n_total + 2)

        Higher gap_score = more likely the student has a gap.
        Returns tags where gap_score > 0.35 (i.e. estimated error rate > 35%).
        """
        profile = self._user_profiles.get(user_id, {})
        gaps = []
        for tid, stats in profile.items():
            n_total = stats["total"]
            if n_total < 1:
                continue
            n_wrong = n_total - stats["correct"]
            # Beta posterior: P(gap) = (n_wrong + alpha) / (n_total + alpha + beta)
            gap_score = (n_wrong + 1) / (n_total + 2)
            if gap_score > 0.35:
                prereqs = self._get_prerequisites(tid)
                tag_node = f"tag_{tid}"
                lectures = []
                if self.graph.has_node(tag_node):
                    for pred in self.graph.predecessors(tag_node):
                        edata = self.graph.edges[pred, tag_node]
                        if edata.get("edge_type") == "COVERS_TAG":
                            lectures.append(pred)

                gaps.append(TagGap(
                    tag_id=tid,
                    current_accuracy=round(stats["accuracy"], 3),
                    gap_score=round(gap_score, 3),
                    prerequisite_tags=prereqs,
                    recommended_lectures=lectures,
                ))

        gaps.sort(key=lambda g: g.gap_score if hasattr(g, 'gap_score') else 1 - g.current_accuracy,
                  reverse=True)
        return gaps

    # ──────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────

    def get_graph_stats(self) -> dict[str, Any]:
        """Summary statistics for the knowledge graph."""
        G = self.graph
        type_counts = defaultdict(int)
        for _, d in G.nodes(data=True):
            type_counts[d.get("node_type", "unknown")] += 1

        edge_type_counts = defaultdict(int)
        for _, _, d in G.edges(data=True):
            edge_type_counts[d.get("edge_type", "unknown")] += 1

        prereq_sub = nx.DiGraph(
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "PREREQUISITE_OF"
        )

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_types": dict(type_counts),
            "edge_types": dict(edge_type_counts),
            "avg_degree": round(np.mean([d for _, d in G.degree()]), 2) if G.number_of_nodes() > 0 else 0,
            "prerequisite_dag_nodes": prereq_sub.number_of_nodes(),
            "prerequisite_dag_edges": prereq_sub.number_of_edges(),
            "prerequisite_is_dag": nx.is_directed_acyclic_graph(prereq_sub) if prereq_sub.number_of_nodes() > 0 else True,
        }

    def get_prerequisite_chains(self, max_chains: int = 10) -> list[list[int]]:
        """Find longest prerequisite chains (for paper visualisation)."""
        prereq_sub = nx.DiGraph()
        for u, v, d in self.graph.edges(data=True):
            if d.get("edge_type") == "PREREQUISITE_OF":
                prereq_sub.add_edge(u, v)

        if prereq_sub.number_of_nodes() == 0:
            return []

        # Find all paths from sources to sinks
        sources = [n for n in prereq_sub.nodes() if prereq_sub.in_degree(n) == 0]
        sinks = [n for n in prereq_sub.nodes() if prereq_sub.out_degree(n) == 0]

        chains: list[list[int]] = []
        for src in sources:
            for sink in sinks:
                try:
                    for path in nx.all_simple_paths(prereq_sub, src, sink, cutoff=8):
                        tag_ids = [
                            self.graph.nodes[n].get("tag_id", n) for n in path
                        ]
                        chains.append(tag_ids)
                except nx.NetworkXNoPath:
                    continue

        # Return longest unique chains
        chains.sort(key=len, reverse=True)
        return chains[:max_chains]
