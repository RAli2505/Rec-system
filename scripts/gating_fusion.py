"""
Learned Gating Fusion for MARS multi-agent system.

Instead of fixed weights (W_PRED=0.50, W_IRT=0.15, ...),
a small MLP learns context-dependent weights per user:
  user_features -> softmax -> [w_pred, w_irt, w_kg, w_conf, w_clust]

Training: inner cross-validation on training users,
optimizing NDCG@10 via differentiable surrogate (approxNDCG).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class UserContext:
    """Features extracted from a user's interaction history for gating."""
    n_interactions: float       # number of context interactions
    recent_accuracy: float      # mean accuracy over context
    theta_estimate: float       # IRT ability estimate
    mean_confidence: float      # mean confidence class (0-5)
    n_mastered_tags: float      # tags with accuracy >= 0.7
    n_gap_tags: float           # tags with accuracy < 0.7
    accuracy_trend: float       # slope of accuracy over time (positive = improving)
    interaction_diversity: float # number of unique tags seen / total tags

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.n_interactions,
            self.recent_accuracy,
            self.theta_estimate,
            self.mean_confidence,
            self.n_mastered_tags,
            self.n_gap_tags,
            self.accuracy_trend,
            self.interaction_diversity,
        ], dtype=torch.float32)


NUM_FEATURES = 8
NUM_AGENTS = 5  # pred, irt, kg, conf, clust


class GatingNetwork(nn.Module):
    """
    Small MLP that maps user context features to agent fusion weights.

    Architecture: 8 -> 32 -> 16 -> 5 (softmax)
    ~700 parameters — minimal overfitting risk even with 300 users.
    """

    def __init__(self, n_features: int = NUM_FEATURES, n_agents: int = NUM_AGENTS,
                 hidden: int = 32, dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_agents),
        )
        self.temperature = temperature
        # Initialize bias toward PredictionAgent dominance (prior from grid search)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
        Returns:
            weights: (batch, n_agents) — softmax normalized
        """
        logits = self.net(x) / self.temperature
        return F.softmax(logits, dim=-1)


def extract_user_context(ctx_df, theta_est: float, num_tags: int = 293) -> UserContext:
    """Extract gating features from a user's context DataFrame."""
    n = len(ctx_df)
    correct = ctx_df["correct"].astype(float)
    acc = correct.mean() if n > 0 else 0.5

    # Accuracy trend: linear regression slope over interactions
    if n >= 3:
        x = np.arange(n, dtype=np.float32)
        x_centered = x - x.mean()
        trend = (x_centered * correct.values).sum() / (x_centered ** 2).sum()
    else:
        trend = 0.0

    # Tag mastery from context
    tag_stats = {}
    tags_col = ctx_df.get("tags", None)
    if tags_col is not None:
        for _, row in ctx_df.iterrows():
            raw_tags = row.get("tags", [])
            if isinstance(raw_tags, str):
                raw_tags = [int(t) for t in raw_tags.split(",") if t.strip().isdigit()]
            elif isinstance(raw_tags, (int, float)):
                raw_tags = [int(raw_tags)]
            for t in raw_tags:
                if t not in tag_stats:
                    tag_stats[t] = {"c": 0, "t": 0}
                tag_stats[t]["t"] += 1
                if row["correct"]:
                    tag_stats[t]["c"] += 1

    mastered = sum(1 for s in tag_stats.values() if s["t"] >= 2 and s["c"] / s["t"] >= 0.7)
    gaps = sum(1 for s in tag_stats.values() if s["t"] >= 2 and s["c"] / s["t"] < 0.7)

    # Confidence
    mean_conf = 0.0
    if "confidence_class" in ctx_df.columns:
        mean_conf = ctx_df["confidence_class"].astype(float).mean()

    return UserContext(
        n_interactions=min(n / 100.0, 1.0),  # normalize to ~[0, 1]
        recent_accuracy=acc,
        theta_estimate=(theta_est + 3.0) / 6.0,  # normalize [-3,3] -> [0,1]
        mean_confidence=mean_conf / 5.0,  # normalize [0,5] -> [0,1]
        n_mastered_tags=min(mastered / 50.0, 1.0),
        n_gap_tags=min(gaps / 50.0, 1.0),
        accuracy_trend=np.clip(trend, -1, 1),
        interaction_diversity=min(len(tag_stats) / num_tags, 1.0),
    )


def approx_ndcg_loss(scores: torch.Tensor, relevance: torch.Tensor,
                     k: int = 10, temperature: float = 0.1) -> torch.Tensor:
    """
    Differentiable approximation of NDCG loss (ApproxNDCG from Qin et al. 2010).

    Uses softmax over scores as smooth approximation to sorting.

    Args:
        scores: (n_items,) predicted scores
        relevance: (n_items,) binary relevance labels
        k: cutoff for NDCG
        temperature: lower = sharper approximation
    Returns:
        loss: scalar, 1 - approxNDCG (minimize this)
    """
    n = scores.shape[0]
    if relevance.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Approximate ranks via softmax
    # For each item i, its approximate rank = sum_j softmax(s_j - s_i)
    score_diffs = scores.unsqueeze(1) - scores.unsqueeze(0)  # (n, n)
    approx_ranks = torch.sigmoid(score_diffs / temperature).sum(dim=0)  # (n,)

    # DCG with approximate positions
    discount = 1.0 / torch.log2(approx_ranks + 1.0)  # (n,)
    gains = (2.0 ** relevance - 1.0)  # (n,)

    # Only count top-k (soft cutoff)
    top_k_mask = torch.sigmoid((k + 0.5 - approx_ranks) / temperature)
    dcg = (gains * discount * top_k_mask).sum()

    # Ideal DCG
    sorted_rel, _ = torch.sort(relevance, descending=True)
    ideal_discount = 1.0 / torch.log2(torch.arange(1, n + 1, dtype=torch.float32) + 1.0)
    idcg = (sorted_rel[:k] * ideal_discount[:k]).sum()

    if idcg == 0:
        return torch.tensor(0.0, requires_grad=True)

    return 1.0 - dcg / (idcg + 1e-10)


def train_gating(
    train_examples: list[dict],
    n_epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 0.01,
    val_fraction: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> GatingNetwork:
    """
    Train gating network on collected (user_features, agent_signals, ground_truth).

    Args:
        train_examples: list of dicts with keys:
            - 'features': UserContext
            - 'signals': np.ndarray (NUM_TAGS, NUM_AGENTS) — stacked agent signals
            - 'ground_truth': np.ndarray (NUM_TAGS,) — binary relevance
        n_epochs: training epochs
        lr: learning rate
        val_fraction: fraction of examples for validation
        seed: random seed
        verbose: print progress

    Returns:
        Trained GatingNetwork
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_examples))
    n_val = max(1, int(len(indices) * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train_idx = [i for i in indices if i not in val_idx]

    model = GatingNetwork()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(n_epochs):
        # Training
        model.train()
        rng.shuffle(train_idx)
        train_loss = 0.0

        for i in train_idx:
            ex = train_examples[i]
            features = ex['features'].to_tensor().unsqueeze(0)  # (1, 8)
            signals = torch.tensor(ex['signals'], dtype=torch.float32)  # (NUM_TAGS, 5)
            gt = torch.tensor(ex['ground_truth'], dtype=torch.float32)  # (NUM_TAGS,)

            weights = model(features).squeeze(0)  # (5,)
            combined = (signals * weights.unsqueeze(0)).sum(dim=1)  # (NUM_TAGS,)

            loss = approx_ndcg_loss(combined, gt, k=10)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in val_idx:
                ex = train_examples[i]
                features = ex['features'].to_tensor().unsqueeze(0)
                signals = torch.tensor(ex['signals'], dtype=torch.float32)
                gt = torch.tensor(ex['ground_truth'], dtype=torch.float32)
                weights = model(features).squeeze(0)
                combined = (signals * weights.unsqueeze(0)).sum(dim=1)
                val_loss += approx_ndcg_loss(combined, gt, k=10).item()

        avg_val = val_loss / max(len(val_idx), 1)
        avg_train = train_loss / max(len(train_idx), 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Gating epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def analyze_gating_weights(model: GatingNetwork, examples: list[dict],
                           verbose: bool = True) -> dict:
    """Analyze learned gating weights across user subgroups."""
    model.eval()
    agent_names = ["PRED", "IRT", "KG", "CONF", "CLUST"]
    results = {"cold": [], "moderate": [], "warm": [], "all": []}

    with torch.no_grad():
        for ex in examples:
            feat = ex['features']
            weights = model(feat.to_tensor().unsqueeze(0)).squeeze(0).numpy()

            results["all"].append(weights)
            n_raw = feat.n_interactions * 100  # denormalize
            if n_raw < 5:
                results["cold"].append(weights)
            elif n_raw < 50:
                results["moderate"].append(weights)
            else:
                results["warm"].append(weights)

    if verbose:
        print("\n=== Learned Gating Weights by Subgroup ===")
        for group, wlist in results.items():
            if not wlist:
                continue
            mean_w = np.mean(wlist, axis=0)
            print(f"  {group:>10} (n={len(wlist):3d}): " +
                  "  ".join(f"{name}={w:.3f}" for name, w in zip(agent_names, mean_w)))

    return {k: np.mean(v, axis=0) if v else np.zeros(NUM_AGENTS)
            for k, v in results.items()}
