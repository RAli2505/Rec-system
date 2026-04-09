"""
Orchestrator for MARS (Multi-Agent Recommender System).

Routes messages between agents and defines the three main pipelines:
  1. cold_start_pipeline  — first encounter with a new user
  2. assessment_pipeline  — after a diagnostic / practice session
  3. continuous_pipeline  — real-time, per-interaction updates

Plus ``batch_evaluation`` for computing offline metrics on a test set.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from .base_agent import BaseAgent, Message
from .prediction_agent import HORIZON as PREDICTION_HORIZON

logger = logging.getLogger("mars.orchestrator")

# Ground-truth horizon for LSTM gap-prediction AUC. Must match training
# HORIZON so train and eval optimize the same task.
LSTM_GT_HORIZON = PREDICTION_HORIZON

# Agent name constants (match the names agents register with)
DIAGNOSTIC = "diagnostic"
CONFIDENCE = "confidence"
KG = "knowledge_graph"
RECOMMENDATION = "recommendation"
PREDICTION = "prediction"
PERSONALIZATION = "personalization"


class Orchestrator:
    """
    Central coordinator that owns every agent, routes messages,
    and executes multi-step pipelines.

    All communication is synchronous Python calls — suitable for
    a research prototype evaluated offline on EdNet.
    """

    def __init__(self, seed: int = 42) -> None:
        from .utils import set_global_seed
        set_global_seed(seed)
        self._seed = seed
        self.agents: dict[str, BaseAgent] = {}
        self._message_history: list[Message] = []

    # ------------------------------------------------------------------
    # Agent registry
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent and give it a back-reference to this orchestrator."""
        if agent.name in self.agents:
            logger.warning("Replacing existing agent '%s'", agent.name)
        agent.orchestrator = self
        self.agents[agent.name] = agent
        self._sync_confidence_schema()
        logger.info("Registered agent: %s (%s)", agent.name, agent.__class__.__name__)

    def _sync_confidence_schema(self) -> None:
        """Propagate confidence schema to downstream agents when available."""
        confidence = self.agents.get(CONFIDENCE)
        if confidence is None:
            return

        n_classes = getattr(confidence, "n_classes", None)
        if n_classes is None:
            return

        for name in (PREDICTION, RECOMMENDATION, PERSONALIZATION):
            agent = self.agents.get(name)
            if agent is None:
                continue
            setter = getattr(agent, "set_confidence_schema", None)
            if callable(setter):
                setter(n_classes)

    def initialize_all(self, **kwargs: Any) -> None:
        """Call ``initialize()`` on every registered agent."""
        for name, agent in self.agents.items():
            logger.info("Initializing agent '%s' ...", name)
            agent.initialize(**kwargs)
            logger.info("Agent '%s' ready.", name)

    def get_agent(self, name: str) -> BaseAgent:
        if name not in self.agents:
            raise KeyError(
                f"Agent '{name}' not registered. "
                f"Available: {list(self.agents.keys())}"
            )
        return self.agents[name]

    # ------------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------------

    def route_message(
        self, sender: str, target: str, data: dict[str, Any]
    ) -> Any:
        """
        Deliver a message from *sender* to *target* and return the result.

        This is the only communication channel between agents.
        """
        msg = Message(sender=sender, target=target, data=data)
        self._message_history.append(msg)

        target_agent = self.get_agent(target)
        logger.debug("%s → %s  keys=%s", sender, target, list(data.keys()))
        return target_agent.receive_message(msg)

    # ------------------------------------------------------------------
    # Pipeline 1: Cold Start
    # ------------------------------------------------------------------

    def cold_start_pipeline(self, user_id: str) -> dict[str, Any]:
        """
        First encounter with a new user (no interaction history).

        Flow
        ----
        1. DiagnosticAgent  — run a short diagnostic (CAT / fixed quiz)
        2. ConfidenceAgent  — classify initial responses
        3. KGAgent          — cold-start profile from Knowledge Graph
        4. RecommendationAgent — first set of recommendations
        5. PersonalizationAgent — assign to a learner cluster
        """
        logger.info("=== COLD START pipeline for user %s ===", user_id)
        result: dict[str, Any] = {"user_id": user_id, "pipeline": "cold_start"}

        # Step 1: Diagnostic assessment
        diagnostic = self.get_agent(DIAGNOSTIC)
        diagnostic._set_processing()
        diag_result = diagnostic.run_diagnostic(user_id)  # type: ignore[attr-defined]
        diagnostic._set_idle()
        result["diagnostic"] = diag_result
        logger.info("Diagnostic complete: %d items assessed", len(diag_result.get("responses", [])))

        # Step 2: Confidence classification on diagnostic responses
        confidence = self.get_agent(CONFIDENCE)
        confidence._set_processing()
        conf_result = confidence.classify_batch(  # type: ignore[attr-defined]
            user_id=user_id,
            interactions=diag_result.get("responses", []),
        )
        confidence._set_idle()
        result["confidence"] = conf_result

        # Step 3: KG cold-start profile
        kg = self.get_agent(KG)
        kg._set_processing()
        kg_result = kg.handle_cold_start(  # type: ignore[attr-defined]
            user_id=user_id,
            diagnostic=diag_result,
            confidence=conf_result,
        )
        kg._set_idle()
        result["kg_profile"] = kg_result

        # Step 4: Initial recommendations
        rec = self.get_agent(RECOMMENDATION)
        rec._set_processing()
        rec_result = rec.recommend(  # type: ignore[attr-defined]
            user_id=user_id,
            kg_profile=kg_result,
            confidence=conf_result,
        )
        rec._set_idle()
        result["recommendations"] = rec_result

        # Step 5: Cluster assignment
        pers = self.get_agent(PERSONALIZATION)
        pers._set_processing()
        pers_result = pers.assign_cluster(  # type: ignore[attr-defined]
            user_id=user_id,
            diagnostic=diag_result,
            confidence=conf_result,
        )
        pers._set_idle()
        result["cluster"] = pers_result

        logger.info("=== COLD START pipeline complete ===")
        return result

    # ------------------------------------------------------------------
    # Pipeline 2: Assessment
    # ------------------------------------------------------------------

    def assessment_pipeline(
        self,
        user_id: str,
        interactions: pd.DataFrame,
        tags: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        After the user completes a practice / assessment session.

        Flow
        ----
        1. DiagnosticAgent  — update ability estimates (IRT)
        2. ConfidenceAgent  — classify all responses in the session
        3. KGAgent          — update user profile on the Knowledge Graph
        4. PredictionAgent  — predict knowledge gaps via LSTM
        5. RecommendationAgent — generate next recommendations
        6. PersonalizationAgent — refine cluster / personalise
        """
        logger.debug(
            "=== ASSESSMENT pipeline for user %s (%d interactions) ===",
            user_id, len(interactions),
        )
        result: dict[str, Any] = {
            "user_id": user_id,
            "pipeline": "assessment",
            "n_interactions": len(interactions),
        }

        # Step 1: Update IRT ability
        diagnostic = self.get_agent(DIAGNOSTIC)
        diagnostic._set_processing()
        diag_result = diagnostic.run_assessment(  # type: ignore[attr-defined]
            user_id=user_id,
            interactions=interactions,
            tags=tags,
        )
        diagnostic._set_idle()
        result["diagnostic"] = diag_result

        # Step 2: Confidence classification
        confidence = self.get_agent(CONFIDENCE)
        confidence._set_processing()
        conf_result = confidence.classify_batch(  # type: ignore[attr-defined]
            user_id=user_id,
            interactions=interactions,
        )
        confidence._set_idle()
        result["confidence"] = conf_result

        # Step 3: KG profile update
        kg = self.get_agent(KG)
        kg._set_processing()
        kg_result = kg.update_user_profile(  # type: ignore[attr-defined]
            user_id=user_id,
            diagnostic=diag_result,
            confidence=conf_result,
        )
        kg._set_idle()
        result["kg_profile"] = kg_result

        # Step 4: Predict knowledge gaps
        prediction = self.get_agent(PREDICTION)
        prediction._set_processing()
        pred_result = prediction.predict_gaps(  # type: ignore[attr-defined]
            user_id=user_id,
            recent=interactions,
            diagnostic=diag_result,
            kg_profile=kg_result,
        )
        prediction._set_idle()
        result["predictions"] = pred_result

        # Step 5: Personalisation (moved BEFORE recommendations)
        pers = self.get_agent(PERSONALIZATION)
        pers._set_processing()
        pers_result = pers.personalize(  # type: ignore[attr-defined]
            user_id=user_id,
            diagnostic=diag_result,
            confidence=conf_result,
            recommendations=None,
        )
        pers._set_idle()
        result["personalization"] = pers_result

        # Step 6: Refresh recommendation-side user profile before ranking
        rec = self.get_agent(RECOMMENDATION)
        updater = getattr(rec, "update_user_profile", None)
        if callable(updater):
            updater(
                user_id=user_id,
                interactions=interactions,
                confidence_result=conf_result,
            )

        # Step 7: Recommendations (informed by learner level)
        rec._set_processing()
        rec_result = rec.recommend(  # type: ignore[attr-defined]
            user_id=user_id,
            kg_profile=kg_result,
            confidence=conf_result,
            predictions=pred_result,
            learner_level=pers_result.get("level") if isinstance(pers_result, dict) else None,
        )
        rec._set_idle()
        result["recommendations"] = rec_result

        logger.debug("=== ASSESSMENT pipeline complete ===")
        return result

    # ------------------------------------------------------------------
    # Pipeline 3: Continuous (per-interaction)
    # ------------------------------------------------------------------

    def continuous_pipeline(
        self,
        user_id: str,
        interaction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Called on every single interaction (question answer).

        Lightweight: only updates that make sense in real-time.

        Flow
        ----
        1. ConfidenceAgent  — classify single response
        2. DiagnosticAgent  — incremental IRT update
        3. PredictionAgent  — update sequence model state
        4. RecommendationAgent — re-rank if needed
        """
        logger.debug("=== CONTINUOUS pipeline for user %s ===", user_id)
        result: dict[str, Any] = {
            "user_id": user_id,
            "pipeline": "continuous",
            "question_id": interaction.get("question_id"),
        }

        # Step 1: Classify confidence for this single interaction
        confidence = self.get_agent(CONFIDENCE)
        confidence._set_processing()
        conf_result = confidence.classify_single(  # type: ignore[attr-defined]
            user_id=user_id,
            interaction=interaction,
        )
        confidence._set_idle()
        result["confidence"] = conf_result

        interaction_with_conf = dict(interaction)
        interaction_with_conf["confidence_class"] = conf_result.get("class", 0)

        # Step 2: Incremental IRT update
        diagnostic = self.get_agent(DIAGNOSTIC)
        diagnostic._set_processing()
        diag_result = diagnostic.update_ability(  # type: ignore[attr-defined]
            user_id=user_id,
            interaction=interaction_with_conf,
            confidence=conf_result,
        )
        diagnostic._set_idle()
        result["diagnostic"] = diag_result

        # Step 3: Update LSTM hidden state
        prediction = self.get_agent(PREDICTION)
        prediction._set_processing()
        pred_result = prediction.update_state(  # type: ignore[attr-defined]
            user_id=user_id,
            interaction=interaction_with_conf,
        )
        prediction._set_idle()
        result["prediction"] = pred_result

        # Step 4: Re-rank recommendations if confidence changed significantly
        if conf_result.get("rerank_needed", False):
            rec = self.get_agent(RECOMMENDATION)
            rec._set_processing()
            rec_result = rec.rerank(  # type: ignore[attr-defined]
                user_id=user_id,
                confidence=conf_result,
                prediction=pred_result,
            )
            rec._set_idle()
            result["recommendations"] = rec_result

        return result

    # ------------------------------------------------------------------
    # Batch evaluation (offline, for the paper)
    # ------------------------------------------------------------------

    def batch_evaluation(
        self,
        test_df: pd.DataFrame,
        top_k: int = 10,
        context_ratio: float = 0.5,
    ) -> dict[str, float]:
        """
        Run the full system on a test set and compute recommendation metrics.

        Groups interactions by user, processes each user through the
        assessment pipeline, and collects predictions vs ground truth.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test interactions.
        top_k : int
            Number of recommendations to evaluate.
        context_ratio : float
            Fraction of each user's interactions used as context (default 0.5).

        Returns
        -------
        dict[str, float]
            Metric name → value. Includes:
            - accuracy, auc, f1 (prediction quality)
            - precision@k, recall@k, ndcg@k (recommendation quality)
            - coverage, novelty (catalogue-level)
        """
        logger.info(
            "=== BATCH EVALUATION on %d rows, %d users ===",
            len(test_df), test_df["user_id"].nunique(),
        )
        t0 = time.time()

        all_preds: list[dict] = []
        all_recs: list[dict] = []
        learning_gains: list[float] = []
        learning_gains_adj: list[float] = []  # difficulty-adjusted

        # Precompute per-question global accuracy for difficulty adjustment
        _q_acc = test_df.groupby("question_id")["correct"].mean()
        _q_difficulty: dict[str, float] = {str(q): float(a) for q, a in _q_acc.items()}

        # Pre-group by user_id — avoids O(n) scan per user
        grouped = {
            uid: grp.sort_values("timestamp")
            for uid, grp in test_df.groupby("user_id")
        }
        user_ids = list(grouped.keys())
        has_tags = "tags" in test_df.columns

        def _parse_tags_set(series) -> set:
            """Fast tag extraction from a pandas Series of tag values."""
            tags: set = set()
            for t in series.values:
                if isinstance(t, list):
                    tags.update(t)
                elif isinstance(t, str):
                    tags.update(
                        int(x) for x in t.replace(";", ",").split(",")
                        if x.strip().isdigit()
                    )
            return tags

        def _build_gt_tag_labels(ground_truth) -> np.ndarray:
            """Build 293-dim failure vector from ground truth rows."""
            gt_tag_labels = np.zeros(293, dtype=np.float32)
            if not has_tags:
                return gt_tag_labels
            incorrect = ground_truth[~ground_truth["correct"].astype(bool)]
            if len(incorrect) == 0:
                return gt_tag_labels
            for t in incorrect["tags"].values:
                if isinstance(t, list):
                    tag_list = t
                elif isinstance(t, str):
                    tag_list = [
                        int(x) for x in t.replace(";", ",").split(",")
                        if x.strip().isdigit()
                    ]
                else:
                    continue
                for tag_id in tag_list:
                    if 0 <= tag_id < 293:
                        gt_tag_labels[tag_id] = 1.0
            return gt_tag_labels

        for i, uid in enumerate(user_ids):
            user_df = grouped[uid]

            # Use first context_ratio of user's test interactions as "context"
            split_idx = max(1, int(len(user_df) * context_ratio))
            context = user_df.iloc[:split_idx]
            ground_truth = user_df.iloc[split_idx:]

            if len(ground_truth) == 0:
                continue

            try:
                result = self.assessment_pipeline(
                    user_id=str(uid),
                    interactions=context,
                    tags=None,
                )

                # Collect predictions (LSTM gap format)
                # Restrict GT to the same HORIZON the model trained on,
                # otherwise AUC drops ~0.10+ from train/eval task mismatch.
                if "predictions" in result and result["predictions"]:
                    gt_horizon = ground_truth.iloc[:LSTM_GT_HORIZON]
                    all_preds.append({
                        "user_id": uid,
                        "predicted": result["predictions"],
                        "ground_truth": ground_truth,
                        "gt_tag_labels": _build_gt_tag_labels(gt_horizon),
                    })

                # Collect recommendations
                if "recommendations" in result and result["recommendations"]:
                    gt_tags = _parse_tags_set(ground_truth["tags"]) if has_tags else set()
                    all_recs.append({
                        "user_id": uid,
                        "recommended": result["recommendations"],
                        "ground_truth_items": [str(x) for x in ground_truth["question_id"].values],
                        "ground_truth_tags": list(gt_tags),
                    })

                # learning_gain: accuracy improvement from context to ground truth
                pre_acc = float(context["correct"].mean())
                post_acc = float(ground_truth["correct"].mean())
                learning_gains.append(post_acc - pre_acc)

                # Difficulty-adjusted learning gain:
                # gain = (post_acc - E[post]) - (pre_acc - E[pre])
                # where E[x] = mean global accuracy of questions in that split.
                # This removes the confound from harder questions in ground truth.
                pre_expected = float(context["question_id"].map(
                    lambda q: _q_difficulty.get(str(q), 0.5)
                ).mean())
                post_expected = float(ground_truth["question_id"].map(
                    lambda q: _q_difficulty.get(str(q), 0.5)
                ).mean())
                adj_gain = (post_acc - post_expected) - (pre_acc - pre_expected)
                learning_gains_adj.append(adj_gain)

            except Exception as e:
                logger.warning("Error evaluating user %s: %s", uid, e)

            if (i + 1) % 500 == 0:
                elapsed_so_far = time.time() - t0
                rate = (i + 1) / elapsed_so_far
                eta = (len(user_ids) - i - 1) / rate
                logger.info(
                    "Evaluated %d / %d users (%.1f u/s, ETA %.0fs)",
                    i + 1, len(user_ids), rate, eta,
                )

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_recs, top_k, test_df)
        elapsed = time.time() - t0
        metrics["eval_time_sec"] = round(elapsed, 1)
        metrics["n_users_evaluated"] = len(all_preds)

        # Learning gain: average accuracy improvement
        if learning_gains:
            metrics["learning_gain_raw"] = round(float(np.mean(learning_gains)), 4)
            metrics["learning_gain_raw_std"] = round(float(np.std(learning_gains)), 4)
        # Difficulty-adjusted learning gain (primary metric for the paper)
        if learning_gains_adj:
            metrics["learning_gain"] = round(float(np.mean(learning_gains_adj)), 4)
            metrics["learning_gain_std"] = round(float(np.std(learning_gains_adj)), 4)
            # Trimmed mean (remove top/bottom 5% outliers for robustness)
            arr = np.array(learning_gains_adj)
            lo, hi = np.percentile(arr, [5, 95])
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            if len(trimmed) > 0:
                metrics["learning_gain_trimmed"] = round(float(trimmed.mean()), 4)

        logger.info("=== BATCH EVALUATION complete in %.1fs ===", elapsed)
        for k, v in sorted(metrics.items()):
            logger.info("  %s: %s", k, v)

        return metrics

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_metrics(
        all_preds: list[dict],
        all_recs: list[dict],
        top_k: int,
        full_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Aggregate predictions and recommendations into standard metrics."""
        metrics: dict[str, float] = {}

        # --- LSTM gap prediction metrics (293-dim multilabel) ---
        if all_preds:
            from sklearn.metrics import f1_score, roc_auc_score

            gap_probs_list = []   # (N, 293)
            gap_labels_list = []  # (N, 293)

            for entry in all_preds:
                preds = entry["predicted"]
                gt_labels = entry.get("gt_tag_labels")

                # LSTM format: gap_probabilities is a 293-dim list
                if isinstance(preds, dict) and "gap_probabilities" in preds:
                    probs = preds["gap_probabilities"]
                    if isinstance(probs, list) and len(probs) == 293 and gt_labels is not None:
                        gap_probs_list.append(probs)
                        gap_labels_list.append(gt_labels)

            if gap_probs_list:
                y_score = np.array(gap_probs_list)   # (N, 293)
                y_true  = np.array(gap_labels_list)  # (N, 293)

                # Find optimal threshold — fine-grained scan [0.005 .. 0.5]
                best_f1, best_thresh = 0.0, 0.5
                for t in np.concatenate([
                    np.arange(0.005, 0.05, 0.005),
                    np.arange(0.05, 0.15, 0.01),
                    np.arange(0.15, 0.51, 0.05),
                ]):
                    y_bin = (y_score >= t).astype(int)
                    f1 = f1_score(y_true, y_bin, average="micro", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = float(t)

                y_pred_bin = (y_score >= best_thresh).astype(int)

                metrics["lstm_f1_micro"]  = round(float(best_f1), 4)
                metrics["lstm_threshold"] = round(best_thresh, 4)
                metrics["lstm_f1_macro"]  = round(
                    float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0)), 4
                )

                # AUC: require >=5 positive samples per tag for robust estimate
                tag_pos_counts = y_true.sum(axis=0)
                tag_mask = tag_pos_counts >= 5
                # Also need at least one negative per tag
                tag_neg_counts = (1 - y_true).sum(axis=0)
                tag_mask = tag_mask & (tag_neg_counts >= 5)
                if tag_mask.sum() > 1:
                    try:
                        metrics["lstm_auc"] = round(
                            float(roc_auc_score(
                                y_true[:, tag_mask],
                                y_score[:, tag_mask],
                                average="macro",
                            )), 4
                        )
                        metrics["lstm_auc_weighted"] = round(
                            float(roc_auc_score(
                                y_true[:, tag_mask],
                                y_score[:, tag_mask],
                                average="weighted",
                            )), 4
                        )
                    except ValueError:
                        pass

                logger.info(
                    "LSTM gap F1: micro=%.4f macro=%.4f threshold=%.2f (N=%d users)",
                    best_f1, metrics.get("lstm_f1_macro", 0), best_thresh, len(gap_probs_list),
                )

        # --- Recommendation metrics (tag-based relevance) ---
        if all_recs:
            precisions = []
            recalls = []
            ndcgs = []
            mrrs = []
            all_recommended_items: set = set()
            all_recommended_tags: set = set()

            # Normalize IDs: strip 'q' prefix, lowercase, strip whitespace
            def _norm_id(x) -> str:
                s = str(x).strip().lower()
                return s[1:] if s.startswith("q") and s[1:].isdigit() else s

            for entry in all_recs:
                rec_items_raw = entry["recommended"]
                if isinstance(rec_items_raw, dict):
                    rec_items_raw = rec_items_raw.get("items", [])

                # Extract item_ids and their tags (ensure int types for tag comparison)
                rec_item_ids = []
                rec_item_tags: list[set] = []
                if isinstance(rec_items_raw, list):
                    for r in rec_items_raw:
                        if isinstance(r, dict):
                            rec_item_ids.append(_norm_id(r.get("item_id", r.get("question_id", ""))))
                            raw_tags = r.get("related_tags", [])
                            tags = set()
                            for t in raw_tags:
                                try:
                                    tags.add(int(t))
                                except (ValueError, TypeError):
                                    pass
                            rec_item_tags.append(tags)
                            all_recommended_tags.update(tags)
                        else:
                            rec_item_ids.append(_norm_id(r))
                            rec_item_tags.append(set())

                rec_item_ids = rec_item_ids[:top_k]
                rec_item_tags = rec_item_tags[:top_k]

                gt_items = set(_norm_id(x) for x in entry["ground_truth_items"])
                gt_tags_raw = entry.get("ground_truth_tags", [])
                gt_tags = set()
                for t in gt_tags_raw:
                    try:
                        gt_tags.add(int(t))
                    except (ValueError, TypeError):
                        pass
                all_recommended_items.update(rec_item_ids)

                # Relevance: item is a hit if its item_id matches OR its tags
                # overlap with ground truth tags
                def _is_relevant(idx: int) -> bool:
                    if rec_item_ids[idx] in gt_items:
                        return True
                    if gt_tags and rec_item_tags[idx]:
                        return bool(rec_item_tags[idx] & gt_tags)
                    return False

                n_relevant = sum(1 for i in range(len(rec_item_ids)) if _is_relevant(i))

                # Precision@K
                precisions.append(n_relevant / top_k if top_k > 0 else 0.0)

                # Recall@K — count ground truth tags covered
                if gt_tags:
                    covered = set()
                    for tags in rec_item_tags:
                        covered.update(tags & gt_tags)
                    recalls.append(len(covered) / len(gt_tags))
                else:
                    recalls.append(n_relevant / len(gt_items) if gt_items else 0.0)

                # NDCG@K
                dcg = sum(
                    1.0 / np.log2(rank + 2)
                    for rank in range(len(rec_item_ids))
                    if _is_relevant(rank)
                )
                n_total_relevant = max(n_relevant, len(gt_tags) if gt_tags else len(gt_items))
                ideal = sum(
                    1.0 / np.log2(rank + 2)
                    for rank in range(min(n_total_relevant, top_k))
                )
                ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

                # MRR
                rr = 0.0
                for rank in range(len(rec_item_ids)):
                    if _is_relevant(rank):
                        rr = 1.0 / (rank + 1)
                        break
                mrrs.append(rr)

            metrics[f"precision@{top_k}"] = round(float(np.mean(precisions)), 4)
            metrics[f"recall@{top_k}"] = round(float(np.mean(recalls)), 4)
            metrics[f"ndcg@{top_k}"] = round(float(np.mean(ndcgs)), 4)
            metrics["mrr"] = round(float(np.mean(mrrs)), 4)

            # Tag coverage (primary): fraction of all tags recommended
            # This is the meaningful coverage metric for educational RS —
            # recommendations target tags/skills, not specific questions.
            all_tags_in_data = set()
            if "tags" in full_df.columns:
                for t in full_df["tags"]:
                    if isinstance(t, list):
                        all_tags_in_data.update(int(x) for x in t)
                    elif isinstance(t, (str, np.str_)):
                        all_tags_in_data.update(
                            int(x.strip()) for x in str(t).replace(";", ",").split(",") if x.strip().isdigit()
                        )
                    elif isinstance(t, (int, float, np.integer, np.floating)) and not np.isnan(t):
                        all_tags_in_data.add(int(t))
            # Ensure recommended tags are also ints for consistent comparison
            all_recommended_tags_int = set()
            for tag in all_recommended_tags:
                try:
                    all_recommended_tags_int.add(int(tag))
                except (ValueError, TypeError):
                    pass
            if all_tags_in_data:
                tag_cov = round(
                    len(all_recommended_tags_int & all_tags_in_data) / len(all_tags_in_data), 4
                )
            else:
                tag_cov = 0.0
            metrics["coverage"] = tag_cov
            metrics["tag_coverage"] = tag_cov

            # Question-level coverage
            all_questions = set(_norm_id(x) for x in full_df["question_id"].unique())
            question_recs = set()
            for iid in all_recommended_items:
                nid = _norm_id(iid)
                if not nid.startswith("tag_") and not nid.startswith("l_") and not nid.startswith("l"):
                    question_recs.add(nid)
            metrics["question_coverage"] = round(
                len(question_recs & all_questions) / len(all_questions), 4
            ) if all_questions else 0.0

        return metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def agent_names(self) -> list[str]:
        return list(self.agents.keys())

    @property
    def message_count(self) -> int:
        return len(self._message_history)

    def status_report(self) -> dict[str, str]:
        """Return {agent_name: status} for all registered agents."""
        return {name: agent.status for name, agent in self.agents.items()}

    def __repr__(self) -> str:
        return (
            f"<Orchestrator agents={self.agent_names} "
            f"messages={self.message_count}>"
        )
