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

logger = logging.getLogger("mars.orchestrator")

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
        logger.info(
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
            diagnostic=diag_result,
            kg_profile=kg_result,
        )
        prediction._set_idle()
        result["predictions"] = pred_result

        # Step 5: Recommendations (Thompson Sampling + LambdaMART)
        rec = self.get_agent(RECOMMENDATION)
        rec._set_processing()
        rec_result = rec.recommend(  # type: ignore[attr-defined]
            user_id=user_id,
            kg_profile=kg_result,
            confidence=conf_result,
            predictions=pred_result,
        )
        rec._set_idle()
        result["recommendations"] = rec_result

        # Step 6: Personalisation
        pers = self.get_agent(PERSONALIZATION)
        pers._set_processing()
        pers_result = pers.personalize(  # type: ignore[attr-defined]
            user_id=user_id,
            diagnostic=diag_result,
            confidence=conf_result,
            recommendations=rec_result,
        )
        pers._set_idle()
        result["personalization"] = pers_result

        logger.info("=== ASSESSMENT pipeline complete ===")
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
        user_ids = test_df["user_id"].unique()

        for i, uid in enumerate(user_ids):
            user_df = test_df[test_df["user_id"] == uid].sort_values("timestamp")

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

                # Collect predictions
                if "predictions" in result and result["predictions"]:
                    all_preds.append({
                        "user_id": uid,
                        "predicted": result["predictions"],
                        "ground_truth": ground_truth,
                    })

                # Collect recommendations
                if "recommendations" in result and result["recommendations"]:
                    all_recs.append({
                        "user_id": uid,
                        "recommended": result["recommendations"],
                        "ground_truth_items": ground_truth["question_id"].tolist(),
                    })

            except Exception as e:
                logger.warning("Error evaluating user %s: %s", uid, e)

            if (i + 1) % 100 == 0:
                logger.info("Evaluated %d / %d users", i + 1, len(user_ids))

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_recs, top_k, test_df)
        elapsed = time.time() - t0
        metrics["eval_time_sec"] = round(elapsed, 1)
        metrics["n_users_evaluated"] = len(all_preds)

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

        # --- Prediction metrics (if agents return correctness predictions) ---
        if all_preds:
            y_true_list = []
            y_pred_list = []
            for entry in all_preds:
                preds = entry["predicted"]
                gt = entry["ground_truth"]
                if isinstance(preds, dict) and "correct_prob" in preds:
                    probs = preds["correct_prob"]
                    truths = gt["correct"].astype(int).tolist()
                    # Align lengths: predictions may cover fewer items than ground truth
                    n = min(len(probs), len(truths))
                    if n > 0:
                        y_pred_list.extend(probs[:n])
                        y_true_list.extend(truths[:n])

            if y_true_list and y_pred_list:
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

                y_true = np.array(y_true_list)
                y_score = np.array(y_pred_list)
                y_pred_bin = (y_score >= 0.5).astype(int)

                metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred_bin)), 4)
                metrics["f1"] = round(float(f1_score(y_true, y_pred_bin, zero_division=0)), 4)
                try:
                    metrics["auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
                except ValueError:
                    metrics["auc"] = 0.0

        # --- Recommendation metrics ---
        if all_recs:
            precisions = []
            recalls = []
            ndcgs = []
            all_recommended_items: set = set()

            for entry in all_recs:
                rec_items = entry["recommended"]
                if isinstance(rec_items, dict):
                    rec_items = rec_items.get("items", [])
                if isinstance(rec_items, list) and len(rec_items) > 0:
                    # Extract question_ids if items are dicts
                    if isinstance(rec_items[0], dict):
                        rec_items = [r.get("question_id", r) for r in rec_items]

                rec_items = rec_items[:top_k]
                gt_items = set(entry["ground_truth_items"])
                all_recommended_items.update(rec_items)

                # Precision@K
                hits = len(set(rec_items) & gt_items)
                precisions.append(hits / top_k if top_k > 0 else 0.0)

                # Recall@K
                recalls.append(hits / len(gt_items) if gt_items else 0.0)

                # NDCG@K
                dcg = sum(
                    1.0 / np.log2(rank + 2)
                    for rank, item in enumerate(rec_items)
                    if item in gt_items
                )
                ideal = sum(
                    1.0 / np.log2(rank + 2)
                    for rank in range(min(len(gt_items), top_k))
                )
                ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

            metrics[f"precision@{top_k}"] = round(float(np.mean(precisions)), 4)
            metrics[f"recall@{top_k}"] = round(float(np.mean(recalls)), 4)
            metrics[f"ndcg@{top_k}"] = round(float(np.mean(ndcgs)), 4)

            # Coverage: fraction of catalogue recommended at least once
            all_items = set(full_df["question_id"].unique())
            metrics["coverage"] = round(
                len(all_recommended_items & all_items) / len(all_items), 4
            ) if all_items else 0.0

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
