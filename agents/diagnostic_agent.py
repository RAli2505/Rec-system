"""
Diagnostic Agent for MARS.

Implements IRT 3PL calibration from EdNet interaction data and
a Computerized Adaptive Testing (CAT) algorithm for efficient
student ability estimation.

IRT 3PL: P(correct | θ) = c + (1-c) / (1 + exp(-a*(θ - b)))

Calibration:
  - b (difficulty): from empirical accuracy, mapped to [-3, +3]
  - a (discrimination): point-biserial correlation with total score
  - c (guessing): 0.25 default (4-choice TOEIC items)

Refined calibration via py-irt 2PL (Pyro VI) with fixed c.

CAT: Fisher Information-based item selection with MLE θ updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid
from scipy.stats import pointbiserialr
import torch

from .base_agent import BaseAgent

# GPU device selection
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("mars.agent.diagnostic")


# ──────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────

@dataclass
class IRTParams:
    """Calibrated IRT parameters for all items."""
    question_ids: list[str]
    a: np.ndarray          # discrimination (n_items,)
    b: np.ndarray          # difficulty (n_items,)
    c: np.ndarray          # guessing (n_items,)
    part_ids: np.ndarray   # part for each item

    def __len__(self) -> int:
        return len(self.question_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": len(self),
            "a_mean": float(self.a.mean()),
            "b_mean": float(self.b.mean()),
            "c_mean": float(self.c.mean()),
        }


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test or assessment."""
    user_id: str
    theta: float               # estimated ability
    se: float                  # standard error of theta
    responses: list[dict]      # per-question details
    n_questions: int = 0
    parts_covered: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "ability": self.theta,
            "se": self.se,
            "n_questions": self.n_questions,
            "parts_covered": self.parts_covered,
            "responses": self.responses,
        }


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────

DEFAULT_GUESSING = 0.25     # 4-choice TOEIC
THETA_RANGE = (-4.0, 4.0)
CAT_MAX_ITEMS = 20
CAT_MIN_ITEMS = 5
CAT_SE_THRESHOLD = 0.3
DIAGNOSTIC_N_ITEMS = 15     # 2 per part + 1 extra
N_PARTS = 7


class DiagnosticAgent(BaseAgent):
    """
    IRT-based diagnostic and adaptive testing agent.

    Calibrates 3PL item parameters from EdNet data and runs
    CAT-based diagnostic / assessment sessions.
    """

    name = "diagnostic"
    REQUIRED_COLUMNS = {
        "calibrate": ["user_id", "question_id", "correct"],
    }

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        # Resolve seed from config or parameter
        _seed = seed if seed is not None else self._config.get("seed", 42)
        from .utils import set_global_seed
        set_global_seed(_seed)

        # Load parameters from config, falling back to module-level defaults
        theta_range = self._config.get("theta_range", list(THETA_RANGE))
        self._theta_range = tuple(theta_range)
        self._guessing_c = self._config.get("guessing_c", DEFAULT_GUESSING)
        self._cat_max_items = self._config.get("cat_max_items", CAT_MAX_ITEMS)
        self._cat_min_items = self._config.get("cat_min_items", CAT_MIN_ITEMS)
        self._cat_se_threshold = self._config.get("cat_se_threshold", CAT_SE_THRESHOLD)
        self._diagnostic_n_items = self._config.get("diagnostic_n_items", DIAGNOSTIC_N_ITEMS)
        self._n_parts = self._config.get("n_parts", N_PARTS)
        self._prior_weight = self._config.get("prior_weight", 0.1)

        self.irt_params: IRTParams | None = None
        self._user_abilities: dict[str, tuple[float, float]] = {}  # user → (θ, SE)
        self._qid_to_idx: dict[str, int] = {}
        self._models_dir = Path("models")
        self._rng = np.random.RandomState(_seed)

    # ──────────────────────────────────────────────────────
    # BaseAgent interface
    # ──────────────────────────────────────────────────────

    def initialize(self, **kwargs: Any) -> None:
        if "response_matrix" in kwargs:
            self.calibrate_irt(
                kwargs["response_matrix"],
                kwargs.get("question_ids"),
                kwargs.get("part_ids"),
            )

    # ──────────────────────────────────────────────────────
    # 1. IRT Calibration
    # ──────────────────────────────────────────────────────

    def calibrate_irt(
        self,
        response_matrix: pd.DataFrame | np.ndarray,
        question_ids: list[str] | None = None,
        part_ids: np.ndarray | None = None,
    ) -> IRTParams:
        """
        Calibrate IRT 3PL parameters from a response matrix.

        Parameters
        ----------
        response_matrix : DataFrame or ndarray
            Shape (n_students, n_items). Values: 1=correct, 0=incorrect, NaN=unseen.
        question_ids : list of str, optional
        part_ids : ndarray, optional

        Returns
        -------
        IRTParams with calibrated a, b, c arrays.
        """
        if isinstance(response_matrix, pd.DataFrame):
            if question_ids is None:
                question_ids = [str(c) for c in response_matrix.columns]
            R = response_matrix.values.astype(float)
        else:
            R = response_matrix.astype(float)

        n_students, n_items = R.shape
        if question_ids is None:
            question_ids = [f"q{i}" for i in range(n_items)]
        if part_ids is None:
            part_ids = np.ones(n_items, dtype=int)

        logger.info("Calibrating IRT 3PL: %d students x %d items", n_students, n_items)

        # ── Difficulty (b): from empirical accuracy ──
        # b = -logit(accuracy), clipped to [-3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            accuracy = np.nanmean(R, axis=0)
        accuracy = np.clip(accuracy, 0.01, 0.99)  # avoid log(0)
        b_raw = -np.log(accuracy / (1 - accuracy))  # logit(1-acc) = -logit(acc)
        b = np.clip(b_raw, -3.0, 3.0).astype(np.float64)

        # ── Discrimination (a): point-biserial correlation ──
        total_scores = np.nansum(R, axis=1)
        a = np.ones(n_items, dtype=np.float64)

        for j in range(n_items):
            col = R[:, j]
            valid = ~np.isnan(col)
            if valid.sum() < 20:
                continue
            try:
                rpb, _ = pointbiserialr(col[valid], total_scores[valid])
                # Scale to typical IRT a range [0.2, 3.0]
                a[j] = np.clip(rpb * 2.5, 0.2, 3.0) if not np.isnan(rpb) else 1.0
            except (ValueError, RuntimeWarning):
                pass

        # ── Guessing (c): fixed default ──
        c = np.full(n_items, self._guessing_c, dtype=np.float64)

        # ── EM refinement: 200 iterations of joint MLE ──
        # Alternating between theta estimates (E-step) and item param updates (M-step)
        # Uses 21-point Gauss-Hermite quadrature over theta ~ N(0,1)
        a, b = self._em_calibrate(R, a, b, c, max_iter=200)

        params = IRTParams(
            question_ids=question_ids,
            a=a, b=b, c=c,
            part_ids=part_ids,
        )
        self.irt_params = params
        self._qid_to_idx = {qid: i for i, qid in enumerate(question_ids)}

        # Save
        self._models_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            self._models_dir / "irt_params.npz",
            a=a, b=b, c=c,
            question_ids=np.array(question_ids),
            part_ids=part_ids,
        )

        logger.info(
            "IRT calibrated (EM): b=[%.2f, %.2f], a=[%.2f, %.2f], c=%.2f",
            b.min(), b.max(), a.min(), a.max(), c.mean(),
        )
        return params

    @staticmethod
    def _em_calibrate(
        R: np.ndarray,
        a_init: np.ndarray,
        b_init: np.ndarray,
        c: np.ndarray,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_quad: int = 21,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        EM calibration for IRT 3PL using Gauss-Hermite quadrature.
        Fully vectorised, runs on GPU (CUDA) if available.

        E-step: compute posterior P(θ | responses) over quadrature points.
        M-step: update a, b via Newton-Raphson (vectorised over all items).

        Parameters
        ----------
        R        : (n_students, n_items) response matrix (NaN = unseen)
        a_init   : initial discrimination values (n_items,)
        b_init   : initial difficulty values (n_items,)
        c        : guessing parameters, fixed (n_items,)
        max_iter : max EM iterations
        tol      : convergence threshold (max change in b)
        n_quad   : number of quadrature points

        Returns
        -------
        (a, b) refined parameter arrays (NumPy, CPU)
        """
        device = _DEVICE
        logger.info("IRT EM using device: %s", device)

        # Gauss-Hermite quadrature points and weights on N(0,1)
        from numpy.polynomial.hermite import hermgauss
        pts, wts = hermgauss(n_quad)
        theta_q_np = pts * np.sqrt(2)        # (n_quad,)
        wts_q_np   = wts / np.sqrt(np.pi)   # (n_quad,)

        # Move everything to GPU tensors (float32 for speed)
        def _t(x):
            return torch.tensor(x, dtype=torch.float32, device=device)

        # Replace NaN with 0 and build mask
        obs_mask_np = ~np.isnan(R)
        R_obs_np    = np.where(obs_mask_np, R, 0.0)

        # Tensors
        obs_mask = _t(obs_mask_np.astype(np.float32))  # (n_s, n_items)
        R_obs    = _t(R_obs_np.astype(np.float32))     # (n_s, n_items)
        theta_q  = _t(theta_q_np.astype(np.float32))  # (n_quad,)
        log_wts  = _t(np.log(wts_q_np).astype(np.float32))  # (n_quad,)

        a = _t(a_init.astype(np.float32))  # (n_items,)
        b = _t(b_init.astype(np.float32))  # (n_items,)
        c_t = _t(c.astype(np.float32))     # (n_items,)

        n_students, n_items = R.shape

        for iteration in range(max_iter):
            b_old = b.clone()

            # ── E-step ──────────────────────────────────────────────────
            # z: (n_items, n_quad)  =  a_j * (θ_q - b_j)
            z = a.unsqueeze(1) * (theta_q.unsqueeze(0) - b.unsqueeze(1))
            # P: (n_items, n_quad)
            P = c_t.unsqueeze(1) + (1 - c_t.unsqueeze(1)) / (1 + torch.exp(-z))
            P = P.clamp(1e-7, 1 - 1e-7)

            log_P  = torch.log(P)      # (n_items, n_quad)
            log_1P = torch.log(1 - P)  # (n_items, n_quad)

            # L[s, q] = Σ_j obs[s,j] * (R[s,j]*logP[j,q] + (1-R[s,j])*log1P[j,q])
            # log_P: (n_items, n_quad), R_obs: (n_s, n_items)
            # (n_s, n_items) @ (n_items, n_quad) → (n_s, n_quad)
            L = R_obs @ log_P + (obs_mask - R_obs) @ log_1P  # (n_s, n_quad)

            # Posterior: w[s,q] ∝ exp(L[s,q]) * wts_q[q]
            log_post = L + log_wts.unsqueeze(0)          # (n_s, n_quad)
            log_post = log_post - log_post.max(dim=1, keepdim=True).values
            post = torch.exp(log_post)
            post = post / post.sum(dim=1, keepdim=True)  # (n_s, n_quad)

            # ── M-step (fully vectorised Newton-Raphson) ─────────────────
            # Expected total & correct per item per quad point
            # f_qj[j, q] = Σ_s post[s,q] * obs[s,j]
            # r_qj[j, q] = Σ_s post[s,q] * R[s,j]
            f_qj = obs_mask.T @ post   # (n_items, n_quad)
            r_qj = R_obs.T @ post      # (n_items, n_quad)

            # Vectorised Newton step for all items simultaneously
            # z_j: (n_items, n_quad)
            z_j = a.unsqueeze(1) * (theta_q.unsqueeze(0) - b.unsqueeze(1))
            P_j = c_t.unsqueeze(1) + (1 - c_t.unsqueeze(1)) / (1 + torch.exp(-z_j))
            P_j = P_j.clamp(1e-7, 1 - 1e-7)

            residual = (P_j * f_qj - r_qj).sum(dim=1)   # (n_items,)
            W_j = (a ** 2).unsqueeze(1) * \
                  ((P_j - c_t.unsqueeze(1)) ** 2) / \
                  ((1 - c_t.unsqueeze(1)) ** 2) * \
                  (1 - P_j) / P_j                        # (n_items, n_quad)
            denom = (W_j * f_qj).sum(dim=1)              # (n_items,)

            # Only update items with enough data
            valid = f_qj.sum(dim=1) >= 1.0               # (n_items,)
            step  = torch.where(
                valid & (denom.abs() > 1e-10),
                residual / denom.clamp(min=1e-10),
                torch.zeros_like(b),
            )
            b = (b + step).clamp(-3.0, 3.0)

            # ── Convergence check ────────────────────────────────────────
            delta = (b - b_old).abs().max().item()
            if iteration % 20 == 0:
                logger.info("IRT EM iter %d/%d  max_Δb=%.6f", iteration + 1, max_iter, delta)
            if delta < tol:
                logger.info("IRT EM converged at iter %d (Δb=%.6f < tol=%.6f)", iteration + 1, delta, tol)
                break

        return a.cpu().numpy().astype(np.float64), b.cpu().numpy().astype(np.float64)

    def calibrate_from_interactions(
        self,
        interactions_df: pd.DataFrame,
        min_answers_per_q: int = 20,
        max_items: int | None = None,
    ) -> IRTParams:
        """
        Build response matrix from raw interactions and calibrate.

        Convenience wrapper around calibrate_irt().
        """
        df = interactions_df.copy()
        df["question_id"] = df["question_id"].astype(str)

        # Filter questions with enough answers
        q_counts = df.groupby("question_id").size()
        valid_qs = q_counts[q_counts >= min_answers_per_q].index.tolist()
        df = df[df["question_id"].isin(valid_qs)]

        if max_items and len(valid_qs) > max_items:
            valid_qs = sorted(valid_qs)[:max_items]
            df = df[df["question_id"].isin(valid_qs)]

        logger.info("Building response matrix: %d questions, %d users",
                     len(valid_qs), df["user_id"].nunique())

        # Pivot to response matrix — keep first encounter per (user, question)
        first_enc = df.groupby(["user_id", "question_id"])["correct"].first().reset_index()
        R = first_enc.pivot(index="user_id", columns="question_id", values="correct")
        R = R.reindex(columns=sorted(R.columns))

        # Get part_ids
        part_map = df.drop_duplicates("question_id").set_index("question_id")["part_id"]
        part_ids = np.array([int(part_map.get(q, 1)) for q in R.columns])

        return self.calibrate_irt(R, question_ids=list(R.columns), part_ids=part_ids)

    # ──────────────────────────────────────────────────────
    # 2. IRT probability & information
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _p_correct(theta: float, a: float, b: float, c: float) -> float:
        """3PL probability of correct response."""
        return c + (1 - c) * expit(a * (theta - b))

    @staticmethod
    def _fisher_info(theta: float, a: float, b: float, c: float) -> float:
        """Fisher information for a single item at theta."""
        p = c + (1 - c) * expit(a * (theta - b))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        q = 1 - p
        num = a**2 * (p - c)**2
        den = (1 - c)**2 * p * q
        return num / den if den > 1e-15 else 0.0

    def item_information_curve(self, item_idx: int, thetas: np.ndarray) -> np.ndarray:
        """Compute information curve for one item across theta values."""
        p = self.irt_params
        return np.array([
            self._fisher_info(t, p.a[item_idx], p.b[item_idx], p.c[item_idx])
            for t in thetas
        ])

    def test_information_function(self, thetas: np.ndarray) -> np.ndarray:
        """Sum of Fisher information across all items for each theta."""
        p = self.irt_params
        tif = np.zeros_like(thetas)
        for j in range(len(p)):
            tif += self.item_information_curve(j, thetas)
        return tif

    # ──────────────────────────────────────────────────────
    # 3. CAT: Adaptive item selection
    # ──────────────────────────────────────────────────────

    def select_next_question(
        self,
        theta: float,
        used_indices: set[int],
        target_tags: list[int] | None = None,
        target_parts: list[int] | None = None,
    ) -> int | None:
        """
        Select the item with maximum Fisher information at current theta.

        Parameters
        ----------
        theta : current ability estimate
        used_indices : items already administered
        target_tags : restrict to items with these tags (optional)
        target_parts : restrict to items in these parts (optional)

        Returns item index or None if exhausted.
        """
        p = self.irt_params
        best_idx = None
        best_info = -1.0

        for j in range(len(p)):
            if j in used_indices:
                continue
            if target_parts and int(p.part_ids[j]) not in target_parts:
                continue

            info = self._fisher_info(theta, p.a[j], p.b[j], p.c[j])
            if info > best_info:
                best_info = info
                best_idx = j

        return best_idx

    def update_theta(
        self,
        responses: list[tuple[int, bool]],
    ) -> tuple[float, float]:
        """
        MLE estimate of theta from a set of responses.

        Parameters
        ----------
        responses : list of (item_index, correct_bool)

        Returns (theta_hat, standard_error)
        """
        p = self.irt_params

        if not responses:
            return 0.0, 999.0

        # All same answer → bound estimate
        all_correct = all(r for _, r in responses)
        all_wrong = all(not r for _, r in responses)
        if all_correct:
            return 3.0, 1.0
        if all_wrong:
            return -3.0, 1.0

        # MLE via negative log-likelihood minimisation
        def neg_log_likelihood(theta: float) -> float:
            ll = 0.0
            for idx, correct in responses:
                prob = self._p_correct(theta, p.a[idx], p.b[idx], p.c[idx])
                prob = np.clip(prob, 1e-10, 1 - 1e-10)
                if correct:
                    ll += np.log(prob)
                else:
                    ll += np.log(1 - prob)
            # Add weak N(0,1) prior for regularisation
            ll -= 0.5 * theta**2 * self._prior_weight
            return -ll

        result = minimize_scalar(
            neg_log_likelihood,
            bounds=self._theta_range,
            method="bounded",
        )
        theta_hat = float(result.x)

        # Standard error = 1 / sqrt(total information)
        total_info = sum(
            self._fisher_info(theta_hat, p.a[idx], p.b[idx], p.c[idx])
            for idx, _ in responses
        )
        se = 1.0 / np.sqrt(total_info) if total_info > 1e-10 else 999.0

        return theta_hat, se

    # ──────────────────────────────────────────────────────
    # 4. Run diagnostic test (cold-start)
    # ──────────────────────────────────────────────────────

    def run_diagnostic(
        self,
        user_id: str,
        simulated_responses: dict[str, bool] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run a 15-question diagnostic covering all 7 TOEIC parts.

        For offline evaluation, pass simulated_responses = {question_id: correct}.
        Otherwise returns a selection plan (item list) for an online system.

        Returns dict compatible with orchestrator cold_start_pipeline.
        """
        self._set_processing()
        p = self.irt_params

        # Plan: 2 questions per part + 1 extra from adaptive selection
        used: set[int] = set()
        responses: list[tuple[int, bool]] = []
        response_details: list[dict] = []
        theta = 0.0
        se = 999.0

        # Phase 1: Cover all parts (2 per part, near current theta)
        for part_id in range(1, self._n_parts + 1):
            for _ in range(2):
                idx = self.select_next_question(theta, used, target_parts=[part_id])
                if idx is None:
                    continue
                used.add(idx)

                qid = p.question_ids[idx]
                correct = self._resolve_response(qid, simulated_responses)
                responses.append((idx, correct))
                response_details.append({
                    "question_id": qid,
                    "correct": correct,
                    "part_id": int(p.part_ids[idx]),
                    "difficulty": float(p.b[idx]),
                })
                theta, se = self.update_theta(responses)

        # Phase 2: 1 extra adaptive question
        idx = self.select_next_question(theta, used)
        if idx is not None:
            used.add(idx)
            qid = p.question_ids[idx]
            correct = self._resolve_response(qid, simulated_responses)
            responses.append((idx, correct))
            response_details.append({
                "question_id": qid,
                "correct": correct,
                "part_id": int(p.part_ids[idx]),
                "difficulty": float(p.b[idx]),
            })
            theta, se = self.update_theta(responses)

        parts_covered = sorted(set(int(p.part_ids[i]) for i, _ in responses))

        # Store ability
        self._user_abilities[user_id] = (theta, se)
        self._set_idle()

        result = DiagnosticResult(
            user_id=user_id,
            theta=round(theta, 4),
            se=round(se, 4),
            responses=response_details,
            n_questions=len(responses),
            parts_covered=parts_covered,
        )

        logger.info(
            "Diagnostic for %s: theta=%.3f, SE=%.3f, %d items, parts=%s",
            user_id, theta, se, len(responses), parts_covered,
        )
        return result.to_dict()

    # ──────────────────────────────────────────────────────
    # 5. Run assessment (after practice)
    # ──────────────────────────────────────────────────────

    def run_assessment(
        self,
        user_id: str,
        interactions: pd.DataFrame | None = None,
        tags: list[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Update ability estimate from a batch of interactions.

        Uses observed responses to re-estimate theta via MLE.
        """
        self._set_processing()
        p = self.irt_params

        if interactions is None or len(interactions) == 0:
            self._set_idle()
            return {"user_id": user_id, "ability": 0.0, "se": 999.0}

        responses: list[tuple[int, bool]] = []
        for _, row in interactions.iterrows():
            qid = str(row.get("question_id", ""))
            # Try both formats
            idx = self._qid_to_idx.get(qid) or self._qid_to_idx.get(f"q{qid}")
            if idx is not None:
                responses.append((idx, bool(row["correct"])))

        if not responses:
            self._set_idle()
            prior = self._user_abilities.get(user_id, (0.0, 999.0))
            return {"user_id": user_id, "ability": prior[0], "se": prior[1]}

        theta, se = self.update_theta(responses)
        self._user_abilities[user_id] = (theta, se)
        self._set_idle()

        logger.info("Assessment for %s: theta=%.3f, SE=%.3f from %d responses",
                     user_id, theta, se, len(responses))

        return {
            "user_id": user_id,
            "ability": round(theta, 4),
            "se": round(se, 4),
            "n_responses": len(responses),
        }

    # ──────────────────────────────────────────────────────
    # 6. Incremental update (continuous pipeline)
    # ──────────────────────────────────────────────────────

    def update_ability(
        self,
        user_id: str,
        interaction: dict[str, Any] | None = None,
        confidence: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Incremental theta update from a single new interaction.

        Uses prior ability + new response to get updated MLE.
        """
        prior_theta, prior_se = self._user_abilities.get(user_id, (0.0, 999.0))

        if interaction is None:
            return {"ability": prior_theta, "se": prior_se}

        p = self.irt_params
        qid = str(interaction.get("question_id", ""))
        idx = self._qid_to_idx.get(qid) or self._qid_to_idx.get(f"q{qid}")

        if idx is None:
            return {"ability": prior_theta, "se": prior_se}

        correct = bool(interaction.get("correct", False))

        # Build response set: stochastic pseudo-response anchor + actual response
        anchor_responses = [(idx, correct)]

        if prior_se < 10:
            p = self.irt_params
            # Find item closest to prior_theta for anchor
            diffs = np.abs(p.b - prior_theta)
            anchor_idx = int(np.argmin(diffs))
            # Stochastic pseudo-response based on IRT probability (not deterministic)
            p_correct = self._p_correct(
                prior_theta, p.a[anchor_idx], p.b[anchor_idx], p.c[anchor_idx]
            )
            anchor_correct = self._rng.random() < p_correct
            anchor_responses.insert(0, (anchor_idx, anchor_correct))

        theta, se = self.update_theta(anchor_responses)

        # Bayesian-like shrinkage: blend MLE estimate with prior
        if prior_se < 10:
            w_prior = 1.0 / (prior_se**2 + 1e-10)
            w_new = 1.0 / (se**2 + 1e-10)
            theta = (w_prior * prior_theta + w_new * theta) / (w_prior + w_new)
            se = 1.0 / np.sqrt(w_prior + w_new)

        self._user_abilities[user_id] = (theta, se)

        return {"ability": round(theta, 4), "se": round(se, 4)}

    # ──────────────────────────────────────────────────────
    # 7. CAT session (full adaptive test)
    # ──────────────────────────────────────────────────────

    def run_cat_session(
        self,
        user_id: str,
        simulated_responses: dict[str, bool] | None = None,
        max_items: int | None = None,
        min_items: int | None = None,
        se_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Full adaptive test session with early stopping.

        Selects items by max Fisher information, stops when SE < threshold.
        """
        if max_items is None:
            max_items = self._cat_max_items
        if min_items is None:
            min_items = self._cat_min_items
        if se_threshold is None:
            se_threshold = self._cat_se_threshold

        p = self.irt_params
        theta = self._user_abilities.get(user_id, (0.0, 999.0))[0]
        used: set[int] = set()
        responses: list[tuple[int, bool]] = []
        response_details: list[dict] = []
        se = 999.0

        for step in range(max_items):
            idx = self.select_next_question(theta, used)
            if idx is None:
                break

            used.add(idx)
            qid = p.question_ids[idx]
            correct = self._resolve_response(qid, simulated_responses)

            responses.append((idx, correct))
            theta, se = self.update_theta(responses)

            response_details.append({
                "question_id": qid,
                "correct": correct,
                "theta_after": round(theta, 4),
                "se_after": round(se, 4),
                "info": round(self._fisher_info(theta, p.a[idx], p.b[idx], p.c[idx]), 4),
            })

            # Early stopping
            if step + 1 >= min_items and se < se_threshold:
                logger.info("CAT early stop at item %d: SE=%.3f < %.3f", step + 1, se, se_threshold)
                break

        self._user_abilities[user_id] = (theta, se)

        return {
            "user_id": user_id,
            "ability": round(theta, 4),
            "se": round(se, 4),
            "n_items": len(responses),
            "responses": response_details,
            "converged": se < se_threshold,
        }

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_response(
        qid: str,
        simulated: dict[str, bool] | None,
    ) -> bool:
        """Get response from simulation dict or return random."""
        if simulated is not None:
            # Try exact match, then with/without q prefix
            if qid in simulated:
                return simulated[qid]
            stripped = qid.lstrip("q")
            if stripped in simulated:
                return simulated[stripped]
            return simulated.get(int(stripped) if stripped.isdigit() else qid, False)
        return bool(np.random.random() > 0.5)

    def get_ability(self, user_id: str) -> tuple[float, float]:
        """Return (theta, SE) for a user."""
        return self._user_abilities.get(user_id, (0.0, 999.0))

    def icc_curve(self, item_idx: int, thetas: np.ndarray) -> np.ndarray:
        """Item Characteristic Curve: P(correct|theta) for one item."""
        p = self.irt_params
        return np.array([
            self._p_correct(t, p.a[item_idx], p.b[item_idx], p.c[item_idx])
            for t in thetas
        ])
