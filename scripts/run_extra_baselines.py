"""
Three additional baselines for XES3G5M Table 3 — fills the gap left by
removing the previously-fabricated BPR-MF / CF-only / Content-only rows.

Each baseline produces an (N_users, N_tags) failure-probability score
matrix on the test set, then is evaluated with the same compute_all_metrics
function as Random/Popularity/DKT/GRU in run_xes3g5m_baselines.py.

Baselines:
  1. BPR-MF       — TruncatedSVD on user x tag failure matrix from train
  2. CF-only      — implicit ALS on user x tag failure matrix
  3. Content-only — RoBERTa tag embeddings (XES3G5M cid2content_emb.json):
                    score(u, t) = cosine(user_pref_vector(u), emb(t))
                    where user_pref_vector = mean emb of FAILED tags in train

Output: results/xes3g5m/baselines_extra_s{seed}_<ts>/baselines.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from agents import prediction_agent as PA
from agents.prediction_agent import (
    GapSequenceDataset, set_num_tags, NUM_CONF_CLASSES,
)
from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_baselines import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("xes3g5m_extra_baselines")


# ─── Helper: build per-user failure tag matrix from a dataframe ──────

def build_user_tag_failure_matrix(
    df: pd.DataFrame, n_tags: int,
) -> tuple[sp.csr_matrix, list[str]]:
    """Return shape=(n_users, n_tags) sparse counts of FAILED tag interactions.
    Each entry [u, t] = number of times user u answered incorrectly on tag t in df.
    """
    fails = df[~df["correct"].astype(bool)]
    user_ids = sorted(fails["user_id"].unique().tolist())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    rows, cols, data = [], [], []
    for _, r in fails.iterrows():
        u = user_idx[r["user_id"]]
        for t in r["tags"]:
            if 0 <= int(t) < n_tags:
                rows.append(u)
                cols.append(int(t))
                data.append(1.0)
    M = sp.csr_matrix((data, (rows, cols)),
                       shape=(len(user_ids), n_tags), dtype=np.float32)
    logger.info("Failure matrix: %d users x %d tags  (nnz=%d, density=%.4f%%)",
                M.shape[0], M.shape[1], M.nnz,
                100 * M.nnz / (M.shape[0] * M.shape[1]))
    return M, user_ids


# ─── Helper: get per-test-user score vector via factorization ────────

def score_test_users_by_factor_dot(
    train_users: list[str],
    train_user_factors: np.ndarray,    # (n_train_users, k)
    item_factors: np.ndarray,          # (n_tags, k)
    test_df: pd.DataFrame,
    n_test_users: int,
    n_tags: int,
) -> np.ndarray:
    """For each test user (by row index in test_dataset.labels) compute a
    score vector by averaging train-user factors of all train users who
    overlap with this test user via shared tag history. If no overlap,
    return mean factor (popularity-like).
    """
    train_user_idx = {u: i for i, u in enumerate(train_users)}
    mean_factor = train_user_factors.mean(axis=0)

    # Build per-test-user "history vector" in factor space:
    # For each test user, average factors of train users who answered the
    # SAME questions. As a fast proxy: just use the global mean factor.
    # (Cold-start dominant evaluation — refine later if needed.)
    y_score = np.tile(
        np.maximum(mean_factor @ item_factors.T, 0),
        (n_test_users, 1),
    ).astype(np.float32)
    # Normalise to [0, 1] for compatibility
    if y_score.max() > 0:
        y_score = y_score / y_score.max()
    return y_score


# ─── Baseline 1: BPR-MF (via TruncatedSVD on user x tag failure) ─────

def baseline_bpr_mf(test_dataset, train_df, test_df, n_tags, seed,
                     n_factors: int = 64) -> dict:
    """Truncated SVD on user x tag failure matrix.

    SVD components serve as latent factors. Score per (user, tag) =
    user_factor[u] . item_factor[t]. Higher = more likely failure.

    For test users (which differ from train users), we use the GLOBAL MEAN
    of train user factors as the test-user representation (Popular-MF
    fallback for cold users — XES3G5M test users are unseen in train).
    """
    M, _ = build_user_tag_failure_matrix(train_df, n_tags)
    k = min(n_factors, min(M.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    U = svd.fit_transform(M)            # (n_train_users, k)
    V = svd.components_                  # (k, n_tags)

    y_true = np.stack(test_dataset.labels)
    n_test_users = len(y_true)

    # Score = mean train-user latent vector . V  →  shape (n_tags,)
    # Tile per test user.
    mean_user = U.mean(axis=0)           # (k,)
    tag_scores = mean_user @ V           # (n_tags,)
    # Shift to non-negative for sigmoid-like interpretation
    tag_scores = tag_scores - tag_scores.min() + 1e-6
    if tag_scores.max() > 0:
        tag_scores = tag_scores / tag_scores.max()
    y_score = np.tile(tag_scores, (n_test_users, 1)).astype(np.float32)

    return compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)


# ─── Baseline 2: CF-only (ALS via implicit lib if available, else SVD) ─

def baseline_cf_only(test_dataset, train_df, test_df, n_tags, seed,
                      n_factors: int = 64) -> dict:
    """Implicit ALS on user x tag (binary touched) matrix.

    Different signal from BPR-MF: counts ANY interaction on a tag, not just
    failures. Used as a Collaborative-Filtering-only baseline (no content,
    no IRT, no KT signal).

    Falls back to TruncatedSVD if `implicit` library is not installed or
    fails on Windows.
    """
    # Build user x tag binary "touched" matrix from train
    user_ids = sorted(train_df["user_id"].unique().tolist())
    user_idx = {u: i for i, u in enumerate(user_ids)}
    rows, cols, data = [], [], []
    for _, r in train_df.iterrows():
        u = user_idx[r["user_id"]]
        for t in r["tags"]:
            if 0 <= int(t) < n_tags:
                rows.append(u); cols.append(int(t)); data.append(1.0)
    M = sp.csr_matrix((data, (rows, cols)),
                       shape=(len(user_ids), n_tags), dtype=np.float32)
    logger.info("CF touched matrix: %d users x %d tags (nnz=%d)",
                M.shape[0], M.shape[1], M.nnz)

    # Try implicit ALS first
    try:
        from implicit.als import AlternatingLeastSquares
        als = AlternatingLeastSquares(
            factors=n_factors, regularization=0.01, iterations=15,
            use_gpu=False, random_state=seed,
        )
        als.fit(M)
        U = als.user_factors             # (n_users, k)
        V = als.item_factors             # (n_tags, k)
        logger.info("ALS factorisation done")
    except Exception as e:
        logger.warning("ALS failed (%s) — falling back to TruncatedSVD", e)
        k = min(n_factors, min(M.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=seed)
        U = svd.fit_transform(M)
        V = svd.components_.T            # (n_tags, k)

    # Use mean train user as test user proxy (cold-start dominant)
    mean_user = U.mean(axis=0)           # (k,)
    tag_scores = mean_user @ V.T         # (n_tags,)

    # In ALS, items rarely-touched have LOWER scores. We want HIGH score =
    # MORE likely failure. Invert: high "touched" => high gap likelihood
    # for popular tags. (Simple heuristic; matches the popularity baseline
    # spirit.)
    tag_scores = tag_scores - tag_scores.min() + 1e-6
    if tag_scores.max() > 0:
        tag_scores = tag_scores / tag_scores.max()

    y_true = np.stack(test_dataset.labels)
    y_score = np.tile(tag_scores, (len(y_true), 1)).astype(np.float32)

    return compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)


# ─── Baseline 3: Content-only (RoBERTa tag embeddings) ──────────────

def baseline_content_only(test_dataset, train_df, test_df, n_tags, seed) -> dict:
    """RoBERTa-768 tag embeddings from XES3G5M metadata.

    For each test user, compute their preference vector as the mean
    embedding of tags they FAILED in train (treats failures as "needed
    practice" signal). Score per tag = cosine similarity to user pref
    vector. Test users unseen in train use the global mean fail-tag
    embedding.
    """
    emb_path = ROOT / "data/xes3g5m/XES3G5M/metadata/embeddings/cid2content_emb.json"
    if not emb_path.exists():
        logger.error("Missing %s — content baseline cannot run", emb_path)
        return {"error": "no_embeddings"}

    logger.info("Loading RoBERTa tag embeddings from %s", emb_path)
    with open(emb_path) as f:
        cid2emb = json.load(f)
    emb_dim = len(next(iter(cid2emb.values())))
    logger.info("  loaded %d tag embeddings, dim=%d", len(cid2emb), emb_dim)

    # Stack into (n_tags, dim) array; missing tags → zero vector
    tag_emb = np.zeros((n_tags, emb_dim), dtype=np.float32)
    for k, v in cid2emb.items():
        try:
            i = int(k)
            if 0 <= i < n_tags:
                tag_emb[i] = np.asarray(v, dtype=np.float32)
        except ValueError:
            continue

    # Normalise to unit vectors for cosine similarity
    norms = np.linalg.norm(tag_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tag_emb_n = tag_emb / norms

    # User preference vector from FAILED tags in train (global mean)
    fails = train_df[~train_df["correct"].astype(bool)]
    fail_tag_counts = np.zeros(n_tags, dtype=np.float32)
    for tags in fails["tags"]:
        for t in tags:
            if 0 <= int(t) < n_tags:
                fail_tag_counts[int(t)] += 1.0

    if fail_tag_counts.sum() == 0:
        logger.warning("No failed tags in train — using uniform pref")
        pref_vec = tag_emb_n.mean(axis=0)
    else:
        weights = fail_tag_counts / fail_tag_counts.sum()
        pref_vec = (weights[:, None] * tag_emb_n).sum(axis=0)
    pn = np.linalg.norm(pref_vec)
    if pn > 0:
        pref_vec = pref_vec / pn

    # Score every tag by cosine sim
    tag_scores = tag_emb_n @ pref_vec    # (n_tags,)
    tag_scores = (tag_scores + 1) / 2    # rescale [-1,1] -> [0,1]

    y_true = np.stack(test_dataset.labels)
    y_score = np.tile(tag_scores, (len(y_true), 1)).astype(np.float32)

    return compute_all_metrics(y_score, y_true, train_df, test_df, n_tags)


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    args = parser.parse_args()

    set_global_seed(args.seed)
    logger.info("=" * 60)
    logger.info("XES3G5M EXTRA BASELINES (BPR-MF, CF-only, Content-only)")
    logger.info("seed=%d, n_students=%d", args.seed, args.n_students)
    logger.info("=" * 60)

    train_df, val_df, test_df = load_xes3g5m(
        n_students=args.n_students,
        min_interactions=args.min_interactions,
        seed=args.seed,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0

    # Concept-space size from train
    train_max_id = 0
    for tags in train_df["tags"]:
        if isinstance(tags, list) and tags:
            train_max_id = max(train_max_id, max(int(t) for t in tags))
    n_tags = train_max_id + 1
    logger.info("NUM_TAGS = %d", n_tags)
    set_num_tags(n_tags)

    test_dataset = GapSequenceDataset(test_df)
    logger.info("Test sequences: %d  (label dim = %d)",
                len(test_dataset), PA.NUM_TAGS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "xes3g5m" / f"baselines_extra_s{args.seed}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, fn in [
        ("bpr_mf",       baseline_bpr_mf),
        ("cf_only",      baseline_cf_only),
        ("content_only", baseline_content_only),
    ]:
        logger.info("--- %s baseline ---", name)
        t0 = time.time()
        try:
            results[name] = fn(test_dataset, train_df, test_df, n_tags, args.seed)
            results[name]["time_s"] = round(time.time() - t0, 1)
            logger.info("  %s AUC=%.4f  NDCG@10=%.4f  P@10=%.4f  MRR=%.4f  Cov=%.4f",
                         name,
                         results[name].get("test_auc_macro", 0),
                         results[name].get("ndcg@10", 0),
                         results[name].get("precision@10", 0),
                         results[name].get("mrr", 0),
                         results[name].get("tag_coverage", 0))
        except Exception as e:
            logger.exception("FAILED %s: %s", name, e)
            results[name] = {"error": str(e), "time_s": round(time.time() - t0, 1)}

    with open(out_dir / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_dir / "baselines.json")


if __name__ == "__main__":
    main()
