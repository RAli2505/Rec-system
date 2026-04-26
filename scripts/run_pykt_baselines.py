"""
Canonical SAINT and AKT baselines via pykt-toolkit on XES3G5M
(reviewer item #8, complement to scripts/run_attention_kt_baselines.py).

This script feeds XES3G5M into the canonical pykt implementations of
SAINT (Choi et al., 2020) and AKT (Ghosh et al., 2020) in their
native (concept_id, response) format. Predictions are per-step
probability of correctness on the next question, exactly as the
original papers define the task.

Output is a separate sub-table — these numbers are NOT directly
comparable to the MARS NDCG@10 / MRR / Tag Coverage in main Table 4
because the published task here is per-step correctness prediction,
not multi-label failure ranking. They are reported alongside
because:
  (a) pykt is the de-facto canonical implementation, so the numbers
      can be benchmarked against the literature without
      reimplementation risk;
  (b) per-step AUC-ROC is the metric every KT paper reports, so this
      gives the reader an immediately comparable single number.

For an apples-to-apples comparison against MARS at the multi-label
task, see scripts/run_attention_kt_baselines.py.

Note on SimpleKT
----------------
The pip release pykt-toolkit==0.0.38 does not include SimpleKT.
SimpleKT is therefore reported only by the matched-input script
(scripts/run_attention_kt_baselines.py).

Usage
-----
    python scripts/run_pykt_baselines.py \
        --models SAINT AKT --seeds 42 123 456 789 2024 \
        --epochs 20 --batch_size 64 --patience 5 --seq_len 100

Output
------
results/xes3g5m/pykt_baselines_<ts>/baselines_s{seed}.json
results/xes3g5m/pykt_baselines_<ts>/summary.csv
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from agents.utils import set_global_seed
from data.xes3g5m_loader import load_xes3g5m

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("pykt_baselines")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_MODELS = ["SAINT", "AKT"]


# ─────────────────────────────────────────────────────────────────────
# Dataset adapter — XES3G5M to pykt (concept_id, response, problem_id)
# ─────────────────────────────────────────────────────────────────────

class PyKTSequenceDataset(Dataset):
    """Per-user windows of length seq_len yielding (q, c, r) triples.

    For multi-tag XES3G5M, uses the FIRST tag in each question's tag
    list as the canonical KC (standard practice in pykt's own
    XES3G5M preprocessor; alternative settings are equivalent up to
    a fixed bijection on multi-tag).

    Loss mask: padding positions are 0; r ∈ {0, 1, padding=2}.
    """

    PAD = 0
    PAD_RESP = 2

    def __init__(self, df: pd.DataFrame, seq_len: int = 100):
        self.seq_len = seq_len
        self.q_seqs: list[np.ndarray] = []
        self.c_seqs: list[np.ndarray] = []
        self.r_seqs: list[np.ndarray] = []
        self.mask_seqs: list[np.ndarray] = []

        for uid, grp in df.groupby("user_id"):
            grp = grp.sort_values("timestamp")
            q_list = grp["question_id"].astype(str).tolist()
            tags_col = grp["tags"].tolist()
            correct = grp["correct"].astype(int).tolist()

            # First tag per row as canonical KC; +1 so 0 stays as PAD.
            kcs = []
            qs = []
            for q, t, c in zip(q_list, tags_col, correct):
                if isinstance(t, list) and t:
                    kc = int(t[0]) + 1
                elif isinstance(t, (int, float)):
                    kc = int(t) + 1
                else:
                    continue
                # question id mapped externally — keep raw string for now,
                # convert to int via dictionary in __init__-level fit
                qs.append(q)
                kcs.append(kc)
            r = correct[: len(kcs)]

            # Slice into non-overlapping windows of seq_len
            n = len(kcs)
            step = self.seq_len
            for s in range(0, n, step):
                end = min(s + step, n)
                win_len = end - s
                if win_len < 5:
                    continue
                q_pad = qs[s:end] + ["__PAD__"] * (step - win_len)
                c_pad = kcs[s:end] + [self.PAD] * (step - win_len)
                r_pad = r[s:end] + [self.PAD_RESP] * (step - win_len)
                m_pad = [1] * win_len + [0] * (step - win_len)
                self.q_seqs.append(np.array(q_pad, dtype=object))
                self.c_seqs.append(np.array(c_pad, dtype=np.int64))
                self.r_seqs.append(np.array(r_pad, dtype=np.int64))
                self.mask_seqs.append(np.array(m_pad, dtype=np.int64))

        # Build problem-id table from accumulated q strings
        all_q = set()
        for arr in self.q_seqs:
            all_q.update(arr.tolist())
        all_q.discard("__PAD__")
        self.q2idx = {q: i + 1 for i, q in enumerate(sorted(all_q))}
        self.q2idx["__PAD__"] = 0
        for i, arr in enumerate(self.q_seqs):
            self.q_seqs[i] = np.array([self.q2idx[q] for q in arr.tolist()],
                                       dtype=np.int64)

    @property
    def n_questions(self) -> int:
        return len(self.q2idx)

    @property
    def n_concepts(self) -> int:
        # +1 for PAD column at index 0
        max_kc = 0
        for arr in self.c_seqs:
            if len(arr):
                m = int(arr.max())
                if m > max_kc:
                    max_kc = m
        return max_kc + 1

    def __len__(self) -> int:
        return len(self.q_seqs)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.q_seqs[idx]),
            torch.from_numpy(self.c_seqs[idx]),
            torch.from_numpy(self.r_seqs[idx]),
            torch.from_numpy(self.mask_seqs[idx]),
        )


# ─────────────────────────────────────────────────────────────────────
# Training loop — canonical per-step BCE on next-step correctness
# ─────────────────────────────────────────────────────────────────────

def _build_pykt_model(name: str, n_q: int, n_c: int, seq_len: int,
                      d_model: int = 256, n_heads: int = 8,
                      n_blocks: int = 2, dropout: float = 0.2):
    """Instantiate canonical pykt SAINT / AKT."""
    if name == "SAINT":
        from pykt.models.saint import SAINT
        return SAINT(
            num_q=n_q, num_c=n_c, seq_len=seq_len,
            emb_size=d_model, num_attn_heads=n_heads,
            dropout=dropout, n_blocks=n_blocks, emb_type="qid",
        )
    if name == "AKT":
        from pykt.models.akt import AKT
        return AKT(
            n_question=n_c, n_pid=n_q, d_model=d_model,
            n_blocks=n_blocks, dropout=dropout, num_attn_heads=n_heads,
            emb_type="qid",
        )
    raise ValueError(f"unsupported pykt model: {name}")


def _forward_one(model_name: str, model, q, c, r):
    """Wrap pykt-specific forward signatures into a uniform call."""
    q = q.to(DEVICE)
    c = c.to(DEVICE)
    r = r.to(DEVICE)
    if model_name == "SAINT":
        # SAINT uses (in_ex=q, in_cat=c, in_res=r); response shifted by start token.
        # SAINT eats r WITHOUT the last position, predicts T positions for
        # next-step correctness. We feed r[:, :-1] and compare with r[:, 1:].
        in_ex = q[:, :-1]
        in_cat = c[:, :-1]
        in_res = r[:, :-1]
        preds = model(in_ex, in_cat, in_res)
        target = r[:, 1:]
        mask = (target != PyKTSequenceDataset.PAD_RESP)
        return preds, target, mask
    if model_name == "AKT":
        # AKT canonical: q_data = c (concepts), pid_data = q (problems),
        # target = response. Predicts correctness of NEXT step.
        preds, c_reg = model(c, r, pid_data=q)
        target = r
        mask = (target != PyKTSequenceDataset.PAD_RESP)
        return preds, target, mask, c_reg
    raise ValueError(f"unknown model: {model_name}")


def train_one(name: str, seed: int, train_df, val_df, test_df,
              epochs: int, batch_size: int, patience: int,
              seq_len: int, lr: float = 1e-3) -> dict:
    set_global_seed(seed)
    train_ds = PyKTSequenceDataset(train_df, seq_len=seq_len)
    val_ds   = PyKTSequenceDataset(val_df,   seq_len=seq_len)
    test_ds  = PyKTSequenceDataset(test_df,  seq_len=seq_len)
    if not len(train_ds) or not len(val_ds):
        return {"error": "empty dataset"}

    n_q = max(train_ds.n_questions, val_ds.n_questions, test_ds.n_questions)
    n_c = max(train_ds.n_concepts,  val_ds.n_concepts,  test_ds.n_concepts)
    logger.info("%-6s seed=%d  n_q=%d  n_c=%d  train_seqs=%d",
                name, seed, n_q, n_c, len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = _build_pykt_model(
        name, n_q=n_q, n_c=n_c, seq_len=seq_len,
        d_model=256, n_heads=8, n_blocks=2, dropout=0.2,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%-6s seed=%d  params=%.2fM", name, seed, n_params / 1e6)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc, best_state, best_ep, no_improve = -1.0, None, 0, 0
    from sklearn.metrics import roc_auc_score, accuracy_score
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for q, c, r, _m in train_loader:
            opt.zero_grad()
            out = _forward_one(name, model, q, c, r)
            if name == "AKT":
                preds, target, mask, c_reg = out
                loss = F.binary_cross_entropy(
                    preds[mask], target[mask].float()) + c_reg
            else:
                preds, target, mask = out
                loss = F.binary_cross_entropy(
                    preds[mask], target[mask].float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        # ── validation ──
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for q, c, r, _m in val_loader:
                out = _forward_one(name, model, q, c, r)
                if name == "AKT":
                    preds, target, mask, _ = out
                else:
                    preds, target, mask = out
                ps.append(preds[mask].cpu().numpy())
                ts.append(target[mask].cpu().numpy())
        yp = np.concatenate(ps); yt = np.concatenate(ts)
        try:
            val_auc = float(roc_auc_score(yt, yp))
        except ValueError:
            val_auc = 0.0
        logger.info("  %-6s seed=%d ep=%2d  loss=%.4f  val_auc=%.4f",
                    name, seed, ep, float(np.mean(losses)), val_auc)
        if val_auc > best_val_auc:
            best_val_auc, best_ep, no_improve = val_auc, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  %-6s seed=%d early stop ep=%d best=%.4f",
                            name, seed, ep, best_val_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ── test ──
    ps, ts = [], []
    with torch.no_grad():
        for q, c, r, _m in test_loader:
            out = _forward_one(name, model, q, c, r)
            if name == "AKT":
                preds, target, mask, _ = out
            else:
                preds, target, mask = out
            ps.append(preds[mask].cpu().numpy())
            ts.append(target[mask].cpu().numpy())
    yp_test = np.concatenate(ps); yt_test = np.concatenate(ts)
    test_auc = float(roc_auc_score(yt_test, yp_test)) if len(yt_test) else 0.0
    test_acc = float(accuracy_score(yt_test, (yp_test > 0.5).astype(int)))
    test_brier = float(np.mean((yp_test - yt_test) ** 2))
    return {
        "model": name, "seed": seed,
        "n_params": int(n_params),
        "n_questions": int(n_q),
        "n_concepts": int(n_c),
        "val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_brier": round(test_brier, 4),
        "best_epoch": best_ep,
        "n_test_steps": int(len(yt_test)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["SAINT", "AKT"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / f"results/xes3g5m/pykt_baselines_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("=== PYKT CANONICAL BASELINES — out=%s ===", out_root)
    logger.info("models=%s  seeds=%s", args.models, args.seeds)

    all_rows = []
    for seed in args.seeds:
        train_df, val_df, test_df = load_xes3g5m(
            n_students=args.n_students,
            min_interactions=args.min_interactions, seed=seed,
        )
        seed_out = {}
        for name in args.models:
            logger.info("━━━ %s seed=%d ━━━", name, seed)
            t0 = time.time()
            try:
                m = train_one(
                    name, seed, train_df, val_df, test_df,
                    epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, seq_len=args.seq_len,
                )
                m["runtime_s"] = round(time.time() - t0, 1)
                seed_out[name] = m
                if "error" not in m:
                    all_rows.append(m)
            except Exception as e:
                logger.exception("%s seed=%d failed: %s", name, seed, e)
                seed_out[name] = {"error": str(e)}
        (out_root / f"baselines_s{seed}.json").write_text(
            json.dumps(seed_out, indent=2, default=str), encoding="utf-8"
        )

    if all_rows:
        df = pd.DataFrame(all_rows)
        summary = (df.groupby("model")
                   .agg(n_params=("n_params", "first"),
                        val_auc_mean=("val_auc", "mean"),
                        val_auc_std=("val_auc", "std"),
                        test_auc_mean=("test_auc", "mean"),
                        test_auc_std=("test_auc", "std"),
                        test_acc_mean=("test_accuracy", "mean"),
                        test_acc_std=("test_accuracy", "std"),
                        brier_mean=("test_brier", "mean"),
                        n_seeds=("seed", "count"))
                   .reset_index())
        summary_path = out_root / "summary.csv"
        summary.to_csv(summary_path, index=False)
        (ROOT / "results/xes3g5m/pykt_baselines_summary.csv"
         ).write_bytes(summary_path.read_bytes())
        logger.info("wrote summary\n%s", summary.to_string(index=False))

    logger.info("DONE at %s", datetime.now().isoformat(timespec="seconds"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
