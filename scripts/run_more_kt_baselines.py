"""
Extra KT baselines for MARS Table 4: SAINT+, DKVMN, LPKT, IEKT, GKT
(reviewer item #8 — full coverage of attention/memory/graph KT families).

Per-(model, seed) JSON is written IMMEDIATELY after each training
finishes — so a crash in model X seed Y cannot lose results from
already-completed pairs.

Models (all from pykt-toolkit 0.0.38):
  - SAINT+   (Shin et al., LAK 2021) — SAINT with temporal features
  - DKVMN    (Zhang et al., WWW 2017) — Dynamic key-value memory
  - LPKT     (Shen et al., KDD 2021) — Learning process consistent KT
  - IEKT     (Long et al., SIGIR 2021) — Individual cognition + acquisition
  - GKT      (Nakagawa et al., WI 2019) — Graph-based KT

DTransformer (Yin et al., WWW 2023) is not in pykt 0.0.38 and is
explicitly deferred with a footnote in the camera-ready.

Usage
-----
    python scripts/run_more_kt_baselines.py \
        --models SAINTPlus DKVMN LPKT IEKT GKT \
        --seeds 42 123 456 789 2024 \
        --epochs 20 --batch_size 64 --patience 5

Output
------
results/xes3g5m/more_kt_baselines_<ts>/{model}_seed{s}.json   ← per pair
results/xes3g5m/more_kt_baselines_<ts>/summary.csv            ← at the end
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
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
logger = logging.getLogger("more_kt_baselines")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_MODELS = ["SAINTPlus", "DKVMN", "IEKT", "GKT"]
# LPKT excluded: requires q_matrix (concept-question incidence) and real
# elapsed-time data; XES3G5M's synthesised elapsed_time is degenerate
# (constant 15 s after fallback), making LPKT's time module non-informative.
# Deferred to camera-ready with a footnote.


# ─────────────────────────────────────────────────────────────────────
# Dataset adapter (with shared q2idx + optional time data for LPKT)
# ─────────────────────────────────────────────────────────────────────

class PyKTDataset(Dataset):
    """Per-user windows yielding (q_idx, c, r, mask) and optional
    (interval_time, answer_time) for LPKT.

    Shared q2idx across train/val/test prevents OOR embedding lookup
    after model is sized on train.
    """

    PAD = 0
    PAD_RESP = 2

    def __init__(self, df: pd.DataFrame, seq_len: int = 100,
                 q2idx: dict | None = None,
                 build_time: bool = False):
        self.seq_len = seq_len
        self._fixed_q2idx = q2idx
        self.build_time = build_time

        self.q_seqs: list = []
        self.c_seqs: list = []
        self.r_seqs: list = []
        self.mask_seqs: list = []
        self.it_seqs: list = []
        self.at_seqs: list = []

        for uid, grp in df.groupby("user_id"):
            grp = grp.sort_values("timestamp")
            q_list = grp["question_id"].astype(str).tolist()
            tags_col = grp["tags"].tolist()
            correct = grp["correct"].astype(int).tolist()
            ts = grp["timestamp"].astype(np.int64).tolist() if "timestamp" in grp.columns else None

            kcs, qs = [], []
            for q, t in zip(q_list, tags_col):
                if isinstance(t, list) and t:
                    kc = int(t[0]) + 1
                elif isinstance(t, (int, float)) and not pd.isna(t):
                    kc = int(t) + 1
                else:
                    continue
                qs.append(q); kcs.append(kc)
            r = correct[: len(kcs)]

            # Interval time: difference to previous timestamp, log-binned
            if build_time and ts is not None:
                dt = np.diff(np.array(ts[: len(kcs)], dtype=np.int64),
                              prepend=ts[0] if ts else 0)
                # bin into discrete buckets (LPKT expects integer ids)
                it_buckets = np.clip(
                    np.log1p(np.maximum(dt / 1000.0, 0)).astype(np.int64),
                    0, 199,
                ).tolist()
                # Answer time: synthetic — XES3G5M elapsed_time is degenerate
                # (constant 15 s after synthesis); LPKT requires non-zero ids,
                # use elapsed bucket if present
                at_buckets = [10] * len(kcs)
            else:
                it_buckets = [0] * len(kcs)
                at_buckets = [0] * len(kcs)

            n = len(kcs)
            for s in range(0, n, seq_len):
                end = min(s + seq_len, n)
                if end - s < 5:
                    continue
                pad = seq_len - (end - s)
                self.q_seqs.append(qs[s:end] + ["__PAD__"] * pad)
                self.c_seqs.append(np.array(kcs[s:end] + [self.PAD] * pad,
                                            dtype=np.int64))
                self.r_seqs.append(np.array(r[s:end] + [self.PAD_RESP] * pad,
                                            dtype=np.int64))
                self.mask_seqs.append(np.array([1] * (end - s) + [0] * pad,
                                                dtype=np.int64))
                self.it_seqs.append(np.array(it_buckets[s:end] + [0] * pad,
                                              dtype=np.int64))
                self.at_seqs.append(np.array(at_buckets[s:end] + [0] * pad,
                                              dtype=np.int64))

        # Build / inherit q2idx
        if self._fixed_q2idx is None:
            all_q = set()
            for arr in self.q_seqs:
                all_q.update(arr)
            all_q.discard("__PAD__")
            self.q2idx = {q: i + 1 for i, q in enumerate(sorted(all_q))}
            self.q2idx["__PAD__"] = 0
        else:
            self.q2idx = self._fixed_q2idx
        self.q_seqs = [
            np.array([self.q2idx.get(q, 0) for q in arr], dtype=np.int64)
            for arr in self.q_seqs
        ]

    @property
    def n_questions(self) -> int:
        return len(self.q2idx)

    @property
    def n_concepts(self) -> int:
        return max((int(arr.max()) for arr in self.c_seqs if len(arr)),
                   default=0) + 1

    @property
    def n_intervals(self) -> int:
        return max((int(arr.max()) for arr in self.it_seqs if len(arr)),
                   default=0) + 1

    def __len__(self) -> int:
        return len(self.q_seqs)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.q_seqs[idx]),
            torch.from_numpy(self.c_seqs[idx]),
            torch.from_numpy(self.r_seqs[idx]),
            torch.from_numpy(self.mask_seqs[idx]),
            torch.from_numpy(self.it_seqs[idx]),
            torch.from_numpy(self.at_seqs[idx]),
        )


# ─────────────────────────────────────────────────────────────────────
# Per-model adapter — handles different pykt forward signatures
# ─────────────────────────────────────────────────────────────────────

def build_pykt_model(name: str, n_q: int, n_c: int, seq_len: int,
                     n_intervals: int = 200,
                     d_model: int = 256, n_heads: int = 8,
                     n_blocks: int = 2, dropout: float = 0.2):
    if name == "SAINTPlus":
        from pykt.models.saint_plus_plus import SAINT
        return SAINT(num_q=n_q, num_c=n_c, seq_len=seq_len,
                     emb_size=d_model, num_attn_heads=n_heads,
                     dropout=dropout, n_blocks=n_blocks, emb_type="qid")
    if name == "DKVMN":
        from pykt.models.dkvmn import DKVMN
        return DKVMN(num_c=n_c, dim_s=d_model, size_m=50,
                     dropout=dropout, emb_type="qid")
    if name == "LPKT":
        from pykt.models.lpkt import LPKT
        return LPKT(n_at=512, n_it=n_intervals, n_exercise=n_q,
                    n_question=n_c, d_a=d_model // 4,
                    d_e=d_model // 2, d_k=d_model // 2,
                    dropout=dropout, emb_type="qid", use_time=True)
    if name == "IEKT":
        from pykt.models.iekt import IEKT
        return IEKT(num_q=n_q, num_c=n_c, emb_size=d_model,
                    max_concepts=1, n_layer=1, dropout=dropout,
                    emb_type="qid", device=str(DEVICE))
    if name == "GKT":
        from pykt.models.gkt import GKT
        return GKT(num_c=n_c, hidden_dim=d_model, emb_size=d_model,
                   graph_type="dense", dropout=dropout, emb_type="qid")
    raise ValueError(name)


def model_forward(name: str, model, batch):
    """Returns (preds, target, mask). Different models predict different
    things, but all reduce to per-step P(correct) on the next step.

    CRITICAL FIX: many pykt models compute combined indices like
    q + n*target (DKVMN, GKT) or q + n_question*target (AKT) inside
    embedding lookups. If target contains PAD_RESP=2, that index
    exceeds the embedding table size (2n+1) and triggers CUDA OOR
    "vectorized gather kernel index out of bounds". We pass a clamped
    response (PAD→0) into the model and keep the original PAD-aware
    mask for the loss.
    """
    q, c, r, mask, it, at = [x.to(DEVICE) for x in batch]
    r_safe = torch.where(
        r == PyKTDataset.PAD_RESP, torch.zeros_like(r), r,
    )

    if name == "SAINTPlus":
        in_ex = q
        in_cat = c
        in_res = r_safe[:, :-1]
        preds = model(in_ex, in_cat, in_res)        # (B, T)
        return preds[:, 1:], r[:, 1:], (r[:, 1:] != PyKTDataset.PAD_RESP)

    if name == "DKVMN":
        preds = model(c, r_safe)
        if isinstance(preds, tuple):
            preds = preds[0]
        return preds, r, (r != PyKTDataset.PAD_RESP)

    if name == "LPKT":
        preds = model(q, r_safe, it, at)
        if isinstance(preds, tuple):
            preds = preds[0]
        return preds, r, (r != PyKTDataset.PAD_RESP)

    if name == "IEKT":
        out = model(q, c, r_safe)
        preds = out[0] if isinstance(out, tuple) else out
        return preds, r, (r != PyKTDataset.PAD_RESP)

    if name == "GKT":
        out = model(c, r_safe)
        preds = out[0] if isinstance(out, tuple) else out
        if preds.shape[1] == r.shape[1] - 1:
            return preds, r[:, 1:], (r[:, 1:] != PyKTDataset.PAD_RESP)
        return preds, r, (r != PyKTDataset.PAD_RESP)

    raise ValueError(name)


def _params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ─────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────

def train_eval(name: str, seed: int, train_df, val_df, test_df,
               epochs: int, batch_size: int, patience: int,
               seq_len: int, lr: float = 1e-3) -> dict:
    set_global_seed(seed)
    needs_time = name == "LPKT"
    train_ds = PyKTDataset(train_df, seq_len=seq_len, build_time=needs_time)
    val_ds   = PyKTDataset(val_df,   seq_len=seq_len, q2idx=train_ds.q2idx,
                            build_time=needs_time)
    test_ds  = PyKTDataset(test_df,  seq_len=seq_len, q2idx=train_ds.q2idx,
                            build_time=needs_time)
    if not len(train_ds) or not len(val_ds):
        return {"error": "empty dataset"}

    n_q = train_ds.n_questions
    n_c = max(train_ds.n_concepts, val_ds.n_concepts, test_ds.n_concepts)
    n_intervals = max(
        train_ds.n_intervals, val_ds.n_intervals, test_ds.n_intervals
    ) if needs_time else 200

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = build_pykt_model(name, n_q=n_q, n_c=n_c, seq_len=seq_len,
                             n_intervals=n_intervals).to(DEVICE)
    n_params = _params(model)
    logger.info("%-10s seed=%d params=%.2fM  n_q=%d n_c=%d",
                name, seed, n_params / 1e6, n_q, n_c)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    from sklearn.metrics import roc_auc_score, accuracy_score

    best_val_auc, best_state, best_ep, no_improve = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad()
            try:
                preds, target, mask = model_forward(name, model, batch)
            except Exception as e:
                raise RuntimeError(f"forward failed for {name}: {e}") from e
            target_f = target.float()
            target_f = torch.where(mask, target_f, torch.zeros_like(target_f))
            preds_safe = torch.where(mask, preds.clamp(1e-6, 1-1e-6),
                                       0.5 * torch.ones_like(preds))
            loss = F.binary_cross_entropy(preds_safe[mask], target_f[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds, target, mask = model_forward(name, model, batch)
                ps.append(preds[mask].cpu().numpy())
                ts.append(target[mask].cpu().numpy())
        yp, yt = np.concatenate(ps), np.concatenate(ts)
        try:
            val_auc = float(roc_auc_score(yt, yp))
        except ValueError:
            val_auc = 0.0
        logger.info("  %-10s seed=%d ep=%2d  loss=%.4f  val_auc=%.4f",
                    name, seed, ep, float(np.mean(losses)), val_auc)
        if val_auc > best_val_auc:
            best_val_auc, best_ep, no_improve = val_auc, ep, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  %-10s seed=%d early stop ep=%d best=%.4f",
                            name, seed, ep, best_val_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ps, ts = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds, target, mask = model_forward(name, model, batch)
            ps.append(preds[mask].cpu().numpy())
            ts.append(target[mask].cpu().numpy())
    yp_test, yt_test = np.concatenate(ps), np.concatenate(ts)
    test_auc = float(roc_auc_score(yt_test, yp_test)) if len(yt_test) else 0.0
    test_acc = float(accuracy_score(yt_test, (yp_test > 0.5).astype(int)))
    test_brier = float(np.mean((yp_test - yt_test) ** 2))
    return {
        "model": name, "seed": seed,
        "n_params": int(n_params), "n_questions": int(n_q),
        "n_concepts": int(n_c),
        "val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_brier": round(test_brier, 4),
        "best_epoch": best_ep,
        "n_test_steps": int(len(yt_test)),
    }


def save_immediate(out_root: Path, model: str, seed: int, payload: dict):
    """Per-(model, seed) immediate write — survives any subsequent crash."""
    safe = model.replace("+", "_plus").replace(" ", "_")
    fname = out_root / f"{safe}_seed{seed}.json"
    fname.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("→ saved %s", fname.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_students", type=int, default=6000)
    parser.add_argument("--min_interactions", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="If a JSON for (model, seed) already exists in any prior "
             "more_kt_baselines_* directory or the current out_root, "
             "skip retraining that pair.",
    )
    parser.add_argument(
        "--reuse-dir", default=None,
        help="Reuse this existing more_kt_baselines_<ts> directory "
             "instead of creating a new one (combine with --skip-existing).",
    )
    args = parser.parse_args()

    if args.reuse_dir:
        out_root = ROOT / f"results/xes3g5m/{args.reuse_dir}"
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info("Reusing existing dir %s", out_root)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ROOT / f"results/xes3g5m/more_kt_baselines_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)
    # stable pointer
    (ROOT / "results/xes3g5m/more_kt_baselines_latest.json").write_text(
        json.dumps({"latest_dir": out_root.name}), encoding="utf-8")
    logger.info("=== EXTRA KT BASELINES — out=%s ===", out_root)
    logger.info("models=%s  seeds=%s", args.models, args.seeds)

    # Build set of completed (model, seed) pairs from existing JSON files
    # under EVERY prior more_kt_baselines_* dir + the current out_root.
    # An entry is "completed" only if the saved JSON has no "error" key,
    # so failed runs from a previous attempt will be retried.
    done_pairs: set[tuple[str, int]] = set()
    if args.skip_existing:
        from glob import glob
        candidates = list((ROOT / "results/xes3g5m").glob("more_kt_baselines_*"))
        for d in candidates + [out_root]:
            for jf in d.glob("*_seed*.json"):
                try:
                    payload = json.loads(jf.read_text())
                    if "error" not in payload:
                        m = payload.get("model")
                        s = payload.get("seed")
                        if m is not None and s is not None:
                            done_pairs.add((m, int(s)))
                except Exception:
                    pass
        logger.info("skip-existing: %d pairs already complete", len(done_pairs))

    all_rows = []
    for seed in args.seeds:
        # Lazy-load data only if any model still needs this seed
        need_seed = any((name, seed) not in done_pairs for name in args.models)
        if not need_seed and args.skip_existing:
            logger.info("seed=%d: all models complete, skipping data load", seed)
            continue
        train_df, val_df, test_df = load_xes3g5m(
            n_students=args.n_students,
            min_interactions=args.min_interactions, seed=seed,
        )
        for name in args.models:
            if (name, seed) in done_pairs:
                logger.info("━━━ %s seed=%d  SKIPPED (existing JSON) ━━━",
                            name, seed)
                continue
            logger.info("━━━ %s seed=%d ━━━", name, seed)
            t0 = time.time()
            try:
                m = train_eval(
                    name, seed, train_df, val_df, test_df,
                    epochs=args.epochs, batch_size=args.batch_size,
                    patience=args.patience, seq_len=args.seq_len,
                )
                m["runtime_s"] = round(time.time() - t0, 1)
                save_immediate(out_root, name, seed, m)
                if "error" not in m:
                    all_rows.append(m)
            except Exception as e:
                tb = traceback.format_exc()
                logger.exception("%s seed=%d failed: %s", name, seed, e)
                save_immediate(out_root, name, seed, {
                    "model": name, "seed": seed,
                    "error": str(e), "traceback": tb,
                    "runtime_s": round(time.time() - t0, 1),
                })

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
        logger.info("wrote summary\n%s", summary.to_string(index=False))

    logger.info("DONE at %s", datetime.now().isoformat(timespec="seconds"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
