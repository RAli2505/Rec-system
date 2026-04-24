"""
Aggregate XES3G5M retraining results into the 3 paper tables.

Reads from:
  results/xes3g5m/xes3g5m_full_s{SEED}_*/metrics.json   (5 seeds)
  results/xes3g5m/baselines_s*/baselines.json           (newest)
  results/xes3g5m/ablation_s*/ablation.json             (newest)

Writes:
  results/xes3g5m/tables/table_main_results.csv      (Table 3 source)
  results/xes3g5m/tables/table_seed_stability.csv    (Table 5 source)
  results/xes3g5m/tables/table_ablation.csv          (Table 4 source)

Picks the newest run per seed if multiple exist.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

ROOT = Path(".")
RES = ROOT / "results" / "xes3g5m"
TBL = RES / "tables"
TBL.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]


def newest(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(files[0]) if files else None


# ─── 1. Per-seed MARS metrics (from xes3g5m_full_s{SEED}_*/metrics.json) ──

mars_per_seed: dict[int, dict] = {}
for seed in SEEDS:
    p = newest(str(RES / f"xes3g5m_full_s{seed}_*" / "metrics.json"))
    if p is None:
        print(f"[WARN] No run found for seed {seed}")
        continue
    with open(p) as f:
        m = json.load(f)
    mars_per_seed[seed] = m.get("eval_metrics", {})
    print(f"  seed {seed}: loaded from {p.parent.name}")

if not mars_per_seed:
    raise SystemExit("No MARS runs found — abort aggregation.")

# Build seed_stability table (rows = metrics, columns = Mean/Std/Min/Max/Seeds)
metric_keys = ["lstm_auc", "lstm_auc_weighted", "ndcg@10", "precision@10",
               "recall@10", "mrr", "tag_coverage", "learning_gain"]

stab_rows = []
for k in metric_keys:
    vals = [mars_per_seed[s][k] for s in SEEDS if k in mars_per_seed.get(s, {})]
    if not vals:
        continue
    vals = np.array(vals, dtype=float)
    stab_rows.append({
        "Metric": k,
        "Mean": round(float(vals.mean()), 4),
        "Std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),
        "Min": round(float(vals.min()), 4),
        "Max": round(float(vals.max()), 4),
        "Seeds": " ".join(f"{v:.4f}" for v in vals),
    })

# Add val_auc from agent_metrics
val_aucs = []
for seed in SEEDS:
    p = newest(str(RES / f"xes3g5m_full_s{seed}_*" / "metrics.json"))
    if p:
        with open(p) as f:
            m = json.load(f)
        v = m.get("agent_metrics", {}).get("prediction", {}).get("val_auc")
        if v is not None:
            val_aucs.append(float(v))
if val_aucs:
    arr = np.array(val_aucs)
    stab_rows.append({
        "Metric": "val_auc",
        "Mean": round(float(arr.mean()), 4),
        "Std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "Min": round(float(arr.min()), 4),
        "Max": round(float(arr.max()), 4),
        "Seeds": " ".join(f"{v:.4f}" for v in arr),
    })

pd.DataFrame(stab_rows).to_csv(TBL / "table_seed_stability.csv", index=False)
print(f"\nWrote {TBL / 'table_seed_stability.csv'}  ({len(stab_rows)} rows)")


# ─── 2. Main results: MARS (mean ± std) + baselines (single-seed) ─────

bl_path = newest(str(RES / "baselines_s*" / "baselines.json"))
if bl_path is None:
    raise SystemExit("No baselines.json found — run baselines first.")
with open(bl_path) as f:
    bl = json.load(f)
print(f"  baselines from: {bl_path.parent.name}")

# Merge in extra baselines (BPR-MF, CF-only, Content-only) if available
extra_path = newest(str(RES / "baselines_extra_s*" / "baselines.json"))
if extra_path is not None:
    with open(extra_path) as f:
        extra = json.load(f)
    print(f"  extra baselines from: {extra_path.parent.name}")
    # Don't overwrite existing keys — only add new ones
    for k, v in extra.items():
        if k not in bl:
            bl[k] = v

# Map between baseline keys in baselines.json and Table 3 method labels.
# Order here = column order in the output CSV.
BL_LABEL = {
    "random":       "Random",
    "popularity":   "Popularity",
    "bpr_mf":       "BPR-MF",
    "cf_only":      "CF-only",
    "content_only": "Content-only",
    "dkt_lstm":     "DKT (LSTM)",
    "gru":          "GRU",
}

# Map between baselines.json metric keys and Table 3 metric names
METRIC_MAP = [
    ("AUC-ROC",      "test_auc_macro", "lstm_auc"),
    ("NDCG@10",      "ndcg@10",        "ndcg@10"),
    ("Precision@10", "precision@10",   "precision@10"),
    ("Recall@10",    "recall@10",      "recall@10"),
    ("MRR",          "mrr",            "mrr"),
    ("Coverage",     "tag_coverage",   "tag_coverage"),
]

main_rows = []
for label, bl_key, mars_key in METRIC_MAP:
    row = {"Metric": label}
    # Baselines (single seed → just the value)
    for bl_id, bl_name in BL_LABEL.items():
        v = bl.get(bl_id, {}).get(bl_key)
        row[bl_name] = round(float(v), 4) if v is not None else None
    # MARS — mean ± std across 5 seeds
    vals = [mars_per_seed[s].get(mars_key) for s in SEEDS
            if mars_per_seed.get(s, {}).get(mars_key) is not None]
    vals = np.array(vals, dtype=float)
    if len(vals):
        mean = vals.mean()
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        row["MARS (ours)"] = f"{mean:.4f} +/- {std:.4f}"
    else:
        row["MARS (ours)"] = None
    main_rows.append(row)

pd.DataFrame(main_rows).to_csv(TBL / "table_main_results.csv", index=False)
print(f"Wrote {TBL / 'table_main_results.csv'}")


# ─── 3. Ablation table ────────────────────────────────────────────────

# Look in BOTH ablation_s*/ (vanilla output) AND ablation_reconstructed_s*/
# (built from log when master was killed mid-loop). Newest wins.
_abl_candidates = sorted(
    glob.glob(str(RES / "ablation_s*" / "ablation.json")) +
    glob.glob(str(RES / "ablation_reconstructed_s*" / "ablation.json")),
    key=os.path.getmtime, reverse=True,
)
abl_path = Path(_abl_candidates[0]) if _abl_candidates else None
if abl_path is None:
    print("[WARN] No ablation.json found — skipping table_ablation.csv")
else:
    with open(abl_path) as f:
        abl = json.load(f)
    print(f"  ablation from: {abl_path.parent.name}")

    # Merge in any single-config runs (ablation_one_*) — they take precedence
    # since they were typically launched in parallel after the main loop.
    one_paths = sorted(
        glob.glob(str(RES / "ablation_one_s*" / "ablation_one.json")),
        key=os.path.getmtime, reverse=True,
    )
    for p in one_paths:
        with open(p) as f:
            one = json.load(f)
        for k, v in one.items():
            if k not in abl or os.path.getmtime(p) > abl_path.stat().st_mtime:
                abl[k] = v
                print(f"  merged {k} from: {Path(p).parent.name}")

    abl_metrics = [
        ("AUC-ROC",  "lstm_auc"),
        ("NDCG@10",  "ndcg@10"),
        ("P@10",     "precision@10"),
        ("MRR",      "mrr"),
        ("Coverage", "tag_coverage"),
    ]
    full = abl.get("Full MARS", {})
    abl_rows = []
    for cfg_name, cfg_metrics in abl.items():
        row = {"Configuration": cfg_name}
        for col_label, key in abl_metrics:
            row[col_label] = round(float(cfg_metrics.get(key, 0)), 4)
        # NDCG / MRR deltas vs Full MARS for downstream heatmap convenience
        if cfg_name != "Full MARS":
            row["delta_NDCG"] = round(row["NDCG@10"] - full.get("ndcg@10", 0), 4)
            row["delta_MRR"]  = round(row["MRR"]     - full.get("mrr", 0), 4)
        else:
            row["delta_NDCG"] = 0.0
            row["delta_MRR"]  = 0.0
        abl_rows.append(row)

    pd.DataFrame(abl_rows).to_csv(TBL / "table_ablation.csv", index=False)
    print(f"Wrote {TBL / 'table_ablation.csv'}")


# ─── Summary printout ─────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Aggregation complete.")
print(f"  table_main_results.csv     ({len(main_rows)} metrics)")
print(f"  table_seed_stability.csv   ({len(stab_rows)} metrics)")
if abl_path:
    print(f"  table_ablation.csv         ({len(abl_rows)} configs)")
print("Next: regenerate paper figures/tables with the new CSVs.")
