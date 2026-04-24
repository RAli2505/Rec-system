"""
Compute per-subgroup NDCG@10 / MRR on XES3G5M using today's trained MARS
prediction model (saved best.pt). Subgroups are by interaction count in
the test user's history (cold / moderate / warm), matching paper §4.6.

Reads:  newest results/xes3g5m/xes3g5m_full_s42_*/best.pt + test split
Writes: results/xes3g5m/subgroup_xes3g5m_s42.json
        results/xes3g5m/figures/fig9_subgroup_analysis.{png,pdf}
"""

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from agents import prediction_agent as PA
from agents.prediction_agent import (
    GapSequenceDataset, create_model, set_num_tags,
    DEVICE, NUM_CONF_CLASSES,
)
from data.xes3g5m_loader import load_xes3g5m

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.plot_style import setup_publication_style, save_figure

setup_publication_style()

SEED = 42

# ─── Load data + checkpoint ──────────────────────────────────────────

train_df, val_df, test_df = load_xes3g5m(n_students=6000, seed=SEED)
for df in [train_df, val_df, test_df]:
    df["confidence_class"] = 0

train_max_id = max(max(int(t) for t in tags) for tags in train_df["tags"]
                    if isinstance(tags, list) and tags)
n_tags = train_max_id + 1
set_num_tags(n_tags)
print(f"NUM_TAGS = {n_tags}")

# Find newest best.pt for seed_42 from today
runs = sorted(
    [p for p in glob.glob(f"results/xes3g5m/xes3g5m_full_s{SEED}_*")
     if "20260423_06" in p or "20260423_07" in p or "20260423_08" in p],
    key=os.path.getmtime, reverse=True,
)
best_pt = None
for r in runs:
    cand = Path(r) / "best.pt"
    if cand.exists():
        best_pt = cand
        break
if best_pt is None:
    sys.exit("No best.pt found for seed_42 in today's runs")
print(f"Loading model from {best_pt}")

state = torch.load(best_pt, map_location=DEVICE)
model = create_model("transformer", num_conf_classes=NUM_CONF_CLASSES).to(DEVICE)
model.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state
                       else state["state_dict"])
model.eval()

# ─── Build test dataset, track which user each sequence belongs to ───

# GapSequenceDataset shuffles user order internally via groupby; rebuild
# in the same order so we can index each sequence back to its user.
test_dataset = GapSequenceDataset(test_df)
print(f"Test sequences: {len(test_dataset)}")

# Per-user interaction count from train_df (proxy for "experience")
train_int_count = train_df.groupby("user_id").size()

# Re-derive (user_id list per sequence) by reproducing GapSequenceDataset
# logic: it groups by user_id and slides windows. We map sequence index
# back to user_id by repeating the user's id for each window it generates.
SEQ_LEN = 100
HORIZON = 20
seq_user_ids = []
test_grouped = test_df.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False)
for uid, grp in test_grouped:
    n = len(grp)
    stride = max(1, HORIZON // 2)
    for _i in range(0, n - SEQ_LEN - HORIZON + 1, stride):
        seq_user_ids.append(uid)
assert len(seq_user_ids) == len(test_dataset), \
    f"Mismatch: {len(seq_user_ids)} ids vs {len(test_dataset)} sequences"

# ─── Forward pass on test set ────────────────────────────────────────

loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
all_score, all_true = [], []
with torch.no_grad():
    for X, y in loader:
        X = X.to(DEVICE)
        p = torch.sigmoid(model(X)).cpu().numpy()
        all_score.append(p)
        all_true.append(y.numpy())
y_score = np.concatenate(all_score)
y_true = np.concatenate(all_true)
print(f"Eval: {y_score.shape}")

# ─── Per-sequence ranking metrics ────────────────────────────────────

ndcg10s, mrrs = [], []
for i in range(len(y_score)):
    gt = set(np.where(y_true[i] > 0)[0])
    if not gt:
        ndcg10s.append(np.nan); mrrs.append(np.nan); continue
    ranked = np.argsort(-y_score[i])[:10].tolist()
    dcg = sum(1.0 / np.log2(r + 2) for r, t in enumerate(ranked) if t in gt)
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), 10)))
    ndcg10s.append(dcg / ideal if ideal > 0 else 0.0)
    rr = 0.0
    for r, t in enumerate(ranked):
        if t in gt:
            rr = 1.0 / (r + 1); break
    mrrs.append(rr)

# ─── Group by user's TRAIN interaction count (cold/moderate/warm) ────

per_seq = pd.DataFrame({
    "user_id": seq_user_ids,
    "ndcg10": ndcg10s,
    "mrr": mrrs,
})
# For test users (not in train), use TEST interaction count as the
# "experience" proxy — counts what the model has seen so far
test_int_count = test_df.groupby("user_id").size()
per_seq["int_count"] = per_seq["user_id"].map(test_int_count).fillna(0)

per_seq = per_seq.dropna(subset=["ndcg10"])

# XES3G5M filters min_interactions=20 — every test user has plenty of
# history. Partition by tertiles of TOTAL test-user interaction count
# instead of fixed cutoffs (paper §4.6's cold/moderate/warm buckets
# would lump everyone into "warm").
test_users = per_seq["user_id"].unique()
counts = test_int_count.reindex(test_users).fillna(0).values
q33, q66 = np.percentile(counts, [33, 66])
print(f"Tertiles of test-user interaction count: q33={q33:.0f}, q66={q66:.0f}")

def bucket(n):
    if n <= q33:  return f"low ({int(counts.min())}-{int(q33)})"
    if n <= q66:  return f"mid ({int(q33)+1}-{int(q66)})"
    return f"high ({int(q66)+1}+)"

per_seq["bucket"] = per_seq["int_count"].apply(bucket)

summary = per_seq.groupby("bucket").agg(
    n_sequences=("user_id", "size"),
    n_users=("user_id", "nunique"),
    ndcg10_mean=("ndcg10", "mean"),
    ndcg10_std=("ndcg10", "std"),
    mrr_mean=("mrr", "mean"),
    mrr_std=("mrr", "std"),
).reset_index()
print("\n=== Subgroup summary ===")
print(summary.to_string(index=False))

# Order by bucket label (low < mid < high after sorting alphabetically)
def _order_key(s: str) -> int:
    if s.startswith("low"):  return 0
    if s.startswith("mid"):  return 1
    return 2
summary = summary.iloc[summary["bucket"].map(_order_key).argsort()].reset_index(drop=True)

# ─── Save JSON ───────────────────────────────────────────────────────

out_json = ROOT / "results/xes3g5m/subgroup_xes3g5m_s42.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(
    {row["bucket"]: {
        "n_sequences": int(row["n_sequences"]),
        "n_users":     int(row["n_users"]),
        "ndcg10":      float(row["ndcg10_mean"]),
        "ndcg10_std":  float(row["ndcg10_std"]),
        "mrr":         float(row["mrr_mean"]),
        "mrr_std":     float(row["mrr_std"]),
    } for _, row in summary.iterrows()},
    indent=2,
), encoding="utf-8")
print(f"\nWrote {out_json}")

# ─── Plot fig9 ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.6, 3.2))
labels = [f"{r['bucket']}\n(n={r['n_users']:,} users)" for _, r in summary.iterrows()]
ndcg = summary["ndcg10_mean"].values
err  = summary["ndcg10_std"].values

bars = ax.bar(range(len(labels)), ndcg, yerr=err, capsize=4,
              color=["#E76F51", "#F4A261", "#2A9D8F"],
              edgecolor="black", linewidth=0.8)
for bar, v, e in zip(bars, ndcg, err):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + e + 0.01,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("NDCG@10")
ax.set_xlabel("Test-user interaction count (tertile of experience)")
ax.set_title("MARS NDCG@10 by user experience on XES3G5M (n=899 test users)",
             fontsize=10)
ax.set_ylim(0, max(ndcg + err) * 1.18)
ax.grid(axis="y", linewidth=0.5, alpha=0.4)

fig.tight_layout()
save_figure(fig, "fig9_subgroup_analysis",
             results_dir="results/xes3g5m/figures")
plt.close(fig)
print("Updated fig9_subgroup_analysis.{png,pdf}")
