"""
Generate two separate training-curve figures from today's seed_42 history.json
to replace the combined Fig. 6 (training curves: loss + AUC together).

Outputs (saved to BOTH results/xes3g5m/figures/ and sn-article-template/):
  fig_training_loss.{png,pdf}  — train vs val loss per epoch
  fig_training_auc.{png,pdf}   — val_auc per epoch
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import setup_publication_style, save_figure

setup_publication_style()

# Newest seed_42 run from today
import glob
runs = sorted(
    [p for p in glob.glob("results/xes3g5m/xes3g5m_full_s42_*")
     if "20260423_06" in p or "20260423_07" in p or "20260423_08" in p],
    key=os.path.getmtime, reverse=True,
)
hist_path = None
for r in runs:
    p = Path(r) / "history.json"
    if p.exists():
        hist_path = p
        break
if hist_path is None:
    sys.exit("No history.json found for seed_42 today")

print(f"Loading {hist_path}")
hist = json.loads(hist_path.read_text())
n = len(hist["train_loss"])
ep = np.arange(1, n + 1)
print(f"  epochs={n}")

best_ep = int(np.argmax(hist["val_auc"])) + 1
best_auc = float(np.max(hist["val_auc"]))
print(f"  best_epoch={best_ep}, val_auc={best_auc:.4f}")

# ─── Fig 1: Loss curves ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(ep, hist["train_loss"], "o-", color="#0173B2",
        linewidth=1.6, markersize=5, label="Train loss")
ax.plot(ep, hist["val_loss"], "s--", color="#DE8F05",
        linewidth=1.6, markersize=5, label="Validation loss")
ax.axvline(best_ep, color="#999", linestyle=":", linewidth=1.0)
ax.text(best_ep, ax.get_ylim()[1] * 0.95, f"best ep {best_ep}",
        ha="center", va="top", fontsize=8, color="#555")

ax.set_xlabel("Epoch")
ax.set_ylabel("Focal BCE loss")
ax.set_title("Training and validation loss on XES3G5M (seed=42)", fontsize=10)
ax.set_xticks(ep)
ax.grid(linewidth=0.4, alpha=0.5)
ax.legend(loc="upper right", frameon=False, fontsize=9)

fig.tight_layout()
save_figure(fig, "fig_training_loss",
             results_dir="results/xes3g5m/figures")
plt.close(fig)

# ─── Fig 2: Val AUC ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(ep, hist["val_auc"], "^-", color="#029E73",
        linewidth=1.8, markersize=6, label="Validation AUC-ROC")
ax.axvline(best_ep, color="#999", linestyle=":", linewidth=1.0)
ax.scatter([best_ep], [best_auc], s=120, marker="*",
            color="#D55E00", zorder=5, label=f"best (ep {best_ep}: {best_auc:.4f})")

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation AUC-ROC")
ax.set_title("Validation AUC-ROC on XES3G5M (seed=42)", fontsize=10)
ax.set_xticks(ep)
ax.grid(linewidth=0.4, alpha=0.5)
ax.legend(loc="lower right", frameon=False, fontsize=9)
# Tighter y-range so the curve is readable
y_lo = min(hist["val_auc"]) - 0.005
y_hi = max(hist["val_auc"]) + 0.003
ax.set_ylim(y_lo, y_hi)

fig.tight_layout()
save_figure(fig, "fig_training_auc",
             results_dir="results/xes3g5m/figures")
plt.close(fig)

# ─── Copy to article template ────────────────────────────────────────

TEMPLATE = Path("/c/Users/user/Documents/sn-article-template")
if TEMPLATE.exists():
    for src_name, dst_name in [
        ("fig_training_loss.pdf", "fig6.pdf"),       # replaces old combined
        ("fig_training_auc.pdf",  "fig6_auc.pdf"),   # new sibling figure
    ]:
        src = Path(f"results/xes3g5m/figures/{src_name}")
        dst = TEMPLATE / dst_name
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
else:
    print(f"Template dir {TEMPLATE} not found — skipped copying")
