"""
Generate Data Pipeline diagram for Methodology section.
Output: results/fig_data_pipeline.{png,pdf}, diagrams/data_pipeline.svg
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from utils.plot_style import setup_publication_style, save_figure

setup_publication_style()
plt.rcParams["axes.grid"] = False

# ── Colors ───────────────────────────────────────────────────────────
C_RAW    = "#AED6F1"
C_PROC   = "#D5DBDB"
C_SPLIT  = "#FDEBD0"
C_TRAIN  = "#D5F5E3"
C_VAL    = "#FCF3CF"
C_TEST   = "#FADBD8"
C_AGENT  = "#EBF0F5"
C_AGENT_BD = "#5B7BA0"
C_ARROW  = "#2C3E50"
C_TEXT   = "#2C3E50"
C_NOTE   = "#7F8C8D"

fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, w, h, text, detail=None, fc="#EBF0F5", ec="#5B7BA0", lw=1.0,
        fs=8.5, dfs=6.5, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    ty = y + h / 2 + (0.13 if detail else 0)
    ax.text(x + w / 2, ty, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=C_TEXT, zorder=3)
    if detail:
        ax.text(x + w / 2, ty - 0.26, detail, ha="center", va="center",
                fontsize=dfs, color=C_NOTE, fontstyle="italic", zorder=3)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=4)


def side_arrow(x1, y1, x2, y2, color=C_ARROW, lw=0.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.15"), zorder=4)


# ═════════════════════════════════════════════════════════════════════
# TOP: Raw Data → Sampling → Preprocessing → Split
# ═════════════════════════════════════════════════════════════════════
bw, bh = 5.0, 0.55
bx = 2.5

box(bx, 10.2, bw, bh, "EdNet Raw Data",
    "784K students,  131M+ interactions", fc=C_RAW, ec="#5DADE2")
arrow(5.0, 10.2, 5.0, 9.85)

box(bx, 9.25, bw, bh, "User Sampling",
    "sample_users=1000  →  filter min_answers ≥ 10", fc=C_PROC, ec="#ABB2B9")
arrow(5.0, 9.25, 5.0, 8.9)

box(bx, 8.15, bw, 0.7, "Preprocessing & Feature Engineering",
    "elapsed_time clip [1s, 300s]  ·  session gap 30 min  ·  16 features",
    fc=C_PROC, ec="#ABB2B9")
arrow(5.0, 8.15, 5.0, 7.8)

box(bx, 7.1, bw, 0.65, "Chronological Split (per student)",
    "first 70% → Train  /  next 15% → Val  /  last 15% → Test",
    fc=C_SPLIT, ec="#E67E22")

# ═════════════════════════════════════════════════════════════════════
# THREE SPLIT BRANCHES
# ═════════════════════════════════════════════════════════════════════
# Train (left-center), Val (right), Test (right)
train_x, train_y = 1.0, 5.7
val_x, val_y = 6.8, 6.35
test_x, test_y = 6.8, 5.05

# Arrows from split box to three branches
arrow(3.5, 7.1, 2.8, 6.35)   # → Train
arrow(6.5, 7.1, 7.8, 6.9)    # → Val
arrow(6.5, 7.1, 7.8, 5.6)    # → Test

box(train_x, train_y, 3.2, 0.55, "Train Data",
    "70%  ·  32,754 interactions", fc=C_TRAIN, ec="#27AE60", fs=8)

box(val_x, val_y, 2.8, 0.45, "Validation Data",
    "15%", fc=C_VAL, ec="#F1C40F", fs=7.5, dfs=6)

box(test_x, test_y, 2.8, 0.45, "Test Data",
    "15%", fc=C_TEST, ec="#E74C3C", fs=7.5, dfs=6)

# Val annotation (right of box, no arrow)
ax.text(val_x + 2.8 + 0.15, val_y + 0.22, "HP tuning",
        ha="left", va="center", fontsize=6, color=C_NOTE, fontstyle="italic")

# Test → Final evaluation
arrow(8.2, test_y, 8.2, test_y - 0.3)
ax.text(8.2, test_y - 0.4, "Final evaluation\n5 seeds · 16 metrics",
        ha="center", va="top", fontsize=6, color=C_NOTE, fontstyle="italic")

# ═════════════════════════════════════════════════════════════════════
# AGENT TRAINING (from Train Data)
# ═════════════════════════════════════════════════════════════════════
agents = [
    ("DiagnosticAgent",      "IRT 3PL calibration"),
    ("ConfidenceAgent",      "XGBoost 6-class training"),
    ("KnowledgeGraphAgent",  "GraphSAGE + prerequisite mining"),
    ("PredictionAgent",      "LSTM seq-to-set training"),
    ("RecommendationAgent",  "TS priors + LambdaMART fit"),
    ("PersonalizationAgent", "K-Means clustering (K=6)"),
]

agent_x = 0.3
agent_w = 4.8
agent_h = 0.45
start_y = 4.8
gap = 0.6

# Arrow from Train → agents area
arrow(2.6, train_y, 2.6, start_y + 0.5)

for i, (name, detail) in enumerate(agents):
    ay = start_y - i * gap
    box(agent_x, ay, agent_w, agent_h, name, detail,
        fc=C_AGENT, ec=C_AGENT_BD, fs=7.5, dfs=6, lw=0.8)
    # Small arrow from left margin
    if i > 0:
        ax.plot([0.15, agent_x], [ay + agent_h / 2, ay + agent_h / 2],
                color=C_AGENT_BD, lw=0.6, zorder=1)

# Vertical line connecting agents
ax.plot([0.15, 0.15], [start_y + agent_h / 2, start_y - (len(agents) - 1) * gap + agent_h / 2],
        color=C_AGENT_BD, lw=0.8, zorder=1)
# Connector from train
ax.plot([2.6, 0.15], [start_y + 0.5, start_y + agent_h / 2],
        color=C_AGENT_BD, lw=0.8, zorder=1)

# ── "train only!" annotation
ax.text(5.3, 3.0, "All agents trained\non train split only\n(no data leakage)",
        fontsize=6.5, color="#E74C3C", fontstyle="italic",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="#FEF9E7", ec="#E74C3C",
                  lw=0.8, alpha=0.9), zorder=5)

save_figure(fig, "fig_data_pipeline")
fig.savefig("diagrams/data_pipeline.svg", bbox_inches="tight", format="svg")
print("Saved: diagrams/data_pipeline.svg")
plt.close(fig)
print("Data Pipeline diagram done.")
