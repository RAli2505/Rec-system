"""
Generate Figure 1: MARS Architecture Diagram.
Programmatic matplotlib version with FancyBboxPatch + annotated arrows.
Output: results/fig_mars_architecture.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from utils.plot_style import setup_publication_style, save_figure

setup_publication_style()
plt.rcParams["axes.grid"] = False  # no grid for diagram

# ── Colors ───────────────────────────────────────────────────────────
C_BG       = "#F7F9FC"       # orchestrator background
C_PIPE_CS  = "#D6EAF8"       # cold-start pipeline
C_PIPE_AS  = "#D5F5E3"       # assessment pipeline
C_PIPE_CT  = "#FDEBD0"       # continuous pipeline
C_AGENT    = "#EBF0F5"       # agent box fill
C_AGENT_BD = "#5B7BA0"       # agent box border
C_INPUT    = "#AED6F1"       # input boxes
C_OUTPUT   = "#A9DFBF"       # output boxes
C_ARROW    = "#2C3E50"       # arrow color
C_LABEL    = "#E74C3C"       # data-flow label color
C_TEXT     = "#2C3E50"

fig, ax = plt.subplots(1, 1, figsize=(7.2, 8.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

# ── Helpers ──────────────────────────────────────────────────────────
def box(x, y, w, h, text, subtitle=None, fc=C_AGENT, ec=C_AGENT_BD, lw=1.0,
        fontsize=8, subtitlesize=6.5, bold=True, rounded=True):
    """Draw a rounded rectangle with centered text."""
    style = "round,pad=0.08" if rounded else "square,pad=0.02"
    p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                        facecolor=fc, edgecolor=ec, linewidth=lw,
                        transform=ax.transData, zorder=2)
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    ty = y + h / 2 + (0.12 if subtitle else 0)
    ax.text(x + w / 2, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=C_TEXT, zorder=3)
    if subtitle:
        ax.text(x + w / 2, ty - 0.28, subtitle, ha="center", va="center",
                fontsize=subtitlesize, color="#555555", fontstyle="italic", zorder=3)
    return (x + w / 2, y + h / 2)  # center

def arrow(x1, y1, x2, y2, label=None, color=C_ARROW, lw=1.2, style="->",
          label_offset=(0, 0.12), label_color=C_LABEL, label_fs=6):
    """Draw arrow with optional label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"),
                zorder=4)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_fs, color=label_color,
                ha="center", va="center", fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5)

# ═════════════════════════════════════════════════════════════════════
# INPUT LAYER (top)
# ═════════════════════════════════════════════════════════════════════
box(2.5, 11.3, 5, 0.5, "EdNet KT2 Dataset", fc=C_INPUT, ec="#5DADE2",
    fontsize=9, bold=True)
arrow(5.0, 11.3, 5.0, 10.95)
box(2.5, 10.4, 5, 0.5, "Data Preprocessing & Feature Engineering",
    fc=C_INPUT, ec="#5DADE2", fontsize=7.5)
arrow(5.0, 10.4, 5.0, 10.05)
box(1.5, 9.5, 7, 0.5, "Chronological Split:  70% Train  /  15% Val  /  15% Test",
    fc="#D5DBDB", ec="#ABB2B9", fontsize=7.5)

# ═════════════════════════════════════════════════════════════════════
# ORCHESTRATOR (main box)
# ═════════════════════════════════════════════════════════════════════
orch_y = 1.8
orch_h = 7.3
orch = FancyBboxPatch((0.3, orch_y), 9.4, orch_h,
                       boxstyle="round,pad=0.15",
                       facecolor=C_BG, edgecolor="#34495E",
                       linewidth=1.8, zorder=1)
ax.add_patch(orch)
ax.text(5.0, orch_y + orch_h - 0.25, "ORCHESTRATOR",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#2C3E50", zorder=3)

# ── Pipelines row ───────────────────────────────────────────────────
pw, ph = 2.4, 0.55
py = 8.15
box(1.0, py, pw, ph, "Cold-Start Pipeline", fc=C_PIPE_CS, ec="#5DADE2",
    fontsize=7, bold=False)
box(3.8, py, pw, ph, "Assessment Pipeline", fc=C_PIPE_AS, ec="#27AE60",
    fontsize=7, bold=False)
box(6.6, py, pw, ph, "Continuous Pipeline", fc=C_PIPE_CT, ec="#E67E22",
    fontsize=7, bold=False)

arrow(5.0, 9.5, 5.0, 8.75)

# ── Agent Row 1: Diagnostic + Confidence ─────────────────────────────
aw, ah = 3.5, 0.8
r1y = 6.9

diag_cx, diag_cy = box(0.8, r1y, aw, ah,
    "DiagnosticAgent", "IRT 3PL  +  Computerized Adaptive Testing",
    fontsize=8.5, subtitlesize=6)

conf_cx, conf_cy = box(5.7, r1y, aw, ah,
    "ConfidenceAgent", "XGBoost  →  6-class classification",
    fontsize=8.5, subtitlesize=6)

# ── Agent Row 2: KG + Prediction ─────────────────────────────────────
r2y = 5.4

kg_cx, kg_cy = box(0.8, r2y, aw, ah,
    "KnowledgeGraphAgent", "GraphSAGE embeddings  +  Prerequisites",
    fontsize=8.5, subtitlesize=6)

pred_cx, pred_cy = box(5.7, r2y, aw, ah,
    "PredictionAgent", "LSTM seq-to-set  →  293-dim gap vector",
    fontsize=8.5, subtitlesize=6)

# Arrows Row 1 → Row 2
arrow(diag_cx, r1y, kg_cx, r2y + ah, label="θ (ability)",
      label_offset=(-0.0, 0.12))
arrow(conf_cx, r1y, pred_cx, r2y + ah, label="confidence\nclass",
      label_offset=(0.0, 0.12))

# ── Agent Row 3: RecommendationAgent (wide) ──────────────────────────
r3y = 3.8
rec_cx, rec_cy = box(1.2, r3y, 7.6, 0.85,
    "RecommendationAgent",
    "Thompson Sampling strategy selection  +  LambdaMART re-ranking  (12 features)",
    fontsize=9, subtitlesize=6.5)

# Arrows Row 2 → Row 3
arrow(kg_cx, r2y, rec_cx - 1.5, r3y + 0.85, label="prerequisites\n& gaps",
      label_offset=(-0.6, 0.1))
arrow(pred_cx, r2y, rec_cx + 1.5, r3y + 0.85, label="gap_probs[293]",
      label_offset=(0.6, 0.1))

# Cross arrows (Diag → Rec, Conf → Rec)
arrow(diag_cx + 0.3, r1y, rec_cx - 2.5, r3y + 0.85,
      color="#7F8C8D", lw=0.8, label="θ", label_offset=(-1.8, -0.3),
      label_color="#7F8C8D", label_fs=5.5)
arrow(conf_cx - 0.3, r1y, rec_cx + 2.5, r3y + 0.85,
      color="#7F8C8D", lw=0.8, label="conf_class", label_offset=(1.8, -0.3),
      label_color="#7F8C8D", label_fs=5.5)

# ── Agent Row 4: PersonalizationAgent ────────────────────────────────
r4y = 2.5
pers_cx, pers_cy = box(1.8, r4y, 6.4, 0.75,
    "PersonalizationAgent",
    "K-Means clustering (K=6)  +  adaptive difficulty & strategy params",
    fontsize=8.5, subtitlesize=6)

arrow(rec_cx, r3y, pers_cx, r4y + 0.75, label="ranked items",
      label_offset=(0, 0.12))

# ═════════════════════════════════════════════════════════════════════
# OUTPUT LAYER (bottom)
# ═════════════════════════════════════════════════════════════════════
arrow(pers_cx, orch_y, pers_cx, 1.45)
box(2.0, 0.85, 6, 0.55, "Personalized Recommendation List",
    fc=C_OUTPUT, ec="#27AE60", fontsize=9)
arrow(5.0, 0.85, 5.0, 0.5)
ax.text(5.0, 0.25, "Student receives ranked items with explanations",
        ha="center", va="center", fontsize=7, color="#555555", fontstyle="italic")

# ── Title ────────────────────────────────────────────────────────────
# (no suptitle — figure caption goes in LaTeX)

save_figure(fig, "fig_mars_architecture")

# Also save as SVG
fig.savefig("diagrams/mars_architecture.svg", bbox_inches="tight", format="svg")
print("Saved: diagrams/mars_architecture.svg")

plt.close(fig)
print("Figure 1 (MARS Architecture) done.")
