"""Generate architecture diagrams via matplotlib (no mermaid needed)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

FDIR = Path('results/xes3g5m/figures')
FDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.linewidth': 0.5,
})
DOUBLE_COL = 7.16
COL_WIDTH = 3.5


def draw_box(ax, x, y, w, h, text, color='#EBF0F5', border='#5B7BA0', fontsize=7, bold_first=True):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        weight = 'bold' if (i == 0 and bold_first) else 'normal'
        fs = fontsize if i == 0 else fontsize - 1
        style = 'normal' if i == 0 else 'italic'
        ax.text(x + w/2, y + h - 0.15 - i*0.18, line,
                ha='center', va='top', fontsize=fs, fontweight=weight, fontstyle=style)


def draw_arrow(ax, x1, y1, x2, y2, label='', color='#2C3E50', label_offset=(0.12, 0)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                fontsize=6, color='#555', style='italic', ha='left', va='center')


# =====================================================
# FIG 1: MARS System Architecture
# =====================================================
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.2))
ax.set_xlim(-0.3, 7.5)
ax.set_ylim(-1.3, 5.3)
ax.axis('off')

# Input
draw_box(ax, 0.5, 4.6, 6.2, 0.5, 'Datasets\nEdNet KT2 (TOEIC) + XES3G5M (Math)', '#AED6F1', '#5DADE2')

# Orchestrator frame
rect = mpatches.FancyBboxPatch((0.2, 0.1), 6.8, 4.1, boxstyle="round,pad=0.1",
                                 facecolor='#FAFAFA', edgecolor='#2C3E50', linewidth=1.5, linestyle='--')
ax.add_patch(rect)
ax.text(3.6, 4.05, 'ORCHESTRATOR', ha='center', fontsize=10, fontweight='bold', color='#2C3E50')

# Pipelines row
for i, (name, col) in enumerate([('Cold-Start', '#D6EAF8'), ('Assessment', '#D5F5E3'), ('Continuous', '#FDEBD0')]):
    draw_box(ax, 0.5 + i*2.2, 3.45, 1.8, 0.35, name, col, '#888', fontsize=7)

# Row 1: Diagnostic + Confidence  (y = 2.5 .. 3.2)
draw_box(ax, 0.5, 2.5, 2.8, 0.7, 'DiagnosticAgent\nIRT 3PL + CAT\ntheta estimation', '#EBF0F5', '#5B7BA0')
draw_box(ax, 3.9, 2.5, 2.8, 0.7, 'ConfidenceAgent\nRule-based 6-class\nbehavioral confidence', '#EBF0F5', '#5B7BA0')

# Row 2: KG + Prediction  (y = 1.4 .. 2.1)
draw_box(ax, 0.5, 1.4, 2.8, 0.7, 'KnowledgeGraphAgent\nGraphSAGE + Prerequisites\ngap_tags, prereq_map', '#EBF0F5', '#5B7BA0')
draw_box(ax, 3.9, 1.4, 2.8, 0.7, 'PredictionAgent\nSAINT Transformer 4L/256d\n293-dim gap_probs', '#EBF0F5', '#5B7BA0')

# Row 3: Recommendation + Personalization  (y = 0.3 .. 1.0)
draw_box(ax, 0.5, 0.3, 3.3, 0.7, 'RecommendationAgent\nThompson Sampling + 6-feature scoring\nMMR diversity + prereq filter', '#E8DAEF', '#8E44AD')
draw_box(ax, 4.3, 0.3, 2.4, 0.7, 'PersonalizationAgent\nIRT 5-level stratification', '#EBF0F5', '#5B7BA0')

# Arrows (vertical gaps are 0.3 wide, enough room for labels)
draw_arrow(ax, 3.6, 4.6, 3.6, 4.22)                               # input -> orchestrator
draw_arrow(ax, 1.9, 2.5, 1.9, 2.12, 'theta')                      # diag -> KG
draw_arrow(ax, 5.3, 2.5, 5.3, 2.12, 'conf_class')                 # conf -> pred
draw_arrow(ax, 1.9, 1.4, 1.9, 1.02, 'prereq_map')                 # KG -> rec
draw_arrow(ax, 5.0, 1.4, 3.5, 1.02, 'gap_probs', label_offset=(0.08, 0.08))  # pred -> rec (diagonal)
draw_arrow(ax, 5.5, 0.3, 5.5, -0.12)                              # pers -> out

# Output
draw_box(ax, 1.5, -0.6, 4.2, 0.35, 'Personalized Recommendation List (Top-K)', '#A9DFBF', '#27AE60')
draw_arrow(ax, 2.2, 0.3, 2.8, -0.22)                              # rec -> out

fig.savefig(FDIR / 'fig1_mars_architecture.png', dpi=600, bbox_inches='tight')
fig.savefig(FDIR / 'fig1_mars_architecture.pdf', bbox_inches='tight')
plt.close()
print('OK: fig1_mars_architecture')


# =====================================================
# FIG 2: Data Pipeline
# =====================================================
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.5))
ax.set_xlim(-0.2, 7.4)
ax.set_ylim(-0.2, 3.8)
ax.axis('off')

# Top row
draw_box(ax, 0.3, 3.0, 2.0, 0.6, 'Raw Datasets\nEdNet + XES3G5M', '#AED6F1', '#5DADE2')
draw_box(ax, 2.7, 3.0, 2.0, 0.6, 'Preprocessing\nFeature Engineering', '#D5DBDB', '#ABB2B9')
draw_box(ax, 5.1, 3.0, 2.0, 0.6, 'User-Level Split\n70/15/15%', '#FDEBD0', '#E67E22')
draw_arrow(ax, 2.3, 3.3, 2.7, 3.3)
draw_arrow(ax, 4.7, 3.3, 5.1, 3.3)

# Splits
draw_box(ax, 0.5, 2.0, 1.5, 0.5, 'Train\n70% users', '#D5F5E3', '#27AE60')
draw_box(ax, 2.7, 2.0, 1.5, 0.5, 'Validation\n15% users', '#FCF3CF', '#F1C40F')
draw_box(ax, 4.9, 2.0, 1.5, 0.5, 'Test\n15% users', '#FADBD8', '#E74C3C')
draw_arrow(ax, 5.5, 3.0, 1.25, 2.55)
draw_arrow(ax, 6.1, 3.0, 3.45, 2.55)
draw_arrow(ax, 6.1, 3.0, 5.65, 2.55)

# Agents
agents = [
    ('Diagnostic\nIRT 3PL', 0.0), ('Confidence\nRule-based', 1.2),
    ('KG Agent\nGraphSAGE', 2.4), ('Prediction\nTransformer', 3.6),
    ('Recommend\nTS+MMR', 4.8), ('Personal.\n5-level', 6.0),
]
for name, xp in agents:
    draw_box(ax, xp, 0.8, 1.0, 0.6, name, '#EBF0F5', '#5B7BA0', fontsize=6)
    draw_arrow(ax, 1.25, 2.0, xp+0.5, 1.45)

# Eval
draw_box(ax, 2.0, 0.0, 3.2, 0.45, 'Evaluation: 5 seeds x 15 metrics\nOrchestrator batch_evaluation', '#E8DAEF', '#8E44AD', fontsize=6)
draw_arrow(ax, 5.65, 2.0, 3.6, 0.5)

fig.savefig(FDIR / 'fig2_data_pipeline.png', dpi=600, bbox_inches='tight')
fig.savefig(FDIR / 'fig2_data_pipeline.pdf', bbox_inches='tight')
plt.close()
print('OK: fig2_data_pipeline')


# =====================================================
# FIG 4: Orchestrator Pipelines (simplified sequence)
# =====================================================
fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))
ax.set_xlim(-0.3, 8.1)
ax.set_ylim(-0.8, 3.5)
ax.axis('off')

# Three pipeline columns
pipelines = [
    ('Cold-Start Pipeline', '#D6EAF8', '#5DADE2',
     ['DiagnosticAgent\ntheta=0, SE', 'KGAgent\ncold_start()', 'RecommendAgent\nexplore mode']),
    ('Assessment Pipeline', '#D5F5E3', '#27AE60',
     ['DiagnosticAgent\nIRT update', 'ConfidenceAgent\n6-class', 'KGAgent\nupdate profile',
      'PredictionAgent\ngap_probs[293]', 'PersonalizationAgent\nlevel', 'RecommendAgent\nranked top-10']),
    ('Continuous Pipeline', '#FDEBD0', '#E67E22',
     ['PredictionAgent\nupdate_state()', 'RecommendAgent\nre_rank()']),
]

COL_W = 2.4
COL_SPACE = 0.4
STEP_H = 0.45
STEP_GAP = 0.08

for pi, (title, bg, border, steps) in enumerate(pipelines):
    xbase = pi * (COL_W + COL_SPACE)
    # Title
    draw_box(ax, xbase, 2.85, COL_W, 0.42, title, bg, border, fontsize=7)
    # Steps (clear gap between boxes, no arrows)
    for si, step in enumerate(steps):
        y = 2.3 - si * (STEP_H + STEP_GAP)
        draw_box(ax, xbase + 0.1, y, COL_W - 0.2, STEP_H, step,
                 '#FAFAFA', '#AAA', fontsize=5.5)

fig.savefig(FDIR / 'fig4_orchestrator_pipelines.png', dpi=600, bbox_inches='tight')
fig.savefig(FDIR / 'fig4_orchestrator_pipelines.pdf', bbox_inches='tight')
plt.close()
print('OK: fig4_orchestrator_pipelines')

print(f'\nArchitecture figures saved to {FDIR}')
