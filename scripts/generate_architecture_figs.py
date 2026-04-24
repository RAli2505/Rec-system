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
# FIG 1: MARS System Architecture (clean pipeline-oriented)
# =====================================================
# Three-column layout: each column = one pipeline, agents listed in execution order.
# No orchestrator frame, no arrow labels, uniform arrows.
ARROW_KW = dict(arrowstyle='-|>', color='#2C3E50', lw=1.0,
                mutation_scale=10, shrinkA=2, shrinkB=2)


def uarrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=ARROW_KW)


AGENT_DESC = {
    'Diagnostic':      'IRT 3PL + CAT',
    'Confidence':      'Rule-based 6-class',
    'KnowledgeGraph':  'GraphSAGE + Prereq.',
    'Prediction':      'SAINT Transformer',
    'Personalization': 'IRT 5-level strat.',
    'Recommendation':  'Thompson + MMR',
}

fig, ax = plt.subplots(figsize=(DOUBLE_COL, 6.6))
ax.set_xlim(-0.2, 8.2)
ax.set_ylim(-1.8, 5.8)
ax.axis('off')

# --- Input row (shared) ---
draw_box(ax, 1.0, 5.1, 6.0, 0.5, 'Datasets: EdNet KT2 (TOEIC) + XES3G5M (Math)',
         '#AED6F1', '#5DADE2')

# --- Three pipeline columns ---
COL_W, COL_GAP = 2.4, 0.35
COL_X = [0.2 + i*(COL_W + COL_GAP) for i in range(3)]

pipelines = [
    ('Cold-Start Pipeline',  '#D6EAF8', '#5DADE2',
     ['Diagnostic', 'KnowledgeGraph', 'Recommendation']),
    ('Assessment Pipeline',  '#D5F5E3', '#27AE60',
     ['Diagnostic', 'Confidence', 'KnowledgeGraph',
      'Prediction', 'Personalization', 'Recommendation']),
    ('Continuous Pipeline',  '#FDEBD0', '#E67E22',
     ['Prediction', 'Recommendation']),
]

STEP_H, STEP_GAP = 0.55, 0.22
TOP_Y = 4.35  # title box top sits here

for i, (title, bg, border, agents) in enumerate(pipelines):
    x = COL_X[i]
    # Title box
    draw_box(ax, x, TOP_Y, COL_W, 0.45, title, bg, border, fontsize=8)

    # Agent boxes in execution order
    prev_center = None
    for si, name in enumerate(agents):
        y = TOP_Y - 0.25 - (si + 1) * (STEP_H + STEP_GAP)
        fill   = '#E8DAEF' if name == 'Recommendation' else '#F5F5F5'
        bcolor = '#8E44AD' if name == 'Recommendation' else '#9FB2C6'
        label = f'{name}Agent\n{AGENT_DESC[name]}'
        draw_box(ax, x + 0.1, y, COL_W - 0.2, STEP_H, label, fill, bcolor, fontsize=7)
        cx = x + COL_W / 2
        top = y + STEP_H
        if prev_center is not None:
            # uniform vertical arrow between consecutive agent boxes
            uarrow(ax, cx, prev_center, cx, top)
        prev_center = y  # bottom of current box for next arrow

# Dataset -> top of every pipeline title
for x in COL_X:
    uarrow(ax, x + COL_W/2, 5.1, x + COL_W/2, TOP_Y + 0.45)

# --- Output row (shared) ---
last_y = TOP_Y - 0.25 - (6) * (STEP_H + STEP_GAP)  # bottom of Assessment (deepest column)
out_y = last_y - 0.25
draw_box(ax, 1.0, out_y - 0.45, 6.0, 0.45,
         'Personalized Recommendation List (Top-K)', '#A9DFBF', '#27AE60')
# Arrow from each pipeline's last agent to output
for i, (_, _, _, agents) in enumerate(pipelines):
    depth = len(agents)
    bottom_y = TOP_Y - 0.25 - depth * (STEP_H + STEP_GAP)
    uarrow(ax, COL_X[i] + COL_W/2, bottom_y, COL_X[i] + COL_W/2, out_y - 0.0)

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


# =====================================================
# FIG 11: MARS detailed architecture (Backbone/Neck/Head style)
# =====================================================
ARROW_KW2 = dict(arrowstyle='-|>', color='#2C3E50', lw=0.9,
                 mutation_scale=9, shrinkA=2, shrinkB=2)


def arr(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=ARROW_KW2)


def panel(ax, x, y, w, h, title, bg='#F7F7F7', border='#B0B0B0'):
    """Outer group panel with a dedicated title band at the top."""
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                   facecolor=bg, edgecolor=border,
                                   linewidth=1.0)
    ax.add_patch(rect)
    # Title band (solid color strip at top, so title never overlaps content)
    band_h = 0.38
    band = FancyBboxPatch((x + 0.02, y + h - band_h - 0.02), w - 0.04, band_h,
                          boxstyle='round,pad=0.02',
                          facecolor=border, edgecolor=border, linewidth=0)
    ax.add_patch(band)
    ax.text(x + w/2, y + h - band_h/2 - 0.02, title, ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')


def minibox(ax, x, y, w, h, text, fill='#FFFFFF', border='#9FB2C6', fontsize=6):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         facecolor=fill, edgecolor=border, linewidth=0.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color='#2C3E50')


# ----------- clean agent-flow diagram (no detail panels / no groupings) -----------
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(10, 7.6, 'MARS - Multi-Agent Pipeline',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#2C3E50')

# Each block: (x, y, w, h, name, summary, arrow-label-above)
BH = 1.4
BW = 2.4
ROW_Y = 3.4
BLOCKS = [
    ('Input\nDataset',          'XES3G5M + EdNet\n6K users',               '#AED6F1', '#5DADE2'),
    ('DiagnosticAgent',         'IRT 3PL + CAT\n-> theta',                 '#EBF0F5', '#5B7BA0'),
    ('ConfidenceAgent',         '6-class rules\n-> conf_class',            '#EBF0F5', '#5B7BA0'),
    ('KnowledgeGraphAgent',     'GraphSAGE + prereq.\n-> gap_tags',        '#EBF0F5', '#5B7BA0'),
    ('PredictionAgent',         'SAINT Transformer\n-> gap_probs[293]',    '#EBF0F5', '#5B7BA0'),
    ('PersonalizationAgent',    'IRT 5-level\n-> learner level',           '#EBF0F5', '#5B7BA0'),
    ('RecommendationAgent',     'Thompson + MMR\n+ LambdaMART',            '#E8DAEF', '#8E44AD'),
    ('Top-K\nRecommendation',   'prereq-filtered\nranked list',            '#A9DFBF', '#27AE60'),
]

# Two rows of 4 blocks each with clear gaps
GAP_X = 0.2
for i, (name, desc, fill, border) in enumerate(BLOCKS):
    row = i // 4            # 0 = top row, 1 = bottom row
    col = i % 4
    y = ROW_Y if row == 1 else ROW_Y + BH + 1.2
    x = 0.4 + col * (BW + GAP_X * 2 + 0.6)

    # small-bordered rounded rectangle (not long borders, not touching)
    box = FancyBboxPatch((x, y), BW, BH, boxstyle='round,pad=0.06',
                         facecolor=fill, edgecolor=border, linewidth=1.0)
    ax.add_patch(box)
    ax.text(x + BW/2, y + BH - 0.30, name,
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color='#2C3E50')
    ax.text(x + BW/2, y + BH/2 - 0.35, desc,
            ha='center', va='center', fontsize=7, style='italic',
            color='#444')

# ---- arrows between blocks ----
# top row left->right (indices 0..3)
for i in range(3):
    x1 = 0.4 + i * (BW + GAP_X * 2 + 0.6) + BW
    x2 = 0.4 + (i + 1) * (BW + GAP_X * 2 + 0.6)
    y = ROW_Y + BH + 1.2 + BH/2
    arr(ax, x1 + 0.05, y, x2 - 0.05, y)

# top row last (i=3) -> bottom row first (i=4): vertical turn arrow
x_end_top = 0.4 + 3 * (BW + GAP_X * 2 + 0.6) + BW/2
x_start_bot = 0.4 + 0 * (BW + GAP_X * 2 + 0.6) + BW/2
y_top = ROW_Y + BH + 1.2            # bottom edge of top row
y_bot = ROW_Y + BH                  # top edge of bottom row

# draw a U-turn: down from end-of-top, left, then down into start-of-bottom
# simplified: one diagonal arrow
arr(ax, x_end_top, y_top, x_start_bot, y_bot + 0.15)
# also add a small tag near the turning arrow so the flow is clear
ax.text((x_end_top + x_start_bot) / 2, (y_top + y_bot) / 2 + 0.05,
        'continue',
        ha='center', va='center', fontsize=7, style='italic', color='#777')

# bottom row left->right (indices 4..7)
for i in range(4, 7):
    x1 = 0.4 + (i - 4) * (BW + GAP_X * 2 + 0.6) + BW
    x2 = 0.4 + (i - 4 + 1) * (BW + GAP_X * 2 + 0.6)
    y = ROW_Y + BH/2
    arr(ax, x1 + 0.05, y, x2 - 0.05, y)

fig.savefig(FDIR / 'fig_mars_detailed_arch.png', dpi=600, bbox_inches='tight')
fig.savefig(FDIR / 'fig_mars_detailed_arch.pdf', bbox_inches='tight')
plt.close()
print('OK: fig_mars_detailed_arch')


print(f'\nArchitecture figures saved to {FDIR}')
