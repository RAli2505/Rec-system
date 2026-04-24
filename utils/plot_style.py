"""
Unified publication style for MARS project figures.

Usage:
    from utils.plot_style import setup_publication_style, save_figure, MARS_COLORS
    setup_publication_style()
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Colorblind-safe palette (Wong / Tol) ─────────────────────────────

MARS_COLORS = {
    "primary": [
        "#0173B2",  # blue
        "#DE8F05",  # orange
        "#029E73",  # green
        "#D55E00",  # vermillion
        "#CC78BC",  # purple
        "#CA9161",  # brown
    ],
    "mars_main": "#0173B2",
    "baseline": "#999999",
    "ablation": "#DE8F05",
}

METHOD_COLORS = {
    "MARS (ours)": "#0173B2",
    "MARS (full)": "#0173B2",
    "Random":      "#999999",
    "Popular":     "#CA9161",
    "DKT":         "#DE8F05",
    "SAKT":        "#029E73",
    "SAINT":       "#D55E00",
    "AKT":         "#CC78BC",
    "BPR-MF":      "#56B4E9",
    "CF-only":     "#F0E442",
    "Content-only":"#CC6677",
    "Binary-conf": "#88CCEE",
    "Monolithic":  "#DDCC77",
}

CONFIDENCE_COLORS = [
    "#0173B2",  # SOLID
    "#029E73",  # CORRECT_HESITANT
    "#DE8F05",  # CORRECT_SLOW
    "#CC78BC",  # WRONG_FAST
    "#D55E00",  # WRONG_HESITANT
    "#CA9161",  # GUESSING
]

# ── Figure sizes (inches) ────────────────────────────────────────────

SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.2, 3.5)
DOUBLE_COL_TALL = (7.2, 5.5)
SQUARE = (3.5, 3.5)


# ── Style setup ──────────────────────────────────────────────────────

def _best_serif_font():
    """Return Times New Roman if available, else DejaVu Serif."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Times New Roman", "DejaVu Serif"):
        if name in available:
            return name
    return "serif"


def setup_publication_style():
    """Configure matplotlib rcParams for journal-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family":       "serif",
        "font.serif":        [_best_serif_font()],
        "font.size":         10,
        "axes.labelsize":    10,
        "axes.titlesize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        # DPI
        "figure.dpi":        300,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
        # Lines
        "lines.linewidth":   1.5,
        "lines.markersize":  6,
        # Grid
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
        # Spines
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # Mathtext
        "mathtext.fontset":  "dejavuserif",
    })


# ── Save helper ──────────────────────────────────────────────────────

def save_figure(fig, name, results_dir="results", dpi=600):
    """Save figure as PNG (600 DPI journal-grade) + PDF (vector).

    Raised from 300 → 600 to satisfy Springer Nature's submission
    guideline for bitmap figures (reviewer item #37).
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {name}.png + {name}.pdf @ {dpi} dpi")


# ── Annotation helpers ───────────────────────────────────────────────

def annotate_significance(ax, x1, x2, y, p_value, dh=0.02, fs=9):
    """Draw a significance bracket between two bar positions."""
    if p_value < 0.001:
        text = "***"
    elif p_value < 0.01:
        text = "**"
    elif p_value < 0.05:
        text = "*"
    else:
        text = "ns"

    ax.plot([x1, x1, x2, x2], [y, y + dh, y + dh, y], lw=1.0, c="black")
    ax.text((x1 + x2) / 2, y + dh, text, ha="center", va="bottom", fontsize=fs)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add (a), (b), (c) panel labels for composite figures."""
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")
