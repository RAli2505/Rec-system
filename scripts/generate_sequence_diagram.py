"""
Generate Orchestrator Pipelines sequence diagram (3 panels).
Output: results/fig_orchestrator_pipelines.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

from utils.plot_style import setup_publication_style, save_figure, add_panel_label

setup_publication_style()
plt.rcParams["axes.grid"] = False

# ── Colors ───────────────────────────────────────────────────────────
C_CS  = "#D6EAF8"  # cold-start bg
C_AS  = "#D5F5E3"  # assessment bg
C_CT  = "#FDEBD0"  # continuous bg
C_BD  = {"cs": "#5DADE2", "as": "#27AE60", "ct": "#E67E22"}

ACTORS = {
    "Student":       ("#AED6F1", "#2980B9"),
    "Orchestrator":  ("#D5DBDB", "#34495E"),
    "Diagnostic":    ("#E8DAEF", "#8E44AD"),
    "Confidence":    ("#FCF3CF", "#F39C12"),
    "KG":            ("#D5F5E3", "#27AE60"),
    "Prediction":    ("#FADBD8", "#E74C3C"),
    "Recommend":     ("#D6EAF8", "#2471A3"),
    "Personal":      ("#FDEBD0", "#E67E22"),
}

def draw_pipeline(ax, title, bg_color, border_color, messages, actors_used):
    """Draw a simplified sequence diagram panel."""
    ax.set_xlim(-0.5, len(actors_used) - 0.5)
    n_msg = len(messages)
    ax.set_ylim(-n_msg - 0.5, 1.5)
    ax.axis("off")

    # Title
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)

    # Background
    bg = FancyBboxPatch((-0.7, -n_msg - 0.3), len(actors_used) + 0.4, n_msg + 1.5,
                         boxstyle="round,pad=0.1", facecolor=bg_color,
                         edgecolor=border_color, linewidth=1.5, alpha=0.3, zorder=0)
    ax.add_patch(bg)

    # Actor positions
    actor_x = {name: i for i, name in enumerate(actors_used)}

    # Draw actor boxes at top
    for name, xi in actor_x.items():
        fc, ec = ACTORS[name]
        p = FancyBboxPatch((xi - 0.35, 0.7), 0.7, 0.5, boxstyle="round,pad=0.05",
                            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2)
        ax.add_patch(p)
        # Short name
        short = name[:6] if len(name) > 8 else name
        ax.text(xi, 0.95, short, ha="center", va="center",
                fontsize=5.5, fontweight="bold", zorder=3)

    # Lifelines
    for name, xi in actor_x.items():
        ax.plot([xi, xi], [0.7, -n_msg - 0.1], color="#BDC3C7",
                linewidth=0.6, linestyle="--", zorder=1)

    # Messages
    for i, (src, dst, label, is_return) in enumerate(messages):
        y = -i - 0.3
        x1 = actor_x[src]
        x2 = actor_x[dst]

        if src == dst:
            # Self-message: rectangular UML-style loop
            jut = 0.3  # horizontal jut-out
            drop = 0.2  # vertical drop
            ax.plot([x1, x1 + jut, x1 + jut], [y, y, y - drop],
                    color="#2C3E50", lw=0.8, solid_capstyle="round", zorder=4)
            ax.annotate("", xy=(x1 + 0.02, y - drop),
                        xytext=(x1 + jut, y - drop),
                        arrowprops=dict(arrowstyle="->", color="#2C3E50",
                                        lw=0.8), zorder=4)
            ax.text(x1 + jut + 0.05, y - drop / 2, label,
                    fontsize=4.5, va="center", ha="left",
                    fontstyle="italic", color="#555", zorder=5)
        else:
            style = "-->" if is_return else "->"
            ls = "--" if is_return else "-"
            color = "#7F8C8D" if is_return else "#2C3E50"
            ax.annotate("", xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8,
                                        linestyle=ls), zorder=4)
            # Label above arrow
            mx = (x1 + x2) / 2
            ax.text(mx, y + 0.12, label, fontsize=4.2, ha="center", va="bottom",
                    color="#2C3E50", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))


# ═════════════════════════════════════════════════════════════════════
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 4.5))

# ── (a) Cold-Start ──────────────────────────────────────────────────
cs_actors = ["Student", "Orchestrator", "Diagnostic", "KG", "Recommend"]
cs_msgs = [
    ("Student", "Orchestrator", "new_student()", False),
    ("Orchestrator", "Diagnostic", "estimate_θ(prior=0)", False),
    ("Diagnostic", "Orchestrator", "θ_init, SE", True),
    ("Orchestrator", "KG", "get_prerequisites()", False),
    ("KG", "Orchestrator", "prereq_graph", True),
    ("Orchestrator", "Recommend", "recommend(explore)", False),
    ("Recommend", "Orchestrator", "ranked_items", True),
    ("Orchestrator", "Student", "initial_items", True),
]
draw_pipeline(ax1, "(a) Cold-Start Pipeline", C_CS, C_BD["cs"], cs_msgs, cs_actors)

# ── (b) Assessment ──────────────────────────────────────────────────
as_actors = ["Student", "Orchestrator", "Diagnostic", "Confidence", "KG", "Prediction", "Recommend"]
as_msgs = [
    ("Student", "Orchestrator", "submit_answers()", False),
    ("Orchestrator", "Diagnostic", "update_θ(resp)", False),
    ("Diagnostic", "Orchestrator", "θ, SE, mastery", True),
    ("Orchestrator", "Confidence", "classify(resp, θ)", False),
    ("Confidence", "Orchestrator", "conf_class[6]", True),
    ("Orchestrator", "KG", "identify_gaps()", False),
    ("KG", "Orchestrator", "gaps, prereqs", True),
    ("Orchestrator", "Prediction", "predict_gaps(seq)", False),
    ("Prediction", "Orchestrator", "gaps[293]", True),
    ("Orchestrator", "Recommend", "recommend(all)", False),
    ("Recommend", "Orchestrator", "ranked (TS+LM)", True),
    ("Orchestrator", "Student", "personalized", True),
]
draw_pipeline(ax2, "(b) Assessment Pipeline", C_AS, C_BD["as"], as_msgs, as_actors)

# ── (c) Continuous ──────────────────────────────────────────────────
ct_actors = ["Orchestrator", "Personal", "Recommend"]
ct_msgs = [
    ("Orchestrator", "Personal", "cluster(features)", False),
    ("Personal", "Orchestrator", "cluster_id, params", True),
    ("Orchestrator", "Orchestrator", "adjust_params()", False),
    ("Orchestrator", "Recommend", "update_weights()", False),
    ("Recommend", "Orchestrator", "adapted", True),
    ("Orchestrator", "Orchestrator", "continue assessment", False),
]
draw_pipeline(ax3, "(c) Continuous Pipeline", C_CT, C_BD["ct"], ct_msgs, ct_actors)

plt.tight_layout(w_pad=0.5)
save_figure(fig, "fig_orchestrator_pipelines")
fig.savefig("diagrams/orchestrator_pipelines.svg", bbox_inches="tight", format="svg")
print("Saved: diagrams/orchestrator_pipelines.svg")
plt.close(fig)
print("Orchestrator Pipelines diagram done.")
