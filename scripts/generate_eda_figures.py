"""
Generate publication-quality composite EDA figures for MARS paper (Section 4.1).

F3: Dataset Overview (2×2) — fig_dataset_overview.{png,pdf}
F4: Engineered Features (2×2) — fig_engineered_features_composite.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_style import (
    setup_publication_style, save_figure, add_panel_label,
    MARS_COLORS, DOUBLE_COL_TALL, DOUBLE_COL,
)

setup_publication_style()

# ── Load data ────────────────────────────────────────────────────────
train_df = pd.read_parquet("data/splits/train.parquet")
val_df = pd.read_parquet("data/splits/val.parquet")
test_df = pd.read_parquet("data/splits/test.parquet")
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Total: {len(df):,} interactions, {df['user_id'].nunique()} users")

PART_NAMES = {
    1: "Part 1: Photographs\n(Listening)",
    2: "Part 2: Question-Response\n(Listening)",
    3: "Part 3: Short Conversations\n(Listening)",
    4: "Part 4: Short Talks\n(Listening)",
    5: "Part 5: Incomplete Sentences\n(Reading)",
    6: "Part 6: Text Completion\n(Reading)",
    7: "Part 7: Reading Comprehension\n(Reading)",
}

palette = MARS_COLORS["primary"]

# ═════════════════════════════════════════════════════════════════════
# FIGURE F3: Dataset Overview (2×2)
# ═════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COL_TALL)

# ── (a) Activity by TOEIC Part ───────────────────────────────────────
ax = axes[0, 0]
part_counts = df["part_id"].value_counts().sort_index()
parts = [PART_NAMES.get(int(p), f"Part {p}") for p in part_counts.index]
bars = ax.bar(range(len(parts)), part_counts.values, color=palette[:len(parts)],
              edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(parts)))
ax.set_xticklabels(parts, rotation=35, ha="right", fontsize=7)
ax.set_ylabel("Number of Interactions")
ax.set_title("Activity by TOEIC Part")
add_panel_label(ax, "a")

# ── (b) Answers per Student ──────────────────────────────────────────
ax = axes[0, 1]
answers_per_user = df.groupby("user_id").size()
ax.hist(answers_per_user, bins=50, color=palette[0], edgecolor="white",
        linewidth=0.5, alpha=0.85)
median_val = answers_per_user.median()
ax.axvline(median_val, color=palette[3], linestyle="--", linewidth=1.5)
ax.text(median_val + 2, ax.get_ylim()[1] * 0.85,
        f"median = {median_val:.0f}", fontsize=8, color=palette[3])
ax.set_xlabel("Answers per Student")
ax.set_ylabel("Number of Students")
ax.set_title("Answers per Student Distribution")
add_panel_label(ax, "b")

# ── (c) Elapsed Time Distribution ────────────────────────────────────
ax = axes[1, 0]
elapsed_sec = df["elapsed_time"].clip(upper=300000) / 1000  # ms → seconds
ax.hist(elapsed_sec, bins=80, color=palette[2], edgecolor="white",
        linewidth=0.3, alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel("Response Time (seconds, log scale)")
ax.set_ylabel("Frequency")
ax.set_title("Response Time Distribution")
add_panel_label(ax, "c")

# ── (d) Correct/Incorrect Distribution ───────────────────────────────
ax = axes[1, 1]
correct_counts = df["correct"].value_counts()
labels = ["Correct", "Incorrect"]
values = [correct_counts.get(True, 0), correct_counts.get(False, 0)]
colors_cb = [palette[0], palette[1]]  # blue/orange (colorblind-safe)
wedges, texts, autotexts = ax.pie(
    values, labels=labels, colors=colors_cb, autopct="%1.1f%%",
    startangle=90, textprops={"fontsize": 9},
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight("bold")
ax.set_title("Correct Answer Distribution")
add_panel_label(ax, "d", x=-0.05)

fig.suptitle("", fontsize=1)  # clear any default
plt.tight_layout()
save_figure(fig, "fig_dataset_overview")
plt.close(fig)
print("F3 done.")


# ═════════════════════════════════════════════════════════════════════
# FIGURE F4: Engineered Features (2×2)
# ═════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COL_TALL)

# ── (a) Correlation Matrix ───────────────────────────────────────────
ax = axes[0, 0]
feature_cols = ["tag_accuracy", "avg_elapsed_by_tag", "rolling_accuracy",
                "time_since_last", "elapsed_time", "response_count"]
# Truncate names > 15 chars
display_names = [c[:15] for c in feature_cols]
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            xticklabels=display_names, yticklabels=display_names,
            annot_kws={"fontsize": 7}, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
ax.set_title("Feature Correlation")
add_panel_label(ax, "a")

# ── (b) Top-20 Tags ─────────────────────────────────────────────────
ax = axes[0, 1]
# Parse tags and count
from collections import Counter
tag_counter = Counter()
for tags_str in df["tags"].dropna():
    if isinstance(tags_str, str):
        for t in tags_str.split(";"):
            t = t.strip()
            if t.isdigit():
                tag_counter[int(t)] += 1
    elif isinstance(tags_str, (int, np.integer)):
        tag_counter[int(tags_str)] += 1

top20 = tag_counter.most_common(20)
tag_ids = [f"Tag {t[0]}" for t in top20]
tag_vals = [t[1] for t in top20]
ax.barh(range(len(tag_ids)), tag_vals, color=palette[0], edgecolor="white", linewidth=0.3)
ax.set_yticks(range(len(tag_ids)))
ax.set_yticklabels(tag_ids, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Frequency")
ax.set_title("Top-20 Tags (TOEIC Item Bank)")
add_panel_label(ax, "b")

# ── (c) Tag Accuracy Distribution ────────────────────────────────────
ax = axes[1, 0]
tag_acc = df["tag_accuracy"].dropna()
ax.hist(tag_acc, bins=60, color=palette[2], edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_xlabel("Tag Accuracy")
ax.set_ylabel("Frequency")
ax.set_title("Tag Accuracy Distribution")
# Annotate spikes
ax.annotate("Cold-start\n(acc=0 or 1)", xy=(0.02, ax.get_ylim()[1] * 0.7),
            fontsize=7, fontstyle="italic", color="gray")
add_panel_label(ax, "c")

# ── (d) Rolling Accuracy Distribution ────────────────────────────────
ax = axes[1, 1]
rolling = df["rolling_accuracy"].dropna()
ax.hist(rolling, bins=50, color=palette[4], edgecolor="white", linewidth=0.3, alpha=0.85)
ax.set_xlabel("Rolling Accuracy (window=20)")
ax.set_ylabel("Frequency")
ax.set_title("Rolling Accuracy Distribution")
add_panel_label(ax, "d")

plt.tight_layout()
save_figure(fig, "fig_engineered_features_composite")
plt.close(fig)
print("F4 done.")

print("\nAll composite EDA figures generated.")
