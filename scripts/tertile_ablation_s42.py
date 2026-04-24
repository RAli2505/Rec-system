"""
Per-tertile NDCG@10 / MRR for each ablation config, seed 42 only
(reviewer A.1 #29, closing the missing piece in posthoc_calibration).

Strategy:
  1. Reuse run_one_config from run_ablation_inference_5seeds with
     save_per_user=True so every user's NDCG@10 and MRR come back.
  2. Run 5 configs (Full MARS + 4 ablations) on seed 42.
  3. Compute each user's ability proxy = mean correctness on the
     first 30% of their test sequence (same 30/70 context/eval split
     as the paper's eval protocol).
  4. Define ability tertiles from Full MARS's user list ONCE, so every
     config is evaluated against the SAME bucket edges. This is the
     subgroup split the reviewer asked for.
  5. For each config × tertile, report mean ± std NDCG@10 and MRR
     plus n_users.

Output
------
results/xes3g5m/tertile_ablation_s42.json
results/xes3g5m/tables/table_tertile_ablation.md
results/xes3g5m/tables/table_tertile_ablation.tex

Runtime: ~20 min on a single GPU (5 configs × ~3 min each).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from agents.prediction_agent import set_num_tags
from data.xes3g5m_loader import load_xes3g5m
from scripts.run_xes3g5m_full import (
    build_xes3g5m_questions_df, build_xes3g5m_lectures_df,
)
from scripts.run_ablation_inference_5seeds import (
    newest_main_run, run_one_config,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("tertile_ablation")

SEED = 42
CONTEXT_RATIO = 0.3

# Full MARS has special handling — we still run it through the
# orchestrator so per-user metrics come back. If memory is tight the
# script can be trivially changed to rely on a cached per-user file.
CONFIGS: list[tuple[str, dict]] = [
    ("Full MARS",         {}),
    ("- Prediction",       {"disable_prediction": True}),
    ("- Knowledge Graph",  {"disable_kg":         True}),
    ("- Confidence",       {"disable_confidence": True}),
    ("- IRT (Diagnostic)", {"disable_irt":        True}),
]


def per_user_ability(test_df: pd.DataFrame, context_ratio: float) -> dict[str, float]:
    """Ability proxy = accuracy on the first `context_ratio` fraction of each
    user's chronologically ordered test interactions. Identical split to the
    one batch_evaluation uses, so the user set is the same."""
    out: dict[str, float] = {}
    for uid, grp in test_df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        split = max(1, int(len(grp) * context_ratio))
        ctx = grp.iloc[:split]
        if len(ctx) == 0:
            continue
        out[str(uid)] = float(ctx["correct"].astype(bool).mean())
    return out


def bucketize(acc: float, low_hi: float, mid_hi: float) -> str:
    if acc <= low_hi:
        return "low"
    if acc <= mid_hi:
        return "mid"
    return "high"


def summarize_config(
    per_user: list[dict],
    ability: dict[str, float],
    edges: tuple[float, float],
) -> dict:
    low_hi, mid_hi = edges
    rows = []
    for row in per_user:
        uid = row["user_id"]
        if uid not in ability:
            continue
        rows.append({
            "user_id": uid,
            "ability": ability[uid],
            "bucket": bucketize(ability[uid], low_hi, mid_hi),
            "ndcg_at_10": row["ndcg_at_k"],
            "mrr": row["mrr"],
            "precision_at_10": row["precision_at_k"],
        })
    df = pd.DataFrame(rows)
    summary = {}
    for bucket in ["low", "mid", "high"]:
        sub = df[df["bucket"] == bucket]
        if len(sub) == 0:
            summary[bucket] = {"n": 0}
            continue
        summary[bucket] = {
            "n_users":      int(len(sub)),
            "ability_mean": round(float(sub["ability"].mean()), 4),
            "ndcg@10_mean": round(float(sub["ndcg_at_10"].mean()), 4),
            "ndcg@10_std":  round(float(sub["ndcg_at_10"].std(ddof=1)), 4),
            "mrr_mean":     round(float(sub["mrr"].mean()), 4),
            "mrr_std":      round(float(sub["mrr"].std(ddof=1)), 4),
            "p@10_mean":    round(float(sub["precision_at_10"].mean()), 4),
        }
    summary["overall"] = {
        "n_users":      int(len(df)),
        "ndcg@10_mean": round(float(df["ndcg_at_10"].mean()), 4),
        "mrr_mean":     round(float(df["mrr"].mean()), 4),
    }
    return summary


def main() -> int:
    # 1. Find pretrained seed_42 main-pipeline checkpoint
    run = newest_main_run(SEED)
    if run is None:
        logger.error("No main-pipeline run with best.pt for seed %d", SEED)
        return 1
    pretrained_pt = run / "best.pt"
    logger.info("Using pretrained checkpoint: %s", pretrained_pt)

    # 2. Load XES3G5M once
    train_df, val_df, test_df = load_xes3g5m(
        n_students=6000, min_interactions=20, seed=SEED,
    )
    for df in [train_df, val_df, test_df]:
        df["confidence_class"] = 0
    train_max_id = max(int(t) for tags in train_df["tags"]
                        if isinstance(tags, list) and tags
                        for t in tags)
    set_num_tags(train_max_id + 1)

    questions_df = build_xes3g5m_questions_df("data/xes3g5m/XES3G5M")
    for col, default in [("bundle_id", lambda: questions_df["question_id"]),
                         ("correct_answer", lambda: "A"),
                         ("deployed_at", lambda: 0)]:
        if col not in questions_df.columns:
            questions_df[col] = default() if callable(default) else default
    lectures_df = pd.DataFrame({
        "lecture_id": [], "tags": [], "part_id": [],
        "type_of": [], "bundle_id": [],
    })

    # 3. Compute ability per user and the tertile edges from Full MARS universe
    ability = per_user_ability(test_df, CONTEXT_RATIO)
    abilities = np.array(list(ability.values()))
    low_hi, mid_hi = np.quantile(abilities, [1/3, 2/3])
    edges = (float(low_hi), float(mid_hi))
    logger.info("Tertile edges (33/66 percentiles): low<=%.3f, mid<=%.3f",
                *edges)

    # 4. Run each config with save_per_user=True
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: dict[str, dict] = {
        "meta": {
            "seed": SEED,
            "context_ratio": CONTEXT_RATIO,
            "n_users_with_ability": len(ability),
            "tertile_edges": {"low_max": edges[0], "mid_max": edges[1]},
            "pretrained_checkpoint": str(pretrained_pt),
            "timestamp": ts,
        },
    }
    for name, kwargs in CONFIGS:
        logger.info("─── running config: %s ───", name)
        t0 = time.time()
        metrics = run_one_config(
            name, SEED, pretrained_pt,
            train_df, val_df, test_df,
            questions_df, lectures_df,
            save_per_user=True,
            **kwargs,
        )
        dt = time.time() - t0
        per_user = metrics.pop("per_user", [])
        logger.info("  eval done in %.1fs, %d per-user rows", dt, len(per_user))
        agg = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
        results[name] = {
            "aggregate": agg,
            "tertile_summary": summarize_config(per_user, ability, edges),
            "runtime_s": round(dt, 1),
        }

    # 5. Persist
    out_json = ROOT / "results/xes3g5m/tertile_ablation_s42.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_json)

    # 6. Render markdown + latex
    render_markdown(results, ROOT / "results/xes3g5m/tables/table_tertile_ablation.md")
    render_latex(results,    ROOT / "results/xes3g5m/tables/table_tertile_ablation.tex")
    return 0


def render_markdown(results: dict, out: Path) -> None:
    cfgs = [c for c, _ in CONFIGS]
    low_hi = results["meta"]["tertile_edges"]["low_max"]
    mid_hi = results["meta"]["tertile_edges"]["mid_max"]

    lines = [
        "# Per-tertile NDCG@10 / MRR for each ablation config (seed 42)",
        "",
        f"Ability proxy = mean correctness on the first {int(results['meta']['context_ratio']*100)}% of each user's test sequence.",
        f"Tertile edges from the XES3G5M test set (equal-frequency, 33/66 percentiles):",
        f"- **low** ≤ {low_hi:.3f}",
        f"- **mid** ∈ ({low_hi:.3f}, {mid_hi:.3f}]",
        f"- **high** > {mid_hi:.3f}",
        "",
        "## NDCG@10 by ability tertile",
        "",
        "| Config | Low (n) | Mid (n) | High (n) | Overall |",
        "|---|---:|---:|---:|---:|",
    ]
    for cfg in cfgs:
        s = results[cfg]["tertile_summary"]
        def fmt(bucket):
            if s[bucket].get("n_users", 0) == 0:
                return "—"
            return f"{s[bucket]['ndcg@10_mean']:.3f}±{s[bucket]['ndcg@10_std']:.3f} (n={s[bucket]['n_users']})"
        lines.append(
            f"| {cfg} | {fmt('low')} | {fmt('mid')} | {fmt('high')} "
            f"| {s['overall']['ndcg@10_mean']:.3f} |"
        )

    lines += ["", "## MRR by ability tertile", "",
              "| Config | Low | Mid | High | Overall |",
              "|---|---:|---:|---:|---:|"]
    for cfg in cfgs:
        s = results[cfg]["tertile_summary"]
        def fmt(bucket):
            if s[bucket].get("n_users", 0) == 0:
                return "—"
            return f"{s[bucket]['mrr_mean']:.3f}±{s[bucket]['mrr_std']:.3f}"
        lines.append(
            f"| {cfg} | {fmt('low')} | {fmt('mid')} | {fmt('high')} "
            f"| {s['overall']['mrr_mean']:.3f} |"
        )

    # Delta vs Full within each tertile
    lines += [
        "",
        "## Δ NDCG@10 within each tertile (Full MARS − ablated)",
        "",
        "Positive Δ = ablated variant hurts vs Full. ",
        "Negative Δ = ablated variant *improves* NDCG@10 in that tertile — the ",
        "subgroup evidence of the IRT coverage/accuracy trade-off reported in §4.5.",
        "",
        "| Ablation | Δ Low | Δ Mid | Δ High |",
        "|---|---:|---:|---:|",
    ]
    full = results["Full MARS"]["tertile_summary"]
    for cfg in cfgs:
        if cfg == "Full MARS":
            continue
        s = results[cfg]["tertile_summary"]
        def delta(bucket):
            if s[bucket].get("n_users", 0) == 0 or full[bucket].get("n_users", 0) == 0:
                return "—"
            return f"{full[bucket]['ndcg@10_mean'] - s[bucket]['ndcg@10_mean']:+.3f}"
        lines.append(
            f"| {cfg} | {delta('low')} | {delta('mid')} | {delta('high')} |"
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out)


def render_latex(results: dict, out: Path) -> None:
    cfgs = [c for c, _ in CONFIGS]
    low_hi = results["meta"]["tertile_edges"]["low_max"]
    mid_hi = results["meta"]["tertile_edges"]["mid_max"]

    lines = [
        "% Auto-generated by scripts/tertile_ablation_s42.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-tertile NDCG@10 by ablation config on XES3G5M "
        f"(seed~42). Ability proxy is mean correctness on the first 30\\% of "
        f"each user's test sequence; tertile edges are "
        f"$[0, {low_hi:.3f}]$ (low), "
        f"$({low_hi:.3f}, {mid_hi:.3f}]$ (mid), "
        f"$({mid_hi:.3f}, 1]$ (high). A negative subgroup delta against "
        "Full~MARS (e.g. $-$IRT in the low tertile) indicates that the ablated "
        "variant ranks better for the subgroup it targets, which is the "
        "expected behaviour of the coverage-accuracy trade-off described "
        "in \\S\\ref{subsec4.5}.}",
        "\\label{tab:tertile_ablation}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Config & Low & Mid & High & Overall\\\\",
        "\\midrule",
    ]
    for cfg in cfgs:
        s = results[cfg]["tertile_summary"]
        cells = []
        for b in ["low", "mid", "high"]:
            if s[b].get("n_users", 0) == 0:
                cells.append("---")
            else:
                cells.append(f"{s[b]['ndcg@10_mean']:.3f}$\\pm${s[b]['ndcg@10_std']:.3f}")
        name_tex = cfg.replace("-", "$-$").strip()
        lines.append(
            f"{name_tex} & {cells[0]} & {cells[1]} & {cells[2]} "
            f"& {s['overall']['ndcg@10_mean']:.3f}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    sys.exit(main())
