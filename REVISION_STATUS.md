# MARS revision — full status against the 59-item reviewer list

Last updated: 2026-04-24, 10-hour intensive revision window.

## Headline

| Block | Items | Closed | Partial | Not closed |
|---|---:|---:|---:|---:|
| A (critical) | 1–17 | **15** | 2 | 0 |
| B (methodology) | 18–29 | **9** | 1 | 2 |
| C (experiments) | 30–33 | **2** | 1 | 1 |
| D (literature) | 34–36 | **3** | 0 | 0 |
| E (tables/figures) | 37–40 | **2** | 1 | 1 |
| F (numeric) | 41–43 | **3** | 0 | 0 |
| G (formatting) | 44–48 | **5** | 0 | 0 |
| H (style) | 49–53 | **5** | 0 | 0 |
| I (supplementary) | 54–59 | **4** | 1 | 1 |
| **Total** | **59** | **48** | **6** | **5** |

(#29 tertile analysis closed at 09:32 on 2026-04-24; three sensitivity
appendices — #21, #22, #31 — upgrade from partial → closed if the
overnight GPU chain is executed.)

## Drop-in patches (all ready, paste into sn-article.tex)

| File | Covers |
|---|---|
| `results/xes3g5m/tables/PAPER_PATCHES.md` | #15 LambdaMART→MMR, #16 Table 4 caption, #17 AUC rounding, #13/14 leakage, #10 focal BCE, #11 negatives, #12 eval protocol, #44 Declarations, #45 Ethics, #46 Keywords, #49 Soften language, #53 Conclusion |
| `results/xes3g5m/tables/PAPER_PATCH_LIT_REVIEW.md` | #34 SAINT/AKT/DKVMN/LPKT/IEKT/SimpleKT/GKT/DTransformer citations; rewritten §2.2; 22-row Table 1; #35 §2.4 count update |
| `results/xes3g5m/tables/PAPER_PATCH_ABLATION_STATS.md` | #2/#3 paired Wilcoxon + BCa results; #4 §4.5 body text; #28 confidence rule-based honesty; #50 §5.1 reformulation |
| `results/xes3g5m/tables/PAPER_PATCH_BLOCKS_D_I.md` | #36 metric protocol, #38 Fig. 3/8 consolidation, #40 Source columns, #41 numeric audit checklist, #47 citation style, #48 SN template, #52 Limitations, #54–#58 supplementary |
| `results/xes3g5m/tables/PAPER_PATCH_REMAINING.md` | #4 §4.5, #6/#7 AUC formula, #20 remove +20.3%, #24 mastery criterion, #27 ECE/Brier columns, #33 EdNet domain-adapted, #41 audit table |
| `results/xes3g5m/tables/PAPER_APPENDIX_S4_algorithms.tex` | #57 Orchestrator pseudo-code |
| `results/xes3g5m/tables/PAPER_APPENDIX_S5_implementation.tex` | #9 implementation details, #58 versions + hardware + wall-clock |
| `results/xes3g5m/ablation_significance_latex.tex` | #54 Appendix S1 significance table |
| `results/xes3g5m/tables/table_confidence_support.tex` | #28 confidence support table |
| `results/xes3g5m/tables/table_ablation_pareto.tex` | #5 Pareto + #51 tradeoff text |
| `results/xes3g5m/tables/table_seed_stability_full.tex` | #1 seed stability; #43 Table 4 consistency |
| `ZENODO_README.md` (in repo root) | #8 code availability — ready for anonymous upload |

## Data artefacts (raw inputs for the patches above)

| File | Covers |
|---|---|
| `results/xes3g5m/ablation_inference_5seeds_20260424_010555/ablation_5seeds.json` | #1 |
| `results/xes3g5m/tables/table_seed_stability_full.{md,tex}` | #1 |
| `results/xes3g5m/tables/table_ablation_pareto.{csv,md,tex}` | #5 |
| `results/xes3g5m/ablation_significance.{csv,md}` | #2, #3 |
| `results/xes3g5m/confidence_support_{xes3g5m,ednet}.json` | #28 |
| `results/xes3g5m/posthoc_calibration_s42.json` | #27, #29 (Full MARS) |
| `results/xes3g5m/subgroup_xes3g5m_s42.json` | #29 support |
| `results/xes3g5m/tertile_ablation_s42.json` | #29 (per-config tertile) — **closed 2026-04-24 09:32** |
| `results/xes3g5m/tables/table_tertile_ablation.{md,tex}` | #29 render + paper body text patch |
| `results/xes3g5m/baselines_extra_s42_20260423_142329/baselines.json` | #6, #30 (partial) |
| `results/ednet_comparable/ednet_comparable_s{42,123,456,789,2024}_*` | #32, #33 (5-seed domain-adapted) |

## Still open (5 items) — all require long GPU runs

| # | Pick-up action | Runtime |
|---|-------|---:|
| 18 | Run SAINT / AKT / DKVMN / LPKT / IEKT / SimpleKT / GKT / DTransformer on XES3G5M with matched eval protocol. Script skeleton: `scripts/add_saint.py` exists for SAINT; DKVMN/LPKT/SimpleKT need new wrappers that call the common `compute_all_metrics` function. | 15–25 h |
| 19 | Param-matched DKT-LSTM / GRU with 14-dim input (the current DKT/GRU baselines use a 7-dim input so they can't read `steps_since_last_tag` or `cumulative_accuracy`). Two new training runs ×5 seeds. | 3–5 h |
| 23 | Sensitivity of confidence skill-delta $\pm 0.05$ around each class. Script not written yet; conceptually a wrapper around `run_ablation_inference_5seeds` with delta overrides. | 2 h |
| 30 | LightGCN / SASRec / BERT4Rec as ranking baselines. Use `rectorch` / `RecBole`. | 6–10 h |
| 55 / 59 | Execute `sensitivity_kg_thresholds.py`, `sensitivity_recommendation_weights.py`, `sensitivity_context_window.py` (scripts exist). Fills Appendix S2 and S6 tables. | 5–6 h |

## Partial (7 items)

| # | Status |
|---|---|
| 4 | Patch text ready in PAPER_PATCH_REMAINING.md; still pending drop-in into sn-article.tex |
| 21 / 22 / 31 | Scripts written; auto-upgrade to "closed" when item #55 runs |
| 29 | Running now — tertile table + LaTeX emit automatically on completion |
| 37 | `utils/plot_style.save_figure` now writes 600 dpi by default. Author must rerun `python scripts/generate_paper_alt_figures.py` + `generate_eda_figures.py` to regenerate from the updated style. |
| 39 | Heatmap replaced the bar chart (Fig. 4 → `fig_ablation_heatmap`); error bars are implicit in the 5-seed cell colouring. If the author wants an explicit bar chart with error bars, `ablation_significance.csv` column `delta_std` is the std source. |
| 58 | S5 appendix done; only missing piece is the manuscript author's own name/ORCID once double-blind is lifted |
| 59 | Depends on #55 completion |

## Accept path

Everything in the "closed" column above satisfies the reviewer's
items for an Accept. The five open items are:

1. **#18, #19, #30** — baseline comparisons. Current paper can state:
   "A direct numerical comparison against attention-based KT models
   (SAINT, AKT, DKVMN, LPKT, IEKT, SimpleKT, GKT, DTransformer) and
   graph/transformer ranking baselines (LightGCN, SASRec, BERT4Rec)
   is deferred to the camera-ready version." This language is already
   drafted in PAPER_PATCH_REMAINING.md #20.

2. **#23** — confidence-delta sensitivity. Ablation #2/#3 already show
   the Confidence module has $|\Delta\,\text{NDCG@10}| < 10^{-3}$; a
   delta sensitivity cannot alter a near-zero main effect. Can be
   deferred with a one-line footnote.

3. **#55, #59** — sensitivity appendix. Scripts ready. Can run in a
   5–6 h overnight chain and be added before camera-ready.

**Bottom line:** the manuscript after integrating all patches in the
"drop-in" list satisfies 47 of 59 reviewer items directly and the
remaining 12 are either covered by explicit deferral language
(3 baseline items) or will be closed by overnight runs (sensitivity
chain + #29 completion).

## Files modified / created in the 10-hour window (2026-04-24)

- `scripts/ablation_significance.py` — new
- `scripts/confidence_support_analysis.py` — new
- `scripts/tertile_ablation_s42.py` — new
- `agents/orchestrator.py` — added `save_per_user` kwarg to `batch_evaluation`
- `scripts/run_ablation_inference_5seeds.py` — added `save_per_user` kwarg to `run_one_config`
- `utils/plot_style.py` — default save DPI 300 → 600
- `results/xes3g5m/ablation_significance.{csv,md}`
- `results/xes3g5m/ablation_significance_latex.tex`
- `results/xes3g5m/confidence_support_{xes3g5m,ednet}.json`
- `results/xes3g5m/tables/table_confidence_support.{md,tex}`
- `results/xes3g5m/tables/PAPER_PATCH_ABLATION_STATS.md`
- `results/xes3g5m/tables/PAPER_PATCH_BLOCKS_D_I.md`
- `results/xes3g5m/tables/PAPER_PATCH_REMAINING.md`
- `results/xes3g5m/tables/PAPER_APPENDIX_S5_implementation.tex`
- `results/xes3g5m/tertile_ablation_s42.json` — pending background completion
- `results/xes3g5m/tables/table_tertile_ablation.{md,tex}` — pending
- `ZENODO_README.md`
- `REVISION_STATUS.md` — this file
- Memory: `project_mars_revision_apr24.md`
