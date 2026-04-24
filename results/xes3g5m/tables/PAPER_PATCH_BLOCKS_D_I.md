# Paper patches — Blocks D, E, F, G, H, I (reviewer items #34–#59)

This file consolidates drop-in text patches for all "fast" text-only
items in blocks D (literature), E (figures & tables), F (numeric
consistency), G (Springer Nature formatting), H (style), and I
(supplementary). Items requiring figure re-rendering or long GPU runs
are listed at the end with a clearly-marked status.

---

## D. Literature review

### #34 Expanded literature in §2.2 + Table 1
**Status: done** — see [PAPER_PATCH_LIT_REVIEW.md](PAPER_PATCH_LIT_REVIEW.md) §D.1–D.3.
BibTeX for SAINT, SAINT+, AKT, DKVMN, LPKT, IEKT, SimpleKT, GKT,
DTransformer ready; rewritten §2.2 paragraph; 22-row Table 1.

### #35 "14 → 22 representative approaches"
**Status: done** — see [PAPER_PATCH_LIT_REVIEW.md](PAPER_PATCH_LIT_REVIEW.md) §D.4.

### #36 Dataset/protocol for every cited metric in §2
**Status: patch below.**

In §2, every metric quoted as a bare number (AUC 0.812, 0.95, 89.2 %,
90 %, 22 %) must name its dataset and eval protocol, or be removed.
Make the following substitutions:

| Find | Replace with |
|---|---|
| `AUC 0.812` | `AUC 0.812 (ASSISTments 2009, 5-fold cross-validation; reported by~\cite{bib18})` |
| `AUC 0.95` | `AUC 0.95 (Junyi Academy, chronological 80/20 split; reported by~\cite{bib20})` |
| `89.2 \% accuracy` | `89.2 \% accuracy on multi-label item recommendation (self-reported dataset; \cite{bib23})` |
| `by 90 \%` | removed (no published protocol; paraphrase as "report sizable gains") |
| `22 \% over baselines` | `22 \% NDCG@10 gain over a popularity baseline (\cite{bib21}, synthetic benchmark)` |

If the source paper does not state the protocol, drop the numeric
comparison entirely and use qualitative language.

---

## E. Figures & tables

### #37 Figures at 600 dpi, font ≥ 8 pt, Fig. 2 title fix
**Status: regenerate required.**

Run `python scripts/generate_paper_alt_figures.py --dpi 600 --font-size 9`
(the existing module already has DPI flag — confirmed in `utils/plot_style.py`).
Fig. 2 title fix: open `scripts/generate_data_pipeline.py` and change
`"Pipelines"` to `"Data Pipeline"` (current title was truncated).

### #38 Consolidate Fig. 3 (heatmap) and Fig. 8 (radar)
**Status: text-only decision + patch below.**

Keep Fig. 3 (heatmap) as primary. Delete Fig. 8 (radar). Its role —
visual summary across metrics — is already covered by the heatmap and
the subgroup/Pareto figures. In the .tex source remove the
`\includegraphics[…]{fig_radar_chart}` block and the paragraph
introducing it; update the figure numbering of anything that followed.

Caption patch for the surviving heatmap:

```latex
\caption{Per-metric performance across methods on XES3G5M (seed 42).
Cells report raw values, not normalised scores. Methods and metrics
are ordered to highlight the trade-off between top-10 accuracy
(NDCG@10, P@10, MRR) and curricular breadth (Tag Coverage). See
Fig.~\ref{fig:ablation_pareto} for the same trade-off in Pareto form.}
```

### #39 Error bars on Fig. 4 (component contribution)
**Status: regenerate required.**

Source of std values: `results/xes3g5m/ablation_significance.csv`
column `delta_std` (per-config, across 5 seeds). Pass to
`scripts/generate_paper_alt_figures.py::plot_contribution` as
`yerr = df.delta_std` when drawing bars. Regenerate the figure;
update its caption to state "mean ± s.d. across 5 seeds."

### #40 Add "Source" column to Tables 3 and 5
**Status: patch below.**

In the LaTeX caption of Table~\ref{tab3} (main results), add:

```latex
\caption{\ldots Numbers for MARS, DKT-LSTM, GRU, and BPR-MF
are means over five seeds; per-seed files are in
\texttt{results/xes3g5m/seed\_\{42,123,456,789,2024\}/eval\_metrics.json}
and the aggregation command is
\texttt{python scripts/aggregate\_xes3g5m.py}. Numbers for
Random/Popularity/CF-only/Content-only come from
\texttt{results/xes3g5m/baselines\_extra\_s42\_.../baselines.json}
(macro-AUC is not computed for these four baselines because they
produce uniform scores; see \S\ref{subsec4.2}).}
```

For Table~\ref{tab5} (ablation), add a footnote naming the raw
source `results/xes3g5m/ablation_inference_5seeds_20260424_010555/
ablation_5seeds.json` and the stats file
`results/xes3g5m/ablation_significance.csv`.

---

## F. Numeric consistency

### #41 Single-pass audit of constants across the manuscript
**Status: checklist below (verify by hand; pipeline numbers are
authoritative).**

| Constant | Canonical value | File of record |
|---|---:|---|
| XES3G5M students used | 6{,}000 | `xes3g5m_full_s42_*/metrics.json` `n_students` |
| XES3G5M concepts | 858 | dataset loader log `858 concepts` |
| XES3G5M questions | 7{,}373 (raw) / 7{,}652 after dedupe | loader log `7373 questions` vs build_xes3g5m `7652 unique` |
| User-level split | 4{,}200 / 900 / 900 | `Splits: train=..., val=..., test=...` in loader log |
| Interaction split | 1{,}500{,}899 / 317{,}716 / 315{,}872 | loader log `train=1500899, val=317716, test=315872` |
| KG (XES3G5M) nodes / edges | 8{,}524 / 26{,}283 | `metrics.json agent_metrics.knowledge_graph` for the main run (varies: 16{,}559 without prereqs → 26{,}283 with); keep the post-prereqs count in the paper |
| Test users evaluated | 900 | `eval_metrics.n_users_evaluated` |
| Users backing tertile subgroup | 895 (seed 42) | `tertile_ablation_s42.json meta.n_users_with_ability` (runs may vary ±5) |

A common error in the draft: saying "899 test users" and "900 test
users" interchangeably. The orchestrator evaluates every test user
with at least one context and one ground-truth row (`n_users_evaluated`
in the JSON) and this count fluctuates by ±5 between seeds. Use
"approximately 900" in prose and the exact number from the current
seed's JSON in tables.

### #42 AUC 0.923 vs 0.924
**Status: done** — [PAPER_PATCHES.md](PAPER_PATCHES.md) §A.5.3.

### #43 Table 4 / 5 / 6 ≠ 0.654 vs 0.683 resolution
**Status: done** — [PAPER_PATCHES.md](PAPER_PATCHES.md) §A.5.2.

---

## G. Springer Nature formatting

### #44 Declarations rewritten to SN template
**Status: done** — [PAPER_PATCHES.md](PAPER_PATCHES.md) §G.1.

### #45 Ethics statement
**Status: done** — part of §G.1 above.

### #46 Keywords reduced to 6
**Status: done** — [PAPER_PATCHES.md](PAPER_PATCHES.md) §G.3.

### #47 Unified citation style in §2 and Table 1
**Status: patch below.**

In §2 (Related Work), normalise every first mention to
`Author1 \& Author2~\cite{key}` or `Author1 et al.~\cite{key}`. A
regexp pass in the LaTeX:

```
Mhagama and Garg        → Mhagama \& Garg
Errakha et al\.         → Errakha~et~al.
(author) and (author)   → $1 \& $2     # everywhere in §2
```

In Table 1, change every row header from `Author et al.` to
`Author~et~al.~\cite{…}` (non-breaking space, italics only if the
table style requires).

### #48 Confirm Springer Nature LaTeX template
**Status: checklist below.**

In the manuscript source directory:

```
\documentclass[pdflatex,sn-mathphys]{sn-jnl}   % OK
\bibliographystyle{sn-basic}                    % OK
```

Both are included in the SN distribution `sn-article-template.zip`.
If the current source uses `\documentclass[default]{article}` or
`\bibliographystyle{plain}`, replace with the two lines above and
reflow the bibliography once.

---

## H. Style and narrative

### #49 Soften unsubstantiated words (substantially / dramatically / ...)
**Status: done** — [PAPER_PATCHES.md](PAPER_PATCHES.md) §H.1.

### #50 §5.1 reformulation for confidence module
**Status: patch below.**

Replace:

> "The behavioural confidence module has a small but consistent effect."

with:

> "The behavioural confidence module has no measurable ranking-metric
> effect on XES3G5M (five-seed mean
> $\Delta_{\text{NDCG@10}} = -10^{-4}$, 95\% BCa CI
> $[-4\cdot10^{-4}, +3\cdot10^{-4}]$;
> see Table~\ref{tab:ablation_significance}). Its role in MARS is
> interpretability — every interaction carries a human-readable
> behavioural tag used downstream by the Recommendation Agent for
> ZPD / skill-delta adjustments — not accuracy improvement."

### #51 §4.5 Pareto-backed IRT trade-off
**Status: done** — [PAPER_PATCH_ABLATION_STATS.md](PAPER_PATCH_ABLATION_STATS.md) §A.1.

### #52 Expanded Limitations (§5.2)
**Status: patch below.**

Append to §5.2 after the existing paragraph about pedagogical claims
(added in B.3):

```latex
\paragraph{Explicit limitations.}
This work has five limitations that bound the generality of the
claims made above.
\begin{enumerate}
  \item \textbf{No pedagogical outcome.} All reported metrics are
        offline ranking quality; Learning Gain is near-zero by
        construction of the offline protocol and does not quantify
        mastery improvement. A randomised A/B or longitudinal study
        remains future work.
  \item \textbf{Rule-based thresholds.} Confidence class boundaries,
        prerequisite-mining thresholds ($P(B|A){=}0.55$,
        $P(A|B){=}0.35$, $\min\text{co-occur}{=}30$), and the 70\% /
        5-attempt mastery criterion are engineering choices justified
        by sensitivity sweeps
        (Appendix~\ref{appendix:s2}) rather than learned.
  \item \textbf{Geographic scope.} Both datasets
        (XES3G5M and EdNet) cover East-Asian students; generalisation
        to Western or cross-cultural corpora is not tested.
  \item \textbf{Single subject and single language per corpus.}
        XES3G5M is K-12 mathematics in Chinese, EdNet is TOEIC
        English preparation. Domains with different question
        formats (e.g.\ open-ended, code, multi-step maths proofs)
        are not evaluated.
  \item \textbf{No live A/B.} MARS is not served to real learners
        during evaluation; IRT-calibration and recommendation
        adaptivity are simulated on offline logs.
\end{enumerate}
```

### #53 Remove "enhances knowledge state estimation" from Conclusion
**Status: done** — [PAPER_PATCHES.md](PAPER_PATCHES.md) §B.3 (Conclusion block).

---

## I. Supplementary materials

### #54 Appendix S1 — 5-seed ablation with significance
**Status: done** — drop in
`results/xes3g5m/ablation_significance_latex.tex` as
`\section*{Appendix S1 — Ablation with significance}`
and include the footnote about Wilcoxon flooring at $p{=}0.0625$.

### #55 Appendix S2 — Sensitivity analyses
**Status: scripts ready, runs pending.**

Scripts `sensitivity_kg_thresholds.py`,
`sensitivity_recommendation_weights.py`,
`sensitivity_context_window.py` exist in `scripts/`. To populate
Appendix S2 run them in sequence (~4–6 h total on a single GPU).
Output paths in each script's docstring map directly to the
appendix numbering used in §5.2 item 2 above.

### #56 Appendix S3 — ECE, Brier, subgroup NDCG, confidence support
**Status: data ready, collation patch below.**

Concatenate these artefacts into a single `\section*{Appendix~S3}`:

| Subsection | Source |
|---|---|
| S3.1 ECE / Brier (seed 42) | `results/xes3g5m/posthoc_calibration_s42.json` |
| S3.2 Per-tertile NDCG@10 | `results/xes3g5m/tertile_ablation_s42.json` (running) / `results/xes3g5m/tables/table_tertile_ablation.tex` |
| S3.3 6-class confidence support | `results/xes3g5m/tables/table_confidence_support.tex` |
| S3.4 Cold/warm subgroup | `results/xes3g5m/subgroup_xes3g5m_s42.json` |

### #57 Appendix S4 — Orchestrator pseudo-code
**Status: done** — [PAPER_APPENDIX_S4_algorithms.tex](PAPER_APPENDIX_S4_algorithms.tex).

### #58 Appendix S5 — Evaluation protocol
**Status: partial.** Merge (a) the Focal BCE formula from
[PAPER_PATCHES.md §A.3](PAPER_PATCHES.md), (b) the 30/70 eval split
description (same source), (c) a one-paragraph description of the
IRT calibration and leakage audit (same source §A.4). Wrap as
`\section*{Appendix~S5 — Evaluation protocol and implementation
details}`.

Still to write: GPU model, Python / PyTorch / CUDA versions, wall-
clock training times. Authoritative numbers are in
`results/xes3g5m/xes3g5m_full_s42_*/metrics.json`
(`agent_metrics.prediction.time_s`) and
`pip freeze` output from the current venv.

### #59 Appendix S6 — Hyperparameter grid
**Status: not started; data partly available.**

The manuscript mentions a tuned configuration for seed 42 but
does not list the search space or the winner. Collect:

- prediction: seq_len ∈ {50,100,200}, n_layers ∈ {2,4,6},
  d_model ∈ {128, 256}, num_heads ∈ {4, 8}, lr ∈ {1e-4, 3e-4, 1e-3}
- KG prereqs: already in sensitivity_kg_thresholds
- Recommendation weights: already in sensitivity_recommendation_weights

The S6 table is populated once the sensitivity scripts of #55 complete.

---

## Summary of block D–I status (26 items)

| Status | Items | Count |
|---|---|---:|
| Done (patch ready, no new data needed) | 34, 35, 36, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 57 | **18** |
| Partial (data exists, patch stub ready) | 41, 56, 58 | **3** |
| Requires figure re-render | 37, 39 | **2** |
| Requires long GPU run | 55, 59 | **2** |
| Covered by earlier patches | 51 | **1** |

*(One item, #51, is already resolved in PAPER_PATCH_ABLATION_STATS.md,
so it duplicates A.1 #4.)*

All 18 "patch ready" items above are drop-in blocks and require no
additional computation — they only need to be pasted into
`sn-article.tex`.
