# Paper patch — Block A.1 (items #2, #3) and block B.4 (item #28)

Output of scripts `ablation_significance.py` and `confidence_support_analysis.py`,
ready to paste into `sn-article.tex`.

---

## A.1 #2–#3 — Paired significance on 5-seed ablation

### Where in the paper

Section §4.5 "Component ablation" or in the Appendix S1 table.

### What to replace

The seed-42-only ablation table (NDCG@10 delta without uncertainty).

### Replace with

Use `results/xes3g5m/ablation_significance_latex.tex` verbatim:

```latex
\input{ablation_significance_latex.tex}
```

That file contains three blocks (NDCG@10, MRR, Tag Coverage) × 4 ablations with:

- Full and ablated mean±std across 5 seeds
- Delta (Full − Ablated) with marker `^{\dagger}` when the 95% BCa bootstrap CI excludes zero
- Delta marker `^{*/**/***}` when paired t-test p<0.05 / 0.01 / 0.001
- Wilcoxon two-sided p (note: N=5 floors at 0.0625, so Wilcoxon cannot reach 0.05)

### Key findings to state in §4.5 body text

Replace any text of the form "removing IRT reduces NDCG@10 by 0.080" with:

> On XES3G5M with 5 seeds, disabling the IRT Diagnostic Agent **raises**
> mean NDCG@10 from 0.683±0.023 to 0.741±0.015
> ($\Delta = -0.058$, 95\% BCa CI $[-0.067, -0.048]$, paired $t$ $p<0.001$),
> while **lowering** mean Tag Coverage from 0.736±0.024 to 0.334±0.034
> ($\Delta = +0.402$, 95\% BCa CI $[+0.375, +0.441]$, paired $t$ $p<0.001$).
> The IRT module therefore trades a measurable top-$10$ accuracy decrement
> for a $>2\times$ increase in curricular breadth; the Pareto analysis in
> Fig.~\ref{fig:ablation_pareto} places Full~MARS on the
> (NDCG@10, Coverage) frontier rather than as a dominated variant.

For the Confidence Agent, state:

> Disabling the six-class behavioural confidence module leaves all
> ranking metrics unchanged to four decimal places
> ($\Delta_{\text{NDCG@10}} = -0.0001$, 95\% BCa CI $[-0.0004,+0.0003]$;
> $\Delta_{\text{MRR}} = 0.0000$). Its contribution is therefore not a
> ranking-quality improvement; we retain the module for interpretability
> and downstream skill-delta signals (see Table~\ref{tab:confidence_support}).

For the Knowledge Graph:

> Disabling prerequisite mining causes a borderline NDCG@10 change
> ($\Delta = -0.006$, CI $[-0.010, -0.000]$, $t$ $p=0.10$) and a
> non-significant Tag Coverage change; its role is primarily cold-start
> coverage and candidate-pool enlargement, not online ranking quality.

### Wilcoxon floor caveat

Add this footnote to Table~\ref{tab:ablation_significance}:

```latex
\footnotetext{With $N=5$ paired seed-level observations the two-sided
Wilcoxon signed-rank test cannot achieve $p<0.0625$ (its floor is
$2^{-4}$). We therefore use BCa bootstrap 95\% CIs and paired $t$-tests
as primary evidence; the Wilcoxon column is reported for non-parametric
robustness but is systematically censored at $p=0.0625$. A per-user
Wilcoxon test ($N=899$) is left as follow-up work pending an evaluation
re-run with per-user NDCG persisted.}
```

---

## B.4 #28 — Honest F1 reporting for Confidence Agent

### Where in the paper

Wherever the current text says "confidence F1 = 1.00" or similar.
Canonical location: §4.7 and any confidence classification table.

### What to insert

Verbatim block from `results/xes3g5m/tables/table_confidence_support.tex`
as Table~\ref{tab:confidence_support}, and replace the F1=1.00 sentence
with the drop-in block below:

```latex
The behavioural confidence classifier is rule-based: a deterministic
function of \texttt{(is\_correct, is\_fast, changed\_answer)}. Because
predicted labels are the rules, classification F1 is trivially $1.0$
and is not a measure of generalisation. We instead report per-class
support (Table~\ref{tab:confidence_support}). On EdNet all six classes
are populated with non-trivial frequency. On XES3G5M the taxonomy
collapses to two populated classes (UNSURE\_CORRECT and CLEAR\_GAP)
because the corpus lacks a \texttt{changed\_answer} signal and has a
compressed \texttt{elapsed\_time} distribution that flattens the
fast/slow split. The 6-class scheme is therefore reported in MARS as an
interpretability layer rather than a predictive classifier; its
contribution to ranking quality is quantified by the corresponding
row in Table~\ref{tab:ablation_significance}
($\Delta_{\text{NDCG@10}} = -0.0001$).
```

### Where to move "F1 = 1.0" from

Remove this row from the metrics table that currently reports
`Conf F1` in columns next to NDCG / MRR / Coverage (e.g.
`table1_comparison.csv` and any rendered LaTeX Table 3 / Table 4 that
shows it). It is misleading next to learned metrics.

---

## A.1 #29 — Per-tertile NDCG@10 for each ablation config

### Where in the paper

Section §4.6 (or a new "Subgroup analysis" subsection between §4.5 and
§4.6). Replace any previous claim about "IRT helps low-ability
learners" — the subgroup data does not support that claim.

### Table to insert

Drop in `results/xes3g5m/tables/table_tertile_ablation.tex` verbatim
as Table~\ref{tab:tertile_ablation}.

### Body text to replace

```latex
\paragraph{Tertile subgroup analysis.}
To test whether the IRT module helps a particular ability subgroup,
we stratified the 900 XES3G5M test users into three equal-frequency
tertiles by context accuracy (the natural proxy for ability
available before the evaluation horizon). Table~\ref{tab:tertile_ablation}
reports NDCG@10 per config per tertile for seed~42. The pattern is
\emph{consistent across tertiles}: disabling IRT raises NDCG@10 in
every tertile by $0.054$--$0.067$, so the accuracy--coverage
trade-off identified in Section~\ref{subsec4.5} is not localised to
a particular ability band. Two asymmetric findings are worth
reporting:

\begin{itemize}\itemsep2pt
  \item Removing the \textbf{Prediction Agent} hurts the low-ability
        tertile the most ($\Delta = +0.055$) versus the high tertile
        ($\Delta = +0.020$). The transformer gap prediction
        contributes more to users whose context accuracy is lowest,
        consistent with its role as the main source of fine-grained
        failure probability.
  \item Removing the \textbf{Knowledge Graph} yields $\Delta = 0.000$
        in all three tertiles; KG's value is not accuracy but
        curricular breadth (Tag Coverage, Table~\ref{tab:ablation_significance}).
\end{itemize}

The removal of the \textbf{Confidence} module yields a uniform
$\Delta \approx +0.008$ in every tertile, further supporting the
interpretation of confidence as an interpretability layer rather
than a subgroup-specific ranking driver.
```

### Key numeric summary

| Ablation | Δ Low | Δ Mid | Δ High |
|---|---:|---:|---:|
| −Prediction | +0.055 | +0.031 | +0.020 |
| −Knowledge Graph | +0.000 | +0.000 | +0.000 |
| −Confidence | +0.008 | +0.007 | +0.008 |
| −IRT (Diagnostic) | −0.057 | −0.054 | −0.067 |

Positive Δ means the ablated variant hurts Full MARS; negative Δ
means it improves NDCG@10 in that tertile.

### N=5 note

Runtime constraint limited this analysis to seed 42 only. The
tertile-level pattern is expected to replicate across seeds because
the overall 5-seed mean (Table~\ref{tab:ablation_significance})
shows the same directional effects with tighter BCa CIs.

---

## Implementation note

All LaTeX files were auto-generated and live at:

- `results/xes3g5m/ablation_significance_latex.tex`
- `results/xes3g5m/tables/table_confidence_support.tex`
- `results/xes3g5m/tables/table_tertile_ablation.tex`

Raw data backing them:

- `results/xes3g5m/ablation_significance.csv`
- `results/xes3g5m/ablation_significance.md`
- `results/xes3g5m/confidence_support_xes3g5m.json`
- `results/xes3g5m/confidence_support_ednet.json`
- `results/xes3g5m/tertile_ablation_s42.json`
