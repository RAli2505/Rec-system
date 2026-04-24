# Confidence Agent — Per-Class Support and Honest F1 Analysis

The 6-class behavioural confidence classifier is **rule-based**: labels are a deterministic function of `(is_correct, is_fast, changed_answer)`. Reporting F1=1.0 against these same rule-generated labels is tautological; this document replaces that claim with a dataset-level support analysis.

## Per-class support

| Class | XES3G5M support | XES3G5M % | EdNet support | EdNet % |
|---|---:|---:|---:|---:|
| SOLID | 0 | 0.00% | 116,700 | 29.91% |
| UNSURE_CORRECT | 1,199,298 | 79.91% | 91,061 | 23.34% |
| FALSE_CONFIDENCE | 0 | 0.00% | 45,879 | 11.76% |
| CLEAR_GAP | 301,601 | 20.09% | 58,619 | 15.02% |
| DOUBT_CORRECT | 0 | 0.00% | 36,650 | 9.39% |
| DOUBT_INCORRECT | 0 | 0.00% | 41,275 | 10.58% |

## Confusion matrix (both datasets are trivially diagonal)

Because the classifier **is** the rule set, every sample is classified to its own label by construction. The two matrices below therefore contain non-zero entries only on the diagonal; they are provided for transparency, not as evidence of predictive performance.

### XES3G5M

| true \\ pred | SOLID | UNSURE_CORRECT | FALSE_CONFIDENCE | CLEAR_GAP | DOUBT_CORRECT | DOUBT_INCORRECT |
|---|---:|---:|---:|---:|---:|---:|
| SOLID | 0 | 0 | 0 | 0 | 0 | 0 |
| UNSURE_CORRECT | 0 | 1,199,298 | 0 | 0 | 0 | 0 |
| FALSE_CONFIDENCE | 0 | 0 | 0 | 0 | 0 | 0 |
| CLEAR_GAP | 0 | 0 | 0 | 301,601 | 0 | 0 |
| DOUBT_CORRECT | 0 | 0 | 0 | 0 | 0 | 0 |
| DOUBT_INCORRECT | 0 | 0 | 0 | 0 | 0 | 0 |

### EdNet

| true \\ pred | SOLID | UNSURE_CORRECT | FALSE_CONFIDENCE | CLEAR_GAP | DOUBT_CORRECT | DOUBT_INCORRECT |
|---|---:|---:|---:|---:|---:|---:|
| SOLID | 116,700 | 0 | 0 | 0 | 0 | 0 |
| UNSURE_CORRECT | 0 | 91,061 | 0 | 0 | 0 | 0 |
| FALSE_CONFIDENCE | 0 | 0 | 45,879 | 0 | 0 | 0 |
| CLEAR_GAP | 0 | 0 | 0 | 58,619 | 0 | 0 |
| DOUBT_CORRECT | 0 | 0 | 0 | 0 | 36,650 | 0 |
| DOUBT_INCORRECT | 0 | 0 | 0 | 0 | 0 | 41,275 |

## Dataset-level diagnosis of the 6-class scheme

- XES3G5M activates **2 / 6** classes (support > 0). The remaining classes do not appear in this corpus because XES3G5M lacks `changed_answer` and has a compressed `elapsed_time` distribution.
- EdNet activates **6 / 6** classes; all six are populated with non-trivial support.

**Implication for the paper.** The 6-class taxonomy is a property of the rule set, not of a learned model. Its value in MARS is interpretability (assigning a human-readable behavioural tag to every interaction) and downstream skill-delta signals, not classification accuracy. The per-seed ablation in `ablation_significance.md` shows that removing the class-derived skill-delta signal changes NDCG@10 by $\approx 10^{-4}$ (Confidence column), consistent with the taxonomy being primarily an interpretability layer.

## Drop-in replacement for the Confidence F1 claim

Replace any text of the form

> "The behavioural confidence classifier achieves F1 = 1.0 on six-class labels."

with:

> "The behavioural confidence classifier is rule-based: a deterministic function of `(is_correct, is_fast, changed_answer)`. Because predicted labels are the rules, classification F1 is trivially 1.0 and is not a measure of generalisation. We instead report per-class support (Appendix~S3.X): on EdNet all six classes are populated; on XES3G5M the taxonomy collapses to two populated classes due to missing answer-change signal. The 6-class scheme is therefore reported as an interpretability layer, not a predictive classifier, and its contribution is quantified via the corresponding row in Table~\ref{tab:ablation_significance} rather than via F1."
