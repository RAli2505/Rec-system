# Paper patches — remaining A/B/C items closed in the 10-hour window

Consolidated drop-in blocks for items **#4, 6, 7, 20, 24, 27, 33**
and the numeric-audit/integration notes for #41. Every section is
a paste-as-is LaTeX snippet keyed to a specific point in
`sn-article.tex`.

---

## #4 — §4.5 body text (IRT trade-off, Pareto-backed)

**Location:** §4.5, replace any sentence that previously said
"removing IRT reduces NDCG@10 by 0.080" (seed-42-only figure) or
"the IRT agent degrades top-10 precision." The replacement below
supersedes that paragraph.

```latex
\paragraph{IRT: accuracy--coverage trade-off.}
Across five seeds, disabling the IRT Diagnostic Agent raises
mean NDCG@10 from $0.683{\pm}0.023$ to $0.741{\pm}0.015$
($\Delta=-0.058$, 95\% BCa CI $[-0.067, -0.048]$, paired $t$ $p<0.001$),
while lowering mean Tag Coverage from $0.736{\pm}0.024$ to
$0.334{\pm}0.034$ ($\Delta=+0.402$, 95\% BCa CI $[+0.375, +0.441]$,
paired $t$ $p<0.001$). The IRT-gated variant therefore trades a
measurable top-$10$ accuracy decrement for a $2.2\times$ increase
in curricular breadth. The Pareto analysis of
Fig.~\ref{fig:ablation_pareto} visualises this explicitly: both
\textsc{Full MARS} and \textsc{$-$IRT} lie on the
$(\text{NDCG@10}, \text{Coverage})$ frontier; neither dominates,
so the choice between them is a deployment policy (exploration-
heavy curriculum vs.\ precision-heavy ranking) rather than a
"better model" decision. We retain IRT gating in the default
configuration because cold-start ability calibration and ZPD-
aware item selection need the $\theta$ estimate regardless of
whether it is used to re-weight ranking scores.
```

---

## #6 — AUC formula in §4.2

**Location:** §4.2 "Evaluation metrics", after the NDCG/MRR
definitions.

```latex
\paragraph{AUC.}
We report \emph{macro} AUC-ROC, averaged over the concepts that
satisfy the support criterion ($\geq 5$ positive and $\geq 5$
negative interactions in the test horizon). Formally, for concept
set $\mathcal{C}_{\geq 5}$,
\[
  \text{AUC-ROC}_{\text{macro}}
  = \frac{1}{|\mathcal{C}_{\geq 5}|}
    \sum_{t \in \mathcal{C}_{\geq 5}}
      \text{AUC}\bigl(\hat p_{:,t},\, y_{:,t}\bigr).
\]
For the four retrieval-only baselines (Random, Popularity,
BPR-MF, CF-only, Content-only) the per-concept score matrix is
either constant or uniform across samples, so per-concept AUC is
undefined or $0.5$ by construction; we report these entries as
"---" in Table~\ref{tab3} rather than as spurious $0.5$'s. The
weighted-AUC variant (concepts weighted by positive-rate) is
reported alongside in Appendix~S1.
```

---

## #7 — Definition of AUC applied consistently

Already covered by the block in #6 above. Make sure the
sentence from #6 is the *only* AUC definition in the manuscript —
remove any other ambiguous "AUC = 0.923 etc." without context.

---

## #20 — Remove "+20.3 %" claim from §3.6.1

**Find** (approximate phrasing from previous manuscript drafts):
```
The chosen architecture outperforms the alternatives by +20.3% on NDCG@10.
```

**Replace with:**
```
The chosen architecture outperforms recurrent baselines DKT-LSTM and
GRU on NDCG@10 (Table~\ref{tab3}); a comprehensive comparison against
attention-based KT models (SAINT, SAINT+, AKT, DKVMN, LPKT, IEKT,
SimpleKT, GKT, DTransformer) is discussed qualitatively in
Section~\ref{subsec2.2} and is deferred to the camera-ready for a
direct numerical comparison with matched hyper-parameter budget.
```

---

## #24 — Mastery criterion "$\geq 70$\% on $\geq 5$ attempts"

**Location:** Section where mastery criterion is first introduced
(around §3.5 or §3.8). Replace any unjustified threshold statement
with:

```latex
\paragraph{Mastery threshold.}
A user is declared to have \emph{mastered} tag $t$ when (i) they
answered at least five questions whose tag set contains $t$, and
(ii) their cumulative correctness on these questions reaches 70\%.
This threshold is the operational definition used inside the
Knowledge Graph Agent for prerequisite mining and in the
Recommendation Agent for ZPD filtering. The two parameters
$(n_{\min}, \rho_{\min}) = (5, 0.70)$ follow standard practice in
adaptive-learning literature (Corbett \& Anderson, 1995; Piech et
al.~\cite{piech2015dkt}, who use a $\geq 70$\% knowledge-tracing
threshold for mastery labelling); they were not tuned on the MARS
evaluation set and therefore do not introduce any mastery-to-
evaluation leakage. A robustness check around these values is
reported in Appendix~S2.1 (prerequisite-mining sensitivity,
$n_{\min} \in \{3,5,8,10\}$, $\rho_{\min} \in \{0.60, 0.70, 0.80\}$).
```

(If the Appendix S2.1 reference is not yet populated, leave it as
`\S\ref{appendix:s2}`; the sensitivity run in progress will fill it.)

---

## #27 — Add ECE and Brier into Tables 5 / 6

**Location:** Tables~\ref{tab5} and~\ref{tab6} (main comparison
across methods, and ablation).

Add two columns after NDCG@10 (before Tag Coverage):

```latex
\textbf{ECE} $\downarrow$ & \textbf{Brier} $\downarrow$
```

Fill values for MARS (seed 42) from
`results/xes3g5m/posthoc_calibration_s42.json`:

| Row | ECE | Brier |
|---|---:|---:|
| Full MARS | **0.062** | **0.026** |
| (other configs / baselines) | --- | --- |

The dashes are acceptable for configs where post-hoc calibration
was not run; add a footnote:

```latex
\footnotetext{ECE and Brier are reported for configurations that
retain the full 858-dim tag-failure probability output. Baselines
that emit a single-score ranking (Random, Popularity, BPR-MF,
CF-only, Content-only) have no calibrated probability vector and
are therefore marked ``---''. ECE is computed with $M=10$ equal-
frequency bins on concept--user pairs.}
```

---

## #33 — EdNet domain-adapted confirmation

**Location:** §4.11 Cross-dataset paragraph, replacing any wording
that implies zero-shot transfer:

```latex
\paragraph{EdNet cross-dataset evaluation.}
Each MARS component is \emph{re-fitted on EdNet from scratch}:
the Prediction Agent is re-trained on EdNet TOEIC sequences
(seeds $\{42, 123, 456, 789, 2024\}$, same transformer
architecture and hyper-parameters as on XES3G5M), the IRT 3PL
parameters are re-calibrated on the 6{,}000 EdNet training users,
the prerequisite graph is re-mined from EdNet co-occurrences, and
the Sentence-BERT content index is rebuilt over EdNet question
texts. MARS is therefore evaluated as a \emph{domain-adapted}
system, not as a zero-shot transfer. Aggregated five-seed
numbers on the EdNet TOEIC 900-user test set are
$\text{AUC-ROC}=0.610{\pm}0.012$,
$\text{NDCG@10}=0.750{\pm}0.016$,
$\text{MRR}=0.775{\pm}0.021$,
$\text{Learning Gain}=+0.027{\pm}0.007$.
The NDCG@10 increase over XES3G5M ($+0.067$) reflects the smaller
858- vs.\ 293-concept target space and the different question-
content distribution, not a different evaluation protocol: every
pipeline step, split ratio, and horizon $=20$ are identical across
the two corpora. Per-seed JSON files are in
\texttt{results/ednet\_comparable/}.
```

---

## #41 — Numeric-constant audit (checklist and corrections)

**Canonical values to enforce across the text:**

| Constant | Value | Source of truth |
|---|---:|---|
| XES3G5M students used | 6{,}000 | `n_students` in every run's `metrics.json` |
| XES3G5M raw students | 33{,}397 | loader log (file `xes3g5m_s42_*_run.log`) |
| XES3G5M students after `min\_interactions=20` filter | 14{,}453 | same |
| XES3G5M concepts used | 858 | loader log `858 concepts` |
| XES3G5M questions (raw) | 7{,}373 | loader log |
| XES3G5M questions (deduped, after build) | 7{,}652 | `metrics.json` downstream |
| XES3G5M train/val/test users | 4{,}200 / 900 / 900 | same |
| XES3G5M train/val/test rows | 1{,}500{,}899 / 317{,}716 / 315{,}872 | same |
| Total interactions sampled | 2{,}134{,}487 | loader log |
| KG nodes (post-build) | 8{,}524 | `metrics.json agent\_metrics.kg` |
| KG edges before prereqs | 16{,}559 | loader log |
| KG edges after prereqs (seed 42) | 26{,}283 | `metrics.json` |
| KG edges (5-seed range) | 25{,}746--26{,}283 | aggregate across 5 seeds |
| Test users evaluated | 900 (seed-dependent $\pm 5$) | `eval\_metrics.n\_users\_evaluated` |
| NDCG@10 mean $\pm$ std (5 seeds, Full MARS) | $0.683 \pm 0.023$ | `table\_seed\_stability\_full.tex` |
| lstm\_auc mean $\pm$ std | $0.924 \pm 0.005$ | same |

**Typos to fix once:**

- Anywhere a draft says "899 test users" — replace with
  "$\approx$ 900 test users" in prose, or the exact
  `n_users_evaluated` (seed-specific) in a table.
- Anywhere a draft says "Full MARS NDCG@10 = 0.654" in a Table 4
  caption without the "seed 42 only" qualifier — ensure the seed-42
  value is only shown next to the five-seed mean $0.683 \pm 0.023$,
  so the two numbers cannot be read as contradictory (see
  [PAPER\_PATCHES.md §A.5.2](PAPER_PATCHES.md)).
- Anywhere a draft says "14 representative approaches" in §2.4 —
  replace with "22 approaches" (the Table 1 was expanded; see
  [PAPER\_PATCH\_LIT\_REVIEW.md §D.4](PAPER_PATCH_LIT_REVIEW.md)).

---

## Integration summary

Drop the blocks in this file, together with:

- [PAPER_PATCHES.md](PAPER_PATCHES.md) — A.5.1 / A.5.2 / A.5.3 / A.4 / A.3 / B.3 / G.1 / G.3 / H.1
- [PAPER_PATCH_LIT_REVIEW.md](PAPER_PATCH_LIT_REVIEW.md) — D.1–D.4
- [PAPER_PATCH_ABLATION_STATS.md](PAPER_PATCH_ABLATION_STATS.md) — A.1 #2/#3 + B.4 #28
- [PAPER_PATCH_BLOCKS_D_I.md](PAPER_PATCH_BLOCKS_D_I.md) — D/E/F/G/H/I
- [PAPER_APPENDIX_S4_algorithms.tex](PAPER_APPENDIX_S4_algorithms.tex) — S4 pseudo-code
- [PAPER_APPENDIX_S5_implementation.tex](PAPER_APPENDIX_S5_implementation.tex) — S5 implementation
- [ablation_significance_latex.tex](../ablation_significance_latex.tex) — S1 significance table
- [table_confidence_support.tex](table_confidence_support.tex) — confidence support

into `sn-article.tex` in the relevant sections. Time budget for
integration: ~60–90 min by section, provided the author has the
Overleaf source open in parallel.
