# Paper patches — reviewer revision

Каждый блок: **где** в `sn-article.tex` → **было** → **стало**.

Применяется как набор Find-Replace.

---

## A.5.1 LambdaMART → MMR (3 места)

### Abstract (line 58)
**Найти**:
```
Thompson Sampling-driven multi-strategy recommendation with LambdaMART re-ranking
```
**Заменить на**:
```
Thompson Sampling-driven multi-strategy recommendation with Maximal Marginal Relevance (MMR) re-ranking
```

### Keywords (line 60) — также см. G.3 (сократить до 6)
**Найти**:
```
LambdaMART, Maximal Marginal Relevance, DKT, Sentence-BERT, ALS, FAISS, NDCG, MRR, XES3G5M, EdNet
```
**Заменить на**:
```
Educational Recommendation
```
(полный сокращённый список см. в G.3 ниже)

### Conclusion (line 432)
**Найти**:
```
while LambdaMART re-ranking and adaptive strategy selection provide complementary gains
```
**Заменить на**:
```
while MMR-based re-ranking and adaptive strategy selection provide complementary gains
```

---

## A.5.2 Table 4 — обозначить «seed 42 only»

### Table 4 caption (around line 286–298)

Найти подпись `\caption{Ablation study results...}` и заменить на:

```latex
\caption{Component ablation on XES3G5M, \textbf{seed 42 only}.
The Full MARS baseline values therefore correspond to the single-seed run
($\text{NDCG@10}=0.654$); the cross-seed mean reported in
Table~\ref{tab3} is $0.683 \pm 0.023$. Δ-values in this table are
within the seed-level standard deviation for the −Knowledge Graph
and −Confidence configurations and therefore should be read as
indicative rather than statistically significant; the Pareto analysis
in Fig.~\ref{fig:ablation_pareto} repositions these configurations
as alternative operating points on the (NDCG@10, Coverage) frontier
rather than as strictly dominated variants.}
```

---

## A.5.3 Микрорасхождение AUC 0.923 vs 0.924

`fig_methods_heatmap` показывает 0.923 (округление 0.9235 ↓), Table 3 показывает 0.924 (округление 0.9235 ↑).

**Опция (a, рекомендую)**: оставить 4 знака везде где есть mean±std → 0.9235 ± 0.0052.

**Опция (b)**: округлять одинаково — Banker's rounding. Либо все 0.923, либо все 0.924.

Минимальное действие: в Table 3 заменить `0.924 \pm 0.005` на `0.923 \pm 0.005`, что соответствует heatmap.

---

## A.4 Data leakage declaration (вставить в §4.1)

После предложения `Each experiment is repeated across five random seeds...` добавить:

```latex
\paragraph{Leakage audit.}
The 70/15/15 user-level split is fixed once per seed and shared by all
agents. IRT calibration (Section~\ref{subsec3c}), the GraphSAGE
embedding training, and prerequisite mining (Section~\ref{subsec3e})
operate exclusively on the 4{,}200 training users. Validation and test
users contribute neither to the IRT item parameters, nor to the mastery
sequences fed into the prerequisite mining heuristics, nor to the
GraphSAGE link-prediction objective. Tag embeddings exposed to the
Prediction Agent through warm initialisation therefore contain no
information about val/test responses, and the reported AUC-ROC is not
inflated by topology-mediated leakage.
```

---

## A.3 Focal BCE + label smoothing — точная формула (§3.6)

Добавить новый абзац после описания `\paragraph{Training objective}`:

```latex
\paragraph{Training objective.}
The prediction head is optimised with a class-weighted focal binary
cross-entropy applied independently to each of the 858 concept
outputs. Let $\hat p_{i,t} = \sigma(z_{i,t})$ denote the predicted
failure probability for user-window $i$ on tag $t$, and let
$y_{i,t} \in \{0,1\}$ be the ground-truth label. Label smoothing is
applied to the targets:
$$
\tilde y_{i,t} = (1-\varepsilon)\, y_{i,t} + \tfrac{\varepsilon}{2},
\qquad \varepsilon = 0.05.
$$
The per-element focal BCE with positive-class weight $w_t$ is
$$
\ell_{i,t} = -\bigl[\,w_t\,\tilde y_{i,t} \log \hat p_{i,t}
                  + (1-\tilde y_{i,t}) \log (1-\hat p_{i,t})\,\bigr]
            \cdot (1-p^{\star}_{i,t})^{\gamma},
$$
where $p^{\star}_{i,t} = \hat p_{i,t}\,\tilde y_{i,t}
+ (1-\hat p_{i,t})(1-\tilde y_{i,t})$ is the probability assigned to
the (smoothed) correct class, $\gamma = 2.0$ is the focusing parameter,
and $w_t = \mathrm{clip}\!\bigl((1-r_t)/r_t,\,1,\,50\bigr)$ with
$r_t$ the empirical positive rate of tag $t$ in the training set.
The total loss is $L = \tfrac{1}{NT}\sum_{i,t} \ell_{i,t}$.
Negative samples are not drawn explicitly: every concept that does not
appear in the next $\text{HORIZON}=20$ interactions is treated as a
negative for that window, which gives an average positive rate of
$\sim 1\%$ per tag and motivates the focal weighting.
```

---

## A.3 Eval protocol formalisation (§4.2)

Добавить новый абзац в начало §4.2:

```latex
\paragraph{Evaluation protocol.}
For each test user, the chronologically-ordered interaction sequence is
split at the 30/70 boundary: the first 30\% of interactions form the
\emph{context} and seed the agent state (CAT estimate of $\theta$,
behavioural confidence history, prerequisite map); the remaining 70\%
form the \emph{evaluation horizon}. The Prediction Agent emits a
distribution over the 858 concepts; the ground-truth top-$k$ label set
for NDCG@10 / MRR / Precision@10 is the set of concepts the user fails
in the next 20 chronologically-ordered interactions of the evaluation
horizon (a re-attempted concept counts once). Tag Coverage is the
fraction of distinct concepts in the union of all top-10 recommendations
across test users, normalised by the number of concepts that satisfy
the macro-AUC support criterion ($\geq 5$ positive and $\geq 5$
negative interactions in the test horizon).
```

---

## B.3 Learning Gain — limitation language

### §4.11 (около line 401)
**Найти**:
```
These results demonstrate that MARS transfers across educational domains and item formats without architectural modifications, while highlighting domain-specific trade-offs
```
**Заменить на**:
```
These results demonstrate that MARS transfers between two East-Asian
educational benchmarks without architectural modifications, while highlighting
domain-specific trade-offs in offline ranking quality. The near-zero
Learning Gain on both datasets ($-0.001$ on XES3G5M, $0.027$ on EdNet)
is consistent with the offline evaluation protocol, in which MARS
recommendations are not actually shown to learners during the test
session; consequently, all numbers in this paper measure
\textbf{ranking quality of candidate generation}, not actual
pedagogical effect. A controlled A/B study with real learners is
required before any claim about \emph{learning} outcomes can be made.
```

### §5.2 Limitations — добавить в начало секции
```latex
\paragraph{Pedagogical claims.}
Learning Gain is computed as the difference in correctness between
the first and second halves of the test horizon and serves only as a
sanity check that the model does not actively harm performance. The
near-zero values reported in Section~\ref{subsec4k} therefore neither
confirm nor refute pedagogical efficacy. Throughout the paper,
``personalised learning'' refers to the personalisation of the
candidate-generation step, not to longitudinal mastery improvement,
which would require a longitudinal field study.
```

### Conclusion (line 432)
**Найти**:
```
The behavioural confidence module enhances knowledge state estimation, and the prerequisite graph improves recommendation diversity and cold-start coverage.
```
**Заменить на**:
```
The behavioural confidence module contributes to interpretability and to a small but non-significant change in ranking metrics; the prerequisite graph mainly enlarges the candidate pool, raising tag coverage by 0.032 in the seed-42 ablation. None of the offline metrics quantify pedagogical efficacy, which is left to a longitudinal user study.
```

### Abstract — убрать «improves learning outcomes» (если есть)
В текущем abstract такой фразы нет, но проверить также:
- §1 Introduction (line 113 area)
- §6 Conclusion

---

## H.1 Soften unsubstantiated language

Везде в тексте:
| Найти | Заменить |
|---|---|
| `substantially higher` | `higher` |
| `dramatically` | (удалить слово) |
| `notably lower` | `lower` |
| `significantly outperform` | `outperform` (только если нет p-value) |

Конкретные строки для проверки: §4.4, §4.11, §5.1.

---

## G.3 Сократить keywords до 6

**Найти** (line 60):
```
\keywords{Knowledge Tracing, Multi-Agent Systems, Recommendation Systems, Behavioral Confidence Modeling, Prerequisite Mining, Personalized Learning, Transformer, GraphSAGE, Thompson Sampling, Item Response Theory, LambdaMART, Maximal Marginal Relevance, DKT, Sentence-BERT, ALS, FAISS, NDCG, MRR, XES3G5M, EdNet}
```
**Заменить на**:
```
\keywords{Knowledge Tracing, Multi-Agent Systems, Educational Recommendation, Prerequisite Mining, Behavioral Confidence Modeling, Personalised Learning}
```

---

## G.1 Declarations Springer Nature template

**Найти** существующий блок Declarations и заменить на:

```latex
\section*{Declarations}

\noindent\textbf{Funding}\\
No funding was received for conducting this study.

\noindent\textbf{Competing interests}\\
The authors declare no competing interests.

\noindent\textbf{Ethics approval and consent to participate}\\
This study uses two publicly released educational datasets
(XES3G5M~\cite{xes3g5m_release} and EdNet~\cite{ednet_release}); no
new data were collected from human subjects. Both datasets were de-
identified by their original publishers; no personally identifiable
information is processed in this work.

\noindent\textbf{Consent for publication}\\
Not applicable.

\noindent\textbf{Data availability}\\
XES3G5M is available from
\url{https://github.com/ai4ed/XES3G5M};
EdNet KT2 is available from
\url{https://github.com/riiid/ednet}.

\noindent\textbf{Code availability}\\
The full implementation, training scripts, evaluation pipeline,
trained checkpoints, and per-seed result JSON files used to produce
all tables and figures in this manuscript are available at
\url{https://anonymous.4open.science/r/mars-recsys-XXXX} (during
review) and will be moved to a permanent DOI-tagged Zenodo release
upon acceptance. See \texttt{README.md} in the repository for the
exact reproduction commands listed in Appendix~S5.

\noindent\textbf{Author contributions}\\
Ramazan Ali conceived the study, designed the methodology,
implemented the system, conducted the experiments, and wrote the
manuscript. Oleg Stelvaga contributed to data analysis, literature
review, and manuscript revision.

\noindent\textbf{Acknowledgements}\\
Not applicable.
```
