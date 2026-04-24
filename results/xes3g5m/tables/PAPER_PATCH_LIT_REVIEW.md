# Block D — Literature additions to §2.2 and Table 1

Reviewer wants explicit comparison with **SAINT, SAINT+, AKT, DKVMN,
LPKT, IEKT, SimpleKT, GKT, DTransformer**. Below are drop-in BibTeX
entries, paragraph rewrites for §2.2, and the new Table 1.

---

## D.1 BibTeX entries (append to `sn-bibliography.bib`)

```bibtex
@inproceedings{choi2020saint,
  title  = {Towards an Appropriate Query, Key, and Value Computation
            for Knowledge Tracing},
  author = {Choi, Youngduck and Lee, Youngnam and Cho, Junghyun and
            Baek, Jineon and Kim, Byungsoo and Cha, Yeongmin and
            Shin, Dongmin and Bae, Chan and Heo, Jaewe},
  booktitle= {Proc. Learning at Scale (L@S)},
  year   = {2020},
  doi    = {10.1145/3386527.3405945}
}

@inproceedings{shin2021saintplus,
  title  = {SAINT+: Integrating Temporal Features for EdNet
            Correctness Prediction},
  author = {Shin, Dongmin and Shim, Yugeun and Yu, Hangyeol and
            Lee, Sangwoo and Kim, Byungsoo and Choi, Youngduck},
  booktitle= {Proc. Learning Analytics and Knowledge (LAK)},
  year   = {2021},
  doi    = {10.1145/3448139.3448188}
}

@inproceedings{ghosh2020akt,
  title  = {Context-Aware Attentive Knowledge Tracing},
  author = {Ghosh, Aritra and Heffernan, Neil and Lan, Andrew S.},
  booktitle= {Proc. ACM SIGKDD},
  pages  = {2330--2339},
  year   = {2020},
  doi    = {10.1145/3394486.3403282}
}

@inproceedings{zhang2017dkvmn,
  title  = {Dynamic Key-Value Memory Networks for Knowledge Tracing},
  author = {Zhang, Jiani and Shi, Xingjian and King, Irwin and
            Yeung, Dit-Yan},
  booktitle= {Proc. WWW},
  pages  = {765--774},
  year   = {2017},
  doi    = {10.1145/3038912.3052580}
}

@inproceedings{shen2021lpkt,
  title  = {Learning Process-Consistent Knowledge Tracing},
  author = {Shen, Shuanghong and Liu, Qi and Chen, Enhong and
            Huang, Zhenya and Huang, Wei and Yin, Yu and
            Su, Yu and Wang, Shijin},
  booktitle= {Proc. ACM SIGKDD},
  pages  = {1452--1460},
  year   = {2021},
  doi    = {10.1145/3447548.3467237}
}

@inproceedings{long2021iekt,
  title  = {Tracing Knowledge State with Individual Cognition and
            Acquisition Estimation},
  author = {Long, Ting and Liu, Yunfei and Shen, Jian and
            Zhang, Weinan and Yu, Yong},
  booktitle= {Proc. ACM SIGIR},
  pages  = {173--182},
  year   = {2021},
  doi    = {10.1145/3404835.3462886}
}

@inproceedings{liu2023simplekt,
  title  = {SimpleKT: A Simple But Tough-to-Beat Baseline for
            Knowledge Tracing},
  author = {Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and
            Huang, Shuyan and Luo, Weiqi},
  booktitle= {Proc. ICLR},
  year   = {2023}
}

@inproceedings{nakagawa2019gkt,
  title  = {Graph-based Knowledge Tracing: Modeling Student
            Proficiency Using Graph Neural Network},
  author = {Nakagawa, Hiromi and Iwasawa, Yusuke and Matsuo, Yutaka},
  booktitle= {Proc. WI/IAT},
  pages  = {156--163},
  year   = {2019},
  doi    = {10.1145/3350546.3352513}
}

@inproceedings{yin2023dtransformer,
  title  = {Tracing Knowledge Instead of Patterns: Stable Knowledge
            Tracing with Diagnostic Transformer},
  author = {Yin, Yu and Dai, Le and Huang, Zhenya and Shen, Shuanghong
            and Wang, Fei and Liu, Qi and Chen, Enhong and Li, Xin},
  booktitle= {Proc. WWW},
  year   = {2023},
  doi    = {10.1145/3543507.3583655}
}
```

---

## D.2 Rewritten paragraph in §2.2 (Knowledge Tracing)

Replace the third and fourth sentences of §2.2 with the following
wider survey paragraph (insertions in **bold**):

> Knowledge Tracing has evolved through three architectural waves.
> The recurrent wave introduced the original DKT-LSTM
> [\cite{piech2015dkt}] and DKVMN with key-value memory
> [**\cite{zhang2017dkvmn}**]. The attention wave brought self-
> attentive variants — SAKT [\cite{pandey2019sakt}], **SAINT
> [\cite{choi2020saint}] and SAINT+ [\cite{shin2021saintplus}]**
> — and context-aware attention with monotonic decay (AKT
> [**\cite{ghosh2020akt}**]). Process-consistent and individualised
> tracing models such as **LPKT [\cite{shen2021lpkt}]** and **IEKT
> [\cite{long2021iekt}]** added latent ability and acquisition
> estimates that resemble the IRT signal used in MARS, while
> **GKT [\cite{nakagawa2019gkt}]** and **DTransformer
> [\cite{yin2023dtransformer}]** explicitly used graph structure,
> mirroring the prerequisite mining direction. The most recent
> entrant, **SimpleKT [\cite{liu2023simplekt}]**, demonstrates that
> a carefully tuned tied-embedding decoder matches or exceeds
> deeper attention models on standard benchmarks. None of these
> systems jointly addresses behavioural confidence taxonomy,
> data-driven prerequisite mining, and multi-strategy
> recommendation; they all stop at the binary correctness prediction
> task that constitutes the input to the MARS Recommendation Agent.

---

## D.3 Updated Table 1 (replace existing table contents)

The original Table 1 listed 14 approaches. The expanded version below
adds 9 KT-specific rows (SAINT, SAINT+, AKT, DKVMN, LPKT, IEKT,
SimpleKT, GKT, DTransformer). Use this as a drop-in replacement for
the `\begin{table}…\end{table}` block in §2.

```latex
\begin{table}[t]
\centering
\caption{Comparative analysis of representative knowledge tracing
and recommendation systems. KT = Knowledge Tracing,
BCM = Behavioral Confidence Modeling, PM = Prerequisite Mining,
MA = Multi-Agent, RS = Recommender System, RT = Real-Time.
$\checkmark$ = supported, $\times$ = not supported,
$\circ$ = partial.}
\label{tab1}
\small
\begin{tabular}{l c c c c c c}
\toprule
Approach & KT & BCM & PM & MA & RS & RT \\
\midrule
\multicolumn{7}{l}{\textit{Recurrent KT}}\\
DKT (Piech et al.~\cite{piech2015dkt})            & \checkmark & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark\\
DKVMN (Zhang et al.~\cite{zhang2017dkvmn})        & \checkmark & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark\\
LPKT (Shen et al.~\cite{shen2021lpkt})            & \checkmark & $\circ$  & $\times$ & $\times$ & $\times$ & \checkmark\\
IEKT (Long et al.~\cite{long2021iekt})            & \checkmark & $\circ$  & $\times$ & $\times$ & $\times$ & \checkmark\\
\multicolumn{7}{l}{\textit{Attention-based KT}}\\
SAKT (Pandey \& Karypis~\cite{pandey2019sakt})    & \checkmark & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark\\
SAINT (Choi et al.~\cite{choi2020saint})          & \checkmark & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark\\
SAINT+ (Shin et al.~\cite{shin2021saintplus})     & \checkmark & $\circ$  & $\times$ & $\times$ & $\times$ & \checkmark\\
AKT (Ghosh et al.~\cite{ghosh2020akt})            & \checkmark & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark\\
SimpleKT (Liu et al.~\cite{liu2023simplekt})      & \checkmark & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark\\
DTransformer (Yin et al.~\cite{yin2023dtransformer}) & \checkmark & $\times$ & $\circ$ & $\times$ & $\times$ & \checkmark\\
\multicolumn{7}{l}{\textit{Graph-based KT}}\\
GKT (Nakagawa et al.~\cite{nakagawa2019gkt})      & \checkmark & $\times$ & $\circ$ & $\times$ & $\times$ & \checkmark\\
Mai et al.~\cite{bib18}                           & \checkmark & $\times$ & \checkmark & $\times$ & $\times$ & \checkmark\\
\multicolumn{7}{l}{\textit{Recommender systems}}\\
Wang et al.~\cite{bib9}                           & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark & $\times$\\
Kaur et al.~\cite{bib10}                          & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark & $\times$\\
Zhang \cite{bib12}                                 & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark & $\times$\\
Onyeke et al.~\cite{bib20}                        & \checkmark & $\circ$  & $\times$ & $\times$ & $\times$ & $\times$\\
Chanaa \& El Faddouli~\cite{bib26}                & $\times$ & $\times$ & \checkmark & $\times$ & \checkmark & $\times$\\
\multicolumn{7}{l}{\textit{Multi-agent}}\\
Feng \cite{bib21}                                  & $\times$ & $\times$ & $\times$ & \checkmark & \checkmark & \checkmark\\
Rida et al.~\cite{bib22}                          & $\times$ & $\times$ & $\times$ & \checkmark & $\times$ & \checkmark\\
Bhushan et al.~\cite{bib23}                       & $\times$ & $\times$ & $\times$ & \checkmark & \checkmark & \checkmark\\
Chaturvedi \cite{bib24}                            & $\times$ & $\times$ & $\times$ & \checkmark & \checkmark & \checkmark\\
Amin et al.~\cite{bib28}                          & $\times$ & $\times$ & $\times$ & \checkmark & \checkmark & \checkmark\\
\midrule
\textbf{MARS (Ours)}                              & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

This expanded Table 1 contains **22 approaches** (was 14) and makes
the uniqueness claim defensible: no single row to the left of MARS
ticks all six columns.

---

## D.4 Update §2.4 last paragraph

Replace `14 representative approaches` with `22 representative
approaches across recurrent KT, attention-based KT, graph-based KT,
recommender systems, and multi-agent educational architectures`.
