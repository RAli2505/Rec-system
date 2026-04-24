# Dual-dataset EDA (XES3G5M vs EdNet KT2)

Both corpora are sampled with the same protocol — 6,000 students with at least 20 interactions each — and share the 70/15/15 user-level split. Full stats are in `table_dual_stats.{md,csv,tex}`.

## Headline numbers

- **Interactions**: XES3G5M ≈ 2,134,487; EdNet ≈ 545,986 (ratio ≈ 3.91×).
- **Avg seq. length**: XES3G5M ≈ 355.7, EdNet ≈ 196.1.
- **Unique concepts**: XES3G5M = 858, EdNet = 187.
- **Unique questions**: XES3G5M = 7,373, EdNet = 10,690.
- **Overall accuracy**: XES3G5M = 0.798, EdNet = 0.626.

## Behavioural signal availability

- **Answer-change rate**: XES3G5M = 0.0000, EdNet = 0.2003. On XES3G5M this field is synthesised as zero (no native signal in the corpus), which is why the 6-class confidence taxonomy collapses to 2 populated classes (UNSURE_CORRECT, CLEAR_GAP). See `confidence_support_*.json`.
- **Median elapsed_time**: XES3G5M = 15000 ms (synthesised from timestamp deltas), EdNet = 24376 ms (native field from the logging pipeline).

## Figures

- `fig_seq_length.pdf` — sequence length distribution
- `fig_accuracy_hist.pdf` — per-user accuracy distribution
- `fig_elapsed_time.pdf` — response-time distribution (log scale)
- `fig_concept_coverage.pdf` — per-student concept coverage
- `fig_correctness_per_user.pdf` — overall vs per-user accuracy

## Reproduction

```bash
python scripts/eda_dual_dataset.py
```