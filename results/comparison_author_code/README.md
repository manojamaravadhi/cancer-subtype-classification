# Accuracy comparison: reference methods' code vs our model

`accuracy_comparison.csv` - test accuracy of the two reference methods' released code (run standalone) and of our
Ensembled Transformer-Graph Streams model, on the same patients and the same 5 test folds (StratifiedKFold, seed 42),
for BRCA (brca_combined), LGG, KIPAN, COADREAD and STAD. Made on 2026-11-08 from the saved predictions (no training).
BRCA and COADREAD rows updated on 2026-11-10 after the re-run with one sample per patient (old file:
`results/archive_with_duplicate_samples/comparison_author_code/`).

| Columns | Source of the predictions |
|---|---|
| `transformer_stream_author_code` | `results/baselines_author_code/baseline_transformer/<cohort>/fold<k>/best_pred_total_DG.csv` |
| `graph_stream_author_code` | `results/baselines_author_code/baseline_graph/<cohort>/fold<k>/test_predictions.csv` |
| `ensembled_streams_ours_fold_mean_sd` | `results/final_ensembled_streams/<cohort>_predictions.csv`, column `pred_ensembled` |

- `n_test_patients` - labeled patients over the 5 test folds (every labeled patient is tested once)
- the three accuracy columns - mean ± sample standard deviation (n-1) of the 5 per-fold accuracies
  (the pooled accuracies were removed from this file on 2026-11-10 at the owner's request; they are in the paper)

Values are rounded to 4 decimals. All three methods have the same test patients and true labels in every fold, and the
per-fold accuracies of both author-code runs equal the accuracies in their own `test_metrics.json`.
