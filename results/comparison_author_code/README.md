# Accuracy comparison: reference methods' code vs our model

`accuracy_comparison.csv` - test accuracy of the two reference methods' released code (run standalone) and of our
Ensembled Transformer-Graph Streams model, on the same patients and the same 5 test folds (StratifiedKFold, seed 42),
for BRCA (brca_combined), LGG, KIPAN, COADREAD and STAD. Made on 2026-11-08 from the saved predictions (no training).
BRCA and COADREAD rows updated on 2026-11-10 after the re-run with one sample per patient (old file:
`results/archive_with_duplicate_samples/comparison_author_code/`).

| Columns | Source of the predictions |
|---|---|
| `ref_transformer_author_code_*` | `results/baselines_author_code/baseline_transformer/<cohort>/fold<k>/best_pred_total_DG.csv` |
| `ref_graph_author_code_*` | `results/baselines_author_code/baseline_graph/<cohort>/fold<k>/test_predictions.csv` |
| `ensembled_streams_ours_*` | `results/final_ensembled_streams/<cohort>_predictions.csv`, column `pred_ensembled` |

- `n_test_patients` - labeled patients over the 5 test folds (every labeled patient is tested once)
- `*_pooled` - correct predictions / all test patients, pooled over the 5 folds
- `*_fold_mean_sd` - mean ± sample standard deviation (n-1) of the 5 per-fold accuracies

Values are rounded to 4 decimals. All three methods have the same test patients and true labels in every fold, and the
per-fold accuracies of both author-code runs equal the accuracies in their own `test_metrics.json`.
