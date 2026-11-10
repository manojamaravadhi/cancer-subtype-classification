# Results

| Folder | What it is |
|---|---|
| **`final_ensembled_streams/`** | FINAL results of the Ensembled Transformer-Graph Streams model (transformer stream + graph stream + bilateral cross-attention fusion, bagged) on brca_combined, lgg, kipan, coadread, stad. Trained 2026-11-01 / 02 on this Windows PC (RTX 2000 Ada); BRCA and COADREAD re-trained 2026-11-10 with one sample per patient. |
| **`baselines_author_code/`** | The two reference methods' released code run standalone on the same patients, folds and features (2026-11-02; BRCA and COADREAD re-run 2026-11-10). Used for the comparison with the reference methods. |
| **`comparison_author_code/`** | Accuracy of both reference codes vs our model on the same test folds (`accuracy_comparison.csv`). |
| **`archive_with_duplicate_samples/`** | The BRCA / COADREAD results, report and figures from before 2026-11-10, when 4 BRCA patients and 1 COADREAD patient also had a metastatic sample in the data. Kept for reference only. |

On 2026-11-03 the folders, file names, keys, column names, report and figures were renamed to neutral names
(transformer stream = `trans`, graph stream = `graph`, ensembled model = `ensembled`). Every number was checked to be
unchanged; the report and figures were regenerated from the saved results (no retraining). Log files keep their original text.

## final_ensembled_streams/

- `<cohort>_5fold_results.json` - settings (`config`) and metrics (mean ± sd over 5 folds) of the transformer stream, graph stream and ensembled model
- `<cohort>_predictions.csv` - out-of-fold predictions of all three (`pred_trans`, `pred_graph`, `pred_ensembled`), class probabilities, risk score, survival time / event
- `<cohort>_feature_importance.csv`, `<cohort>_learning_curves.json`
- `report/` - REPORT.md, classification tables, survival summary, top-30 biomarkers, enrichment/, figures/
- `figures/main`, `figures/supplementary` - paper figures
- `logs/` - training, report, enrichment and plotting logs

Settings: `--trans_lr 1e-4 --trans_pretrain_epochs 50 --member_split split --fusion_unlabeled --interpret`,
all other options default (5 members, 85 % feature subspace, d_model 128, 5-fold CV seed 42).

Ensembled model (accuracy / macro-F1 pooled over the 5 test folds, C-index = mean over folds): BRCA 0.869 / 0.838 / 0.662;
LGG 0.980 / 0.982 / 0.765; KIPAN 0.957 / 0.954 / 0.659; COADREAD 0.758 / 0.388 / 0.564; STAD 0.854 / 0.680 / 0.532.

## baselines_author_code/

- `baseline_transformer/<cohort>/fold<k>/` - test predictions (`best_pred_total_DG.csv`, `best_probs_total_DG.csv`) and `test_metrics.json`
- `baseline_graph/<cohort>/fold<k>/` - `test_predictions.csv`, `test_metrics.json`, the code's own log in `result/` (model weights deleted)
- `logs/` - one log per run

Code and every change made to run it on our data: `external/CHANGES.md`.

A new experiment goes into a new, clearly named folder; nothing is overwritten.
