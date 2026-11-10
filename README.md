# Ensembled Transformer-Graph Streams

Cancer subtype classification and survival prediction from three omics (mRNA, miRNA, DNA methylation).
Two published methods are used unchanged as streams and joined by a bilateral cross-attention fusion:

- **Transformer stream** - the reference transformer method (Int. J. Mol. Sci. 2026, 27, 4237; `papers/ref_transformer_2026.pdf`):
  a transformer per omics over cancer-driver and L1-SVC-selected features.
- **Graph stream** - the reference graph method (Bioinformatics 2025; `papers/ref_graph_2025.pdf`):
  SNF patient-similarity graph, a GCN per omics and a 2-head GAT.
- **Fusion** - each stream's embeddings projected to 128, bilateral cross-attention, a subtype head
  (class-balanced focal loss) and a Cox survival head (weight 0.15). No gates.
- **Ensemble (bagging)** - 5 members per fold, each on its own 70 / 20 / 10 split of the training patients
  (streams / fusion / epoch selection) and an 85 % random subset of the non-driver features.

**Final results: `results/final_ensembled_streams/`** (trained 2026-11-01 / 02; names neutralised 2026-11-03;
BRCA and COADREAD re-trained 2026-11-10 with one sample per patient - 5 metastatic samples of patients who also have a
primary sample removed; old files in `results/archive_with_duplicate_samples/`).

### GitHub repository

The repository holds only our model's code (`src/`, without the stream models - see below), its small input files
(`configs/cohorts.json`, `data/labels/`, `data/biological_priors/`) and the accuracy comparison
(`results/comparison_author_code/`). Folders and files mentioned below that are not in the repository are kept
locally by the authors.

Not included in the GitHub repository:

- `results/final_ensembled_streams/` (our model's predictions, report and figures), `results/baselines_author_code/`
  (outputs of the reference methods' code), `external/CHANGES.md` and `src/export_for_author_code.py`
  (data export for the reference methods' code).

- `src/models/exact_streams.py` - the two stream models, reproduced from the reference methods' released code;
  get that code from the authors' repositories linked in the two papers (Int. J. Mol. Sci. 2026, 27, 4237;
  Bioinformatics 2025). Training cannot run without it.
- `external/ref_*` - the authors' released code itself (only `external/CHANGES.md`, our list of changes, is included).
- `data/raw/` and `data/processed_exact/` - rebuilt from the public UCSC Xena TCGA files with the pipeline commands below.
- The Illumina 450K manifest (GEO GPL13534, 64 MB) - only the source of the included
  `data/biological_priors/hm450_tss1500_probe_to_gene.tsv`; download it from GEO if needed.
- `papers/` (reference PDFs), all log files and `results/archive_with_duplicate_samples/`.
- `paper/` (our manuscript), `configs/base_paper_results.json` (numbers published in the reference papers) and the figure
  `fig1_classification_vs_papers` made from it; `src/report_exact.py` and `src/plot_exact.py` read that file.

## Cohorts (configs/cohorts.json)

| Cohort | Patients (labeled) | Classes | Label source |
|---|---|---|---|
| brca_combined | 618 (564) | LumA, LumB, Basal, Her2, Normal | cBioPortal PanCancer SUBTYPE; missing ones filled from Xena PAM50 |
| lgg | 509 (502) | IDHmut-codel, IDHmut-non-codel, IDHwt | cBioPortal PanCancer SUBTYPE |
| kipan | 512 (512) | KIRC, KIRP, KICH | TCGA project |
| coadread | 336 (322) | 8 classes (COAD/READ x CIN, GS, MSI, POLE) | cBioPortal PanCancer SUBTYPE |
| stad | 367 (343) | CIN, MSI, GS, EBV, POLE | cBioPortal PanCancer SUBTYPE |

Omics and survival: UCSC Xena TCGA legacy hub (HiSeqV2, miRNA_HiSeq_gene, HumanMethylation450 made gene-level by
TSS1500 mean, survival OS / OS.time). Survival is only the Cox target of the fusion, never a stream input.
Age, stage and grade are not used.

## Results (5-fold cross-validation; accuracy / macro-F1 pooled over the test folds, C-index = mean over folds)

| Cohort | Accuracy | Macro-F1 | C-index | Log-rank p (high vs low risk) |
|---|---|---|---|---|
| BRCA | 0.869 | 0.838 | 0.662 | 0.025 |
| LGG | 0.980 | 0.982 | 0.765 | 4e-8 |
| KIPAN | 0.957 | 0.954 | 0.659 | 3e-6 |
| COADREAD | 0.758 | 0.388 | 0.564 | 0.13 |
| STAD | 0.854 | 0.680 | 0.532 | 0.24 |

Full report: `results/final_ensembled_streams/report/REPORT.md`; figures: `results/final_ensembled_streams/figures/{main,supplementary}`.

## Folder structure (as of 2026-11-03)

```
<project>/
├── README.md                        this file
├── requirements.txt                 Python packages
│
├── configs/
│   ├── cohorts.json                 cohort registry: brca_combined, lgg, kipan, coadread, stad
│   │                                (TCGA projects, raw folders, label file, class order, transformer-stream heads, driver-list rule)
│   └── base_paper_results.json      published numbers of the reference methods and FastMI-HGNet (with table numbers)
│
├── data/
│   ├── raw/                         UCSC Xena files: mRNA HiSeqV2, miRNA, methylation 450K, clinicalMatrix, survival
│   │   ├── tcga_brca/  tcga_lgg/  tcga_stad/
│   │   ├── tcga_kipan/{tcga_kirc, tcga_kirp, tcga_kich}/
│   │   ├── tcga_coadread/{tcga_coad, tcga_read}/
│   │   └── download_manifest.json   size, SHA-256, URL and date of every downloaded file
│   ├── labels/                      subtype labels
│   │   ├── brca_combined_subtype.tsv          used (PanCancer PAM50, missing ones from Xena PAM50; column 'source')
│   │   ├── lgg_pancanatlas_subtype.tsv        used
│   │   ├── coadread_subtype.tsv, stad_subtype.tsv   used
│   │   ├── brca_pancanatlas_subtype.tsv       source of brca_combined_subtype.tsv
│   │   └── *_cbioportal.json                  raw cBioPortal downloads
│   ├── biological_priors/
│   │   ├── cancer_drivers_<cohort>.json       cancer-specific driver lists (5 cohorts)
│   │   ├── mirbase_*, GPL13534_*, hm450_tss1500_probe_to_gene.tsv   ID maps
│   │   └── raw_downloads/                     original NCG 7 supplement (zip + the 2 tables used) / OncomiR downloads
│   └── processed_exact/tcga_<cohort>/         model inputs per cohort: X_trans_{mrna,mirna,meth}.npy (transformer stream),
│                                              X_graph_{mrna,mirna,meth}.npy (graph stream), y_subtype, y_os_time,
│                                              y_os_event, sample_ids, feature_names.json, preprocessing_metadata.json
│
├── src/
│   ├── download_tcga.py             download raw data for a cohort
│   ├── build_priors.py              build the miRBase ID map in data/biological_priors
│   ├── build_cancer_drivers.py      cancer-specific driver lists
│   ├── preprocess.py                helper functions (barcode handling, matrix reading) used by preprocess_exact.py
│   ├── preprocess_exact.py          preprocessing exactly as the two reference methods -> data/processed_exact
│   ├── train_exact.py               ensembled-streams training, 5-fold CV (default cohort brca_combined)
│   ├── interpret_exact.py           feature importance (called by train_exact.py --interpret)
│   ├── report_exact.py              REPORT.md, tables, survival summary
│   ├── enrichment_exact.py          pathway enrichment of the top genes
│   ├── plot_exact.py                paper figures
│   ├── export_for_author_code.py    writes our folds in the authors' input format (for the baseline runs)
│   └── models/
│       ├── exact_streams.py         transformer stream and graph stream (reference methods' code)
│       └── ensembled_streams.py     bilateral cross-attention fusion, focal loss, Cox loss
│
├── external/                        the reference methods' released code
│   ├── CHANGES.md                   sources, commits and every change made for the standalone runs
│   ├── ref_transformer_original/    untouched copy of the reference transformer method's code
│   ├── ref_transformer_run/         copy with the minimum changes to run on our data (+ CHANGES.diff)
│   ├── ref_graph_original/          untouched copy of the reference graph method's code (incl. their result logs)
│   └── ref_graph_run/               copy with the minimum changes to run on our data (+ CHANGES.diff)
│
├── papers/                          ref_transformer_2026.pdf, ref_graph_2025.pdf, genes-17-00865.pdf (FastMI-HGNet),
│                                    DeepMoIC.pdf, subtype_former.pdf
│
├── results/
│   ├── README.md                    what each results folder contains
│   ├── final_ensembled_streams/     FINAL model
│   │   ├── <cohort>_5fold_results.json, _predictions.csv, _feature_importance.csv, _learning_curves.json
│   │   ├── report/                  REPORT.md, classification / survival tables, top-30 biomarkers, enrichment/, figures/
│   │   ├── figures/{main, supplementary}/   paper figures (PNG + PDF)
│   │   └── logs/                    training, report, enrichment, plotting logs (original text of the run)
│   └── baselines_author_code/       reference methods' code run standalone on our folds
│       ├── baseline_transformer/<cohort>/fold<k>/   test predictions + test_metrics.json
│       ├── baseline_graph/<cohort>/fold<k>/         test predictions + test_metrics.json + the code's own log
│       └── logs/
│
├── logs/                            download_<cohort>.log, drivers_<cohort>.log, preprocess_exact_<cohort>.log,
│                                    preprocess_exact_pc.log (LGG and KIPAN preprocessing, 2026-10-31),
│                                    export_for_author_code.log (inputs of the baseline runs)
└── .venv/                           Python environment (not part of the results)
```

Log files keep the text exactly as written during the runs, so they still contain the original method names and folder paths.

## Pipeline (commands that produced the final model)

```bash
python -m src.download_tcga --cohort all            # Xena files (skips existing ones)
python -m src.build_cancer_drivers --cohort all     # per-cohort driver lists (NCG 7 + OncomiR)
python -m src.preprocess_exact --cohort all         # stream inputs -> data/processed_exact/

# training (one command per cohort: brca_combined, lgg, kipan, coadread, stad)
python -m src.train_exact --cohort <cohort> --trans_lr 1e-4 --trans_pretrain_epochs 50 \
       --member_split split --fusion_unlabeled --interpret --tag final_ensembled_streams

python -m src.report_exact --tag final_ensembled_streams --results_dir results/final_ensembled_streams
python -m src.enrichment_exact --tag final_ensembled_streams
python -m src.plot_exact --tag final_ensembled_streams --report_dir results/final_ensembled_streams/report --out_dir results/final_ensembled_streams/figures
```

`src/build_priors.py` built `data/biological_priors/mirbase_mimat_to_name.tsv` from the miRBase download.

## Comparison with the reference methods (results/baselines_author_code)

The reference methods' released code was run standalone on the same patients, folds and features:

```bash
python -m src.export_for_author_code --cohort all   # writes data/author_format/ (about 2 GB; deleted after the runs, recreated exactly by this command)
```

then `external/ref_transformer_run/2.src/train_test_dg+nocdg_KIPAN.py` (released defaults; epoch chosen on a 10 %
validation slice instead of the test patients) and `external/ref_graph_run/main.py`, one run per cohort and fold.
Every change to the authors' code is listed in `external/CHANGES.md`.
Results: `results/baselines_author_code/baseline_*/<cohort>/fold<k>/test_metrics.json` and the test predictions next to it.

Published numbers in `configs/base_paper_results.json` are reference only: the papers used other omics or labels,
a single split, and the reference transformer method's released code picked its epoch on the test set.

## Requirements

`requirements.txt` (Python 3.13, torch with CUDA), plus `snfpy` (SNF, used by the graph stream and by the standalone graph-method run).

Deleted on 2026-11-03 (not needed for the final model): earlier result versions, the UCEC cohort, the BRCA-Xena and
BRCA-PanCancer-only cohorts, the October pipeline scripts, `handoff/`, `data/author_format/`
(recreate with `python -m src.export_for_author_code --cohort all`), baseline model weights.
Deleted on 2026-11-08: `download_data.sh` (replaced by `src/download_tcga.py`), the October-pipeline prior files
(`ncg_v7_driver_genes.tsv`, `oncomir_driver_mirnas.tsv`, `illuminaMethyl450_hg19_probeMap.tsv`) and their code in
`src/preprocess.py` / `src/build_priors.py`, and the unused files extracted from the NCG 7 supplement zip.
Deleted on 2026-11-10 after the BRCA / COADREAD re-run: `data/author_format/` again (recreate as above) and the graph
code's model weights (`*.pkl`) of those runs.
