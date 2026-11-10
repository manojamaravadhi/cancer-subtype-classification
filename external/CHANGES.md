# Authors' released code, run standalone on our data

Downloaded 2026-11-02 (code files only, no data or model files).

| Folder | Source | Commit |
|---|---|---|
| `ref_transformer_original/` | the authors' GitHub repository (`Readme.md`, `2.src/*.py`) | af97a5c970f2b846de299e50cc876346d9811689 |
| `ref_graph_original/` | the authors' GitHub repository (`*.py`, `README.md`, `LICENSE`, `result/*/*.log`) | d9ff3c1f4d956e666b918a3c4ca7309fe0190f1b |

The `*_original/` folders are never edited. The `*_run/` folders are copies with the minimum
changes needed to run on our data. Every changed line is marked `RUN-CHANGE n` in the code,
and the full line-by-line difference is in `*_run/CHANGES.diff`.

Input data: written by `src/export_for_author_code.py` into `data/author_format/`. It contains the same
patients, the same 5 test folds and the same per-fold feature preparation as our model
(`src/train_exact.py`, FoldPrep). Only labeled patients are used, because both codes are supervised only.

## reference transformer method: `ref_transformer_run/2.src/train_test_dg+nocdg_KIPAN.py`

`model_Tran0814.py` and `utils.py` are **unchanged**: Transformer per omics for the L1-SVC features and the driver
features, classifier per omics, VCDN fusion, data loader `prepare_trte_data`.

| # | Change | Why |
|---|---|---|
| 1 | `import dgl` / `dgl.random.seed(42)` commented out | DGL is not installed; it was used only for that seed, nowhere else |
| 2 | `cuda:1` → `cuda:0` | this PC has one GPU |
| 3 | data / result folders are arguments `--data_dir`, `--out_dir` | released paths point to the authors' server |
| 4 | **epoch chosen on a validation slice** (10 % of the training fold, loaded with the authors' own `prepare_trte_data`) instead of on the test patients; the test patients are evaluated once, with the chosen model | the released code picked the epoch with the best TEST accuracy, which leaks test information (owner's choice: fair run only) |
| 5 | weights kept in memory, not written to disk at each improvement | released code saved ~1 GB per improvement at d_model 512; the same in-memory copy is used for the final test, as in the released code |
| 6 | test metrics saved to `test_metrics.json` (`import json` added) | released code only printed them |

Consequence of change 4: reference transformer method trains on 90 % of each training fold (the released code trained on 100 % and used
the test fold for epoch choice).

## reference graph method: `ref_graph_run/main.py`

`gcn.py`, `gat.py`, `train.py` and `data_process.py` are **unchanged**.

| # | Change | Why |
|---|---|---|
| 0 | snfpy → scikit-learn argument-name shim (`force_all_finite` → `ensure_all_finite`) | library version only; the same shim is used in our model |
| 1 | the omics are given on the command line (ours: mRNA, miRNA, methylation) and handled in loops, instead of four copied lines for the fixed files cna / met / mrna / rppa | our data has 3 omics, and miRNA instead of CNA and RPPA |
| 2 | train / test patients from our fold file `split.csv` instead of `train_test_split(test_size=0.25, random_state=0)` | same test patients as every other method |
| 3 | omics selection runs when fewer than ALL omics pass the training-accuracy threshold (released: `< 4`) | the released number 4 = their 4 omics |
| 4 | folders as arguments; the reported model's test predictions and metrics saved | released code wrote only a log and confusion-matrix images |

Unchanged method: SNF affinity (K = 20, mu = 0.5, squared Euclidean), top-1 % Laplacian, a GCN per omics
(100 hidden, dropout 0.5, Adam lr 0.001, weight decay 0.01, up to 150 epochs, stop after 20 epochs without a lower
training loss), normalised first-layer embeddings, SNF-fused graph, 2-head GAT, omics kept when training
accuracy > 0.8 (lowered by 0.05 until at least two pass), test patients get their own graph (features only).
Reported result = "selected omics" model if a selection was made, otherwise the all-omics model (their README).
Runs on CPU, as released.

Data difference from the paper (not a code change): the paper used CNA, methylation, mRNA and RPPA from
cBioPortal; here reference graph method gets our mRNA, miRNA and methylation (Xena), prepared with the same rules as its
`data_process.py` (features with > 10 % missing removed, missing set to 0, features with >= 10 % zeros removed;
mRNA / miRNA z-scored with the training patients' mean and standard deviation, like cBioPortal z-score files).
