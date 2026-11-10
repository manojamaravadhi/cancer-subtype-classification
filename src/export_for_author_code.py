"""
Write our data in the input format of the authors' released code, so reference transformer method and reference graph method
can be run standalone on exactly the same patients, folds and features as our model.

Same as src/train_exact.py (imported, not copied):
  * the 5 outer folds (StratifiedKFold, seed 42, over all patients; unlabeled patients
    stratified by death event), so every method is tested on the same test patients;
  * the per-fold feature preparation FoldPrep, fit on the fold's labeled training patients:
      reference transformer method   - min-max scaling, cancer driver features, L1-SVC (C = 0.1) features;
      Graph stream - mRNA / miRNA z-scored with training mean/std, methylation as is.
Only labeled patients are written (both released codes are supervised only).

reference transformer method (the authors' GitHub repository, read by model_Tran0814.prepare_trte_data):
  data/author_format/trans/<cohort>/fold<k>/{select,final}/{WODG,DG}/
      <v>_tr.csv, <v>_te.csv, labels_tr.csv, labels_te.csv   (v: 1 mRNA, 2 miRNA, 3 methylation;
      comma-separated, no header, as in the repo's 1.data/KIPAN example)
  WODG = L1-SVC features, DG = driver features (repo folder "没进行svcl1的dg").
  The released code chose the epoch on the test patients. Fair run instead:
      select/: tr = 90 % of the labeled training patients, te = the other 10 % (validation)
      final/ : tr = the same 90 %,                       te = the test fold
  The epoch is chosen on select/te; final/te is used once for the reported result.

reference graph method (the authors' GitHub repository, read by main.py):
  data/author_format/graph/<cohort>/met_data.csv               (same for every fold)
  data/author_format/graph/<cohort>/fold<k>/mrna_data.csv, mirna_data.csv,
      subtype_data.csv (PATIENT_ID, class), split.csv (PATIENT_ID, set = train / test)
  First column = patient id, header row, as written by the repo's data_process.py.

Usage: python -m src.export_for_author_code --cohort all
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.train_exact import COHORTS, VIEWS, FoldPrep, load

OUT = "data/author_format"
COHORT_LIST = ["brca_combined", "lgg", "kipan", "coadread", "stad"]


def write_matrix(path, X):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, X, fmt="%.6g", delimiter=",")


def export(cohort, n_folds=5, seed=42, val_ratio=0.1):
    data = load(cohort)
    classes = data["meta"]["class_names"]
    n_cls = len(classes)
    y, ev = data["y_subtype"], data["y_os_event"]
    ids = np.array([str(s) for s in data["sample_ids"]])
    n = len(y)
    strat = np.where(y >= 0, y, n_cls + ev)                     # identical to train_exact.run
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n), strat))
    info = {"cohort": cohort, "classes": classes, "n_patients": int(n), "n_labeled": int((y >= 0).sum()),
            "fold_seed": seed, "val_ratio": val_ratio, "folds": []}

    met_written = False
    for k, (tr, te) in enumerate(folds, 1):
        tr_lab, te_lab = tr[y[tr] >= 0], te[y[te] >= 0]
        prep = FoldPrep().fit(data, tr_lab, y)

        # ---------------- reference transformer method: 90 / 10 split of the labeled training patients ----------------
        # per class, round(10 %) of the patients go to validation, but every class keeps >= 1 training
        # patient (same rule as split_member in train_exact.py; a 1-patient class has no validation patient)
        rng = np.random.default_rng(seed + k)
        fit, val = [], []
        for c in np.unique(y[tr_lab]):
            idx = rng.permutation(tr_lab[y[tr_lab] == c])
            n_v = min(int(round(val_ratio * len(idx))), len(idx) - 1)
            val.append(idx[:n_v]); fit.append(idx[n_v:])
        fit, val = np.sort(np.concatenate(fit)), np.sort(np.concatenate(val))
        base = f"{OUT}/trans/{cohort}/fold{k}"
        for part, te_rows in (("select", val), ("final", te_lab)):
            for sub, key in (("WODG", "svc"), ("DG", "dg")):
                d = f"{base}/{part}/{sub}"
                for vi, v in enumerate(VIEWS, 1):
                    cols = prep.trans[v][key]
                    if len(cols) == 0:
                        raise SystemExit(f"STOP: {cohort} fold {k} view {v} has no {sub} features")
                    X = prep.trans[v]["X"][:, cols]
                    write_matrix(f"{d}/{vi}_tr.csv", X[fit])
                    write_matrix(f"{d}/{vi}_te.csv", X[te_rows])
                    with open(f"{d}/{vi}_featname.csv", "w", encoding="utf-8") as f:
                        f.write("\n".join(np.asarray(data["names"][f"trans_{v}"])[cols]) + "\n")
                np.savetxt(f"{d}/labels_tr.csv", y[fit], fmt="%d", delimiter=",")
                np.savetxt(f"{d}/labels_te.csv", y[te_rows], fmt="%d", delimiter=",")
        pd.DataFrame({"PATIENT_ID": np.concatenate([ids[fit], ids[val], ids[te_lab]]),
                      "set": ["train"] * len(fit) + ["validation"] * len(val) + ["test"] * len(te_lab)}
                     ).to_csv(f"{base}/patients.csv", index=False)

        # ---------------- reference graph method: all labeled training patients / test fold ----------------
        rows = np.sort(np.concatenate([tr_lab, te_lab]))
        mbase = f"{OUT}/graph/{cohort}"
        os.makedirs(f"{mbase}/fold{k}", exist_ok=True)

        def frame(v):
            names = np.asarray(data["names"][f"graph_{v}"])
            df = pd.DataFrame(prep.graph[v][rows], columns=names)
            df.insert(0, "index", ids[rows])
            return df
        if not met_written:                      # methylation is not re-scaled per fold
            frame("meth").to_csv(f"{mbase}/met_data.csv", index=False, float_format="%.6g")
            met_all_rows = rows
            met_written = True
        assert np.array_equal(rows, met_all_rows), "labeled patient set differs between folds"
        for v, fn in (("mrna", "mrna_data.csv"), ("mirna", "mirna_data.csv")):
            frame(v).to_csv(f"{mbase}/fold{k}/{fn}", index=False, float_format="%.6g")
        pd.DataFrame({"PATIENT_ID": ids[rows], "class": y[rows]}).to_csv(f"{mbase}/fold{k}/subtype_data.csv", index=False)
        pd.DataFrame({"PATIENT_ID": ids[rows], "set": np.where(np.isin(rows, te_lab), "test", "train")}
                     ).to_csv(f"{mbase}/fold{k}/split.csv", index=False)

        info["folds"].append({"fold": k, "trans_train": int(len(fit)), "trans_validation": int(len(val)),
                              "graph_train": int(len(tr_lab)), "test": int(len(te_lab)),
                              "features": prep.summary()})
        print(f"{cohort} fold {k}: reference transformer method train {len(fit)} / validation {len(val)} | reference graph method train {len(tr_lab)} "
              f"| test {len(te_lab)} | {json.dumps(prep.summary())}", flush=True)

    with open(f"{OUT}/{cohort}_export_info.json", "w") as f:
        json.dump(info, f, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="all", choices=COHORT_LIST + ["all"])
    a = p.parse_args()
    for c in (COHORT_LIST if a.cohort == "all" else [a.cohort]):
        if c not in COHORTS:
            raise SystemExit(f"unknown cohort {c}")
        if os.path.exists(f"{OUT}/{c}_export_info.json"):
            raise SystemExit(f"STOP: {OUT}/{c}_export_info.json exists - export already done, not overwriting")
        export(c)


if __name__ == "__main__":
    main()
