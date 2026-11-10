"""
Bagged Ensembled Transformer-Graph Streams ensemble with the two reference methods as streams.

  Stream A = transformer stream, as released (src/models/exact_streams.py)
  Stream B = graph stream, as released (src/models/exact_streams.py)
  Fusion   = bilateral cross-attention over both streams' patient representations,
             with a subtype head and a Cox survival head (this project's addition).

Protocol (leak-free), per outer fold of a stratified 5-fold CV:
  1. Feature preparation is fit on the fold's TRAINING patients only:
     transformer stream  - min-max scaling to [0, 1]; driver features from the cancer-specific
              lists; non-driver features by L1-SVC (C = 0.1) on training labels.
     Graph stream - mRNA / miRNA z-scored with training mean/std (cBioPortal-style
              z-scores); methylation beta values as they are.
  2. M bagging members. Each draws a bootstrap sample of the labeled training
     patients and an 85 % random subset of the non-driver features.
       * transformer stream is trained exactly as released (5 pre-training epochs per view,
         then view classifiers + VCDN; Adam, lr 1e-5, batch 128, up to 100 epochs).
         The epoch is chosen by OUT-OF-BAG accuracy (the transformer method's own criterion) and
         training stops after 50 epochs without improvement (paper: early stop 50).
         The released code chose the epoch on the TEST fold; we never do that.
       * graph stream is trained exactly as released on the member's unique training
         patients (snfpy affinity K=20, mu=0.5; top-1 % Laplacian; GCN per omics;
         omics chosen by training accuracy > 0.8; SNF; 2-head GAT). As in the paper,
         a held-out patient set (out-of-bag or test fold) gets its own graph built
         from that set only (features only, no labels).
       * The fusion layer is trained on the member's training patients and its
         epoch chosen on the out-of-bag patients.
  3. Each output (transformer stream alone, graph stream alone, fused) is averaged over members,
     weighted by that output's out-of-bag macro-F1. The test fold is used once.

Options added 2026-11-01 (defaults keep the design above unchanged):
  --member_split split   each member cuts its labeled training patients (per class) into
                         disjoint stream / fusion / selection sets (--split_ratios, default
                         0.7 0.2 0.1) instead of a bootstrap sample: the streams train on the
                         first set, the fusion on the second (so it only sees stream outputs
                         of patients the streams did not train on), and the third picks the
                         transformer stream epoch, the fusion epoch and the member weights.
  --fusion_unlabeled     the fold's unlabeled training patients are added to the fusion's
                         training set; they enter only the Cox survival loss.

Results: results/<tag>/<cohort>_5fold_results.json, <cohort>_predictions.csv,
         <cohort>_learning_curves.json
         with --interpret also <cohort>_feature_importance.csv (src/interpret_exact.py;
         measured after training on each test fold, does not change training or results)
Usage:   .venv/bin/python -m src.train_exact --cohort brca
"""

import os
import sys
import json
import time
import math
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import snf
import snf.compute
from sklearn.utils.validation import check_array as _check_array
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.models.exact_streams import (TransformerModel, cal_sample_weight, make_laplacian, normalize, graph_train_model)
from src.models.ensembled_streams import (BilateralCrossAttentionFusion, ClassBalancedFocalLoss, cox_ph_loss,
                                    compute_c_index)
from src.interpret_exact import member_importance

def _check_array_compat(*a, **kw):
    """snfpy 0.2.2 passes `force_all_finite`, renamed `ensure_all_finite` in scikit-learn >= 1.6.
    Only the argument name is translated; the check itself is unchanged."""
    if "force_all_finite" in kw:
        kw["ensure_all_finite"] = kw.pop("force_all_finite")
    return _check_array(*a, **kw)


snf.compute.check_array = _check_array_compat

VIEWS = ["mrna", "mirna", "meth"]
DRIVER_KEY = {"mrna": "genes", "mirna": "mirnas", "meth": "genes"}
# cohort registry (configs/cohorts.json); transformer stream heads: paper Sec. 4.5 (BRCA 4, KIPAN 2; LGG not in paper -> 4)
COHORTS = {k: v for k, v in json.load(open("configs/cohorts.json")).items() if not k.startswith("_")}
TRANS_HEADS = {c: spec["trans_heads"] for c, spec in COHORTS.items()}


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def metrics(y, pred, n_cls):
    m = y >= 0
    if m.sum() == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan"), "n_labeled": 0}
    lab = list(range(n_cls))
    return {"accuracy": float(accuracy_score(y[m], pred[m])),
            "macro_f1": float(f1_score(y[m], pred[m], labels=lab, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y[m], pred[m], labels=lab, average="weighted", zero_division=0)),
            "n_labeled": int(m.sum())}


def load(cohort):
    d = f"data/processed_exact/tcga_{cohort}"
    data = {k: np.load(f"{d}/{k}.npy") for k in ["y_subtype", "y_os_event", "y_os_time", "sample_ids"]}
    for v in VIEWS:
        data[f"trans_{v}"] = np.load(f"{d}/X_trans_{v}.npy")
        data[f"graph_{v}"] = np.load(f"{d}/X_graph_{v}.npy")
    data["names"] = json.load(open(f"{d}/feature_names.json"))
    data["meta"] = json.load(open(f"{d}/preprocessing_metadata.json"))
    data["drivers"] = json.load(open(f"data/biological_priors/cancer_drivers_{cohort}.json"))
    return data


# ---------------------------------------------------------------------------
# Step 1: per-fold feature preparation (training patients only)
# ---------------------------------------------------------------------------
class FoldPrep:
    def fit(self, data, tr_lab, y):
        self.trans, self.graph = {}, {}
        for v in VIEWS:
            X = data[f"trans_{v}"]
            mn, mx = X[tr_lab].min(0), X[tr_lab].max(0)
            rng = np.where(mx > mn, mx - mn, 1.0)
            Xs = (X - mn) / rng
            ds = set(data["drivers"][DRIVER_KEY[v]])
            is_d = np.array([n in ds for n in data["names"][f"trans_{v}"]])
            nd = np.where(~is_d)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                svc = LinearSVC(penalty="l1", dual=False, C=0.1, max_iter=2000, random_state=42)
                svc.fit(Xs[tr_lab][:, nd], y[tr_lab])
            w = np.abs(svc.coef_).sum(0)
            svc_idx = nd[np.where(w > 1e-5)[0]]
            self.trans[v] = {"X": Xs.astype(np.float32), "dg": np.where(is_d)[0], "svc": svc_idx}

            Xb = data[f"graph_{v}"]
            if v in ("mrna", "mirna"):
                mu, sd = Xb[tr_lab].mean(0), Xb[tr_lab].std(0)
                Xb = (Xb - mu) / np.where(sd > 1e-8, sd, 1.0)
            self.graph[v] = Xb.astype(np.float32)
        return self

    def summary(self):
        return {v: {"trans_driver": int(len(self.trans[v]["dg"])), "trans_l1svc": int(len(self.trans[v]["svc"])),
                    "graph": int(self.graph[v].shape[1])} for v in VIEWS}


# ---------------------------------------------------------------------------
# Step 2a: transformer stream (training loop of train_test_dg+nocdg_KIPAN.py)
# ---------------------------------------------------------------------------
class TransformerStream:
    def __init__(self, prep, rng, args, n_cls, cohort, device):
        self.dev = device
        self.cols_svc, self.cols_dg = {}, {}
        for v in VIEWS:
            s = prep.trans[v]["svc"]
            k = max(1, int(round(args.subspace_ratio * len(s)))) if len(s) else 0
            self.cols_svc[v] = np.sort(rng.choice(s, size=k, replace=False)) if k else s
            self.cols_dg[v] = prep.trans[v]["dg"]
        self.X = {v: torch.tensor(prep.trans[v]["X"], device=device) for v in VIEWS}
        self.model = TransformerModel([len(self.cols_svc[v]) for v in VIEWS], [len(self.cols_dg[v]) for v in VIEWS], n_cls,
                           args.d_model, args.dim_feedforward, args.trans_dropout, TRANS_HEADS[cohort]).to(device)
        self.ci = {v: torch.tensor(self.cols_svc[v], device=device) for v in VIEWS}
        self.cd = {v: torch.tensor(self.cols_dg[v], device=device) for v in VIEWS}

    def inputs(self, idx_t):
        xs = [self.X[v][idx_t][:, self.ci[v]] for v in VIEWS]
        xd = [self.X[v][idx_t][:, self.cd[v]] for v in VIEWS]
        return xs, xd

    @torch.no_grad()
    def predict(self, rows, bs=32):
        self.model.eval()
        P, E = [], []
        for s in range(0, len(rows), bs):
            idx = torch.tensor(rows[s:s + bs], device=self.dev)
            emb, logits = self.model.embed(*self.inputs(idx))
            P.append(torch.softmax(logits, -1).cpu().numpy())
            E.append(torch.stack(emb, 1).cpu().numpy())
        return np.concatenate(P), np.concatenate(E)

    def fit(self, boot, oob, y, n_cls, args, log, seed):
        dev, m = self.dev, self.model
        torch.manual_seed(seed)
        if args.trans_class_weight == "released":     # default: transformer stream as released (weight = class share)
            w_np = cal_sample_weight(y[boot], n_cls)
        else:                                          # ablation "balanced": weight = N / (classes x class count)
            cnt = np.bincount(y[boot], minlength=n_cls).astype(float)
            w_np = len(boot) / (n_cls * np.maximum(cnt, 1))[y[boot]]
        sw = torch.tensor(w_np, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y, device=dev)
        crit = nn.CrossEntropyLoss(reduction="none")

        def make_opts():
            o = {i: torch.optim.Adam(list(m.E[i].parameters()) + list(m.C[i].parameters()), lr=args.trans_lr)
                 for i in range(3)}
            o["C"] = torch.optim.Adam(m.vcdn.parameters(), lr=args.trans_lr)
            return o

        def epoch(opts, use_vcdn):
            m.train()
            perm = np.random.default_rng(seed + len(curve["loss"])).permutation(len(boot))
            tot = 0.0
            for s in range(0, len(perm), args.trans_batch):
                pos = perm[s:s + args.trans_batch]
                if len(pos) < 2:
                    continue
                # One transformer stream batch (128) is processed as micro-batches whose gradients are
                # accumulated before each optimiser step (needed to fit 8 GB of GPU memory).
                # Loss = mean over the full batch, as in the released code.
                # near-equal micro-batches, so none has a single patient (BatchNorm needs >= 2)
                micro = np.array_split(pos, math.ceil(len(pos) / args.trans_micro))
                ci_parts = [[None, None, None] for _ in micro]
                for i in range(3):
                    opts[i].zero_grad()
                    for j, mp in enumerate(micro):
                        idx = torch.tensor(boot[mp], device=dev)
                        xs, xd = self.inputs(idx)
                        ci = m.C[i](m.E[i](xs[i], xd[i]))
                        ci_parts[j][i] = ci.detach()
                        loss = torch.sum(crit(ci, y_t[idx]) * sw[torch.tensor(mp, device=dev)]) / len(pos)
                        loss.backward()
                        tot += loss.item()
                    opts[i].step()
                if use_vcdn:
                    opts["C"].zero_grad()
                    for j, mp in enumerate(micro):
                        idx = torch.tensor(boot[mp], device=dev)
                        loss = torch.sum(crit(m.vcdn(list(ci_parts[j])), y_t[idx]) * sw[torch.tensor(mp, device=dev)]) / len(pos)
                        loss.backward()
                        tot += loss.item()
                    opts["C"].step()
            return tot / math.ceil(len(perm) / args.trans_batch)

        curve = {"loss": [], "oob_acc": [], "oob_macro_f1": []}
        pre = make_opts()
        for _ in range(args.trans_pretrain_epochs):
            curve["loss"].append(epoch(pre, False)); curve["oob_acc"].append(None); curve["oob_macro_f1"].append(None)
        opts = make_opts()
        best = {"acc": -1, "state": None, "epoch": 0}
        bad = 0
        for ep in range(1, args.trans_epochs + 1):
            curve["loss"].append(epoch(opts, True))
            p, _ = self.predict(oob)
            mt = metrics(y[oob], p.argmax(1), n_cls)
            curve["oob_acc"].append(mt["accuracy"]); curve["oob_macro_f1"].append(mt["macro_f1"])
            if mt["accuracy"] > best["acc"]:
                best = {"acc": mt["accuracy"], "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}}
                bad = 0
            else:
                bad += 1
            if ep % 10 == 0:
                log(f"transformer stream ep {ep:3d} loss {curve['loss'][-1]:.4f} OOB acc {mt['accuracy']:.3f} mF1 {mt['macro_f1']:.3f}")
            if bad >= args.trans_patience:
                break
        m.load_state_dict(best["state"])
        return best["epoch"], curve


# ---------------------------------------------------------------------------
# Step 2b: graph stream (main.py)
# ---------------------------------------------------------------------------
class GraphStream:
    def __init__(self, prep, rng, args, device):
        self.dev, self.args = device, args
        self.cols = {}
        for v in VIEWS:
            n = prep.graph[v].shape[1]
            k = max(1, int(round(args.subspace_ratio * n)))
            self.cols[v] = np.sort(rng.choice(n, size=k, replace=False))
        self.X = {v: prep.graph[v][:, self.cols[v]] for v in VIEWS}

    def _affinity(self, rows):
        return snf.compute.make_affinity(*[self.X[v][rows].astype(np.float64) for v in VIEWS],
                                         metric="sqeuclidean", K=20, mu=0.5)

    def _t(self, a):
        return torch.tensor(a, dtype=torch.float32, device=self.dev)

    def fit(self, train_rows, y, n_cls, seed):
        aff = self._affinity(train_rows)
        laps = [self._t(make_laplacian(a)) for a in aff]
        lab = torch.tensor(y[train_rows], device=self.dev)
        self.gcns, accs = [], []
        self.loss_curves = {"gcn_loss": {}, "gat_loss": []}      # saved for the learning-curve plots only
        for i, v in enumerate(VIEWS):
            g, acc, gcn_losses = graph_train_model("GCN", 100, self._t(self.X[v][train_rows]), laps[i], lab, n_cls, self.dev, seed + i)
            self.gcns.append(g); accs.append(acc)
            self.loss_curves["gcn_loss"][v] = gcn_losses
        thr = 0.8
        sel = [i for i, a in enumerate(accs) if a > thr]
        while len(sel) < 2:
            thr -= 0.05
            sel = [i for i, a in enumerate(accs) if a > thr]
        self.sel, self.gcn_train_acc = sel, accs
        H, fused = self._embed(train_rows, aff, laps)
        self.gat, self.gat_train_acc, losses = graph_train_model("GAT", 100, H, fused, lab, n_cls, self.dev, seed + 9)
        self.loss_curves["gat_loss"] = losses
        return {"gcn_train_acc": dict(zip(VIEWS, accs)), "selected_omics": [VIEWS[i] for i in sel],
                "gat_train_acc": self.gat_train_acc, "gat_epochs": len(losses)}

    @torch.no_grad()
    def _embed(self, rows, aff=None, laps=None):
        if aff is None:
            aff = self._affinity(rows)
            laps = [self._t(make_laplacian(a)) for a in aff]
        embs = [normalize(self.gcns[i].forward_1(self._t(self.X[VIEWS[i]][rows]), laps[i])) for i in self.sel]
        fused = self._t(make_laplacian(snf.snf([aff[i] for i in self.sel], K=20)))
        self._last_embs = embs
        return torch.cat(embs, 1), fused

    @torch.no_grad()
    def predict(self, rows):
        """Graph built from `rows` only (graph stream test procedure)."""
        H, fused = self._embed(rows)
        self.gat.eval()
        probs = torch.exp(self.gat(H, fused)).cpu().numpy()
        hid = self.gat.hidden(H, fused).cpu().numpy()
        gcn = torch.stack(self._last_embs, 1).cpu().numpy()
        return probs, hid, gcn


# ---------------------------------------------------------------------------
# Step 2c: fusion of the two streams
# ---------------------------------------------------------------------------
class Fusion(nn.Module):
    def __init__(self, n_cls, n_gcn_tokens, hidden=128, dropout=0.3):
        super().__init__()
        self.pa = nn.Linear(64, hidden)          # transformer stream per-view 64-d embeddings -> tokens
        self.pz_a = nn.Linear(64 * 3, hidden)
        self.pb_gat = nn.Linear(200, hidden)     # graph stream GAT hidden (2 heads x 100)
        self.pb_gcn = nn.Linear(100, hidden)     # graph stream per-omics GCN latent (100)
        self.fuse = BilateralCrossAttentionFusion(hidden, 4, dropout)

        def head(o):
            return nn.Sequential(nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden // 2, o))
        self.cls, self.surv = head(n_cls), head(1)

    def forward(self, ea, gat, gcn):
        tok_a = self.pa(ea)
        z_a = self.pz_a(ea.flatten(1))
        z_b = self.pb_gat(gat)
        tok_b = torch.cat([z_b.unsqueeze(1), self.pb_gcn(gcn)], 1)
        z, _ = self.fuse(z_a, tok_a, z_b, tok_b)
        return self.cls(z), self.surv(z).squeeze(-1)


def train_fusion(tr, va, y, ev, tm, n_cls, args, device, seed, log):
    torch.manual_seed(seed)
    T = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in tr.items() if k in ("ea", "gat", "gcn")}
    V = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in va.items() if k in ("ea", "gat", "gcn")}
    f = Fusion(n_cls, tr["gcn"].shape[1], dropout=args.dropout).to(device)
    y_fit = y[tr["rows"]]
    counts = np.bincount(y_fit[y_fit >= 0], minlength=n_cls)   # unlabeled rows (-1) only enter the Cox loss
    crit = ClassBalancedFocalLoss(counts).to(device)
    opt = torch.optim.AdamW(f.parameters(), lr=args.fusion_lr, weight_decay=1e-4)
    yt = torch.tensor(y[tr["rows"]], device=device)
    et = torch.tensor(ev[tr["rows"]], device=device)
    tt = torch.tensor(tm[tr["rows"]], dtype=torch.float32, device=device)
    best, bad, curve = {"score": -1e9, "state": None, "epoch": 0}, 0, {"loss": [], "oob_macro_f1": [], "oob_c_index": []}
    for ep in range(1, args.fusion_epochs + 1):
        f.train()
        opt.zero_grad()
        logits, risk = f(T["ea"], T["gat"], T["gcn"])
        loss = crit(logits, yt) + args.alpha * cox_ph_loss(risk, et, tt)
        loss.backward()
        opt.step()
        f.eval()
        with torch.no_grad():
            lv, rv = f(V["ea"], V["gat"], V["gcn"])
        mt = metrics(y[va["rows"]], lv.argmax(1).cpu().numpy(), n_cls)
        c = compute_c_index(rv.cpu().numpy(), ev[va["rows"]], tm[va["rows"]])
        curve["loss"].append(loss.item()); curve["oob_macro_f1"].append(mt["macro_f1"]); curve["oob_c_index"].append(c)
        score = mt["macro_f1"] + (0.5 if np.isnan(c) else c)
        if score > best["score"]:
            best = {"score": score, "epoch": ep, "state": {k: v.detach().clone() for k, v in f.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        if bad >= args.fusion_patience:
            break
    f.load_state_dict(best["state"])
    log(f"Fusion best epoch {best['epoch']} (OOB macro-F1 + C-index = {best['score']:.3f})")
    return f, best["epoch"], curve


def split_member(tr_lab, y, ratios, rng):
    """--member_split split: per class, shuffle the labeled training patients and cut them into
    stream / fusion / selection sets (ratios e.g. 0.7 0.2 0.1). No patient is in two sets.
    Every class present gets >= 1 stream patient; tiny classes may have none in the other sets."""
    s, f, v = [], [], []
    for c in np.unique(y[tr_lab]):
        idx = rng.permutation(tr_lab[y[tr_lab] == c])
        n = len(idx)
        n_s = max(1, int(round(ratios[0] * n)))
        n_v = min(int(round(ratios[2] * n)), n - n_s)
        n_f = n - n_s - n_v
        s.append(idx[:n_s]); f.append(idx[n_s:n_s + n_f]); v.append(idx[n_s + n_f:])
    return tuple(np.sort(np.concatenate(p)) for p in (s, f, v))


# ---------------------------------------------------------------------------
# Step 3: cross-validation driver
# ---------------------------------------------------------------------------
def run(cohort, args):
    t0 = time.time()
    dev = get_device(args.device)
    data = load(cohort)
    classes = data["meta"]["class_names"]
    n_cls = len(classes)
    y, ev, tm = data["y_subtype"], data["y_os_event"], data["y_os_time"].astype(np.float32)
    n = len(y)
    print("=" * 78)
    print(f"TCGA-{cohort.upper()} | {n} patients ({(y >= 0).sum()} labeled) | {classes} | {dev}")
    print(f"members={args.ensemble_size} subspace={args.subspace_ratio} d_model={args.d_model} "
          f"transformer stream lr={args.trans_lr} batch={args.trans_batch} epochs<={args.trans_epochs}")
    print("=" * 78)
    out_dir = os.path.join("results", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # stratified 5-fold over all patients (unlabeled stratified by death event)
    strat = np.where(y >= 0, y, n_cls + ev)
    folds = list(StratifiedKFold(args.n_folds, shuffle=True, random_state=args.seed).split(np.zeros(n), strat))
    outs = ["trans", "graph", "ensembled"]
    oof = {o: np.full((n, n_cls), np.nan, np.float32) for o in outs}
    oof_risk = np.full(n, np.nan, np.float32)
    fold_res, curves = [], {}
    imp_rows = []

    for k, (tr, te) in enumerate(folds, 1):
        log0 = lambda s: print(f"  [F{k}] {s}", flush=True)
        tr_lab = tr[y[tr] >= 0]
        prep = FoldPrep().fit(data, tr_lab, y)
        log0(f"train {len(tr)} ({len(tr_lab)} labeled) | test {len(te)} | features {json.dumps(prep.summary())}")
        P = {o: [] for o in outs}; R, W = [], {o: [] for o in outs}; minfo = []
        for mi in range(args.ensemble_size):
            seed = args.seed + 1000 * k + mi
            rng = np.random.default_rng(seed)
            log = lambda s, mi=mi: print(f"  [F{k} M{mi + 1}] {s}", flush=True)
            if args.member_split == "bootstrap":      # v2 design
                boot = rng.choice(tr_lab, size=len(tr_lab), replace=True)
                uniq = np.unique(boot)
                oob = np.setdiff1d(tr_lab, uniq)
                fus_rows = uniq                       # fusion trained on the streams' own training patients
            else:                                     # "split": stream / fusion / selection sets, no overlap
                uniq, fus_rows, oob = split_member(tr_lab, y, args.split_ratios, rng)
                boot = uniq
            if args.fusion_unlabeled:                 # unlabeled training patients: Cox loss only
                fus_rows = np.sort(np.concatenate([fus_rows, tr[y[tr] < 0]]))
            if mi == 0:
                log(f"member sets: streams {len(np.unique(boot))} unique | fusion {len(fus_rows)} "
                    f"({int((y[fus_rows] >= 0).sum())} labeled) | selection {len(oob)}")

            A = TransformerStream(prep, rng, args, n_cls, cohort, dev)
            a_epoch, a_curve = A.fit(boot, oob, y, n_cls, args, log, seed)
            B = GraphStream(prep, rng, args, dev)
            b_info = B.fit(uniq, y, n_cls, seed)
            log(f"graph stream: {b_info}")

            def stream_outputs(rows):
                pa, ea = A.predict(rows)
                pb, gat, gcn = B.predict(rows)
                return {"rows": rows, "pa": pa, "ea": ea, "pb": pb, "gat": gat, "gcn": gcn}
            S_tr, S_oob, S_te = stream_outputs(fus_rows), stream_outputs(oob), stream_outputs(te)
            fus, f_epoch, f_curve = train_fusion(S_tr, S_oob, y, ev, tm, n_cls, args, dev, seed, log)

            @torch.no_grad()
            def fuse_out(S):
                fus.eval()
                l, r = fus(*[torch.tensor(S[x], dtype=torch.float32, device=dev) for x in ("ea", "gat", "gcn")])
                return torch.softmax(l, -1).cpu().numpy(), r.cpu().numpy()
            pf_oob, _ = fuse_out(S_oob)
            pf_te, r_te = fuse_out(S_te)
            _, r_tr = fuse_out(S_tr)
            r_te = (r_te - r_tr.mean()) / (r_tr.std() + 1e-8)
            if args.interpret:      # after training; forward passes only (src/interpret_exact.py)
                rows = member_importance(A, fus, S_te, te, y, ev, tm, n_cls, data["names"], k, mi + 1, dev)
                imp_rows += rows
                log(f"feature importance recorded for {len(rows)} transformer stream-input features / omics")

            info = {"member": mi + 1, "n_boot_unique": int(len(uniq)), "n_oob": int(len(oob)),
                    "n_fusion": int(len(fus_rows)), "n_fusion_labeled": int((y[fus_rows] >= 0).sum()),
                    "trans_best_epoch": a_epoch, "fusion_best_epoch": f_epoch, "graph": b_info}
            for o, po, pt in [("trans", S_oob["pa"], S_te["pa"]), ("graph", S_oob["pb"], S_te["pb"]),
                              ("ensembled", pf_oob, pf_te)]:
                mo = metrics(y[oob], po.argmax(1), n_cls)
                w = mo["macro_f1"] if mo["macro_f1"] and mo["macro_f1"] > 0 else 1e-3
                W[o].append(w); P[o].append(pt)
                info[f"{o}_oob"] = mo
                info[f"{o}_test_member_alone"] = metrics(y[te], pt.argmax(1), n_cls)
            R.append(r_te)
            minfo.append(info)
            curves[f"fold{k}_member{mi + 1}"] = {"trans": a_curve, "fusion": f_curve, "graph": B.loss_curves}
            log(f"OOB macro-F1  transformer stream {info['trans_oob']['macro_f1']:.3f} | graph stream {info['graph_oob']['macro_f1']:.3f} "
                f"| ensembled {info['ensembled_oob']['macro_f1']:.3f}")
            del A, B, fus
            if dev.type == "mps":
                torch.mps.empty_cache()

        fr = {"fold": k, "n_test": int(len(te)), "members": minfo}
        for o in outs:
            w = np.asarray(W[o]) / np.sum(W[o])
            Pe = np.tensordot(w, np.stack(P[o]), axes=1)
            oof[o][te] = Pe
            fr[o] = metrics(y[te], Pe.argmax(1), n_cls)
        w = np.asarray(W["ensembled"]) / np.sum(W["ensembled"])
        Re = np.tensordot(w, np.stack(R), axes=1)
        oof_risk[te] = Re
        fr["ensembled"]["c_index"] = compute_c_index(Re, ev[te], tm[te])
        fold_res.append(fr)
        log0("ENSEMBLE test fold: " + " | ".join(
            f"{o} acc {fr[o]['accuracy']:.4f} mF1 {fr[o]['macro_f1']:.4f}" for o in outs)
            + f" | C-index {fr['ensembled']['c_index']:.4f}")

    def agg(o, key):
        v = np.array([f[o][key] for f in fold_res], dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
    summary = {"cohort": f"TCGA-{cohort.upper()}", "num_samples": n, "num_labeled": int((y >= 0).sum()),
               "class_names": classes, "config": vars(args),
               "label_source": data["meta"]["label_source"],
               "results": {o: {m: agg(o, m) for m in ["accuracy", "macro_f1", "weighted_f1"]} for o in outs},
               "pooled_out_of_fold": {o: metrics(y, np.nan_to_num(oof[o]).argmax(1), n_cls) for o in outs},
               "folds": fold_res, "elapsed_minutes": round((time.time() - t0) / 60, 1)}
    summary["results"]["ensembled"]["c_index"] = agg("ensembled", "c_index")
    summary["pooled_out_of_fold"]["ensembled"]["c_index"] = compute_c_index(oof_risk, ev, tm)
    json.dump(summary, open(f"{out_dir}/{cohort}_5fold_results.json", "w"), indent=2, default=float)
    json.dump(curves, open(f"{out_dir}/{cohort}_learning_curves.json", "w"), default=float)
    df = pd.DataFrame({"sample_id": data["sample_ids"], "true_label": y,
                       "true_name": [classes[i] if i >= 0 else "unlabeled" for i in y],
                       "risk_score": oof_risk, "os_event": ev, "os_time": tm})
    for o in outs:
        df[f"pred_{o}"] = np.nan_to_num(oof[o]).argmax(1)
    df["pred_label"] = df["pred_ensembled"]
    df["pred_name"] = [classes[i] for i in df["pred_label"]]
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = oof["ensembled"][:, i]
    df.to_csv(f"{out_dir}/{cohort}_predictions.csv", index=False)
    if args.interpret:
        pd.DataFrame(imp_rows).to_csv(f"{out_dir}/{cohort}_feature_importance.csv", index=False)

    print("\n" + "=" * 78 + f"\n5-FOLD SUMMARY TCGA-{cohort.upper()} ({summary['num_labeled']} labeled)")
    for o, name in [("trans", "transformer stream alone (bagged)"), ("graph", "graph stream alone (bagged)"),
                    ("ensembled", "Ensembled Transformer-Graph Streams fused (bagged)")]:
        r = summary["results"][o]
        print(f"  {name:32s} acc {r['accuracy']['mean'] * 100:6.2f} ± {r['accuracy']['std'] * 100:5.2f} | "
              f"macro-F1 {r['macro_f1']['mean'] * 100:6.2f} ± {r['macro_f1']['std'] * 100:5.2f}")
    print(f"  C-index (ensembled) {summary['results']['ensembled']['c_index']['mean']:.4f} ± "
          f"{summary['results']['ensembled']['c_index']['std']:.4f}   ({summary['elapsed_minutes']} min)")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="brca_combined", choices=list(COHORTS) + ["all"])
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--subspace_ratio", type=float, default=0.85)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    # transformer stream (paper Sec. 4.5 / released code)
    p.add_argument("--d_model", type=int, default=128, help="not stated in the paper; code default 512 does not fit 8 GB")
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--trans_dropout", type=float, default=0.1)
    p.add_argument("--trans_lr", type=float, default=1e-5)
    p.add_argument("--trans_batch", type=int, default=128)
    p.add_argument("--trans_micro", type=int, default=16, help="micro-batch for gradient accumulation (memory)")
    p.add_argument("--trans_epochs", type=int, default=100)
    p.add_argument("--trans_pretrain_epochs", type=int, default=5)
    p.add_argument("--trans_patience", type=int, default=50)
    p.add_argument("--trans_class_weight", default="released", choices=["released", "balanced"],
                   help="released = transformer stream sample weight as published (class share); "
                        "balanced = inverse class frequency (ablation, NOT the original transformer stream)")
    # fusion (this project)
    p.add_argument("--member_split", default="bootstrap", choices=["bootstrap", "split"],
                   help="bootstrap = v2 design (streams on a bootstrap sample, fusion on the same patients, "
                        "out-of-bag patients pick epochs); split = disjoint stream / fusion / selection sets "
                        "(--split_ratios), so the fusion only sees stream outputs of patients the streams did not train on")
    p.add_argument("--split_ratios", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                   help="stream / fusion / selection shares of the labeled training patients (--member_split split)")
    p.add_argument("--fusion_unlabeled", action="store_true",
                   help="also give the fusion the fold's unlabeled training patients (Cox survival loss only)")
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--fusion_lr", type=float, default=1e-3)
    p.add_argument("--fusion_epochs", type=int, default=300)
    p.add_argument("--fusion_patience", type=int, default=30)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="exact_streams")
    p.add_argument("--interpret", action="store_true",
                   help="also record transformer stream-style feature importance (reference transformer paper Sec. 2.5) on each test fold; "
                        "measured after training, does not change training or results")
    return p.parse_args(argv)


def main():
    a = parse_args()
    cohorts = list(COHORTS) if a.cohort == "all" else [a.cohort]
    # never overwrite a finished run: every experiment gets its own tag (results/versions/vN)
    done = [c for c in cohorts if os.path.exists(os.path.join("results", a.tag, f"{c}_5fold_results.json"))]
    if done:
        sys.exit(f"STOP: results/{a.tag} already has results for {done} - choose a new --tag (e.g. versions/vN)")
    for c in cohorts:
        run(c, a)


if __name__ == "__main__":
    main()
