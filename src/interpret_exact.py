"""
Feature importance for the exact-stream ensembled-streams model, using the reference transformer paper's method
(reference transformer paper Sec. 2.5: "we assigned each feature to 0 and calculated the model's F1 score
after ... The greater the difference, the more important the feature is").

Called from src/train_exact.py only when --interpret is given. For one trained bagging
member and its outer-fold TEST patients, every transformer stream input feature (cancer driver or
L1-SVC selected; min-max scaled to [0, 1]) is set to 0 for all test patients, one at a
time, and the change of these outputs is recorded:
  ensembled-streams model : subtype macro-F1 (transformer stream criterion), mean probability of the true class,
               survival C-index (this project's addition)
  transformer stream alone: subtype macro-F1 (as in the reference transformer paper)
Whole-omics rows set all transformer stream features of one omics to 0.

Only forward passes in eval mode: the trained models are not changed, no random numbers
are drawn, and nothing here is used for training, epoch choice or vote weights.
graph stream inputs are not perturbed (16-17k features per omics, each change would need a new
patient graph); the graph stream outputs of the member are held fixed.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.models.ensembled_streams import compute_c_index

VIEWS = ["mrna", "mirna", "meth"]


def _macro_f1(y, pred, n_cls):
    m = y >= 0
    if not m.any():
        return float("nan")
    return float(f1_score(y[m], pred[m], labels=list(range(n_cls)), average="macro", zero_division=0))


@torch.no_grad()
def member_importance(A, fus, S_te, te, y, ev, tm, n_cls, names, fold, member, device):
    """A = trained TransformerStream, fus = trained Fusion, S_te = stream outputs on the test rows."""
    m = A.model
    m.eval()
    fus.eval()
    xs, xd = A.inputs(torch.tensor(te, device=device))
    enc = [m.E[i](xs[i], xd[i]) for i in range(3)]
    emb = [m.C[i].embed(enc[i]) for i in range(3)]
    vlog = [m.C[i](enc[i]) for i in range(3)]
    gat = torch.tensor(S_te["gat"], dtype=torch.float32, device=device)
    gcn = torch.tensor(S_te["gcn"], dtype=torch.float32, device=device)
    yt = y[te]
    lab = yt >= 0

    def evaluate(emb_list, vlog_list):
        logits, risk = fus(torch.stack(emb_list, 1), gat, gcn)
        p = torch.softmax(logits, -1).cpu().numpy()
        p_true = float(p[lab, yt[lab]].mean()) if lab.any() else float("nan")
        p_trans = m.vcdn(vlog_list).argmax(1).cpu().numpy()
        return (_macro_f1(yt, p.argmax(1), n_cls), p_true,
                compute_c_index(risk.cpu().numpy(), ev[te], tm[te]), _macro_f1(yt, p_trans, n_cls))

    base = evaluate(emb, vlog)
    rows = []

    def record(v, kind, col, name, res):
        rows.append({"fold": fold, "member": member, "omics": v, "kind": kind, "feature_index": col,
                     "feature": name, "n_test": int(len(te)), "n_test_labeled": int(lab.sum()),
                     "base_ensembled_macro_f1": base[0], "base_ensembled_true_prob": base[1],
                     "base_ensembled_c_index": base[2], "base_trans_macro_f1": base[3],
                     "ensembled_macro_f1_drop": base[0] - res[0], "ensembled_true_prob_drop": base[1] - res[1],
                     "ensembled_c_index_drop": base[2] - res[2], "trans_macro_f1_drop": base[3] - res[3]})

    for i, v in enumerate(VIEWS):
        for kind, cols, X, is_svc in (("l1svc", A.cols_svc[v], xs[i], True), ("driver", A.cols_dg[v], xd[i], False)):
            for j, col in enumerate(cols):
                Xz = X.clone()
                Xz[:, j] = 0.0
                e = m.E[i](Xz, xd[i]) if is_svc else m.E[i](xs[i], Xz)
                e2, l2 = list(emb), list(vlog)
                e2[i], l2[i] = m.C[i].embed(e), m.C[i](e)
                record(v, kind, int(col), names[f"trans_{v}"][int(col)], evaluate(e2, l2))
        e = m.E[i](torch.zeros_like(xs[i]), torch.zeros_like(xd[i]))
        e2, l2 = list(emb), list(vlog)
        e2[i], l2[i] = m.C[i].embed(e), m.C[i](e)
        record(v, "whole_omics", -1, "ALL", evaluate(e2, l2))
    return rows
