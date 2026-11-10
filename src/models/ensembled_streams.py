"""
Ensembled Transformer-Graph Streams Multi-Task Model (final design from the project history, turn 108)

Stream A - gene-to-gene learner (transformer stream idea)
    Inputs: driver features + L1-SVC selected non-driver features (3 omics).
    Upgrade 1 "multi-token": each omics is projected to several learned tokens
    (mRNA 4, miRNA 2, methylation 4 = 10 tokens) with learnable token embeddings,
    then a Transformer encoder layer (self-attention + feed-forward) learns
    cross-omics interactions. `multi_token=False` gives the earlier 3-token model.
    (The tokens are learned projections; they are not tied to named pathways.)

Stream B - patient-to-patient learner (graph stream idea)
    Inputs: high-variance features (drivers and non-drivers) of the 3 omics.
    Upgrade 2: each patient is linked to its nearest TRAINING patients with
    the graph method's scaled exponential affinity (src/graph.py); one graph-convolution
    step L = D^-1 (A + I) aggregates neighbour embeddings (graph stream Eq. 5-6).

Fusion - bilateral cross-attention
    Stream A summary attends over Stream B tokens [self, neighbourhood] and the
    Stream B summary attends over the 10 Stream A tokens.

Heads - subtype classifier + Cox survival risk.
Losses - class-balanced focal loss (Cui et al. 2019) + alpha * Cox partial likelihood.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOKENS = {"mrna": 4, "mirna": 2, "meth": 4}


class ModalityEncoder(nn.Module):
    def __init__(self, in_features: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        hid = max(out_dim, 128) * 2
        self.net = nn.Sequential(
            nn.Linear(in_features, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.net(x)


class StreamAGeneNetwork(nn.Module):
    def __init__(self, in_dims: dict, hidden: int = 128, nhead: int = 4, dropout: float = 0.3,
                 multi_token: bool = True):
        super().__init__()
        self.n_tok = {m: (TOKENS[m] if multi_token else 1) for m in ["mrna", "mirna", "meth"]}
        self.hidden = hidden
        self.enc = nn.ModuleDict({m: ModalityEncoder(in_dims[m], hidden * self.n_tok[m], dropout)
                                  for m in self.n_tok})
        T = sum(self.n_tok.values())
        self.tok_emb = nn.Parameter(torch.zeros(1, T, hidden))
        nn.init.normal_(self.tok_emb, std=0.02)
        self.attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.Linear(hidden * T, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Dropout(dropout))

    def forward(self, x):
        toks = [self.enc[m](x[m]).view(x[m].size(0), self.n_tok[m], self.hidden) for m in self.n_tok]
        t = torch.cat(toks, dim=1) + self.tok_emb
        a, w = self.attn(t, t, t, need_weights=True, average_attn_weights=True)
        t = self.norm1(t + a)
        t = self.norm2(t + self.ffn(t))
        z = self.out(t.reshape(t.size(0), -1))
        return z, t, w


class StreamBPatientGraph(nn.Module):
    def __init__(self, in_dims: dict, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        dims = {"mrna": hidden // 2, "mirna": hidden // 4, "meth": hidden // 2}
        self.enc = nn.ModuleDict({m: ModalityEncoder(in_dims[m], d, dropout) for m, d in dims.items()})
        self.proj = nn.Sequential(nn.Linear(sum(dims.values()), hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gcn = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def embed(self, x):
        return self.proj(torch.cat([self.enc[m](x[m]) for m in self.enc], dim=-1))

    def forward(self, x_self, x_nbr, w_nbr):
        """x_self: {m: [B, d]}, x_nbr: {m: [B, k, d]}, w_nbr: [B, k] raw affinities."""
        B, k = w_nbr.shape
        h_self = self.embed(x_self)
        h_nbr = self.embed({m: v.reshape(B * k, -1) for m, v in x_nbr.items()}).view(B, k, -1)
        # L = D^-1 (A + I): affinities scaled to [0, 1] per row, self-loop weight 1
        w = w_nbr / w_nbr.max(dim=1, keepdim=True).values.clamp(min=1e-12)
        w_all = torch.cat([torch.ones(B, 1, device=w.device), w], dim=1)
        w_all = w_all / w_all.sum(1, keepdim=True)
        h_all = torch.cat([h_self.unsqueeze(1), h_nbr], dim=1)
        agg = (w_all.unsqueeze(-1) * h_all).sum(1)
        g = F.elu(self.gcn(agg))
        z = self.norm(h_self + self.drop(g))
        tokens = torch.stack([h_self, agg], dim=1)          # [B, 2, H]
        return z, tokens


class BilateralCrossAttentionFusion(nn.Module):
    def __init__(self, hidden: int = 128, nhead: int = 4, dropout: float = 0.3):
        super().__init__()
        self.a2b = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.b2a = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.na, self.nb = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden * 4, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden),
                                 nn.GELU())

    def forward(self, z_A, tok_A, z_B, tok_B):
        a, _ = self.a2b(z_A.unsqueeze(1), tok_B, tok_B)      # gene view asks the patient graph
        b, w_ba = self.b2a(z_B.unsqueeze(1), tok_A, tok_A)   # patient view asks the gene tokens
        a = self.na(z_A + a.squeeze(1))
        b = self.nb(z_B + b.squeeze(1))
        return self.mlp(torch.cat([z_A, z_B, a, b], dim=-1)), w_ba


class EnsembledStreamsMultiTaskModel(nn.Module):
    def __init__(self, in_dims_A, in_dims_B, num_classes, hidden=128, dropout=0.3, multi_token=True):
        super().__init__()
        self.stream_A = StreamAGeneNetwork(in_dims_A, hidden, 4, dropout, multi_token)
        self.stream_B = StreamBPatientGraph(in_dims_B, hidden, dropout)
        self.fusion = BilateralCrossAttentionFusion(hidden, 4, dropout)

        def head(o):
            return nn.Sequential(nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden // 2, o))
        self.classifier = head(num_classes)
        self.survival_head = head(1)

    def forward(self, xA, xB_self, xB_nbr, w_nbr):
        z_A, tok_A, attn_A = self.stream_A(xA)
        z_B, tok_B = self.stream_B(xB_self, xB_nbr, w_nbr)
        z, attn_BA = self.fusion(z_A, tok_A, z_B, tok_B)
        return {"logits": self.classifier(z), "risk": self.survival_head(z).squeeze(-1),
                "attn_A": attn_A, "attn_BA": attn_BA}


# ---------------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------------
class ClassBalancedFocalLoss(nn.Module):
    """Cui et al. (CVPR 2019): w_c = (1 - beta) / (1 - beta^n_c), focal term (1 - p_t)^gamma.
    Labels < 0 (unlabeled patients) are ignored."""

    def __init__(self, class_counts, beta: float = 0.999, gamma: float = 1.5):
        super().__init__()
        counts = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
        w = w / w.sum() * len(counts)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, y):
        mask = y >= 0
        if mask.sum() == 0:
            return logits.sum() * 0.0
        logits, y = logits[mask], y[mask]
        logp = F.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
        p = logp.exp()
        loss = -self.w[y] * (1 - p) ** self.gamma * logp
        return loss.sum() / self.w[y].sum()


class MaskedCrossEntropy(nn.Module):
    def forward(self, logits, y):
        if (y >= 0).sum() == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(logits, y, ignore_index=-1)


def cox_ph_loss(risk, events, times):
    """Negative Cox partial log-likelihood (Breslow, sorted by descending time)."""
    if events.sum() == 0:
        return risk.sum() * 0.0
    order = torch.argsort(times, descending=True)
    r, e = risk[order], events[order].float()
    log_cum = torch.logcumsumexp(r, dim=0)
    return -((r - log_cum) * e).sum() / e.sum()


def compute_c_index(risk, events, times) -> float:
    """Harrell's C-index (higher risk should mean shorter survival); ties in risk count 0.5."""
    risk, events, times = map(np.asarray, (risk, events, times))
    ev = events == 1
    if ev.sum() == 0:
        return float("nan")
    ti, ri = times[ev][:, None], risk[ev][:, None]
    comparable = times[None, :] > ti
    conc = ((ri > risk[None, :]) & comparable).sum() + 0.5 * ((ri == risk[None, :]) & comparable).sum()
    n = comparable.sum()
    return float(conc / n) if n > 0 else float("nan")
