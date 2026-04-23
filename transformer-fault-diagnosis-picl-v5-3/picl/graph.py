# Hybrid causal graph: fixed hard edges + learnable discoverable edges
# (plausible / unknown, per the IEC 60599 prior knowledge file).
#
# Each discoverable edge carries a Beta(a, b) posterior, sampled via Binary
# Concrete so gradients flow through. Hard edges use softplus(theta) to
# enforce positivity.

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PICLConfig, EdgeSpec


def _softplus_inv(y):
    # Numerically stable inverse of softplus.
    return y + torch.log1p(-torch.exp(-y))


class HybridCausalGraph(nn.Module):

    def __init__(self, cfg: PICLConfig):
        super().__init__()
        self.cfg = cfg
        self.n_vars = cfg.n_vars
        self.n_faults = cfg.n_faults
        self.n_gases = cfg.n_gases

        # Three edge groups: hard (always on), discoverable (= plausible + unknown,
        # subject to learning), forbidden (kept at zero).
        self.hard_edges: List[EdgeSpec] = list(cfg.hard_edges)
        self.discoverable_edges: List[EdgeSpec] = list(cfg.plausible_edges) + list(cfg.unknown_edges)

        # Pre-compute (src, tgt) index tensors for scatter operations.
        if self.hard_edges:
            hi = [[cfg.var_index[e.src], cfg.var_index[e.tgt]] for e in self.hard_edges]
            self.hard_idx = torch.tensor(hi, dtype=torch.long)
        else:
            self.hard_idx = torch.zeros(0, 2, dtype=torch.long)

        if self.discoverable_edges:
            di = [[cfg.var_index[e.src], cfg.var_index[e.tgt]] for e in self.discoverable_edges]
            self.disc_idx = torch.tensor(di, dtype=torch.long)
        else:
            self.disc_idx = torch.zeros(0, 2, dtype=torch.long)

        self.register_buffer("_hard_idx_buf", self.hard_idx, persistent=False)
        self.register_buffer("_disc_idx_buf", self.disc_idx, persistent=False)

        # Beta priors: plausible edges -> Beta(pi*kappa, (1-pi)*kappa);
        # unknown edges -> Beta(1, 1).
        kappa = cfg.kappa
        pa, pb = [], []
        for e in self.discoverable_edges:
            if e.kind == "plausible":
                pa.append(e.pi * kappa)
                pb.append((1.0 - e.pi) * kappa)
            else:
                pa.append(1.0)
                pb.append(1.0)
        prior_a = torch.tensor(pa, dtype=torch.float32)
        prior_b = torch.tensor(pb, dtype=torch.float32)
        self.register_buffer("prior_a", prior_a, persistent=False)
        self.register_buffer("prior_b", prior_b, persistent=False)

        # Hard-edge weights initialised near 1.0 via softplus.
        if len(self.hard_edges) > 0:
            init_w = torch.full((len(self.hard_edges),), 1.0)
            self.theta_hard = nn.Parameter(_softplus_inv(init_w))
        else:
            self.theta_hard = nn.Parameter(torch.zeros(0))

        # Discoverable weights are unconstrained (negative is allowed, e.g. for
        # a suppression effect). Beta posterior starts at the prior.
        n_disc = len(self.discoverable_edges)
        if n_disc > 0:
            self.w_disc = nn.Parameter(torch.randn(n_disc) * 0.1)
            self.a_raw = nn.Parameter(_softplus_inv(prior_a.clone()))
            self.b_raw = nn.Parameter(_softplus_inv(prior_b.clone()))
        else:
            self.w_disc = nn.Parameter(torch.zeros(0))
            self.a_raw = nn.Parameter(torch.zeros(0))
            self.b_raw = nn.Parameter(torch.zeros(0))

    # --- basic quantities ---

    def hard_weights(self):
        return F.softplus(self.theta_hard)

    def beta_ab(self):
        a = F.softplus(self.a_raw) + 1e-3
        b = F.softplus(self.b_raw) + 1e-3
        return a, b

    def edge_prob_mean(self):
        a, b = self.beta_ab()
        return a / (a + b)

    def edge_prob_var(self):
        a, b = self.beta_ab()
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    # --- adjacency / weight matrices ---

    def _scatter(self, values, idx):
        M = torch.zeros(self.n_vars, self.n_vars,
                        dtype=values.dtype, device=values.device)
        if idx.numel() > 0:
            M[idx[:, 0], idx[:, 1]] = values
        return M

    def weight_matrix(self):
        W = self._scatter(self.hard_weights(), self._hard_idx_buf)
        if self.w_disc.numel() > 0:
            W = W + self._scatter(self.w_disc, self._disc_idx_buf)
        return W

    def sample_adjacency(self, tau: float, hard: bool = False):
        # Hard edges always 1. Discoverable edges sampled via Binary Concrete.
        # hard=True returns the MAP 0/1 mask (pi > 0.5), used for G_hat.
        pi = self.edge_prob_mean().clamp(1e-6, 1 - 1e-6)
        if hard:
            disc_vals = (pi > 0.5).float()
        else:
            u = torch.empty_like(pi).uniform_(1e-6, 1 - 1e-6)
            logistic = torch.log(u) - torch.log1p(-u)
            logit_pi = torch.log(pi) - torch.log1p(-pi)
            disc_vals = torch.sigmoid((logit_pi + logistic) / max(tau, 1e-6))

        A = self._scatter(torch.ones_like(self.theta_hard), self._hard_idx_buf)
        if disc_vals.numel() > 0:
            A = A + self._scatter(disc_vals, self._disc_idx_buf)
        return A

    def expected_adjacency(self):
        # E[A], fed to the NOTEARS acyclicity penalty.
        A = self._scatter(torch.ones_like(self.theta_hard), self._hard_idx_buf)
        if self.w_disc.numel() > 0:
            A = A + self._scatter(self.edge_prob_mean(), self._disc_idx_buf)
        return A

    def effective_weights(self, A):
        return A * self.weight_matrix()

    def final_hard_adjacency(self):
        with torch.no_grad():
            return self.sample_adjacency(tau=1.0, hard=True)

    # --- KL ---

    def kl_divergence(self):
        # Closed-form KL(Beta(a,b) || Beta(a0,b0)).
        if self.w_disc.numel() == 0:
            return torch.zeros((), device=self.w_disc.device)
        a, b = self.beta_ab()
        a0, b0 = self.prior_a, self.prior_b
        lb_prior = torch.lgamma(a0) + torch.lgamma(b0) - torch.lgamma(a0 + b0)
        lb_post = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        kl = (lb_prior - lb_post
              + (a - a0) * torch.digamma(a)
              + (b - b0) * torch.digamma(b)
              + (a0 + b0 - a - b) * torch.digamma(a + b))
        return kl.sum()

    # --- reporting ---

    def hard_edge_summary(self) -> List[Dict]:
        ws = self.hard_weights().detach().cpu().tolist()
        return [{"src": e.src, "tgt": e.tgt, "weight": ws[i], "kind": "hard"}
                for i, e in enumerate(self.hard_edges)]

    def edge_posterior_summary(self) -> List[Dict]:
        # Posterior stats per discoverable edge.
        # kept = large enough weight AND real structural support (pi clearly
        # above 0.5 OR posterior variance noticeably tighter than the prior).
        # Weight alone is not enough - Phase 2b's parameter-only pass can push
        # weights around without moving pi, so we must require both.
        pis = self.edge_prob_mean().detach().cpu().tolist()
        vars_ = self.edge_prob_var().detach().cpu().tolist()
        weights = self.w_disc.detach().cpu().tolist() if self.w_disc.numel() > 0 else []
        a0 = self.prior_a.detach().cpu().tolist()
        b0 = self.prior_b.detach().cpu().tolist()

        rows = []
        for i, e in enumerate(self.discoverable_edges):
            s = a0[i] + b0[i]
            prior_var = (a0[i] * b0[i]) / (s * s * (s + 1.0))
            prior_pi = e.pi if e.kind == "plausible" else 0.5

            has_weight = abs(weights[i]) > 0.10
            has_structure = (pis[i] > 0.55) or (vars_[i] < 0.7 * prior_var)

            if e.kind == "plausible":
                # Plausible prior is already above 0.5 - just require the
                # posterior not to crash below it and a non-trivial weight.
                kept = (pis[i] > 0.5) and (abs(weights[i]) > 0.05)
            else:
                kept = has_weight and has_structure

            rows.append({
                "src": e.src,
                "tgt": e.tgt,
                "kind": e.kind,
                "prior_pi": prior_pi,
                "prior_var": prior_var,
                "post_pi": pis[i],
                "post_var": vars_[i],
                "weight": weights[i],
                "kept": kept,
            })
        return rows

    def all_edges_ranked(self) -> List[Dict]:
        # Hard + discoverable, sorted by |weight|. Handy for dumping a
        # single ranked table of the full causal graph.
        rows = []
        for e in self.hard_edge_summary():
            rows.append({**e, "post_pi": 1.0, "post_var": 0.0, "prior_pi": 1.0, "kept": True})
        rows.extend(self.edge_posterior_summary())
        rows.sort(key=lambda r: -abs(r["weight"]))
        return rows
