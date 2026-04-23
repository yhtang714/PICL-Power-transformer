# ELBO training loop. Objective:
#   L = E[log p(X | A, W, sigma)] - KL(q||p0) - lam_h * h(E[A]) - lam_ce * CE(post, y)
# The last term is a discriminative auxiliary loss that pushes W and sigma
# toward configurations that separate classes. Without it the pure generative
# ELBO tends to pool classes with overlapping gas signatures.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import PICLConfig
from .data import PICLDataset
from .graph import HybridCausalGraph
from .scm import LinearGaussianSCM


def _anneal_tau(epoch, max_epochs, tau0, tau_f):
    # Exponential anneal tau_t = tau0 * (tau_f / tau0)^(t / T).
    progress = min(1.0, max(0.0, epoch / max(1, max_epochs)))
    return tau0 * (tau_f / tau0) ** progress


def _acyclicity(E_A):
    # NOTEARS trace formulation h(A) = tr(exp(A*A)) - d, applied to E[A].
    d = E_A.shape[0]
    return torch.trace(torch.matrix_exp(E_A * E_A)) - d


def _cb_weights(labels, n_classes):
    # w_i = N / (K * count_class(i)). Sums to N so the loss scale is unchanged.
    counts = torch.bincount(labels, minlength=n_classes).float().clamp(min=1.0)
    N = labels.shape[0]
    return N / (float(n_classes) * counts[labels])


def _ce_aux_loss(ds, graph, scm, W_eff, labels, sample_w):
    # Discriminative CE. Uses the same joint-Gaussian formula as inference for
    # P(f=e_k | y) so gradients propagate through W_eff, scm.mu, scm.log_sigma2.
    from .inference import class_posterior_single_graph
    n_faults = graph.n_faults
    mu = scm.mu
    total = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    total_w = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)

    for s in torch.unique(ds.source):
        sel = (ds.source == s)
        if sample_w[sel].sum().item() == 0:
            continue
        sigma2 = scm.noise_variance(int(s.item()))
        log_post = class_posterior_single_graph(
            ds.gas_values[sel], W_eff, sigma2, mu, n_faults)
        nll = -log_post.gather(1, labels[sel].unsqueeze(1)).squeeze(1)
        total = total + (nll * sample_w[sel]).sum()
        total_w = total_w + sample_w[sel].sum()

    if total_w.item() == 0:
        return torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    return total / total_w


@dataclass
class LearnResult:
    final_loss: float
    loss_history: List[float]
    acyclicity_history: List[float]
    kept_edges: List[Dict]


def learn_joint(cfg, graph, scm, ds, epochs, lr,
                exclude_synthetic, anneal_tau, phase_name=""):
    params = list(graph.parameters()) + list(scm.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    tau0 = float(cfg.raw["model"]["temperature_init"])
    tau_f = float(cfg.raw["model"]["temperature_final"])
    lam_h = float(cfg.raw["model"]["acyclicity_weight"])

    # Sample weights: class-balance + synthetic exclusion for Phase 2b structure.
    cb = bool(cfg.raw["model"].get("class_balanced_loss", True))
    cb_w = _cb_weights(ds.labels, graph.n_faults).to(ds.data.device) if cb \
        else torch.ones(ds.data.shape[0], device=ds.data.device)
    sample_w = cb_w.clone()
    if exclude_synthetic:
        sample_w = sample_w * (~ds.is_synthetic).float()

    # Gaussian weight-prior scales per edge type.
    s_hard = float(cfg.raw["model"]["weight_prior_scale_hard"])
    s_plaus = float(cfg.raw["model"]["weight_prior_scale_plausible"])
    s_unk = float(cfg.raw["model"]["weight_prior_scale_unknown"])
    w_prior_scale = torch.ones(len(graph.discoverable_edges),
                               device=ds.data.device)
    for i, e in enumerate(graph.discoverable_edges):
        w_prior_scale[i] = s_plaus if e.kind == "plausible" else s_unk

    n_eff = float(sample_w.sum().item())
    if n_eff == 0:
        raise ValueError(f"{phase_name}: no samples selected")

    beta_kl = float(cfg.raw["model"].get("kl_weight", 1.0))
    w_reg_w = float(cfg.raw["model"].get("weight_reg_weight", 1.0))
    lam_ce = float(cfg.raw["model"].get("discriminative_weight", 0.0))

    history, acy_history = [], []
    grad_clip = float(cfg.raw["training"]["grad_clip"])

    for epoch in range(epochs):
        tau = _anneal_tau(epoch, epochs, tau0, tau_f) if anneal_tau else tau_f

        opt.zero_grad()
        A = graph.sample_adjacency(tau=tau, hard=False)
        W = graph.weight_matrix()
        W_eff = A * W

        ll = scm.log_likelihood(ds.data, ds.source, ds.miss_mask, W_eff,
                                sample_weights=sample_w)
        kl = graph.kl_divergence()
        h = _acyclicity(graph.expected_adjacency())

        w_reg = 0.5 * (graph.w_disc ** 2 / (w_prior_scale ** 2)).sum()
        noise_reg = 0.01 * (scm.log_sigma2 ** 2).sum()
        hard_reg = 0.5 * (graph.theta_hard ** 2).sum() / (s_hard ** 2 + 1e-8)

        if lam_ce > 0:
            ce = _ce_aux_loss(ds, graph, scm, W_eff, ds.labels, sample_w)
        else:
            ce = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)

        loss = (-ll / n_eff
                + beta_kl * kl / n_eff
                + lam_h * h
                + w_reg_w * (w_reg + noise_reg + hard_reg) / n_eff
                + lam_ce * ce)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        history.append(float(loss.item()))
        acy_history.append(float(h.item()))

    kept = [r for r in graph.edge_posterior_summary() if r["kept"]]
    return LearnResult(
        final_loss=history[-1] if history else float("nan"),
        loss_history=history,
        acyclicity_history=acy_history,
        kept_edges=kept,
    )


def learn_parameters_only(cfg, graph, scm, ds, epochs, lr, phase_name=""):
    # Freeze the q(Gamma) Beta parameters, keep the weights (theta_hard,
    # w_disc) and SCM sigma^2 trainable. Freezing theta_hard or w_disc too
    # would reduce this phase to noise-variance tuning only.
    graph.a_raw.requires_grad_(False)
    graph.b_raw.requires_grad_(False)
    try:
        trainable = [p for p in graph.parameters() if p.requires_grad] + \
                    list(scm.parameters())
        opt = torch.optim.Adam(trainable, lr=lr)

        # Hold the MAP 0/1 adjacency mask fixed and rebuild W_eff each epoch.
        A_fixed = graph.final_hard_adjacency().detach()

        s_plaus = float(cfg.raw["model"]["weight_prior_scale_plausible"])
        s_unk = float(cfg.raw["model"]["weight_prior_scale_unknown"])
        s_hard = float(cfg.raw["model"]["weight_prior_scale_hard"])
        w_prior_scale = torch.ones(len(graph.discoverable_edges),
                                   device=ds.data.device)
        for i, e in enumerate(graph.discoverable_edges):
            w_prior_scale[i] = s_plaus if e.kind == "plausible" else s_unk
        grad_clip = float(cfg.raw["training"]["grad_clip"])

        n_eff = ds.data.shape[0]
        history = []
        for epoch in range(epochs):
            opt.zero_grad()
            W_eff = A_fixed * graph.weight_matrix()
            ll = scm.log_likelihood(ds.data, ds.source, ds.miss_mask, W_eff)
            noise_reg = 0.01 * (scm.log_sigma2 ** 2).sum()
            w_reg = 0.5 * (graph.w_disc ** 2 / (w_prior_scale ** 2)).sum()
            hard_reg = 0.5 * (graph.theta_hard ** 2).sum() / (s_hard ** 2 + 1e-8)

            loss = -(ll / n_eff) + (noise_reg + w_reg + hard_reg) / n_eff
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            history.append(float(loss.item()))

        return LearnResult(
            final_loss=history[-1] if history else float("nan"),
            loss_history=history, acyclicity_history=[],
            kept_edges=[r for r in graph.edge_posterior_summary() if r["kept"]],
        )
    finally:
        graph.a_raw.requires_grad_(True)
        graph.b_raw.requires_grad_(True)
