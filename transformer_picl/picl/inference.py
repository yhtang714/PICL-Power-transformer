from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import PICLConfig
from .data import PICLDataset
from .graph import HybridCausalGraph
from .scm import LinearGaussianSCM


_LOG_2PI = math.log(2.0 * math.pi)


# --- class posterior ---

def _class_means(W_eff, n_faults):
    # mu_k = E[Y | do(f = e_k)] without the intercept. Shape [K, G].
    n_vars = W_eff.shape[0]
    W_int = W_eff.clone()
    W_int[:, :n_faults] = 0.0
    I = torch.eye(n_vars, dtype=W_eff.dtype, device=W_eff.device)
    T = torch.linalg.solve(I - W_int, I)
    E_f = torch.eye(n_faults, n_vars, dtype=W_eff.dtype, device=W_eff.device)
    return (E_f @ T)[:, n_faults:]


def class_posterior_single_graph(y, W_eff, sigma2, mu, n_faults):
    # Single-graph posterior. Derivation:
    #   log p(f=e_k, y) = -½ r_k^T Omega r_k + const,   r_k = [e_k - mu_f; y - mu_y]
    # Only two terms depend on k:
    #   (a) (e_k - mu_f)^T Omega_ff (e_k - mu_f)
    #   (b) (e_k - mu_f)^T Omega_fy (y - mu_y)
    # The y^T Omega_yy y term is k-independent and cancels in the softmax.
    n_vars = W_eff.shape[0]
    I = torch.eye(n_vars, dtype=W_eff.dtype, device=W_eff.device)
    T = torch.linalg.solve(I - W_eff, I)
    Sigma = T.T @ torch.diag(sigma2) @ T + 1e-6 * torch.eye(n_vars)
    Omega = torch.linalg.inv(Sigma)

    Omega_ff = Omega[:n_faults, :n_faults]
    Omega_fy = Omega[:n_faults, n_faults:]
    mu_f = mu[:n_faults]
    mu_y = mu[n_faults:]

    # centred_f[k] = e_k - mu_f, built via I - mu_f broadcast.
    I_K = torch.eye(n_faults, dtype=W_eff.dtype, device=W_eff.device)
    centred_f = I_K - mu_f.unsqueeze(0)                   # [K, K]
    centred_y = y - mu_y.unsqueeze(0)                     # [B, G]

    quad_f = (centred_f @ Omega_ff * centred_f).sum(dim=1)   # [K]
    M = centred_f @ Omega_fy                               # [K, G]
    cross = centred_y @ M.T                                # [B, K]

    log_joint = -0.5 * quad_f.unsqueeze(0) - cross
    return log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)


def class_posterior_bma(y, graph, scm, n_samples, source_idx=0, tau=0.05):
    # BMA: draw S graphs from q(Gamma), compute posterior per graph, average
    # the probabilities (not the log-probs).
    n_faults = graph.n_faults
    sigma2 = scm.noise_variance(source_idx).detach()
    mu = scm.mu.detach()

    log_posts = []
    with torch.no_grad():
        for _ in range(n_samples):
            A = graph.sample_adjacency(tau=tau, hard=False)
            W_eff = A * graph.weight_matrix()
            log_posts.append(class_posterior_single_graph(
                y, W_eff.detach(), sigma2, mu, n_faults))
    stacked = torch.stack(log_posts, dim=0)
    return stacked.exp().mean(dim=0).clamp(min=1e-12)


# --- Ed / Es ---

def causal_disablement_and_sufficiency(y, posteriors, W_eff, scm, n_faults):
    # Ed(f_k, y) = || y - E[Y | do(f_k=0, others=post)] || / ||y||
    # Es(f_k, y) = 1 - || y - E[Y | do(f_k=post, others=0)] || / ||y||
    # Interpretation:
    #   high Ed -> disabling f_k moves the prediction far from y, so f_k is
    #              actually active (disablement)
    #   high Es -> keeping only f_k already reproduces y, so f_k is a
    #              sufficient cause (sufficiency)
    B, G = y.shape
    K = n_faults
    y_norm = torch.norm(y, dim=1, keepdim=True).clamp(min=1e-8)

    # Build the do-intervention vectors. Shape [B, K, K].
    f_ed = posteriors.unsqueeze(1).expand(B, K, K).clone()
    for k in range(K):
        f_ed[:, k, k] = 0.0
    f_es = torch.zeros(B, K, K, dtype=posteriors.dtype, device=posteriors.device)
    for k in range(K):
        f_es[:, k, k] = posteriors[:, k]

    mu_ed = scm.intervene_on_faults(W_eff, f_ed.reshape(B * K, K), n_faults).reshape(B, K, G)
    mu_es = scm.intervene_on_faults(W_eff, f_es.reshape(B * K, K), n_faults).reshape(B, K, G)

    Ed = torch.norm(y.unsqueeze(1) - mu_ed, dim=2) / y_norm
    Es = 1.0 - torch.norm(y.unsqueeze(1) - mu_es, dim=2) / y_norm

    Es = Es.clamp(0.0, 1.0)
    Ed = Ed.clamp(0.0, 10.0)
    return Ed, Es


# --- composite score ---

@dataclass
class CompositeScores:
    posteriors: torch.Tensor
    Ed: torch.Tensor
    Es: torch.Tensor
    scores: torch.Tensor
    pred: torch.Tensor
    conf: torch.Tensor


def composite_scores(cfg, y, graph, scm, n_graph_samples,
                     sources=None, classifier_posterior=None):
    # Full scoring pipeline. If classifier_posterior is supplied, it replaces
    # the native SCM posterior (Phase 4 behaviour). Ed/Es always come from
    # the SCM.
    w = cfg.raw["inference"]["gate_weights"]
    wp, wd, ws = float(w["posterior"]), float(w["disablement"]), float(w["sufficiency"])

    B = y.shape[0]
    if sources is None:
        sources = torch.zeros(B, dtype=torch.long, device=y.device)

    with torch.no_grad():
        if classifier_posterior is None:
            post_all = torch.zeros(B, graph.n_faults, dtype=y.dtype, device=y.device)
            for s in torch.unique(sources):
                sel = (sources == s)
                post_all[sel] = class_posterior_bma(
                    y[sel], graph, scm, n_graph_samples, source_idx=int(s.item()))
        else:
            post_all = classifier_posterior.to(y.device)

        # Ed / Es use the MAP adjacency and are source-independent (they are
        # geometric quantities that don't depend on sigma^2).
        A_map = graph.final_hard_adjacency()
        W_eff = (A_map * graph.weight_matrix()).detach()
        Ed, Es = causal_disablement_and_sufficiency(
            y, post_all, W_eff, scm, graph.n_faults)

        scores = wp * post_all + wd * Ed + ws * Es
        conf, pred = scores.max(dim=1)
    return CompositeScores(post_all, Ed, Es, scores, pred, conf)


# --- temperature calibration ---

class TemperatureCalibrator:
    # Single scalar T fit by LBFGS on BCE loss of the 'MAP prediction correct?'
    # binary signal - i.e. we calibrate the quantity that the gate actually
    # thresholds.

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, scores, labels):
        max_s, pred = scores.max(dim=1)
        correct = (pred == labels).float()
        max_s = max_s.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(max_s) - torch.log1p(-max_s)

        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            T = torch.exp(logT)
            loss = bce(logits / T, correct)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            self.temperature = float(torch.exp(logT.detach()).item())
        except Exception:
            self.temperature = 1.0

    def transform(self, scores):
        T = self.temperature
        x = scores.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(x) - torch.log1p(-x)
        return torch.sigmoid(logits / T)


# --- gate threshold search ---

def optimise_threshold(cfg, conf, pred, labels):
    # Two modes:
    #   max_cov_acc - maximise coverage * accuracy (paper default)
    #   target_cov  - pick the gamma whose coverage is closest to target;
    #                 on ties, prefer the higher accuracy.
    inf = cfg.raw["inference"]
    g_lo = float(inf["threshold_grid_min"])
    g_hi = float(inf["threshold_grid_max"])
    n_grid = int(inf["threshold_grid_steps"])
    mode = str(inf.get("threshold_mode", "max_cov_acc"))
    target_cov = float(inf.get("target_coverage", 0.879))

    records = []
    for g in torch.linspace(g_lo, g_hi, n_grid):
        accepted = conf >= g
        cov = accepted.float().mean().item()
        acc = (pred[accepted] == labels[accepted]).float().mean().item() if accepted.any() else 0.0
        records.append((float(g), cov, acc))

    if mode == "target_cov":
        return min(records, key=lambda r: (abs(r[1] - target_cov), -r[2]))[0]

    best_g, best_m = g_lo, 0.0
    for g, cov, acc in records:
        if cov * acc > best_m:
            best_m = cov * acc
            best_g = g
    return best_g


# --- ECE ---

def expected_calibration_error(conf, correct, n_bins=10):
    conf_np = conf.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy().astype(float)
    total = len(conf_np)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = (conf_np > lo) & (conf_np <= hi)
        if in_bin.sum() == 0:
            continue
        acc = correct_np[in_bin].mean()
        avg = conf_np[in_bin].mean()
        ece += (in_bin.sum() / total) * abs(acc - avg)
    return float(ece)


# --- end-to-end evaluation ---

@dataclass
class EvalReport:
    accuracy_all: float
    accuracy_accepted: float
    coverage: float
    ece: float
    per_class: Dict[int, Dict[str, float]]
    n_samples: int
    n_accepted: int


def full_evaluation(cfg, ds, graph, scm, calibrator, threshold,
                    clf_posterior: Optional[torch.Tensor] = None) -> EvalReport:
    n_samples = int(cfg.raw["inference"]["n_graph_samples"])
    y = ds.gas_values
    labels = ds.labels

    with torch.no_grad():
        cs = composite_scores(cfg, y, graph, scm, n_samples,
                              sources=ds.source,
                              classifier_posterior=clf_posterior)
        calibrated_conf = calibrator.transform(cs.scores).max(dim=1).values

        # target_cov mode thresholds on the raw composite score (matching
        # the threshold search in trainer.py).
        mode = str(cfg.raw["inference"].get("threshold_mode", "max_cov_acc"))
        accept_conf = cs.scores.max(dim=1).values if mode == "target_cov" else calibrated_conf
        accepted = accept_conf >= threshold

        correct = (cs.pred == labels)
        acc_all = correct.float().mean().item()
        cov = accepted.float().mean().item()
        acc_accepted = correct[accepted].float().mean().item() if accepted.any() else 0.0
        ece = expected_calibration_error(calibrated_conf, correct)

        per_class = {}
        for k in torch.unique(labels).tolist():
            sel = (labels == k)
            per_class[int(k)] = {
                "accuracy": float(correct[sel].float().mean().item()) if sel.any() else 0.0,
                "coverage": float(accepted[sel].float().mean().item()) if sel.any() else 0.0,
                "accuracy_accepted": float(correct[sel & accepted].float().mean().item())
                                     if (sel & accepted).any() else 0.0,
                "n": int(sel.sum().item()),
            }

    return EvalReport(
        accuracy_all=acc_all,
        accuracy_accepted=acc_accepted,
        coverage=cov,
        ece=ece,
        per_class=per_class,
        n_samples=int(ds.data.shape[0]),
        n_accepted=int(accepted.sum().item()),
    )
