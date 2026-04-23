# Phase 4: non-linear classifier head.
# Use the trained SCM as a feature extractor and concatenate classical DGA
# log-ratio features, then train GB/RF/LR on top. The linear-Gaussian SCM's
# native class posterior (Eq. 16) is linear in y, which caps accuracy at
# around 82% on this dataset. A non-linear classifier breaks that ceiling.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .config import PICLConfig
from .data import PICLDataset
from .graph import HybridCausalGraph
from .inference import causal_disablement_and_sufficiency
from .scm import LinearGaussianSCM


# All C(5,2) = 10 pairs of the five gases; covers every Rogers / Duval /
# Doernenburg ratio without hand-picking.
_GAS_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def _dga_ratios(log_ppm):
    # log(ppm_i / ppm_j) = log1p_ppm_i - log1p_ppm_j (close enough when ppm >> 1).
    cols = [(log_ppm[:, i] - log_ppm[:, j]).unsqueeze(1) for i, j in _GAS_PAIRS]
    return torch.cat(cols, dim=1)


@dataclass
class ClassifierHead:
    models: list           # sklearn estimator list (length > 1 in ensemble mode)
    n_classes: int


def extract_scm_features(ds: PICLDataset,
                         graph: HybridCausalGraph,
                         scm: LinearGaussianSCM) -> np.ndarray:
    # Feature layout:
    #   raw y              (5)
    #   E[f | y]           (6)       joint-Gaussian conditional mean, source 0
    #   y - mu_k           (6x5=30)  residual vectors for each class
    #   ||y - mu_k||       (6)
    #   Ed(y,k), Es(y,k)   (6+6=12)
    #   DGA ratios         (10)
    # Total 69.
    n_faults = graph.n_faults
    n_vars = graph.n_vars
    y = ds.gas_values
    N = y.shape[0]

    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        mu = scm.mu.detach()
        mu_y = mu[n_faults:]

        # Per-class do-intervention means mu_k = E[Y | do(f=e_k)].
        I_K = torch.eye(n_faults, dtype=W_eff.dtype)
        mus_by_class = scm.intervene_on_faults(W_eff, I_K, n_faults)    # [K, G]

        # E[f | y] from the joint-Gaussian conditional mean.
        sigma2 = scm.noise_variance(0).detach()
        I = torch.eye(n_vars, dtype=W_eff.dtype)
        T = torch.linalg.solve(I - W_eff, I)
        Sigma = T.T @ torch.diag(sigma2) @ T + 1e-6 * torch.eye(n_vars)
        S_fy = Sigma[:n_faults, n_faults:]
        S_yy = Sigma[n_faults:, n_faults:]
        rhs = (y - mu_y.unsqueeze(0)).T
        Efy = (S_fy @ torch.linalg.solve(S_yy, rhs)).T + mu[:n_faults].unsqueeze(0)

        # Residuals.
        res = y.unsqueeze(1) - mus_by_class.unsqueeze(0)    # [N, K, G]
        res_norm = torch.norm(res, dim=2)                    # [N, K]

        # Ed / Es need a probability vector; use a clipped + renormalised
        # Efy as a cheap proxy.
        post_proxy = Efy.clamp(min=0.0, max=1.0)
        post_proxy = post_proxy / post_proxy.sum(dim=1, keepdim=True).clamp(min=1e-8)
        Ed, Es = causal_disablement_and_sufficiency(
            y, post_proxy, W_eff, scm, n_faults)

        dga = _dga_ratios(ds.log_ppm) if ds.log_ppm is not None \
            else torch.zeros(N, len(_GAS_PAIRS), dtype=y.dtype)

    feats = torch.cat([y, Efy, res.reshape(N, -1), res_norm, Ed, Es, dga], dim=1)
    return feats.cpu().numpy().astype(np.float32)


def _build(cfg: PICLConfig) -> List[object]:
    # Build one or more classifiers based on config. Ensemble mode returns 3.
    hc = cfg.raw.get("classifier_head", {})
    name = hc.get("model", "gradient_boosting")
    n_est = int(hc.get("n_estimators", 300))
    max_depth = int(hc.get("max_depth", 4))
    lr = float(hc.get("learning_rate", 0.05))
    seed = int(cfg.raw["experiment"]["seed"])
    ensemble = bool(hc.get("ensemble", False))

    def one(name):
        if name == "gradient_boosting":
            # HistGB is ~10x faster than GradientBoosting at similar accuracy.
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=n_est, max_depth=max_depth if max_depth > 0 else None,
                learning_rate=lr, random_state=seed)
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_depth if max_depth > 0 else None,
                random_state=seed, n_jobs=-1)
        if name == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=5000, C=1.0, random_state=seed)
        raise ValueError(f"unknown classifier: {name}")

    if ensemble:
        return [one("gradient_boosting"), one("random_forest"), one("logistic")]
    return [one(name)]


def train_classifier_head(cfg: PICLConfig,
                          ds_train: PICLDataset,
                          graph: HybridCausalGraph,
                          scm: LinearGaussianSCM) -> ClassifierHead:
    # train_on_real_only=True skips synthetic samples: the linear SCM's
    # generated samples don't match the real non-linear class boundary, so
    # training on them tends to hurt rather than help.
    hc = cfg.raw.get("classifier_head", {})
    real_only = bool(hc.get("train_on_real_only", False))

    if real_only and ds_train.is_synthetic is not None:
        from .data import subset
        ds_use = subset(ds_train, ~ds_train.is_synthetic)
    else:
        ds_use = ds_train

    X = extract_scm_features(ds_use, graph, scm)
    y = ds_use.labels.cpu().numpy()

    models = _build(cfg)
    for clf in models:
        clf.fit(X, y)
    return ClassifierHead(models=models, n_classes=graph.n_faults)


def classifier_posterior(head: ClassifierHead,
                         ds: PICLDataset,
                         graph: HybridCausalGraph,
                         scm: LinearGaussianSCM) -> torch.Tensor:
    # Ensemble: average predict_proba outputs. sklearn's classes_ may not
    # cover all K classes (if some class is absent in training), so we
    # realign by class id.
    X = extract_scm_features(ds, graph, scm)
    N = X.shape[0]
    out = np.zeros((N, head.n_classes), dtype=np.float32)
    for clf in head.models:
        proba = clf.predict_proba(X)
        for col, c in enumerate(clf.classes_):
            out[:, int(c)] += proba[:, col]
    out /= len(head.models)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return torch.from_numpy(out / row_sum)
