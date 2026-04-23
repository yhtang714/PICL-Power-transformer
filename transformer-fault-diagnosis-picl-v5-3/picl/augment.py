# Module D: missing-value imputation + counterfactual augmentation.
# Imputation: conditional-Gaussian mean  mu_{M|O} = mu_M + Sigma_{MO} Sigma_{OO}^{-1} (v_O - mu_O)
# Augmentation: propagate do(f = alpha * e_k) through the SCM, add noise,
# filter out implausible samples via a posterior-ratio test.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import PICLConfig
from .data import PICLDataset, concat_datasets
from .graph import HybridCausalGraph
from .inference import class_posterior_bma
from .scm import LinearGaussianSCM


def impute_training_set(ds, graph, scm,
                        log_mu: Optional[torch.Tensor] = None,
                        log_sd: Optional[torch.Tensor] = None) -> PICLDataset:
    # Conditional-mean imputation under the current graph + weights. Returns
    # a new dataset (original unchanged). If log_mu/log_sd are given, also
    # refresh ds.log_ppm at missing positions by inverting the z-score - that
    # way the classifier head's DGA-ratio features don't see a spurious
    # log1p(0)=0 at originally-missing cells.
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        imputed = scm.impute_conditional_mean(ds.data, ds.source, ds.miss_mask, W_eff)

    n_faults = graph.n_faults
    new_log_ppm = None
    if ds.log_ppm is not None:
        new_log_ppm = ds.log_ppm.clone()
        if log_mu is not None and log_sd is not None:
            miss_gas = ds.miss_mask[:, n_faults:]
            if miss_gas.any():
                imputed_gas = imputed[:, n_faults:]
                log_mu_t = log_mu.to(imputed_gas.dtype)
                log_sd_t = log_sd.to(imputed_gas.dtype)
                recovered = imputed_gas * log_sd_t.unsqueeze(0) + log_mu_t.unsqueeze(0)
                new_log_ppm[miss_gas] = recovered[miss_gas]

    return PICLDataset(
        data=imputed,
        gas_values=imputed[:, n_faults:].clone(),
        labels=ds.labels.clone(),
        source=ds.source.clone(),
        miss_mask=torch.zeros_like(ds.miss_mask),   # all observed after imputation
        is_synthetic=ds.is_synthetic.clone(),
        log_ppm=new_log_ppm,
    )


def _target_class_counts(labels, target_total, n_classes):
    # Distribute target_total across classes by empirical frequency. Any
    # rounding slack goes to the largest class.
    counts = torch.bincount(labels, minlength=n_classes).float()
    target = ((counts / counts.sum()) * target_total).round().to(torch.long)
    diff = target_total - int(target.sum().item())
    if diff != 0:
        target[int(target.argmax().item())] += diff
    return target.tolist()


def counterfactual_augment(cfg: PICLConfig,
                           ds: PICLDataset,
                           graph: HybridCausalGraph,
                           scm: LinearGaussianSCM,
                           target_total: Optional[int] = None,
                           rng_seed: int = 0) -> Tuple[PICLDataset, List[int]]:

    aug_cfg = cfg.raw["augmentation"]
    if target_total is None:
        target_total = int(aug_cfg["target_size"])
    intervention_levels = list(aug_cfg["intervention_levels"])
    r_filter = float(aug_cfg["plausibility_ratio"])
    max_ratio = float(aug_cfg["max_synthetic_ratio"])

    # Only apply simplex projection (clip + renormalise) in proportion mode.
    # In log1p_z mode the feature space is R^G and any clipping would corrupt it.
    feature_mode = str(cfg.raw["data"].get("gas_feature_mode", "proportion"))
    enforce_simplex = (feature_mode == "proportion")

    n_faults = graph.n_faults
    n_gases = graph.n_gases
    n_vars = graph.n_vars
    n_bma = int(cfg.raw["inference"]["n_graph_samples"])
    rng = torch.Generator().manual_seed(rng_seed)

    with torch.no_grad():
        A_map = graph.final_hard_adjacency()
        W_eff = (A_map * graph.weight_matrix()).detach()

    target_counts = _target_class_counts(ds.labels, target_total, n_faults)
    real_counts = torch.bincount(ds.labels, minlength=n_faults).tolist()
    needed = [max(0, target_counts[k] - real_counts[k]) for k in range(n_faults)]
    cap = [int(max_ratio * real_counts[k]) for k in range(n_faults)]
    needed = [min(needed[k], cap[k]) for k in range(n_faults)]

    synth_rows, synth_labels, synth_sources = [], [], []
    accepted_count = [0] * n_faults

    for k in range(n_faults):
        n_need = needed[k]
        if n_need <= 0:
            continue

        class_mask = (ds.labels == k)
        if class_mask.sum() == 0:
            continue
        source_pool = ds.source[class_mask].cpu().numpy()

        accepted = 0
        attempts = 0
        max_attempts = max(20 * n_need, 500)
        batch_gen = 256

        while accepted < n_need and attempts < max_attempts:
            this_batch = min(batch_gen, (n_need - accepted) * 4)
            attempts += this_batch

            # Sample one activation level alpha per candidate.
            alpha_idx = torch.randint(len(intervention_levels),
                                      (this_batch,), generator=rng)
            alpha = torch.tensor([intervention_levels[int(i)] for i in alpha_idx],
                                 dtype=ds.data.dtype)
            f_int = torch.zeros(this_batch, n_faults, dtype=ds.data.dtype)
            f_int[:, k] = alpha

            with torch.no_grad():
                mu_gas = scm.intervene_on_faults(W_eff, f_int, n_faults)

                # Sample sources from the in-class distribution.
                src_sel = np.random.default_rng(
                    rng_seed + k * 1000 + attempts).integers(
                        0, len(source_pool), size=this_batch)
                src_idx = torch.from_numpy(source_pool[src_sel].astype(np.int64))

                sigma_y = torch.stack(
                    [scm.noise_variance(int(s.item()))[n_faults:] for s in src_idx],
                    dim=0)
                noise = torch.randn(this_batch, n_gases, generator=rng) * torch.sqrt(sigma_y)
                y_cand = mu_gas + noise

                if enforce_simplex:
                    y_cand = y_cand.clamp(min=0.0)
                    row_sum = y_cand.sum(dim=1, keepdim=True)
                    row_sum = torch.where(row_sum <= 0, torch.ones_like(row_sum), row_sum)
                    y_cand = y_cand / row_sum

                # Plausibility filter: P(k | y_cand) >= r * max_{j!=k} P(j | y_cand)
                post = class_posterior_bma(y_cand, graph, scm,
                                           n_samples=n_bma, source_idx=0)
                target_p = post[:, k]
                others = post.clone()
                others[:, k] = -1.0
                keep = target_p >= r_filter * others.max(dim=1).values
                accept_idx = torch.where(keep)[0]

            slots = n_need - accepted
            if len(accept_idx) > slots:
                accept_idx = accept_idx[:slots]
            if len(accept_idx) == 0:
                continue

            # Build the SCM row: [fault one-hot | gas]
            y_keep = y_cand[accept_idx]
            onehot = torch.zeros(len(accept_idx), n_faults, dtype=ds.data.dtype)
            onehot[:, k] = 1.0

            synth_rows.append(torch.cat([onehot, y_keep], dim=1))
            synth_labels.append(torch.full((len(accept_idx),), k, dtype=torch.long))
            synth_sources.append(src_idx[accept_idx])

            accepted += len(accept_idx)
            accepted_count[k] = accepted

    if not synth_rows:
        return ds, accepted_count

    all_row = torch.cat(synth_rows, dim=0)
    all_lab = torch.cat(synth_labels, dim=0)
    all_src = torch.cat(synth_sources, dim=0)
    synth_gas = all_row[:, n_faults:].clone()

    # Synthetic samples have no raw ppm - reuse gas_values as log_ppm so the
    # concat keeps working. In log1p_z mode both live in the same space; in
    # proportion mode the DGA-ratio features will be uninformative for these
    # samples and the classifier head learns to ignore them.
    synth_ds = PICLDataset(
        data=all_row,
        gas_values=synth_gas,
        labels=all_lab,
        source=all_src,
        miss_mask=torch.zeros(all_row.shape[0], n_vars, dtype=torch.bool),
        is_synthetic=torch.ones(all_row.shape[0], dtype=torch.bool),
        log_ppm=synth_gas.clone(),
    )

    ds_marked = PICLDataset(
        data=ds.data, gas_values=ds.gas_values, labels=ds.labels,
        source=ds.source, miss_mask=ds.miss_mask,
        is_synthetic=torch.zeros(ds.data.shape[0], dtype=torch.bool),
        log_ppm=ds.log_ppm)

    return concat_datasets(ds_marked, synth_ds), accepted_count
