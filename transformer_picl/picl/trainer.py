# Orchestrates the full training pipeline: 4 phases + final evaluation.
#   Phase 1   joint graph + SCM param learning on original data
#   Phase 2a  impute missing values + counterfactual augmentation
#   Phase 2b  refit structure (real samples only), then refit params (all samples)
#   Phase 4   (optional) non-linear classifier head
#   Phase 3   temperature calibration + gate + evaluation

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

from .augment import counterfactual_augment, impute_training_set
from .config import PICLConfig
from .data import PICLDataset, get_log_stats
from .graph import HybridCausalGraph
from .inference import (EvalReport, TemperatureCalibrator, composite_scores,
                        full_evaluation, optimise_threshold)
from .learn import learn_joint, learn_parameters_only
from .scm import LinearGaussianSCM


log = logging.getLogger("picl.trainer")


@dataclass
class PICLBundle:
    cfg: PICLConfig
    graph: HybridCausalGraph
    scm: LinearGaussianSCM
    calibrator: TemperatureCalibrator
    threshold: float


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _log_causal_graph(graph: HybridCausalGraph, title: str):
    # Dump the current causal graph: hard edges (fixed) + kept discoverable
    # edges + a few weak-signal ones for reference.
    log.info("-" * 70)
    log.info("%s", title)
    log.info("-" * 70)

    hard = graph.hard_edge_summary()
    log.info("Hard edges (from IEC 60599):")
    for e in hard:
        log.info("  %-4s -> %-5s   w = %+.4f", e["src"], e["tgt"], e["weight"])

    disc = graph.edge_posterior_summary()
    kept = [r for r in disc if r["kept"]]
    kept.sort(key=lambda r: -abs(r["weight"]))
    if kept:
        log.info("Learned discoverable edges:")
        for r in kept:
            log.info("  %-4s -> %-5s   w = %+.4f   pi = %.3f   var = %.4f   [%s]",
                     r["src"], r["tgt"], r["weight"], r["post_pi"],
                     r["post_var"], r["kind"])
    else:
        log.info("Learned discoverable edges: (none)")

    weak = [r for r in disc if (not r["kept"]) and abs(r["weight"]) > 0.10]
    weak.sort(key=lambda r: -abs(r["weight"]))
    if weak:
        log.info("Weak-signal edges (|w|>0.10, below keep threshold, for reference):")
        for r in weak[:5]:
            log.info("  %-4s -> %-5s   w = %+.4f   pi = %.3f",
                     r["src"], r["tgt"], r["weight"], r["post_pi"])


def train_picl(cfg: PICLConfig,
               train: PICLDataset,
               cal: PICLDataset,
               test: PICLDataset,
               output_dir: Path) -> Tuple[PICLBundle, Dict]:

    n_sources = int(cfg.raw["data"]["n_sources"])
    n_vars = cfg.n_vars

    # Warm-start mu from training means. Zero-mean assumption would bias
    # the class posterior badly since one-hot fault columns are far from zero.
    graph = HybridCausalGraph(cfg)
    scm = LinearGaussianSCM(n_vars=n_vars, n_sources=n_sources,
                            init_log_var=float(cfg.raw["model"]["noise_log_var_init"]))
    scm.initialize_mu_from_data(train.data, train.miss_mask)
    log.info("SCM intercept mu initialised from training means")

    # --- Phase 1 ---
    log.info("=" * 70)
    log.info("PHASE 1  joint structure + weight learning on original data")
    log.info("=" * 70)
    phase1_cfg = cfg.raw["training"]["phase1"]
    res1 = learn_joint(cfg, graph, scm, train,
                       epochs=int(phase1_cfg["epochs"]),
                       lr=float(phase1_cfg["lr"]),
                       exclude_synthetic=False,
                       anneal_tau=True,
                       phase_name="Phase 1")
    log.info("  final loss = %.4f", res1.final_loss)
    _log_causal_graph(graph, "Causal graph after Phase 1")

    # --- Phase 2a: imputation + augmentation ---
    log.info("=" * 70)
    log.info("PHASE 2a  imputation + counterfactual augmentation")
    log.info("=" * 70)
    log_mu, log_sd = get_log_stats()
    train_imputed = impute_training_set(train, graph, scm, log_mu, log_sd)
    n_miss = int(train.miss_mask.sum().item())
    log.info("  imputed %d missing entries using conditional-Gaussian mean", n_miss)

    augmented, synth_counts = counterfactual_augment(
        cfg, train_imputed, graph, scm,
        rng_seed=int(cfg.raw["experiment"]["seed"]))
    n_real = int((~augmented.is_synthetic).sum().item())
    n_synth = int(augmented.is_synthetic.sum().item())
    log.info("  augmented set: %d real + %d synthetic = %d total",
             n_real, n_synth, n_real + n_synth)
    log.info("  synthetic samples per class: %s", synth_counts)

    # --- Phase 2b-1: structure (real samples only) ---
    log.info("=" * 70)
    log.info("PHASE 2b  structure refinement (real samples only)")
    log.info("=" * 70)
    phase2s_cfg = cfg.raw["training"]["phase2_structure"]
    res2s = learn_joint(cfg, graph, scm, augmented,
                        epochs=int(phase2s_cfg["epochs"]),
                        lr=float(phase2s_cfg["lr"]),
                        exclude_synthetic=True,
                        anneal_tau=False,
                        phase_name="Phase 2b structure")
    log.info("  final loss = %.4f", res2s.final_loss)

    # --- Phase 2b-2: params (all samples) ---
    log.info("=" * 70)
    log.info("PHASE 2b  parameter re-estimation (all samples)")
    log.info("=" * 70)
    phase2p_cfg = cfg.raw["training"]["phase2_params"]
    res2p = learn_parameters_only(cfg, graph, scm, augmented,
                                  epochs=int(phase2p_cfg["epochs"]),
                                  lr=float(phase2p_cfg["lr"]),
                                  phase_name="Phase 2b params")
    log.info("  final loss = %.4f", res2p.final_loss)
    _log_causal_graph(graph, "Causal graph after Phase 2b")

    # --- Impute cal/test too (classifier head and Ed/Es both need full data) ---
    cal_imputed = impute_training_set(cal, graph, scm, log_mu, log_sd)
    test_imputed = impute_training_set(test, graph, scm, log_mu, log_sd)

    # --- Phase 4: non-linear classifier head (optional) ---
    head = None
    clf_post_cal = clf_post_test = clf_post_train = None
    if cfg.raw.get("classifier_head", {}).get("enabled", False):
        log.info("=" * 70)
        log.info("PHASE 4  non-linear classifier head on SCM + DGA-ratio features")
        log.info("=" * 70)
        from .classifier_head import (train_classifier_head,
                                      classifier_posterior)
        import time
        t0 = time.time()
        head = train_classifier_head(cfg, augmented, graph, scm)
        clf_post_cal = classifier_posterior(head, cal_imputed, graph, scm)
        clf_post_test = classifier_posterior(head, test_imputed, graph, scm)
        clf_post_train = classifier_posterior(head, train_imputed, graph, scm)
        model_name = cfg.raw["classifier_head"].get("model", "gradient_boosting")
        ensemble = cfg.raw["classifier_head"].get("ensemble", False)
        tag = f"{model_name}+ensemble" if ensemble else model_name
        log.info("  trained classifier head (%s) in %.1fs", tag, time.time() - t0)

    # --- Phase 3: calibration + gate + evaluation ---
    log.info("=" * 70)
    log.info("PHASE 3  temperature calibration + gate + evaluation")
    log.info("=" * 70)
    n_samp = int(cfg.raw["inference"]["n_graph_samples"])

    with torch.no_grad():
        cal_scores = composite_scores(cfg, cal_imputed.gas_values, graph, scm,
                                      n_graph_samples=n_samp,
                                      sources=cal_imputed.source,
                                      classifier_posterior=clf_post_cal)
    calibrator = TemperatureCalibrator()
    if cfg.raw["inference"]["temperature_scaling"]:
        calibrator.fit(cal_scores.scores, cal_imputed.labels)
    log.info("  temperature T = %.4f", calibrator.temperature)

    # In target_cov mode use raw composite scores for threshold selection
    # (HistGB probabilities are near-one-hot, LBFGS collapses T to ~0 which
    # flattens calibrated scores to bimodal 0/1 and breaks the gate search).
    mode = str(cfg.raw["inference"].get("threshold_mode", "max_cov_acc"))
    if mode == "target_cov":
        cal_conf = cal_scores.scores.max(dim=1).values
        cal_pred = cal_scores.scores.argmax(dim=1)
    else:
        calibrated_cal = calibrator.transform(cal_scores.scores)
        cal_conf = calibrated_cal.max(dim=1).values
        cal_pred = calibrated_cal.argmax(dim=1)
    threshold = optimise_threshold(cfg, cal_conf, cal_pred, cal_imputed.labels)
    log.info("  gamma* = %.4f (mode=%s, target_cov=%.3f)",
             threshold, mode,
             cfg.raw["inference"].get("target_coverage", 0.879))

    test_report = full_evaluation(cfg, test_imputed, graph, scm, calibrator,
                                  threshold, clf_posterior=clf_post_test)
    cal_report = full_evaluation(cfg, cal_imputed, graph, scm, calibrator,
                                 threshold, clf_posterior=clf_post_cal)
    train_report = full_evaluation(cfg, train_imputed, graph, scm,
                                   calibrator, threshold,
                                   clf_posterior=clf_post_train)

    log.info("  TEST  accuracy (all)      = %.4f", test_report.accuracy_all)
    log.info("  TEST  accuracy (accepted) = %.4f", test_report.accuracy_accepted)
    log.info("  TEST  coverage            = %.4f", test_report.coverage)
    log.info("  TEST  ECE                 = %.4f", test_report.ece)

    # --- persist ---
    bundle = PICLBundle(cfg=cfg, graph=graph, scm=scm,
                        calibrator=calibrator, threshold=threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "graph_state": graph.state_dict(),
        "scm_state": scm.state_dict(),
        "temperature": calibrator.temperature,
        "threshold": threshold,
    }, output_dir / "models" / "picl_final_model.pt")

    def _report_dict(rep):
        return {"accuracy_all": rep.accuracy_all,
                "accuracy_accepted": rep.accuracy_accepted,
                "coverage": rep.coverage,
                "ece": rep.ece,
                "n_samples": rep.n_samples,
                "n_accepted": rep.n_accepted,
                "per_class": rep.per_class}

    final_disc = graph.edge_posterior_summary()
    final_kept = [r for r in final_disc if r["kept"]]

    results_summary = {
        "phase1": {"final_loss": res1.final_loss,
                   "kept_edges": res1.kept_edges},
        "phase2_structure": {"final_loss": res2s.final_loss,
                             "kept_edges": res2s.kept_edges},
        "phase2_params": {"final_loss": res2p.final_loss},
        "augmentation": {
            "n_real": n_real, "n_synth": n_synth,
            "synthetic_per_class": synth_counts,
        },
        "calibration": {
            "temperature": calibrator.temperature,
            "threshold": threshold,
        },
        "hard_edges": graph.hard_edge_summary(),
        "learned_edges": final_kept,
        "all_discoverable_edges": final_disc,
        "report": {
            "train": _report_dict(train_report),
            "cal": _report_dict(cal_report),
            "test": _report_dict(test_report),
        },
    }
    _save_json(output_dir / "results_summary.json", results_summary)
    log.info("Saved model to %s", output_dir / "models" / "picl_final_model.pt")
    log.info("Saved summary to %s", output_dir / "results_summary.json")

    return bundle, results_summary
