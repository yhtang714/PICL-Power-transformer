#!/usr/bin/env python3
"""PICL training entry point.

Example
-------
    python train.py --config config/config.yaml

Everything else (data paths, hyperparameters, phase schedule) is
controlled from the YAML file.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from picl.config import load_config
from picl.data import load_picl_datasets
from picl.trainer import train_picl


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, mode="w")],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--prior", default="config/prior_knowledge.yaml")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    output_dir = Path(args.output)
    setup_logging(output_dir / "logs" / "training.log")
    log = logging.getLogger("picl.main")

    cfg = load_config(args.config, args.prior)
    set_seed(int(cfg.raw["experiment"]["seed"]))

    log.info("Loading datasets from %s", cfg.raw["data"]["csv_path"])
    train, cal, test = load_picl_datasets(cfg)
    log.info("train=%d  cal=%d  test=%d   missing in train=%d",
             train.data.shape[0], cal.data.shape[0], test.data.shape[0],
             int(train.miss_mask.sum().item()))

    bundle, summary = train_picl(cfg, train, cal, test, output_dir)

    # Concise summary printed to stdout.
    tr = summary["report"]["test"]
    print("\n" + "=" * 62)
    print("FINAL TEST RESULTS")
    print("-" * 62)
    print(f"  Overall accuracy     : {tr['accuracy_all']:7.4f}")
    print(f"  Accepted accuracy    : {tr['accuracy_accepted']:7.4f}")
    print(f"  Coverage             : {tr['coverage']:7.4f}")
    print(f"  ECE                  : {tr['ece']:7.4f}")
    print(f"  N accepted / total   : {tr['n_accepted']} / {tr['n_samples']}")
    print("=" * 62)


if __name__ == "__main__":
    main()
