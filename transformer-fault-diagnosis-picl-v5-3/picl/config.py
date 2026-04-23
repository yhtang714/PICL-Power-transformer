# Config loading. Parses the two YAML files and bins every candidate edge
# into one of: hard / plausible / forbidden / unknown.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class EdgeSpec:
    src: str                # source variable name
    tgt: str                # target variable name
    kind: str               # hard / plausible / unknown / forbidden
    pi: float = 0.5         # prior mean of edge existence
    rationale: str = ""


@dataclass
class PICLConfig:
    raw: Dict
    prior_raw: Dict

    fault_names: List[str] = field(default_factory=list)
    gas_names: List[str] = field(default_factory=list)
    var_names: List[str] = field(default_factory=list)
    var_index: Dict[str, int] = field(default_factory=dict)

    hard_edges: List[EdgeSpec] = field(default_factory=list)
    plausible_edges: List[EdgeSpec] = field(default_factory=list)
    unknown_edges: List[EdgeSpec] = field(default_factory=list)
    forbidden_edges: List[EdgeSpec] = field(default_factory=list)

    @property
    def n_faults(self):
        return len(self.fault_names)

    @property
    def n_gases(self):
        return len(self.gas_names)

    @property
    def n_vars(self):
        return self.n_faults + self.n_gases

    @property
    def kappa(self):
        return float(self.raw["model"]["kappa"])


def _build_edges(cfg: PICLConfig):
    # Enumerate every directed pair (u, v), u != v, and bin by:
    #   hard      - explicit in prior_knowledge.yaml (physically required)
    #   plausible - physically reasonable, Beta prior with pi > 0.5
    #   forbidden - self-loops, gas->fault, optionally fault->fault / gas->gas
    #   unknown   - everything else, Beta(1,1) non-informative prior
    faults = set(cfg.fault_names)
    gases = set(cfg.gas_names)

    forbid_ff = bool(cfg.raw.get("model", {}).get("forbid_fault_to_fault", False))
    forbid_gg = bool(cfg.raw.get("model", {}).get("forbid_gas_to_gas", False))

    hard_pairs, plaus_pairs = set(), set()
    for row in cfg.prior_raw.get("hard_edges", []):
        e = EdgeSpec(row["src"], row["tgt"], "hard",
                     float(row.get("pi", 1.0)), row.get("rationale", ""))
        cfg.hard_edges.append(e)
        hard_pairs.add((e.src, e.tgt))

    for row in cfg.prior_raw.get("plausible_edges", []):
        e = EdgeSpec(row["src"], row["tgt"], "plausible",
                     float(row.get("pi", 0.5)), row.get("rationale", ""))
        cfg.plausible_edges.append(e)
        plaus_pairs.add((e.src, e.tgt))

    for u in cfg.var_names:
        for v in cfg.var_names:
            if u == v:
                cfg.forbidden_edges.append(EdgeSpec(u, v, "forbidden", 0.0, "self-loop"))
                continue
            if u in gases and v in faults:
                cfg.forbidden_edges.append(EdgeSpec(u, v, "forbidden", 0.0,
                                                    "gas -> fault not physical"))
                continue
            if forbid_ff and u in faults and v in faults:
                cfg.forbidden_edges.append(EdgeSpec(u, v, "forbidden", 0.0,
                                                    "fault -> fault disabled"))
                continue
            if forbid_gg and u in gases and v in gases:
                cfg.forbidden_edges.append(EdgeSpec(u, v, "forbidden", 0.0,
                                                    "gas -> gas disabled"))
                continue
            if (u, v) in hard_pairs or (u, v) in plaus_pairs:
                continue
            cfg.unknown_edges.append(EdgeSpec(u, v, "unknown", 0.5, "Beta(1,1) prior"))


def load_config(config_path: str = "config/config.yaml",
                prior_path: str = "config/prior_knowledge.yaml") -> PICLConfig:
    with open(Path(config_path), "r") as f:
        raw = yaml.safe_load(f)
    with open(Path(prior_path), "r") as f:
        prior_raw = yaml.safe_load(f)

    cfg = PICLConfig(raw=raw, prior_raw=prior_raw)
    cfg.fault_names = list(raw["data"]["fault_types"])
    cfg.gas_names = list(raw["data"]["gas_types"])
    cfg.var_names = cfg.fault_names + cfg.gas_names
    cfg.var_index = {name: i for i, name in enumerate(cfg.var_names)}

    _build_edges(cfg)
    return cfg
