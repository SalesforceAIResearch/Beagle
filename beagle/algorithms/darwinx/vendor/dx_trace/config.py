"""QC configuration — a profile is proposers + filters + mergers (blog config).

A config selects which proposers run, with what rubric/chunking, plus the
post-processing filters and mergers. Built-in profiles live in ``configs/*.yaml``
(``default`` = the general taxonomy; ``monet`` = the same taxonomy with rubrics
tuned to monet-on-coding-benchmarks). Pass a path to use your own — the taxonomy
is meant to be customized, which is the whole point of the proposer design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .chunks import ContextSpec
from .filters import MERGERS, RULE_FILTERS, Filter, LLMFilter, Merger
from .model import IssueCategory, Severity
from .proposers import RULE_PROPOSERS, LLMProposer, Proposer

_CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class QCConfig:
    name: str
    proposers: list[Proposer]
    filters: list[Filter]
    mergers: list[Merger]


def load_config(name_or_path: str = "default") -> QCConfig:
    path = _CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        path = Path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no such config {name_or_path!r}; built-ins: {', '.join(builtin_configs())}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return build_config(data, default_name=path.stem)


def builtin_configs() -> list[str]:
    return sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))


def build_config(data: dict[str, Any], *, default_name: str = "custom") -> QCConfig:
    proposers = [_build_proposer(p) for p in data.get("proposers", [])]
    filters = [_build_filter(f) for f in data.get("filters", [])]
    mergers = [_build_merger(m) for m in data.get("issue_mergers", data.get("mergers", []))]
    return QCConfig(data.get("name", default_name), proposers, filters, mergers)


def _build_proposer(p: dict[str, Any]) -> Proposer:
    ptype = p.get("type", "rule")
    if ptype == "rule":
        rule = p.get("rule") or p["name"]
        cls = RULE_PROPOSERS.get(rule)
        if cls is None:
            raise ValueError(f"unknown rule proposer {rule!r}; have {sorted(RULE_PROPOSERS)}")
        return cls(**(p.get("params") or {}))
    if ptype == "llm":
        return LLMProposer(
            name=p["name"],
            category=IssueCategory.parse(p["category"]),
            rubric=p["rubric"],
            chunk_size=int(p.get("chunk_size", 8)),
            chunk_format=p.get("chunk_format", "xml"),
            context=ContextSpec.from_dict(p.get("context")),
            severity=Severity.parse(p.get("severity", "medium")),
        )
    raise ValueError(f"unknown proposer type {ptype!r}")


def _build_filter(f: dict[str, Any]) -> Filter:
    ftype = f.get("type", "rule")
    if ftype == "rule":
        cls = RULE_FILTERS.get(f["name"])
        if cls is None:
            raise ValueError(f"unknown rule filter {f['name']!r}; have {sorted(RULE_FILTERS)}")
        return cls()
    if ftype == "llm":
        return LLMFilter(name=f["name"], prompt=f["prompt"])
    raise ValueError(f"unknown filter type {ftype!r}")


def _build_merger(m: dict[str, Any]) -> Merger:
    cls = MERGERS.get(m["name"])
    if cls is None:
        raise ValueError(f"unknown merger {m['name']!r}; have {sorted(MERGERS)}")
    return cls()
