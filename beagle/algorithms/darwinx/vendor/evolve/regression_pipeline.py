"""Regression resolver pipeline for self-evolve campaigns."""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import pipeline


@dataclass
class RegressionResolverPipelineConfig:
    base: pipeline.PipelineConfig
    target_node_id: str
    pipeline_id_override: str | None = None


class RegressionResolverPipeline(pipeline.SelfEvolvePipeline):
    """A child-node pipeline that starts from a regressed target node."""

    def __init__(self, cfg: RegressionResolverPipelineConfig) -> None:
        base = replace(
            cfg.base,
            parent_id_override=cfg.target_node_id,
            pipeline_id_override=cfg.pipeline_id_override,
            pipeline_kind="regression_resolve",
        )
        super().__init__(base)
        self.regression_cfg = cfg


__all__ = [
    "RegressionResolverPipeline",
    "RegressionResolverPipelineConfig",
]
