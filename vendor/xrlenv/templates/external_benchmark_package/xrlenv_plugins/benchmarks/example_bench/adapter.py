"""EnvAdapter skeleton for the example_bench plug-in.

Replace the ``raise NotImplementedError`` calls with real harness
glue. The adapter runs INSIDE the sandbox — the in-sandbox stub
imports this module and instantiates :class:`ExampleBenchEnvAdapter`
on each ``env_setup`` request. See spec 14 for the full Protocol.
"""

from __future__ import annotations

from typing import Any


class ExampleBenchEnvAdapter:
    """Wraps your benchmark's harness as an xrlenv EnvAdapter."""

    def __init__(self, init_params: dict[str, Any]) -> None:
        self._init_params = init_params
        # ``init_params`` carries platform-supplied fields (timeouts,
        # task-id selectors, ...) merged with consumer-supplied
        # rollout-level kwargs. Pull the ones you need and stash on
        # self for the step/teardown lifecycle.

    def setup(self) -> dict[str, Any]:
        """Return the initial observation. Called once at the start
        of the rollout, before any step()."""
        raise NotImplementedError(
            "example_bench adapter is a skeleton; replace setup() with "
            "your harness's initialisation"
        )

    def step(self, action: Any) -> dict[str, Any]:
        """Apply ``action`` to the harness; return the standard
        ``{obs, reward, done, truncated, info}`` dict."""
        raise NotImplementedError(
            "example_bench adapter is a skeleton; replace step() with "
            "your harness's step routine"
        )

    def teardown(self) -> dict[str, Any]:
        """Clean up. Return ``{"status": "ok"}`` or surface any
        teardown diagnostics in ``info``."""
        raise NotImplementedError(
            "example_bench adapter is a skeleton; replace teardown() "
            "with your harness's cleanup"
        )
