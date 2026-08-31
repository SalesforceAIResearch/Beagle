"""EnvAdapter protocol and the SyncEnvAdapter base (spec 14).

Adapters live *inside* the sandbox alongside the stub. They wrap a benchmark
Environment class (terminal-bench's ``Terminal``, OSWorld's ``DesktopEnv``,
SWE-bench harnesses, or a user-defined class) and expose a uniform
``setup`` / ``step`` / ``teardown`` surface. The stub dynamically imports an
adapter at ``/env/setup`` time and drives it for the rest of the rollout.

``Action`` and ``Observation`` are deliberately opaque — XRLEnv core never
inspects them, only routes the bytes.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

from xrlenv.types import Action, Observation, StepResult

# Reward modes the platform recognizes (spec 02 RewardContract).
REWARD_MODES: frozenset[str] = frozenset(
    {"env_step", "in_sandbox_final", "consumer_final", "external_final", "token_level"}
)


class AdapterCapabilities(BaseModel):
    """Adapter introspection consumed by the stub at setup time (spec 14)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    xrlenv_api_version_supported: tuple[str, ...]
    supported_reward_modes: frozenset[str]
    supports_resume: bool = False
    supports_enumerate_tasks: bool = False
    supports_binary_refs: bool = False
    supports_soft_deadline_signal: bool = False


class StepTimeout(Exception):
    """Raised when an env call exceeds its per-phase timeout (spec 14)."""


class EnvAdapter(ABC):
    """The protocol every in-sandbox adapter implements.

    Concrete adapters declare ``supported_reward_modes`` so the catalog can
    validate the template's ``reward.mode`` against the adapter at register
    time (spec 00 invariant 10).
    """

    supported_reward_modes: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    async def setup(self, init_params: dict[str, Any]) -> Observation:
        """One-time init at sandbox creation; returns the first observation."""

    @abstractmethod
    async def step(self, action: Action) -> StepResult:
        """Apply ``action`` and return ``(obs, reward, done, info, truncated)``."""

    @abstractmethod
    async def teardown(self) -> None:
        """Best-effort cleanup before sandbox destroy."""

    async def time_left(self, soft_deadline_s: float) -> None:
        """Optional: signal the policy that the soft deadline has passed.

        Default no-op so adapters that don't care about soft deadlines (most
        of phase 0) don't have to implement an empty stub.
        """
        return None

    @classmethod
    def capabilities(cls) -> AdapterCapabilities:
        """Override to advertise capabilities; the default is conservative."""
        return AdapterCapabilities(
            xrlenv_api_version_supported=("0.0",),
            supported_reward_modes=cls.supported_reward_modes,
        )


# ──────────────────────────────────────────────────────────────────────────────
# SyncEnvAdapter — the recommended base for wrapping sync / thread-affine envs.
# Pins env calls to a single worker thread per sandbox and bridges async ↔ sync
# via asyncio.to_thread. Spec 14 §"Wrapping sync / non-thread-safe upstream
# envs".
# ──────────────────────────────────────────────────────────────────────────────


class _PendingExecutor(BaseModel):
    """Holds the per-sandbox single-thread executor + the wrapped env."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    executor: ThreadPoolExecutor
    env: Any = None


class SyncEnvAdapter(EnvAdapter):
    """Pin all underlying env calls to a single worker thread per sandbox."""

    def __init__(self, *, sandbox_id: str) -> None:
        self._state = _PendingExecutor(
            executor=ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"env-{sandbox_id}"
            ),
        )

    async def _call(
        self, fn: Any, *args: Any, timeout_s: float | None, **kwargs: Any
    ) -> Any:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._state.executor, lambda: fn(*args, **kwargs)),
                timeout=timeout_s,
            )
        except TimeoutError as exc:
            raise StepTimeout(f"env call exceeded {timeout_s}s") from exc

    @abstractmethod
    def _do_setup(self, init_params: dict[str, Any]) -> Observation:
        """Called on the worker thread; concrete subclasses construct ``self._state.env``."""

    @abstractmethod
    def _do_step(self, action: Action) -> StepResult:
        """Called on the worker thread."""

    def _do_teardown(self) -> None:
        env = self._state.env
        if env is not None and hasattr(env, "close"):
            env.close()

    async def setup(self, init_params: dict[str, Any]) -> Observation:
        timeout_s = float(init_params.get("setup_timeout_s") or 60.0)
        return await self._call(self._do_setup, init_params, timeout_s=timeout_s)

    async def step(self, action: Action) -> StepResult:
        # The stub annotates the action with a ``__step_timeout_s`` envelope
        # before delivering it; honor that if present, else fall back to a
        # generous default.
        timeout_s: float = 30.0
        if isinstance(action, dict):
            ts = action.get("__step_timeout_s")
            if isinstance(ts, (int, float)):
                timeout_s = float(ts)
        return cast(
            StepResult,
            await self._call(self._do_step, action, timeout_s=timeout_s),
        )

    async def teardown(self) -> None:
        try:
            await self._call(self._do_teardown, timeout_s=30.0)
        finally:
            self._state.executor.shutdown(wait=True)


__all__ = [
    "REWARD_MODES",
    "AdapterCapabilities",
    "EnvAdapter",
    "StepTimeout",
    "SyncEnvAdapter",
]
