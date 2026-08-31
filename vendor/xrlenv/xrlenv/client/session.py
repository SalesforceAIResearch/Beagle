"""RolloutSession — async context manager that wraps one rollout (spec 05)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import TracebackType
from typing import TYPE_CHECKING, Any

from xrlenv.types import Action, Observation, StepResult, Trajectory

if TYPE_CHECKING:
    from xrlenv.client.transport import ClientTransport


# Consumer-side reward function shape (spec 02 RewardContract consumer_final).
RewardFn = Callable[[Trajectory], Awaitable[float]]


class RolloutSession:
    """The consumer-facing handle to one in-progress rollout.

    Construction goes through :py:meth:`xrlenv.client.Client.rollout`; do not
    instantiate directly. The async context manager guarantees the sandbox is
    released (finished or cancelled) on exit, regardless of exception path.
    """

    def __init__(
        self,
        *,
        transport: ClientTransport,
        rollout_id: str,
        initial_obs: Observation,
        template: str,
        reward_mode: str = "env_step",
        reward_fn: RewardFn | None = None,
    ) -> None:
        self._transport = transport
        self._rollout_id = rollout_id
        self._template = template
        self._reward_mode = reward_mode
        self._reward_fn = reward_fn
        self._observation: Observation = initial_obs
        self._done: bool = False
        self._truncated: bool = False
        self._reward_sum: float = 0.0
        self._steps: int = 0
        self._info: dict[str, Any] = {}
        self._trajectory: Trajectory | None = None
        self._closed: bool = False

    # ── Public surface ───────────────────────────────────────────────────────

    @property
    def rollout_id(self) -> str:
        return self._rollout_id

    @property
    def template(self) -> str:
        return self._template

    @property
    def observation(self) -> Observation:
        return self._observation

    @property
    def done(self) -> bool:
        return self._done

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def reward_sum(self) -> float:
        return self._reward_sum

    @property
    def steps_taken(self) -> int:
        return self._steps

    @property
    def trajectory(self) -> Trajectory:
        if self._trajectory is None:
            raise RuntimeError(
                f"trajectory for rollout {self._rollout_id} is not sealed yet — "
                "exit the context manager (or await session.finish()) first"
            )
        return self._trajectory

    async def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError(
                f"rollout {self._rollout_id} is already done; cannot step"
            )
        result = await self._transport.step(self._rollout_id, action)
        self._observation = result.obs
        self._reward_sum += result.reward
        self._steps += 1
        self._done = result.done or result.truncated
        self._truncated = result.truncated
        self._info = result.info
        return result

    async def heartbeat(self) -> None:
        """Keep the rollout alive without sending a step (spec 02).

        Each :py:meth:`step` resets the idle-TTL clock implicitly; this
        explicit call is for consumers whose ``policy.act(obs)`` runs longer
        than the configured ``idle_ttl_s`` between steps. The default is
        120 s; use a background task::

            async def keepalive(s):
                while not s.done:
                    await asyncio.sleep(30)
                    await s.heartbeat()
        """
        await self._transport.heartbeat(self._rollout_id)

    async def finish(self) -> Trajectory:
        if self._closed:
            assert self._trajectory is not None
            return self._trajectory
        self._closed = True
        self._trajectory = await self._transport.finish(self._rollout_id)
        await self._maybe_apply_consumer_final_reward()
        return self._trajectory

    async def _maybe_apply_consumer_final_reward(self) -> None:
        """Spec 02 RewardContract consumer_final: call the consumer-supplied
        reward_fn(trajectory) and back-fill the result into the sealed
        trajectory + the canonical record on the control plane.
        """
        if self._reward_mode != "consumer_final" or self._reward_fn is None:
            return
        if self._trajectory is None:
            return
        final_reward = await self._reward_fn(self._trajectory)
        # Pydantic models are mutable by default unless frozen; Trajectory
        # is not frozen, so this in-place update is safe and matches the
        # spec language ("SDK back-fills Trajectory.final_reward").
        self._trajectory.final_reward = final_reward
        await self._transport.set_final_reward(self._rollout_id, final_reward)

    async def cancel(self, reason: str = "consumer_cancelled") -> Trajectory:
        if self._closed:
            assert self._trajectory is not None
            return self._trajectory
        self._closed = True
        self._trajectory = await self._transport.cancel(self._rollout_id, reason)
        return self._trajectory

    # ── Async context manager ────────────────────────────────────────────────

    async def __aenter__(self) -> RolloutSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        if exc is not None:
            # Propagate the user's exception; cancel best-effort.
            # Carry a one-line ``<type>: <message>`` summary into the
            # cancel reason so the admin page / replay caller can see
            # the cause without spelunking coordinator.log. ``reason``
            # is a free-form string in the proto/state-store; consumers
            # that branch on the categorical prefix should match
            # ``startswith("aborted_with_exception")`` rather than
            # equality.
            summary = f"aborted_with_exception: {type(exc).__name__}: {exc}"
            summary = " ".join(summary.split())
            if len(summary) > 500:
                summary = summary[:497] + "..."
            with suppress(Exception):
                await self.cancel(summary)
            return
        await self.finish()
