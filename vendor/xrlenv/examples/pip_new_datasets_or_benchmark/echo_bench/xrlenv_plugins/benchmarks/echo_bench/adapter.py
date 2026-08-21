"""echo_bench — own-benchmark worked example (scenario 1 of B11.6).

Three instances. Each gives the agent a target string in the initial
observation; the agent's only legal action is ``{"output": "<string>"}``.
Reward 1.0 if the agent's output matches the target byte-for-byte,
0.0 otherwise.

This file is the smallest, most-self-contained example of the two
xrlenv plug-in surfaces: the :class:`InstanceResolver` (control-plane
side, picks per-task image + init params) and the :class:`EnvAdapter`
(in-sandbox side, runs the agent loop). If you're shipping a real
benchmark, copy this file and grow the resolver + adapter to taste.
"""

from __future__ import annotations

import contextlib
import io
import tarfile
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar

from xrlenv.control.instance_resolver import (
    InstanceResolver,
    InstanceResolverDecl,
    ResolvedInstance,
    VerifierUpload,
)
from xrlenv.envs.base import EnvAdapter
from xrlenv.types import Action, Observation, StepResult

# ──────────────────────────────────────────────────────────────────────────────
# InstanceResolver — control-plane side
# ──────────────────────────────────────────────────────────────────────────────


# In-process target-string database. A real benchmark would load this
# from a vendored dataset file (``data/instances.jsonl`` or similar);
# echo_bench keeps the data inline so the example fits in one file.
_INSTANCES: dict[str, str] = {
    "echo-hello": "Hello, world!",
    "echo-multiline": "line one\nline two\nline three",
    "echo-symbols": "<>&|$\"' \t!@#%^*()",
}

# Image tag template. The build script
# (``scripts/build-task-images.sh``) tags images with this exact
# pattern; the resolver returns the same per-instance ref so the
# scheduler's pre-flight image check finds them on the node.
_IMAGE_TAG_TEMPLATE = "echo-bench/{instance_id}:0.1"


class EchoBenchInstanceResolver(InstanceResolver):
    """Picks per-task image + the target string for each instance.

    The constructor takes the spec-06 :class:`InstanceResolverDecl`
    by contract; we don't read anything off it because echo_bench
    has no per-deployment options.
    """

    def __init__(self, decl: InstanceResolverDecl) -> None:
        self._decl = decl

    def resolve(self, instance_id: str) -> ResolvedInstance:
        if instance_id not in _INSTANCES:
            raise KeyError(
                f"echo-bench: unknown instance_id {instance_id!r}; "
                f"known instances: {sorted(_INSTANCES)}"
            )
        target = _INSTANCES[instance_id]
        return ResolvedInstance(
            instance_id=instance_id,
            image=_IMAGE_TAG_TEMPLATE.format(instance_id=instance_id),
            init_params={
                "instance_id": instance_id,
                "target": target,
            },
            # D12 stage 1 — upload the grader script at reward time
            # rather than baking it into the image. Closes the
            # adversarial-reward-isolation half (the agent never sees
            # /opt/xrlenv/run-echo-tests.sh during step()).
            verifier_uploads=(
                VerifierUpload(
                    target_dir="/opt/xrlenv",
                    tarball=_grader_tarball(),
                ),
            ),
        )

    def enumerate_instances(self) -> list[str]:
        return sorted(_INSTANCES)


def _grader_tarball() -> bytes:
    """Pack ``run-echo-tests.sh`` into a gzipped tarball with mode 0o755.

    ``importlib.resources.files`` walks the wheel's installed layout,
    which works for both editable installs and built wheels. The
    file lives at ``scripts/run-echo-tests.sh`` host-side and ships
    to ``xrlenv_plugins/benchmarks/echo_bench/scripts/run-echo-tests.sh``
    inside the wheel via the pyproject.toml force-include rule.
    """
    pkg = files("xrlenv_plugins.benchmarks.echo_bench")
    script_path = Path(str(pkg / "scripts" / "run-echo-tests.sh"))
    data = script_path.read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="run-echo-tests.sh")
        info.size = len(data)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# EnvAdapter — in-sandbox side
# ──────────────────────────────────────────────────────────────────────────────


_STATE_DIR = Path("/tmp/echo_bench")


class EchoBenchEnvAdapter(EnvAdapter):
    """Drives the agent loop inside the sandbox.

    Lifecycle:
      - ``setup`` writes the target string to ``/tmp/echo_bench/target.txt``
        and returns the initial observation telling the agent what
        to echo.
      - ``step`` accepts ``{"output": "<string>"}``; writes the agent's
        output to ``/tmp/echo_bench/output.txt`` and sets ``done=True``.
        The benchmark is a 1-step game.
      - The platform fires the manifest's ``reward.cmd`` after step
        returns ``done=True``; the grader script reads target.txt +
        output.txt and emits ``1.0`` / ``0.0`` to stdout.
    """

    supported_reward_modes: ClassVar[frozenset[str]] = frozenset(
        {"in_sandbox_final"},
    )

    def __init__(self) -> None:
        self._target: str | None = None

    async def setup(self, init_params: dict[str, Any]) -> Observation:
        target = init_params.get("target")
        if not isinstance(target, str):
            raise ValueError(
                f"echo-bench setup: missing or non-string ``target`` in "
                f"init_params={init_params!r}"
            )
        self._target = target
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        (_STATE_DIR / "target.txt").write_text(target, encoding="utf-8")
        return {
            "kind": "echo_bench.start",
            "instance_id": init_params.get("instance_id"),
            "target": target,
            "max_steps": 1,
        }

    async def step(self, action: Action) -> StepResult:
        if not isinstance(action, dict):
            raise ValueError(
                f"echo-bench step: action must be a dict with an "
                f"``output`` key, got {type(action).__name__}"
            )
        output = action.get("output")
        if not isinstance(output, str):
            raise ValueError(
                f"echo-bench step: action.output must be a string, "
                f"got {type(output).__name__}"
            )
        (_STATE_DIR / "output.txt").write_text(output, encoding="utf-8")
        return StepResult(
            obs={
                "kind": "echo_bench.submitted",
                "output_len": len(output),
            },
            reward=0.0,  # final reward comes from the in_sandbox_final grader
            done=True,
            truncated=False,
        )

    async def teardown(self) -> None:
        # Best-effort cleanup; sandbox is destroyed seconds after this
        # returns, so the file removal is purely cosmetic.
        for name in ("target.txt", "output.txt"):
            with contextlib.suppress(Exception):
                (_STATE_DIR / name).unlink(missing_ok=True)
