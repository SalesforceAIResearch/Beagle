"""Slice 4.5 reward mode tests.

Three layers of coverage:

1. Pure parser/aggregator unit tests over :mod:`xrlenv.control.reward`
   — fast, no transport, no node.
2. End-to-end ``in_sandbox_final`` through the coordinator with a fake
   node that scripts the ``run_in_sandbox`` results.
3. End-to-end ``consumer_final`` through Client/Session — validation at
   start_rollout (RewardFnRequired), reward_fn invocation after finish,
   back-fill into state + sink meta.json (using PlatformJsonlSink).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from xrlenv.backends.base import (
    ExecResult,
    ResourceSpec,
    ResourceUsage,
    SandboxHandle,
)
from xrlenv.client.client import Client
from xrlenv.control.admission import AdmissionQueue
from xrlenv.control.coordinator import RolloutCoordinator
from xrlenv.control.reward import (
    GraderResult,
    RewardComputation,
    RewardComputationError,
    compute_in_sandbox_final_reward,
)
from xrlenv.control.scheduler import Placement
from xrlenv.control.service import CoordinatorRolloutService
from xrlenv.control.state import InMemoryStateStore
from xrlenv.control.template_catalog import (
    EnvAdapterDecl,
    GraderSpec,
    RewardContract,
    TemplateCatalog,
    TemplateManifest,
)
from xrlenv.control.trajectory_sink import PlatformJsonlSink
from xrlenv.errors import RewardFnRequired, RolloutFailed
from xrlenv.node.hw_probe import HardwareInfo

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _hw() -> HardwareInfo:
    return HardwareInfo(
        vcpus=4, mem_bytes=16 * 1024**3, disk_bytes=200 * 1024**3,
        has_kvm=False, has_gpu=False, gpu_model=None,
        kernel_version="0.0.0", platform="linux",
    )


def _manifest(
    *,
    name: str = "t",
    reward: RewardContract | None = None,
) -> TemplateManifest:
    return TemplateManifest(
        name=name, version="0.1", digest=f"sha256:{name}", image=f"im/{name}:1",
        resources=ResourceSpec(
            cpu_request=0.25, cpu_limit=1.0,
            mem_request_bytes=64_000_000, mem_limit_bytes=128_000_000,
            disk_request_bytes=64_000_000,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=reward or RewardContract(mode="env_step"),
    )


class _FakeNode:
    """NodeTransport stand-in. ``run_in_sandbox`` returns scripted ExecResults
    keyed by the cmd's first arg so each grader can be controlled in tests.
    """

    def __init__(self, *, scripts: dict[str, ExecResult]) -> None:
        self.node_id = "fake-reward"
        self._scripts = scripts
        self._created = 0
        self._max_steps_per_sb: dict[str, int] = {}
        self._steps_per_sb: dict[str, int] = {}
        self.run_in_sandbox_calls: list[tuple[str, list[str], float]] = []
        self.put_archive_calls: list[tuple[str, str, int, bool]] = []

    def supported_backends(self) -> list[str]:
        return ["docker"]

    def hardware(self) -> HardwareInfo:
        return _hw()

    async def create_sandbox(self, **_: Any) -> SandboxHandle:
        self._created += 1
        sid = f"sb-{self._created}"
        return SandboxHandle(
            id=sid, backend="docker", backend_ref=f"cid-{self._created}",
            stub_endpoint="tcp://127.0.0.1:0",
        )

    async def destroy_sandbox(self, sb: SandboxHandle) -> None:
        return None

    async def env_setup(
        self, sb: SandboxHandle, *, adapter_module: str, adapter_class: str,
        init_params: dict[str, Any], **_kw: Any,
    ) -> dict[str, Any]:
        self._max_steps_per_sb[sb.id] = int(init_params.get("max_steps") or 1)
        self._steps_per_sb[sb.id] = 0
        return {"obs": {"first": True}}

    async def env_step(
        self, sb: SandboxHandle, action: Any, **_kw: Any,
    ) -> dict[str, Any]:
        self._steps_per_sb[sb.id] += 1
        done = self._steps_per_sb[sb.id] >= self._max_steps_per_sb[sb.id]
        return {
            "obs": {"step": self._steps_per_sb[sb.id]},
            "reward": 0.0,  # no per-step reward in non-env_step modes
            "done": done,
            "truncated": False,
            "info": {},
        }

    async def env_teardown(self, sb: SandboxHandle, **_kw: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def run_in_sandbox(
        self, sb: SandboxHandle, cmd: list[str], *,
        timeout_s: float = 30.0, cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        self.run_in_sandbox_calls.append((sb.id, list(cmd), timeout_s))
        # D12 stage 1: the wipe-before-upload step issues
        # ``sh -c "rm -rf <target>"`` as a no-op when the dir doesn't
        # exist. Tests that don't script that key get a benign success
        # to avoid forcing every test fixture to know about it.
        if cmd[0] == "sh" and len(cmd) >= 3 and cmd[1] == "-c":
            return ExecResult(exit_code=0, stdout=b"", stderr=b"", timed_out=False)
        # Key on the first non-`cat` token so the cat-for-json_file path
        # routes to the right scripted file.
        key = cmd[1] if cmd[0] == "cat" else cmd[0]
        if key not in self._scripts:
            raise KeyError(f"no scripted ExecResult for cmd={cmd!r}")
        return self._scripts[key]

    async def put_archive(
        self,
        sb: SandboxHandle,
        target_dir: str,
        tarball: bytes,
        *,
        clean_target: bool = False,
    ) -> None:
        # D12 stage 1 verifier-asset injection — record the call so
        # tests can assert the timing-isolation contract (uploads
        # happen before reward.cmd runs). ``clean_target`` is the
        # H1-follow-up signal that the backend must root-wipe the
        # target dir before extraction; we record it on the call so
        # tests can pin the contract.
        self.put_archive_calls.append(
            (sb.id, target_dir, len(tarball), clean_target)
        )

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        return ResourceUsage(cpu_seconds=0.0, rss_bytes=0, disk_bytes=0, rx_bytes=0, tx_bytes=0)

    async def query_image(self, _image: str) -> Any:
        from xrlenv.node.image_cache import ImageQueryResult
        return ImageQueryResult(present=True)


def _build_runtime(
    manifest: TemplateManifest,
    *,
    scripts: dict[str, ExecResult] | None = None,
    sink: PlatformJsonlSink | None = None,
) -> tuple[Client, RolloutCoordinator, InMemoryStateStore, _FakeNode]:
    node = _FakeNode(scripts=scripts or {})
    catalog = TemplateCatalog()
    catalog.register(manifest)
    sched = MagicMock()
    sched.place.return_value = Placement(node=node, backend="docker", score=1)
    sched.nodes = [node]
    state = InMemoryStateStore()
    admission = AdmissionQueue(scheduler=sched, state=state)
    coordinator = RolloutCoordinator(
        catalog=catalog, scheduler=sched, state=state,
        admission=admission, trajectory_sink=sink,
    )
    service = CoordinatorRolloutService(coordinator)
    client = Client.in_process(service)
    return client, coordinator, state, node


async def _drain(coord: RolloutCoordinator, client: Client) -> None:
    """Tear down the coordinator's background watchers + close the client."""
    await coord.deadline_watcher.shutdown()
    await coord.idle_ttl_watcher.shutdown()
    await client.close()


# ──────────────────────────────────────────────────────────────────────────────
# Pure parser / aggregator tests
# ──────────────────────────────────────────────────────────────────────────────


async def test_verifier_uploads_extracted_before_graders_run() -> None:
    """D12 stage 1 + H1 follow-up: when ``compute_in_sandbox_final_reward``
    is given a non-empty ``verifier_uploads`` tuple, every entry's
    tarball is extracted into its target_dir BEFORE the first grader
    runs, with ``clean_target=True`` so the backend wipes the dir as
    root before extraction. No ``rm -rf`` round-trip through the
    in-sandbox stub (which doesn't run as root) — the wipe is
    backend-owned and fail-closed."""
    from xrlenv.control.instance_resolver import VerifierUpload

    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
    )
    node = _FakeNode(
        scripts={"grader.sh": ExecResult(exit_code=0, stdout=b"1.0\n")}
    )
    sb = SandboxHandle(
        id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0",
    )
    uploads = (
        VerifierUpload(target_dir="/tests", tarball=b"\x1f\x8b\x08\x00fake-tar1"),
        VerifierUpload(target_dir="/opt/xrlenv", tarball=b"\x1f\x8b\x08\x00fake-tar2"),
    )

    comp = await compute_in_sandbox_final_reward(
        node=node, sandbox=sb, contract=contract,
        verifier_uploads=uploads,
    )

    # Both uploads landed (in declared order) with clean_target=True.
    assert node.put_archive_calls == [
        ("sb", "/tests", len(uploads[0].tarball), True),
        ("sb", "/opt/xrlenv", len(uploads[1].tarball), True),
    ]
    # No stub-side rm -rf calls — the wipe is now backend/root-backed,
    # so run_in_sandbox should only see the grader command.
    rm_calls = [
        cmd for (_sb, cmd, _t) in node.run_in_sandbox_calls
        if cmd[0] == "sh" and len(cmd) >= 3 and "rm -rf" in cmd[2]
    ]
    assert rm_calls == [], f"expected no stub-side rm -rf, got {rm_calls}"
    # Reward still computed normally.
    assert comp.final_reward == 1.0


async def test_no_verifier_uploads_skips_upload_phase() -> None:
    """When the resolver supplies no verifier uploads (e.g. the
    hello-shell template, or a Pattern-A task that doesn't need
    timing-isolated grader assets), the reward path skips the
    upload phase entirely — preserves the existing contract for
    benchmarks that bake their grader."""
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
    )
    node = _FakeNode(
        scripts={"grader.sh": ExecResult(exit_code=0, stdout=b"0.5\n")}
    )
    sb = SandboxHandle(
        id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0",
    )

    comp = await compute_in_sandbox_final_reward(
        node=node, sandbox=sb, contract=contract,
    )

    assert node.put_archive_calls == []
    assert comp.final_reward == 0.5


async def test_parse_exit_code_format() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("/bin/false",),
        output_format="exit_code",
    )
    node = _FakeNode(scripts={"/bin/false": ExecResult(exit_code=1)})
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert comp.final_reward == 0.0  # exit_code != 0 → 0.0
    assert comp.per_grader[0].score == 0.0

    contract_pass = RewardContract(
        mode="in_sandbox_final",
        cmd=("/bin/true",),
        output_format="exit_code",
    )
    node_pass = _FakeNode(scripts={"/bin/true": ExecResult(exit_code=0)})
    comp_pass = await compute_in_sandbox_final_reward(
        node=node_pass, sandbox=sb, contract=contract_pass
    )
    assert comp_pass.final_reward == 1.0


async def test_parse_stdout_float_format() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
    )
    node = _FakeNode(scripts={"grader.sh": ExecResult(exit_code=0, stdout=b"some logs\n0.875\n")})
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert pytest.approx(comp.final_reward) == 0.875
    # Diagnostic capture: the grader's stdout/stderr survive into the
    # GraderResult so trajectory.metadata.rewards carries the trail
    # (audit follow-up — score=0.0 with no error needs context).
    assert comp.per_grader[0].stdout == "some logs\n0.875\n"
    assert comp.per_grader[0].stderr is None  # no stderr emitted


async def test_grader_output_captured_with_stderr_and_truncation() -> None:
    """A grader that emits a long stderr trace (e.g. test.log) gets
    front-truncated so trajectory metadata stays bounded, and stderr
    is captured even when stdout is empty (typical for stdout_float
    where the wrapper just prints the score)."""
    from xrlenv.control.reward import GRADER_OUTPUT_BYTES_CAP

    huge_stderr = b"x" * (GRADER_OUTPUT_BYTES_CAP + 5000) + b"TAIL_MARKER\n"
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
    )
    node = _FakeNode(scripts={
        "grader.sh": ExecResult(exit_code=0, stdout=b"0\n", stderr=huge_stderr),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    captured = comp.per_grader[0].stderr
    assert captured is not None
    # Bounded by the cap (decoded UTF-8 length ≤ byte cap because
    # the payload is ASCII).
    assert len(captured.encode("utf-8")) <= GRADER_OUTPUT_BYTES_CAP
    # Tail-truncation preserves the trailing diagnostic, which is
    # where most error messages land.
    assert "TAIL_MARKER" in captured


async def test_parse_json_stdout_format() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="json_stdout",
        score_key="my_score",
    )
    node = _FakeNode(
        scripts={"grader.sh": ExecResult(exit_code=0, stdout=b'{"my_score": 0.6, "extra": 1}')}
    )
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert pytest.approx(comp.final_reward) == 0.6


async def test_parse_json_file_format() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("oswold-eval.sh",),
        output_format="json_file",
        output_path="/sandbox/.report.json",
        score_key="score",
    )
    node = _FakeNode(scripts={
        "oswold-eval.sh": ExecResult(exit_code=0, stdout=b""),
        "/sandbox/.report.json": ExecResult(exit_code=0, stdout=b'{"score": 0.42}'),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert pytest.approx(comp.final_reward) == 0.42
    # Verify the cat round-trip happened.
    cmds = [c for _, c, _ in node.run_in_sandbox_calls]
    assert ["oswold-eval.sh"] in cmds
    assert ["cat", "/sandbox/.report.json"] in cmds


# ──────────────────────────────────────────────────────────────────────────────
# Multi-grader + aggregators
# ──────────────────────────────────────────────────────────────────────────────


async def test_multi_grader_weighted_sum() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="correctness", cmd=("c.sh",), output_format="stdout_float", weight=1.0),
            GraderSpec(name="style", cmd=("s.sh",), output_format="exit_code", weight=0.1),
        ),
        aggregator="weighted_sum",
    )
    node = _FakeNode(scripts={
        "c.sh": ExecResult(exit_code=0, stdout=b"0.8"),
        "s.sh": ExecResult(exit_code=0),  # style passes
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    # 0.8 * 1.0 + 1.0 * 0.1 = 0.9
    assert pytest.approx(comp.final_reward) == 0.9
    by_name = {r.name: r for r in comp.per_grader}
    assert pytest.approx(by_name["correctness"].score) == 0.8
    assert pytest.approx(by_name["style"].score) == 1.0


async def test_multi_grader_mean_aggregator() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="a", cmd=("a.sh",), output_format="stdout_float"),
            GraderSpec(name="b", cmd=("b.sh",), output_format="stdout_float"),
        ),
        aggregator="mean",
    )
    node = _FakeNode(scripts={
        "a.sh": ExecResult(exit_code=0, stdout=b"0.4"),
        "b.sh": ExecResult(exit_code=0, stdout=b"0.8"),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert pytest.approx(comp.final_reward) == 0.6


async def test_multi_grader_max_aggregator() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="a", cmd=("a.sh",), output_format="stdout_float"),
            GraderSpec(name="b", cmd=("b.sh",), output_format="stdout_float"),
        ),
        aggregator="max",
    )
    node = _FakeNode(scripts={
        "a.sh": ExecResult(exit_code=0, stdout=b"0.3"),
        "b.sh": ExecResult(exit_code=0, stdout=b"0.9"),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert pytest.approx(comp.final_reward) == 0.9


async def test_multi_grader_first_aggregator() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="canonical", cmd=("a.sh",), output_format="stdout_float"),
            GraderSpec(name="info_only", cmd=("b.sh",), output_format="stdout_float"),
        ),
        aggregator="first",
    )
    node = _FakeNode(scripts={
        "a.sh": ExecResult(exit_code=0, stdout=b"0.5"),
        "b.sh": ExecResult(exit_code=0, stdout=b"99.0"),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert pytest.approx(comp.final_reward) == 0.5
    # The info-only grader's score is preserved in per_grader for inspection.
    assert pytest.approx(comp.per_grader[1].score) == 99.0


# ──────────────────────────────────────────────────────────────────────────────
# on_error semantics
# ──────────────────────────────────────────────────────────────────────────────


async def test_on_error_fail_rollout_raises() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
        on_error="fail_rollout",
    )
    node = _FakeNode(scripts={"grader.sh": ExecResult(exit_code=2, stdout=b"")})
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    with pytest.raises(RewardComputationError) as excinfo:
        await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert excinfo.value.computation.failed is True
    assert "exit_2" in (excinfo.value.computation.error_message or "")


async def test_on_error_zero_reward_substitutes() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="a", cmd=("a.sh",), output_format="stdout_float", weight=1.0),
            GraderSpec(name="b", cmd=("b.sh",), output_format="stdout_float", weight=1.0),
        ),
        aggregator="weighted_sum",
        on_error="zero_reward",
    )
    node = _FakeNode(scripts={
        "a.sh": ExecResult(exit_code=0, stdout=b"0.5"),
        "b.sh": ExecResult(exit_code=2, stdout=b""),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert comp.failed is False
    # b's failure is treated as 0; aggregate = 0.5 * 1.0 + 0 * 1.0 = 0.5
    assert pytest.approx(comp.final_reward) == 0.5
    by_name = {r.name: r for r in comp.per_grader}
    assert by_name["b"].error is not None  # error preserved for inspection


async def test_on_error_partial_drops_failed_graders() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="a", cmd=("a.sh",), output_format="stdout_float", weight=1.0),
            GraderSpec(name="b", cmd=("b.sh",), output_format="stdout_float", weight=1.0),
        ),
        aggregator="mean",
        on_error="partial",
    )
    node = _FakeNode(scripts={
        "a.sh": ExecResult(exit_code=0, stdout=b"0.4"),
        "b.sh": ExecResult(exit_code=2, stdout=b""),
    })
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    # Mean over surviving graders only = 0.4
    assert pytest.approx(comp.final_reward) == 0.4


async def test_timeout_treated_as_grader_error() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
        on_error="zero_reward",
    )
    node = _FakeNode(scripts={"grader.sh": ExecResult(exit_code=124, timed_out=True)})
    sb = SandboxHandle(id="sb", backend="docker", backend_ref="cid", stub_endpoint="tcp://127.0.0.1:0")
    comp = await compute_in_sandbox_final_reward(node=node, sandbox=sb, contract=contract)
    assert comp.per_grader[0].error == "timeout"
    assert pytest.approx(comp.final_reward) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# RewardContract validation
# ──────────────────────────────────────────────────────────────────────────────


def test_reward_contract_rejects_cmd_and_graders_together() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="mutually exclusive"):
        RewardContract(
            mode="in_sandbox_final",
            cmd=("a.sh",),
            graders=(GraderSpec(name="x", cmd=("b.sh",)),),
        )


def test_reward_contract_rejects_duplicate_grader_names() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="duplicate grader name"):
        RewardContract(
            mode="in_sandbox_final",
            graders=(
                GraderSpec(name="dupe", cmd=("a.sh",)),
                GraderSpec(name="dupe", cmd=("b.sh",)),
            ),
        )


def test_reward_contract_in_sandbox_final_requires_cmd_or_graders() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"requires either 'cmd' .* or 'graders'"):
        RewardContract(mode="in_sandbox_final")


def test_grader_spec_json_file_requires_output_path() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="output_path is required"):
        GraderSpec(name="g", cmd=("a.sh",), output_format="json_file")


# ──────────────────────────────────────────────────────────────────────────────
# Coordinator integration: in_sandbox_final fires at done
# ──────────────────────────────────────────────────────────────────────────────


async def test_coordinator_runs_in_sandbox_final_at_done() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
    )
    manifest = _manifest(reward=contract)
    client, coord, state, _node = _build_runtime(manifest, scripts={
        "grader.sh": ExecResult(exit_code=0, stdout=b"0.75"),
    })
    try:
        s = await client.rollout(template="t", init={"max_steps": 1})
        async with s:
            while not s.done:
                await s.step({"cmd": "noop"})
        # Final reward written into state by the coordinator's reward path.
        rec = state.get_rollout(s.rollout_id)
        assert pytest.approx(rec.final_reward) == 0.75
        assert rec.metadata.get("rewards") is not None
        assert pytest.approx(rec.metadata["rewards"]["default"]["score"]) == 0.75
    finally:
        await _drain(coord, client)


async def test_coordinator_in_sandbox_final_fail_rollout_raises_at_step() -> None:
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
        on_error="fail_rollout",
    )
    manifest = _manifest(reward=contract)
    client, coord, _state, _node = _build_runtime(manifest, scripts={
        "grader.sh": ExecResult(exit_code=2, stdout=b""),
    })
    try:
        s = await client.rollout(template="t", init={"max_steps": 1})
        # The terminal step's reward computation raises RolloutFailed; the
        # session.__aexit__ then catches it and cancels — the consumer sees
        # the carrier exception escape.
        with pytest.raises(RolloutFailed, match="reward computation failed"):
            async with s:
                while not s.done:
                    await s.step({"cmd": "noop"})
    finally:
        await _drain(coord, client)


async def test_in_sandbox_final_fail_rollout_seals_as_failed_reward_failed(
    tmp_path: Path,
) -> None:
    """Regression for audit H1 against commit af3b985.

    When in_sandbox_final reward computation raises with ``on_error=fail_rollout``,
    the sealed trajectory must be ``status=FAILED reason=reward_failed``, not
    ``cancelled/aborted_with_exception`` (which is what the bug produced).
    """
    sink = PlatformJsonlSink(tmp_path / "runs")
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
        on_error="fail_rollout",
    )
    manifest = _manifest(reward=contract)
    client, coord, state, _node = _build_runtime(
        manifest,
        sink=sink,
        scripts={"grader.sh": ExecResult(exit_code=2, stdout=b"")},
    )
    try:
        s = await client.rollout(template="t", init={"max_steps": 1})
        rollout_id = s.rollout_id
        with pytest.raises(RolloutFailed, match="reward computation failed"):
            async with s:
                while not s.done:
                    await s.step({"cmd": "noop"})

        # State store: terminal status FAILED, reason reward_failed.
        rec = state.get_rollout(rollout_id)
        assert rec.status.value == "failed", (
            f"expected failed, got {rec.status.value}/{rec.reason}"
        )
        assert rec.reason == "reward_failed"
        # On-disk meta.json mirrors the canonical sealed state.
        replayed = sink.read(rollout_id)
        assert replayed.status.value == "failed"
        assert replayed.reason == "reward_failed"
    finally:
        await _drain(coord, client)


async def test_in_sandbox_final_fail_rollout_batch_buckets_as_failed(
    tmp_path: Path,
) -> None:
    """``batch_rollout`` must bucket reward-failure rollouts in ``failed``,
    not ``truncated`` — the carrier's ``partial`` lets it land in the right
    list with ``reason='reward_failed'``.
    """
    sink = PlatformJsonlSink(tmp_path / "runs")
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
        on_error="fail_rollout",
    )
    manifest = _manifest(reward=contract)
    client, coord, _state, _node = _build_runtime(
        manifest,
        sink=sink,
        scripts={"grader.sh": ExecResult(exit_code=2, stdout=b"")},
    )

    async def policy(_obs: Any) -> dict[str, str]:
        return {"cmd": "noop"}

    try:
        result = await client.batch_rollout(
            template="t",
            inits=[{"max_steps": 1}],
            policy=policy,
        )
        assert result.finished == []
        assert result.truncated == []
        assert len(result.failed) == 1
        assert result.failed[0].reason == "reward_failed"
        assert result.failed[0].partial is not None
        assert result.failed[0].partial.status.value == "failed"
    finally:
        await _drain(coord, client)


async def test_in_sandbox_final_skipped_when_cancelled_before_done() -> None:
    """D1 from notes/deferred_audit_todos.md: cancel_rollout fired
    before the rollout reaches its terminal step must NOT trigger
    grader execution. ``_compute_in_sandbox_final`` is gated on
    ``result.done`` in coordinator.py and the cancel path bypasses
    that branch entirely — pin the contract so a future refactor
    that moves reward computation into a teardown hook can't
    silently start grading partial trajectories.
    """
    contract = RewardContract(
        mode="in_sandbox_final",
        cmd=("grader.sh",),
        output_format="stdout_float",
    )
    manifest = _manifest(reward=contract)
    # Long-horizon rollout so we have plenty of pre-done time to
    # cancel inside.
    client, coord, state, node = _build_runtime(manifest, scripts={
        "grader.sh": ExecResult(exit_code=0, stdout=b"0.99"),
    })
    try:
        s = await client.rollout(template="t", init={"max_steps": 50})
        # Take one step, then cancel — well before the grader would
        # otherwise fire at done.
        await s.step({"cmd": "noop"})
        await client.cancel_rollout(s.rollout_id, reason="test")

        rec = state.get_rollout(s.rollout_id)
        assert rec.status.value == "cancelled"
        # Grader must never have run — the cancel path skips reward
        # computation entirely.
        grader_calls = [
            cmd for (_sb, cmd, _t) in node.run_in_sandbox_calls
            if cmd and cmd[0] == "grader.sh"
        ]
        assert grader_calls == [], (
            f"grader.sh should never run on cancel before done; "
            f"got {grader_calls!r}"
        )
        # And the state record must not carry a final reward written by
        # the in_sandbox_final path (defaults to 0.0).
        assert pytest.approx(rec.final_reward) == 0.0
    finally:
        await _drain(coord, client)


# ──────────────────────────────────────────────────────────────────────────────
# consumer_final via Client + Session
# ──────────────────────────────────────────────────────────────────────────────


async def test_consumer_final_requires_reward_fn_at_start_rollout() -> None:
    contract = RewardContract(mode="consumer_final")
    manifest = _manifest(reward=contract)
    client, coord, _state, _node = _build_runtime(manifest)
    try:
        with pytest.raises(RewardFnRequired):
            await client.rollout(template="t", init={"max_steps": 1})
    finally:
        await _drain(coord, client)


async def test_consumer_final_calls_reward_fn_after_finish_and_backfills(
    tmp_path: Path,
) -> None:
    sink = PlatformJsonlSink(tmp_path / "runs")
    contract = RewardContract(mode="consumer_final")
    manifest = _manifest(reward=contract)
    client, coord, state, _node = _build_runtime(manifest, sink=sink)

    invocations: list[float] = []

    async def reward_fn(traj: Any) -> float:
        invocations.append(float(len(traj.steps)))
        return 0.42

    try:
        s = await client.rollout(
            template="t", init={"max_steps": 2}, reward_fn=reward_fn
        )
        async with s:
            while not s.done:
                await s.step({"cmd": "noop"})
        # reward_fn was invoked exactly once with the sealed trajectory.
        assert invocations == [2.0]
        # In-memory trajectory has the back-filled value.
        assert pytest.approx(s.trajectory.final_reward) == 0.42
        # State store has the canonical value.
        rec = state.get_rollout(s.rollout_id)
        assert pytest.approx(rec.final_reward) == 0.42
        # Sink's meta.json was atomically rewritten.
        replayed = sink.read(s.rollout_id)
        assert pytest.approx(replayed.final_reward) == 0.42
    finally:
        await _drain(coord, client)


async def test_consumer_final_unused_when_template_is_env_step(tmp_path: Path) -> None:
    """env_step templates ignore reward_fn (it's allowed but no-op)."""
    sink = PlatformJsonlSink(tmp_path / "runs")
    manifest = _manifest()  # default env_step
    client, coord, state, _node = _build_runtime(manifest, sink=sink)
    invoked = False

    async def reward_fn(_traj: Any) -> float:
        nonlocal invoked
        invoked = True
        return 99.0

    try:
        s = await client.rollout(
            template="t", init={"max_steps": 1}, reward_fn=reward_fn
        )
        async with s:
            while not s.done:
                await s.step({"cmd": "noop"})
        assert invoked is False  # not called for env_step
        # final_reward stays at the env_step accumulated value (0 here since
        # FakeNode returns reward=0 each step).
        rec = state.get_rollout(s.rollout_id)
        assert rec.final_reward == 0.0
    finally:
        await _drain(coord, client)


async def test_in_sandbox_final_writes_per_grader_to_metadata_rewards() -> None:
    """Sub-scores survive aggregation so the consumer / admin can inspect them."""
    contract = RewardContract(
        mode="in_sandbox_final",
        graders=(
            GraderSpec(name="correctness", cmd=("c.sh",), output_format="stdout_float", weight=1.0),
            GraderSpec(name="style", cmd=("s.sh",), output_format="exit_code", weight=0.1),
        ),
        aggregator="weighted_sum",
    )
    manifest = _manifest(reward=contract)
    client, coord, state, _node = _build_runtime(manifest, scripts={
        "c.sh": ExecResult(exit_code=0, stdout=b"0.8"),
        "s.sh": ExecResult(exit_code=0),
    })
    try:
        s = await client.rollout(template="t", init={"max_steps": 1})
        async with s:
            while not s.done:
                await s.step({"cmd": "noop"})
        rec = state.get_rollout(s.rollout_id)
        rewards = rec.metadata["rewards"]
        assert pytest.approx(rewards["correctness"]["score"]) == 0.8
        assert pytest.approx(rewards["style"]["score"]) == 1.0
        assert pytest.approx(rec.final_reward) == 0.9
    finally:
        await _drain(coord, client)


# Suppress unused-import lint on GraderResult / RewardComputation — the
# pure helpers above import them so callers (and future tests) can construct
# expectations without going through compute_in_sandbox_final_reward.
_ = (GraderResult, RewardComputation)
