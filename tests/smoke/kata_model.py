"""Real mini-swe-agent + local Qwen inference inside offline Kata (opt-in).

Build the image in kata_model/Dockerfile first. Requires the local Docker/KVM
host and permissions to inspect QEMU processes. This is a functional repair
smoke, not a benchmark score or adversarial escape-resistance evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from kata_runtime import check_removed, vm_processes

import beagle as bgl
from beagle.agents.mini_swe import MiniSweAgent
from beagle.benchmarks import Benchmark, DockerHarness, InBandGrader, register
from beagle.config import RunConfig
from beagle.rollout.runtime import BindMount, KataDockerRuntime
from beagle.types import RolloutStatus, Task, TaskContext

MODEL_SHA256 = "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c"


class ModelInputKata(KataDockerRuntime):
    """Supply one verified read-only input to the unchanged Kata acquire method."""

    def __init__(self, *, docker_host: str, model_file: Path):
        super().__init__(docker_host=docker_host)
        self.model_file = model_file.resolve(strict=True)
        with self.model_file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise ValueError("The model does not match the pinned Qwen 7B GGUF")

    def acquire(self, **kwargs):
        mounts = list(kwargs.pop("mounts", None) or [])
        mounts.append(BindMount(host_path=self.model_file, container_path="/models/model.gguf"))
        return super().acquire(mounts=mounts, **kwargs)


class PreinstalledMiniSwe(MiniSweAgent):
    """Smoke-only setup: start the local model instead of doing an online install.

    The inference/tool loop and trajectory/patch capture use the existing
    MiniSweAgent.run_in implementation unchanged.
    """

    def __init__(self, spec, evidence_dir: Path):
        super().__init__(spec)
        self.evidence_dir = evidence_dir
        self.container_ids = []
        self.evidence = {}

    def install(self, handle, task_ctx, *, runtime):
        self.container_ids.append(handle.container_id)
        kernel = runtime.exec(handle, ["uname", "-r"])
        interfaces = runtime.exec(handle, ["ls", "/sys/class/net"])
        assert kernel.ok and kernel.stdout.strip() != os.uname().release
        assert interfaces.ok and interfaces.stdout.split() == ["lo"]
        capacity = runtime.exec(handle, ["python3", "-c", (
            "import os; from pathlib import Path; "
            "mem=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() "
            "if line.startswith('MemTotal:'))); "
            "assert mem > 7*1024*1024 and os.cpu_count() >= 4, "
            "'Configure the Kata guest for 8 GiB and 4 vCPUs before running this smoke'")])
        assert capacity.ok, capacity.stderr
        readonly = runtime.exec(handle, ["python3", "-c", (
            "import errno\n"
            "try:\n"
            "    stream = open('/models/model.gguf', 'r+b')\n"
            "except OSError as exc:\n"
            "    assert exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM)\n"
            "else:\n"
            "    stream.close()\n"
            "    raise AssertionError('Model input is writable')\n")])
        assert readonly.ok, readonly.stderr
        vms = vm_processes(handle.container_id)
        assert vms, "Expected a visible QEMU process associated with this container"
        baseline = runtime.exec(handle, ["python3", "-m", "unittest", "-v"],
                                workdir="/testbed", timeout=30)
        self.evidence_dir.joinpath("baseline-tests.txt").write_text(baseline.stdout + baseline.stderr)
        assert baseline.returncode == 1 and "FAILED (failures=3)" in baseline.stderr
        version = runtime.exec(handle, ["/agent/.venv/bin/python", "-c",
            "import importlib.metadata; print(importlib.metadata.version('mini-swe-agent'))"])
        assert version.ok and version.stdout.strip() == "2.4.6"
        started = runtime.exec(handle, ["sh", "-c",
            ("/app/llama-server -m /models/model.gguf --alias kata-coder "
            "--host 127.0.0.1 --port 8080 -c 4096 -t 4 -ngl 0 --parallel 1 --jinja "
            "</dev/null >/tmp/model.log 2>&1 &")], workdir="/app")
        assert started.ok, started.stderr
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            health = runtime.exec(handle, ["curl", "-fsS", "--max-time", "2",
                                          "http://127.0.0.1:8080/health"], timeout=5)
            if health.ok:
                break
            time.sleep(1)
        else:
            logs = runtime.exec(handle, ["tail", "-c", "6000", "/tmp/model.log"])
            raise RuntimeError(f"Local model failed to become healthy: {logs.stdout}")
        self.evidence = {
            "agent_container": handle.container_id, "qemu_pids": vms,
            "host_kernel": os.uname().release, "guest_kernel": kernel.stdout.strip(),
            "network": "none", "mini_swe_version": version.stdout.strip(),
            "model": "Qwen2.5-Coder-7B-Instruct Q4_K_M", "model_sha256": MODEL_SHA256,
            "model_mount_read_only": True,
            "baseline_failures": 3,
        }
        print("Local Qwen model ready inside Kata; starting mini-swe-agent", flush=True)

    def run_in(self, handle, task, task_ctx, *, runtime):
        result = super().run_in(handle, task, task_ctx, runtime=runtime)
        self.evidence_dir.joinpath("mini.traj.json").write_text(result.trajectory_text or "")
        self.evidence_dir.joinpath("model.log").write_text(
            runtime.exec(handle, ["cat", "/tmp/model.log"]).stdout)
        self.evidence_dir.joinpath("patch.diff").write_text(result.patch or "")
        assert result.status == RolloutStatus.COMPLETED, result.error
        trajectory = json.loads(result.trajectory_text or "{}")
        exit_status = trajectory.get("info", {}).get("exit_status")
        self.evidence.update(agent_exit_status=exit_status, tokens=result.tokens,
                             model_turns=result.num_turns)
        assert exit_status == "Submitted", f"Agent did not submit successfully: {exit_status}"
        assert result.patch, "Agent submitted no patch"
        assert result.tokens.get("completion", 0) > 0 and result.num_turns > 0, "No model usage recorded"
        source = runtime.exec(handle, ["cat", "/testbed/clamp.py"])
        assert source.ok
        # Grade only the generated source in a fresh Kata guest. Agent edits to
        # tests, the Python environment or the first guest cannot affect it.
        verifier = runtime.acquire(image=task_ctx.image, command=["sleep", "infinity"])
        self.container_ids.append(verifier.container_id)
        try:
            assert vm_processes(verifier.container_id)
            copied = runtime.exec(verifier, ["python3", "-c",
                "from pathlib import Path; import sys; Path('/testbed/clamp.py').write_text(sys.argv[1])",
                source.stdout])
            assert copied.ok, copied.stderr
            verified = runtime.exec(verifier, ["python3", "-m", "unittest", "-v"],
                                    workdir="/testbed", timeout=30)
            self.evidence_dir.joinpath("verification-tests.txt").write_text(
                verified.stdout + verified.stderr)
            assert verified.ok and "Ran 5 tests" in verified.stderr, verified.stderr
        finally:
            runtime.destroy(verifier)
        self.evidence.update(tokens=result.tokens, model_turns=result.num_turns,
                             verification_tests=5, verifier_container=self.container_ids[-1])
        result.reward = 1.0
        result.resolved = True
        return result


@register("kata-model-smoke")
class ModelSmokeBenchmark(Benchmark):
    def source(self):
        raise NotImplementedError("The smoke supplies its dataset directly")

    def harness(self, env_import_path=None):
        return DockerHarness()

    def grader(self):
        return InBandGrader()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-host", default="unix:///var/run/docker.sock")
    parser.add_argument("--image", default="beagle-kata-model-smoke:local")
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = RunConfig.from_dict({
        "model": {"name": "openai/kata-coder"},
        "agent": {"name": "mini-swe", "config": {
            "config_path": "kata-mini.yaml", "max_turns": 8,
            "provider": {"type": "gateway", "name": "kata-local", "extra_args": {
                "api_base": "http://127.0.0.1:8080/v1"}}}},
        "benchmark": {"name": "kata-model-smoke"}, "parallelism": 1,
        "runtime": {"kind": "kata", "options": {"docker_host": args.docker_host}},
    })
    runtime = ModelInputKata(docker_host=args.docker_host, model_file=args.model_file)
    agent = PreinstalledMiniSwe(config.agent_spec(), args.output)
    task = Task(task_id="clamp-repair", benchmark="kata-model-smoke", problem_statement=(
        "Fix clamp.py so clamp(value, lower, upper) restricts value to the inclusive "
        "interval [lower, upper]. Assume lower <= upper. Inspect the code, make the "
        "smallest source fix, and run python3 -m unittest -v. Do not modify the tests."))
    try:
        run = bgl.evaluate(config, agent=agent, runtime=runtime, run_dir=args.output / "run",
            dataset=[(task, TaskContext(image=args.image, repo_path="/testbed", agent_timeout_s=600))])
        assert len(run.results) == 1 and run.results[0].resolved, [r.error for r in run.results]
    finally:
        for container_id in agent.container_ids:
            check_removed(runtime, container_id)
        agent.evidence["containers_and_qemu_removed"] = bool(agent.container_ids)
        args.output.joinpath("evidence.json").write_text(json.dumps(agent.evidence, indent=2))
    print(json.dumps(agent.evidence, indent=2), flush=True)
    print("PASS: real mini-swe repair + local Qwen inference + fresh Kata verification + teardown")


if __name__ == "__main__":
    main()
