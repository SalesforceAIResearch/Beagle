"""Unit tests for the EvoClaw docker-CLI interceptor (no cluster needed).

Drives ``docker_shim`` against a fake docker-py-shaped client and asserts the
argv → client routing specified in SHIM-SURFACE.md.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docker_shim


# ---- fakes ------------------------------------------------------------------
class FakeExec:
    def __init__(self, exit_code: int, output):
        self.exit_code = exit_code
        self.output = output


class FakeContainer:
    def __init__(self, cid: str, name: str | None):
        self.id = cid
        self.name = name
        self.execs: list[tuple] = []
        self.put: list[tuple] = []
        self.removed = False
        self.stopped = False
        self.started = False

    def exec_run(self, cmd, user="", workdir=None, environment=None, demux=False):
        self.execs.append((cmd, user, workdir, environment))
        out, err = b"OUT", b""
        return FakeExec(0, (out, err) if demux else out)

    def put_archive(self, path, data):
        self.put.append((path, data))
        return True

    def get_archive(self, path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name=os.path.basename(path))
            content = b"DATA"
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
        return ([buf.getvalue()], {})

    def remove(self, force=False):
        self.removed = True

    def stop(self):
        self.stopped = True

    def start(self):
        self.started = True


class FakeContainers:
    def __init__(self):
        self.created: list[dict] = []

    def run(self, image, command=None, name=None, detach=True, **kw):
        self.created.append({"image": image, "command": command, "name": name, **kw})
        if detach:
            return FakeContainer(cid=f"cid-{name}", name=name)
        return b"oneshot-logs"


class FakeClient:
    def __init__(self):
        self.containers = FakeContainers()


@pytest.fixture()
def client():
    c = FakeClient()
    docker_shim.install(client=c, name_prefix="p-", labels={"benchmark": "evoclaw"})
    try:
        yield c
    finally:
        docker_shim.uninstall()


def _run(argv, **kw):
    return subprocess.run(argv, capture_output=True, **kw)


# ---- run --------------------------------------------------------------------
def test_run_detached_parses_flags_and_registers(client):
    r = _run([
        "docker", "run", "-d", "--init", "--name", "c1",
        "--cap-add=NET_ADMIN", "--cpus", "2", "--ulimit", "nofile=1:1",
        "--sysctl", "net.ipv6.conf.all.disable_ipv6=1",
        "-e", "K=V", "-w", "/testbed", "-v", "/host/ws:/e2e_workspace",
        "img:tag", "tail", "-f", "/dev/null",
    ])
    assert r.returncode == 0
    assert b"cid-p-c1" in r.stdout
    created = client.containers.created[0]
    assert created["name"] == "p-c1"          # name namespaced
    assert created["image"] == "img:tag"
    assert created["command"] == ["tail", "-f", "/dev/null"]
    assert created["cap_add"] == ["NET_ADMIN"]
    assert created["nano_cpus"] == 2_000_000_000
    assert created["environment"] == {"K": "V"}
    assert created["working_dir"] == "/testbed"
    assert "c1" in docker_shim._REGISTRY          # keyed by EvoClaw's own name
    # the dropped -v host bind has its container target pre-created
    cont = docker_shim._REGISTRY["c1"]
    assert (["mkdir", "-p", "/e2e_workspace"], "", None, None) in cont.execs


def test_run_bind_nonempty_host_is_input(client, tmp_path):
    # -v <non-empty host dir>:/golden:ro  -> put_archive in; NOT an output bind
    g = tmp_path / "golden"
    g.mkdir()
    (g / "m1.tar").write_bytes(b"x")
    _run(["docker", "run", "-d", "--name", "c1", "-v", f"{g}:/golden:ro", "img"])
    cont = docker_shim._REGISTRY["c1"]
    assert any(path == "/" for path, _ in cont.put)   # golden dir copied to /golden
    assert "c1" not in docker_shim._OUTPUT_BINDS       # input, not synced back


def test_run_bind_nonempty_rw_is_input_and_output(client, tmp_path):
    # eval container: -v output_dir:/output (read-write) already holds an input
    # (source_snapshot.tar) AND the container writes reports there -> both.
    out = tmp_path / "output"
    out.mkdir()
    (out / "source_snapshot.tar").write_bytes(b"S")
    _run(["docker", "run", "-d", "--name", "c1", "-v", f"{out}:/output", "img"])
    cont = docker_shim._REGISTRY["c1"]
    assert any(path == "/" for path, _ in cont.put)                # input copied in
    assert (str(out), "/output") in docker_shim._OUTPUT_BINDS["c1"]  # AND synced back


def test_run_bind_empty_host_is_output(client, tmp_path):
    # -v <empty host dir>:/output  -> mkdir + track as output bind (synced back)
    out = tmp_path / "output"
    out.mkdir()
    _run(["docker", "run", "-d", "--name", "c1", "-v", f"{out}:/output", "img"])
    cont = docker_shim._REGISTRY["c1"]
    assert (["mkdir", "-p", "/output"], "", None, None) in cont.execs
    assert not cont.put
    assert (str(out), "/output") in docker_shim._OUTPUT_BINDS["c1"]


def test_run_bind_missing_host_is_output(client):
    _run(["docker", "run", "-d", "--name", "c1", "-v", "/no/such/host/path:/e2e_workspace", "img"])
    cont = docker_shim._REGISTRY["c1"]
    assert (["mkdir", "-p", "/e2e_workspace"], "", None, None) in cont.execs
    assert not cont.put
    assert docker_shim._OUTPUT_BINDS["c1"] == [("/no/such/host/path", "/e2e_workspace")]


def test_output_bind_syncs_back_after_exec(client, tmp_path):
    # The core fix: an exec's writes under an output volume reach the host dir.
    out = tmp_path / "output"
    out.mkdir()
    _run(["docker", "run", "-d", "--name", "c1", "-v", f"{out}:/output", "img"])
    cont = docker_shim._REGISTRY["c1"]

    def fake_get_archive(path):  # dir-style tar: "output/eval.json"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            data = b"REPORT"
            info = tarfile.TarInfo(name="output/eval.json")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        return ([buf.getvalue()], {})

    cont.get_archive = fake_get_archive
    _run(["docker", "exec", "c1", "bash", "-c", "run tests -> /output"])
    assert (out / "eval.json").read_bytes() == b"REPORT"   # synced container -> host


def test_run_oneshot_returns_logs(client):
    r = _run(["docker", "run", "--rm", "--init", "-w", "/w", "img", "bash", "-c", "echo hi"])
    assert r.returncode == 0
    assert r.stdout == b"oneshot-logs"
    assert client.containers.created[-1]["name"] is None


# ---- exec -------------------------------------------------------------------
def test_exec_demux(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    r = _run(["docker", "exec", "--user", "fakeroot", "-w", "/testbed", "-e", "H=1", "c1", "echo", "hi"])
    assert r.returncode == 0
    assert r.stdout == b"OUT"
    cmd, user, workdir, env = docker_shim._REGISTRY["c1"].execs[-1]
    assert cmd == ["echo", "hi"] and user == "fakeroot" and workdir == "/testbed"
    assert env == {"H": "1"}


def test_exec_stdout_to_file_for_git_archive(client, tmp_path):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    out = tmp_path / "snap.tar"
    with out.open("wb") as fh:
        res = subprocess.run(
            ["docker", "exec", "--user", "fakeroot", "c1", "git", "archive", "HEAD"],
            stdout=fh, stderr=subprocess.PIPE,
        )
    assert res.returncode == 0
    assert out.read_bytes() == b"OUT"   # exec stdout streamed to the file


# ---- cp ---------------------------------------------------------------------
def test_cp_to_container(client, tmp_path):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    local = tmp_path / "f.txt"
    local.write_text("hello")
    r = _run(["docker", "cp", str(local), "c1:/testbed/f.txt"])
    assert r.returncode == 0
    path, data = docker_shim._REGISTRY["c1"].put[-1]
    assert path == "/testbed"
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        assert tf.getnames() == ["f.txt"]


def test_cp_from_container(client, tmp_path):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    dst = tmp_path / "out"
    r = _run(["docker", "cp", "c1:/logs/report.json", str(dst)])
    assert r.returncode == 0
    assert dst.read_bytes() == b"DATA"


# ---- lifecycle / probes -----------------------------------------------------
def test_rm_force(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    cont = docker_shim._REGISTRY["c1"]
    r = _run(["docker", "rm", "-f", "c1"])
    assert r.returncode == 0 and cont.removed and "c1" not in docker_shim._REGISTRY


def test_inspect_running(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    r = _run(["docker", "inspect", "-f", "{{.State.Running}}", "c1"])
    assert r.returncode == 0 and r.stdout == b"true\n"
    r2 = _run(["docker", "inspect", "-f", "{{.State.Running}}", "nope"])
    assert r2.returncode != 0


def test_ps_filter(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    r = _run(["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=^c1$"])
    assert r.stdout == b"c1\n"
    r2 = _run(["docker", "ps", "-a", "--format", "{{.Names}}", "--filter", "name=^ghost$"])
    assert r2.stdout == b""


def test_images_and_image_inspect_present(client):
    assert _run(["docker", "images", "-q", "img"]).stdout.strip()
    assert _run(["docker", "image", "inspect", "img"]).returncode == 0


def test_build_fails_loud(client):
    r = _run(["docker", "build", "-t", "x", "."])
    assert r.returncode == 127
    assert b"not covered" in r.stderr


# ---- shell=True string forms ------------------------------------------------
def test_shell_string_run(client):
    r = subprocess.run(
        'docker run --rm --init -w /w img bash -c "echo hi"',
        shell=True, capture_output=True,
    )
    assert r.returncode == 0 and r.stdout == b"oneshot-logs"


def test_shell_string_rm(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    r = subprocess.run("docker rm -f c1", shell=True, capture_output=True)
    assert r.returncode == 0 and "c1" not in docker_shim._REGISTRY


# ---- passthrough + streaming Popen ------------------------------------------
def test_non_docker_passes_through(client):
    r = subprocess.run(["printf", "hello"], capture_output=True)
    assert r.stdout == b"hello"


def test_popen_streaming_exec(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    proc = subprocess.Popen(
        ["docker", "exec", "c1", "echo", "hi"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    data = proc.stdout.read()
    rc = proc.wait(timeout=10)
    assert rc == 0 and data == "OUT"


def test_text_mode_decodes(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    r = subprocess.run(["docker", "exec", "c1", "echo"], capture_output=True, text=True)
    assert r.stdout == "OUT"


def test_unknown_run_flag_fails_loud(client):
    with pytest.raises(docker_shim.DockerShimError):
        subprocess.run(["docker", "run", "--totally-unknown", "img"], capture_output=True)


def test_cleanup_containers_removes_registered(client):
    _run(["docker", "run", "-d", "--name", "c1", "img"])
    _run(["docker", "run", "-d", "--name", "c2", "img"])
    c1 = docker_shim._REGISTRY["c1"]
    c2 = docker_shim._REGISTRY["c2"]
    docker_shim.cleanup_containers()
    assert c1.removed and c2.removed          # both force-removed
    assert docker_shim._REGISTRY == {}         # registry emptied
    docker_shim.cleanup_containers()           # idempotent: no raise on empty


# ---- transient cluster-loss retry ------------------------------------------
class ControlPlaneLost(Exception):
    """Named to match xrlenv.errors.ControlPlaneLost (matched by class name)."""


def test_is_transient_by_type_name_and_message():
    assert docker_shim._is_transient_cluster_error(ControlPlaneLost("boom"))
    assert docker_shim._is_transient_cluster_error(RuntimeError("Cancelling all calls"))
    assert docker_shim._is_transient_cluster_error(RuntimeError("UNAVAILABLE: ..."))
    assert not docker_shim._is_transient_cluster_error(ValueError("bad arg"))


def test_with_retry_recovers_after_transient(monkeypatch):
    monkeypatch.setattr(docker_shim.time, "sleep", lambda *_: None)
    monkeypatch.setattr(docker_shim, "_RETRY_ATTEMPTS", 4)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ControlPlaneLost("Cancelling all calls")
        return "ok"

    assert docker_shim._with_retry("exec c", flaky) == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on the third


def test_with_retry_exhausts_to_connectionerror(monkeypatch):
    monkeypatch.setattr(docker_shim.time, "sleep", lambda *_: None)
    monkeypatch.setattr(docker_shim, "_RETRY_ATTEMPTS", 3)

    def always():
        raise ControlPlaneLost("Connection refused")

    with pytest.raises(ConnectionError) as ei:
        docker_shim._with_retry("exec c", always)
    assert isinstance(ei.value, OSError)  # EvoClaw's eval-retry treats OSError as transient
    assert "after 3 attempt" in str(ei.value)


def test_with_retry_passes_through_nontransient(monkeypatch):
    monkeypatch.setattr(docker_shim.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("not a cluster blip")

    with pytest.raises(ValueError):
        docker_shim._with_retry("exec c", bad)
    assert calls["n"] == 1  # non-transient: not retried


# ---- memory: CPU-scaled limit so heavy suites don't OOM the 4GiB default ----
def test_parse_mem():
    assert docker_shim._parse_mem("4g") == 4 * 1024**3
    assert docker_shim._parse_mem("512m") == 512 * 1024**2
    assert docker_shim._parse_mem("4294967296") == 4294967296
    assert docker_shim._parse_mem("2GB") == 2 * 1024**3


def test_effective_mem_cpu_scaled(monkeypatch):
    monkeypatch.setattr(docker_shim, "_MEM_PER_CPU_GB", 2.0)                    # install() default
    assert docker_shim._effective_mem(None, 16 * 10**9) == 16 * 2 * 1024**3  # 2GiB/cpu default
    assert docker_shim._effective_mem(8 * 1024**3, 16 * 10**9) == 8 * 1024**3  # declared wins
    assert docker_shim._effective_mem(None, None) is None                      # no cpus -> none


def test_effective_mem_per_cpu_override(monkeypatch):
    # The per-CPU factor is set by install(container_mem_per_cpu_gb=...) — the
    # --mem-per-cpu-gb flag — into the module global, NOT an env var.
    monkeypatch.setattr(docker_shim, "_MEM_PER_CPU_GB", 4.0)
    assert docker_shim._effective_mem(None, 8 * 10**9) == 8 * 4 * 1024**3
    monkeypatch.setattr(docker_shim, "_MEM_PER_CPU_GB", 0.0)                    # 0 = opt out
    assert docker_shim._effective_mem(None, 8 * 10**9) is None


def test_run_declares_cpu_scaled_memory(client, monkeypatch):
    monkeypatch.setattr(docker_shim, "_MEM_PER_CPU_GB", 2.0)
    _run(["docker", "run", "-d", "--name", "c1", "--cpus", "16", "img"])
    created = client.containers.created[0]
    assert created["mem_limit"] == 16 * 2 * 1024**3   # 32 GiB, not the 4 GiB default
    assert created["nano_cpus"] == 16 * 10**9


def test_run_forwards_explicit_memory(client):
    _run(["docker", "run", "-d", "--name", "c1", "--cpus", "4", "--memory", "8g", "img"])
    assert client.containers.created[0]["mem_limit"] == 8 * 1024**3  # explicit wins over scaling


# ---- fleet reservation (opt-in) --------------------------------------------
_FLEET_ID = "xrlenv.fleet_id"
_FLEET_CPU = "xrlenv.fleet_cpu_request"
_FLEET_MEM = "xrlenv.fleet_mem_request"


def _install_fleet(**kw):
    """Fresh client + install with (optional) fleet params. Caller uninstalls."""
    c = FakeClient()
    docker_shim.install(client=c, name_prefix="p-", labels={"benchmark": "evoclaw"}, **kw)
    return c


def test_fleet_first_container_opens_rest_are_companions():
    """The first `docker run` of the process declares the whole footprint (the
    opener); every later container carries only fleet_id (companions)."""
    c = _install_fleet(
        fleet_id="xrl-123-", fleet_cpu_request=18.0,
        fleet_mem_request_bytes=40 * 1024**3,
    )
    try:
        # 1st: the agent container = opener → all three labels.
        _run(["docker", "run", "-d", "--name", "agent", "img"])
        opener = c.containers.created[0]["labels"]
        assert opener[_FLEET_ID] == "xrl-123-"
        assert float(opener[_FLEET_CPU]) == 18.0
        assert int(opener[_FLEET_MEM]) == 40 * 1024**3
        assert opener["benchmark"] == "evoclaw"  # base labels still merged

        # 2nd (detached eval) + 3rd (one-shot eval) = companions → fleet_id only.
        _run(["docker", "run", "-d", "--name", "eval1", "img"])
        _run(["docker", "run", "--rm", "img", "bash", "-c", "true"])
        for i in (1, 2):
            lbl = c.containers.created[i]["labels"]
            assert lbl[_FLEET_ID] == "xrl-123-"
            assert _FLEET_CPU not in lbl
            assert _FLEET_MEM not in lbl
    finally:
        docker_shim.uninstall()


def test_fleet_off_emits_no_fleet_labels():
    """Without fleet params the container labels are exactly the base labels —
    byte-for-byte the legacy behaviour (fleet is opt-in)."""
    c = _install_fleet()
    try:
        _run(["docker", "run", "-d", "--name", "c1", "img"])
        assert c.containers.created[0]["labels"] == {"benchmark": "evoclaw"}
    finally:
        docker_shim.uninstall()


def test_fleet_id_without_footprint_fails_loud():
    """A fleet_id with a missing footprint value is a hard error — no silent
    partial declaration."""
    c = FakeClient()
    with pytest.raises(docker_shim.DockerShimError, match="incomplete"):
        docker_shim.install(client=c, fleet_id="xrl-1-", fleet_cpu_request=18.0)
    with pytest.raises(docker_shim.DockerShimError, match="incomplete"):
        docker_shim.install(
            client=c, fleet_id="xrl-1-", fleet_mem_request_bytes=1,
        )


def test_fleet_opener_state_resets_on_uninstall():
    """Uninstall clears the 'fleet opened' flag so a re-install opens a fresh
    fleet (the first container is the opener again, not stuck as a companion)."""
    _install_fleet(
        fleet_id="xrl-a-", fleet_cpu_request=18.0, fleet_mem_request_bytes=1,
    )
    _run(["docker", "run", "-d", "--name", "agent", "img"])  # opens fleet a
    docker_shim.uninstall()

    c2 = _install_fleet(
        fleet_id="xrl-b-", fleet_cpu_request=20.0, fleet_mem_request_bytes=2,
    )
    try:
        _run(["docker", "run", "-d", "--name", "agent2", "img"])
        opener = c2.containers.created[0]["labels"]
        assert opener[_FLEET_ID] == "xrl-b-"
        assert _FLEET_CPU in opener  # opener again, footprint re-declared
    finally:
        docker_shim.uninstall()


# ---- exec belt-and-suspenders: a node exec error must not crash the caller ----
def test_exec_transient_node_error_returns_nonzero(client):
    """A non-cluster-loss node exec error (e.g. a docker-py demux 'N is not a valid
    stream' that survived the node-side resync-retry) must come back as a docker-style
    non-zero, NOT raise — otherwise EvoClaw's tag-watcher thread dies and the runner
    hangs on 'Waiting...' forever."""
    _run(["docker", "run", "-d", "--name", "c1", "img"])

    def boom(cmd, **kw):
        raise RuntimeError(
            "gRPC error UNKNOWN: node X: remote command "
            "ValueError: 53 is not a valid stream"
        )

    docker_shim._REGISTRY["c1"].exec_run = boom
    rc, _out, err = docker_shim._exec(["c1", "git", "rev-parse", "HEAD"])
    assert rc == 1                              # docker-style non-zero
    assert b"not a valid stream" in err         # surfaced, not silently swallowed


def test_exec_connectionerror_still_propagates(client):
    """Genuine cluster-loss (ConnectionError from _with_retry) must still propagate so
    EvoClaw's own eval-retry re-runs on a fresh container — the belt only absorbs
    node exec errors, never a real node loss."""
    _run(["docker", "run", "-d", "--name", "c2", "img"])

    def lost(cmd, **kw):
        raise ConnectionError("transient cluster loss on exec c2")

    docker_shim._REGISTRY["c2"].exec_run = lost
    with pytest.raises(ConnectionError):
        docker_shim._exec(["c2", "git", "rev-parse"])
