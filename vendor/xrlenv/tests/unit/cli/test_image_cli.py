"""Tests for ``xrlenv images`` and ``xrlenv warmup`` (Slice 6, spec 15)."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from xrlenv.backends.base import (
    ExecChunk,
    ImageRecord,
    NetworkPolicy,
    ResourceSpec,
    ResourceUsage,
    SandboxBackend,
    SandboxCapabilities,
    SandboxHandle,
    ServiceSpec,
    SnapshotID,
    TemplateRef,
)
from xrlenv.cli.commands import (
    cmd_images,
    cmd_stub_runtime_layer,
    cmd_warmup,
    stub_runtime_dockerfile_path,
)


class _FakeBackend(SandboxBackend):
    name = "fake"
    capabilities = SandboxCapabilities(
        supports_snapshot=False, supports_chainable_snapshot=False,
        live_state_captured=False, supports_port_forward=False,
        supports_gpu=False, isolation_class="container", fast_create_p50_ms=10,
    )

    def __init__(
        self,
        present: list[ImageRecord] | None = None,
        free_bytes: int = 100 * 1024**3,
    ) -> None:
        self._present = list(present or [])
        self._free = free_bytes
        self.pulled: list[str] = []
        self.pull_should_fail: set[str] = set()

    async def list_images(self, *, include_shared_size=False) -> list[ImageRecord]:
        return list(self._present)

    async def pull_image(self, image: str, *, timeout_s: float = 600.0) -> None:
        if image in self.pull_should_fail:
            raise RuntimeError(f"boom: {image}")
        self.pulled.append(image)

    async def free_disk_bytes(self) -> int:
        return self._free

    async def create(
        self, template: TemplateRef, resources: ResourceSpec, network_policy: NetworkPolicy,
    ) -> SandboxHandle:
        raise NotImplementedError

    async def destroy(self, sb: SandboxHandle) -> None:
        raise NotImplementedError

    def exec(self, sb: SandboxHandle, cmd: list[str], stdin: bytes | None = None,
             env: dict[str, str] | None = None, timeout_s: float | None = None) -> AsyncIterator[ExecChunk]:
        raise NotImplementedError

    async def read_file(self, sb: SandboxHandle, path: str) -> bytes:
        raise NotImplementedError

    async def write_file(self, sb: SandboxHandle, path: str, data: bytes) -> None:
        raise NotImplementedError

    async def put_archive(
        self, sb: SandboxHandle, target_dir: str, tarball: bytes, *, clean_target: bool = False,
    ) -> None:
        raise NotImplementedError

    def read_file_stream(self, sb: SandboxHandle, path: str) -> AsyncIterator[bytes]:
        raise NotImplementedError

    async def write_file_stream(self, sb: SandboxHandle, path: str, src: AsyncIterator[bytes]) -> None:
        raise NotImplementedError

    async def spawn_service(self, sb: SandboxHandle, spec: ServiceSpec) -> object:
        raise NotImplementedError

    async def spawn_services(self, sb: SandboxHandle, specs: list[ServiceSpec]) -> list[object]:
        raise NotImplementedError

    async def port_forward(self, sb: SandboxHandle, internal_port: int) -> str:
        raise NotImplementedError

    async def snapshot(self, sb: SandboxHandle) -> SnapshotID:
        raise NotImplementedError

    async def restore(self, snapshot: SnapshotID) -> SandboxHandle:
        raise NotImplementedError

    async def stats(self, sb: SandboxHandle) -> ResourceUsage:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# cmd_images
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_images_text_includes_all_images_and_marks_pinned(tmp_path: Path) -> None:
    pin_file = tmp_path / "pins.yaml"
    pin_file.write_text(yaml.safe_dump({"pins": ["foo:1"]}))
    backend = _FakeBackend(
        present=[
            ImageRecord(name="foo:1", size_bytes=2 * 1024**3),
            ImageRecord(name="bar:1", size_bytes=500 * 1024**2),
        ],
        free_bytes=42 * 1024**3,
    )
    out = io.StringIO()
    rc = cmd_images(pin_file=pin_file, output_format="text", out=out, backend=backend)
    assert rc == 0
    body = out.getvalue()
    assert "foo:1" in body and "bar:1" in body
    assert "yes" in body  # foo:1 is pinned
    assert "free_disk=42.0G" in body


def test_cmd_images_json_carries_full_record(tmp_path: Path) -> None:
    pin_file = tmp_path / "pins.yaml"
    pin_file.write_text(yaml.safe_dump({"pins": ["a:1"]}))
    backend = _FakeBackend(
        present=[
            ImageRecord(name="a:1", size_bytes=1024**3, digest="sha256:abc"),
            ImageRecord(name="b:1", size_bytes=2 * 1024**3),
        ],
        free_bytes=10 * 1024**3,
    )
    out = io.StringIO()
    rc = cmd_images(pin_file=pin_file, output_format="json", out=out, backend=backend)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["summary"]["total_count"] == 2
    assert payload["summary"]["pinned_count"] == 1
    assert payload["summary"]["free_disk_bytes"] == 10 * 1024**3
    by_name = {img["name"]: img for img in payload["images"]}
    assert by_name["a:1"]["pinned"] is True
    assert by_name["a:1"]["digest"] == "sha256:abc"
    assert by_name["b:1"]["pinned"] is False


def test_cmd_images_handles_missing_pin_file(tmp_path: Path) -> None:
    backend = _FakeBackend(
        present=[ImageRecord(name="solo:1", size_bytes=1024)],
    )
    out = io.StringIO()
    rc = cmd_images(
        pin_file=tmp_path / "missing.yaml", output_format="json",
        out=out, backend=backend,
    )
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["summary"]["pinned_count"] == 0
    assert payload["images"][0]["pinned"] is False


# ──────────────────────────────────────────────────────────────────────────────
# cmd_warmup
# ──────────────────────────────────────────────────────────────────────────────


def test_cmd_warmup_pulls_each_image_in_order() -> None:
    backend = _FakeBackend()
    out = io.StringIO()
    rc = cmd_warmup(
        ["a:1", "b:1", "c:1"], deadline_s=5.0, out=out, backend=backend,
    )
    assert rc == 0
    assert backend.pulled == ["a:1", "b:1", "c:1"]
    body = out.getvalue()
    for img in ("a:1", "b:1", "c:1"):
        assert f"pulled {img}" in body


def test_cmd_warmup_rejects_empty_image_list() -> None:
    backend = _FakeBackend()
    out = io.StringIO()
    rc = cmd_warmup([], out=out, backend=backend)
    assert rc == 2
    assert "at least one image" in out.getvalue()


def test_cmd_warmup_returns_nonzero_on_partial_failure() -> None:
    backend = _FakeBackend()
    backend.pull_should_fail = {"b:1"}
    out = io.StringIO()
    rc = cmd_warmup(["a:1", "b:1", "c:1"], out=out, backend=backend)
    assert rc == 1
    body = out.getvalue()
    assert "pulled a:1" in body
    assert "failed  b:1" in body
    assert "pulled c:1" in body


# ──────────────────────────────────────────────────────────────────────────────
# cmd_stub_runtime_layer (D12 stage 1 helper, three-stage build pipeline)
# ──────────────────────────────────────────────────────────────────────────────


def test_stub_runtime_dockerfile_ships_with_the_package() -> None:
    """The canonical Dockerfile snippet must be shipped under
    ``xrlenv/sandbox_stub/`` so plug-ins that call
    ``xrlenv stub-runtime layer`` find it via the platform install
    rather than via a fragile relative path."""
    snippet = stub_runtime_dockerfile_path()
    assert snippet.is_file(), f"missing snippet: {snippet}"
    body = snippet.read_text()
    # Smoke-check the contract — the snippet MUST take BASE_IMAGE
    # as a build-arg and pip-install the platform's three core
    # stub deps. If a future edit drops one of these, this test
    # catches it before plug-ins start failing at run time.
    assert "ARG BASE_IMAGE" in body
    assert "FROM ${BASE_IMAGE}" in body
    assert "pydantic" in body
    assert "aiohttp" in body
    assert "pyyaml" in body
    # Audit M1 (2026-04-29 follow-up): the layer must restore the
    # upstream image's USER at the bottom so installing as root for
    # apt+pip doesn't permanently flip the runtime user away from
    # whatever the upstream image author set.
    assert "ARG UPSTREAM_USER" in body
    assert "USER ${UPSTREAM_USER}" in body


def test_cmd_stub_runtime_layer_invokes_docker_build_with_correct_flags() -> None:
    """Builds the right ``docker build`` invocation: ``--build-arg
    BASE_IMAGE``, ``--file`` pointing at the platform snippet,
    ``--tag`` with the requested out tag, and a positional context
    that's the snippet's parent directory (so the Dockerfile's
    relative-path semantics still work).
    """
    captured: list[list[str]] = []

    def _fake_runner(cmd: list[str]) -> int:
        captured.append(list(cmd))
        return 0

    out = io.StringIO()
    rc = cmd_stub_runtime_layer(
        base="terminal-bench-2-base/fix-git:0.1",
        out_tag="terminal-bench-2/fix-git:0.1",
        runner=_fake_runner,
        upstream_user_resolver=lambda _img: "root",
        out=out,
    )
    assert rc == 0
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:2] == ["docker", "build"]
    # BASE_IMAGE build-arg threaded through.
    arg_indices = [i for i, v in enumerate(cmd) if v == "--build-arg"]
    arg_values = [cmd[i + 1] for i in arg_indices]
    assert "BASE_IMAGE=terminal-bench-2-base/fix-git:0.1" in arg_values
    # UPSTREAM_USER build-arg threaded through (audit M1 follow-up).
    assert "UPSTREAM_USER=root" in arg_values
    assert "--tag" in cmd
    tag_idx = cmd.index("--tag")
    assert cmd[tag_idx + 1] == "terminal-bench-2/fix-git:0.1"
    assert "--file" in cmd
    file_idx = cmd.index("--file")
    assert cmd[file_idx + 1] == str(stub_runtime_dockerfile_path())
    # Last positional arg is the build context; must be the
    # snippet's parent dir so any local files the snippet COPYs
    # would still resolve.
    assert cmd[-1] == str(stub_runtime_dockerfile_path().parent)


def test_cmd_stub_runtime_layer_threads_non_root_upstream_user() -> None:
    """When the base image declares a non-root ``Config.User``, the
    CLI passes it through as the ``UPSTREAM_USER`` build-arg so the
    Dockerfile's final ``USER ${UPSTREAM_USER}`` directive restores
    the upstream identity. This is the load-bearing piece of the
    audit-M1 fix: installing as root for apt+pip during the layer
    build is fine, but the runtime user must NOT be flipped to root
    for tasks that wanted a non-root agent."""
    captured: list[list[str]] = []
    out = io.StringIO()
    rc = cmd_stub_runtime_layer(
        base="custom/non-root-base:1",
        out_tag="custom/with-stub:1",
        runner=lambda c: (captured.append(list(c)), 0)[1],
        upstream_user_resolver=lambda _img: "sandbox",
        out=out,
    )
    assert rc == 0
    arg_values = [
        captured[0][i + 1] for i, v in enumerate(captured[0]) if v == "--build-arg"
    ]
    assert "UPSTREAM_USER=sandbox" in arg_values


def test_cmd_stub_runtime_layer_propagates_runner_exit_code() -> None:
    """Non-zero return from the runner (i.e. ``docker build`` failed)
    must propagate so the plug-in's build-task-images.sh can exit
    early instead of silently producing a half-built tag."""
    out = io.StringIO()
    rc = cmd_stub_runtime_layer(
        base="some-base:1", out_tag="some-out:1",
        runner=lambda _cmd: 7,
        upstream_user_resolver=lambda _img: "root",
        out=out,
    )
    assert rc == 7


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher wiring
# ──────────────────────────────────────────────────────────────────────────────


def test_dispatcher_routes_images_to_cmd_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.cli.__main__ as cli_module

    captured: dict[str, Any] = {}

    def _fake_images(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "cmd_images", _fake_images)
    pin_file = tmp_path / "pins.yaml"
    pin_file.write_text(yaml.safe_dump({"pins": []}))
    rc = cli_module.main(
        [
            "--state-db", str(tmp_path / "state.db"),
            "--runs-root", str(tmp_path / "runs"),
            "images", "--pin-file", str(pin_file), "--format", "json",
        ]
    )
    assert rc == 0
    assert captured["pin_file"] == pin_file
    assert captured["output_format"] == "json"


def test_dispatcher_routes_stub_runtime_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``xrlenv stub-runtime layer --base ... --out ...`` must reach
    ``cmd_stub_runtime_layer`` with the right kwargs."""
    import xrlenv.cli.__main__ as cli_module

    captured: dict[str, Any] = {}

    def _fake_layer(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "cmd_stub_runtime_layer", _fake_layer)
    rc = cli_module.main(
        [
            "--state-db", str(tmp_path / "state.db"),
            "--runs-root", str(tmp_path / "runs"),
            "stub-runtime", "layer",
            "--base", "terminal-bench-2-base/fix-git:0.1",
            "--out", "terminal-bench-2/fix-git:0.1",
        ]
    )
    assert rc == 0
    assert captured["base"] == "terminal-bench-2-base/fix-git:0.1"
    assert captured["out_tag"] == "terminal-bench-2/fix-git:0.1"


def test_dispatcher_routes_warmup_to_cmd_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.cli.__main__ as cli_module

    captured: dict[str, Any] = {}

    def _fake_warmup(images: list[str], **kwargs: Any) -> int:
        captured["images"] = images
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli_module, "cmd_warmup", _fake_warmup)
    rc = cli_module.main(
        [
            "--state-db", str(tmp_path / "state.db"),
            "--runs-root", str(tmp_path / "runs"),
            "warmup", "img1:1", "img2:2", "--deadline", "30.0",
        ]
    )
    assert rc == 0
    assert captured["images"] == ["img1:1", "img2:2"]
    assert captured["deadline_s"] == 30.0
