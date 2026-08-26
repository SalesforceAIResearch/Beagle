"""Tests for the manifest loader and TemplateCatalog (Slice 1).

Covers:
- happy-path load of the bundled hello-shell manifest
- byte-suffix parser (SI vs binary, units the spec mentions)
- digest is stable + content-keyed (invariant 4)
- invalid manifests raise ManifestInvalid
- round-trip catalog register / get / list
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from xrlenv.control.template_catalog import (
    TemplateCatalog,
    load_manifest,
    parse_bytes,
)
from xrlenv.errors import ManifestInvalid


def test_loads_hello_shell(hello_shell_manifest_path: Path) -> None:
    """The hello-shell manifest carries only contract fields — name,
    version, image, adapter, reward. Backend / network / resources /
    deadlines do not appear in the manifest; the loader fills runtime
    defaults so existing scheduler / capacity call-sites keep working
    while the rollout-time wire moves these to the run-config layer."""
    manifest = load_manifest(hello_shell_manifest_path)

    assert manifest.name == "hello-shell"
    assert manifest.version == "0.1"
    assert manifest.image == "xrlenv/hello-shell:0.1"
    assert manifest.env_adapter.module == "xrlenv.templates.hello_shell.adapter"
    assert manifest.env_adapter.class_name == "ShellEnvAdapter"
    assert manifest.reward.mode == "env_step"
    # backend / network are not manifest fields anymore — they're
    # user-side policy in the run-config.
    assert not hasattr(manifest, "backend")
    assert not hasattr(manifest, "network")
    # Resources defaulted by the loader.
    assert manifest.resources is not None
    assert manifest.resources.cpu_request > 0
    assert manifest.digest.startswith("sha256:")


def test_loads_manifest_with_resources_block(tmp_path: Path) -> None:
    """When a manifest does carry a resources: block (legacy / dev
    convenience), the loader honours its values verbatim instead of
    using platform defaults."""
    body = {
        "name": "with-resources",
        "image": "im:1",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "env_step"},
        "resources": {
            "cpu_request": 0.5,
            "cpu_limit": 2.0,
            "mem_request": "256Mi",
            "mem_limit": "512Mi",
            "disk_request": "1Gi",
        },
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(body))
    manifest = load_manifest(p)
    assert manifest.resources is not None
    assert manifest.resources.cpu_request == 0.5
    assert manifest.resources.cpu_limit == 2.0
    assert manifest.resources.mem_limit_bytes == 512 * 1024 * 1024


@pytest.mark.parametrize(
    "value,expected",
    [
        ("256Mi", 256 * 1024 * 1024),
        ("4Gi", 4 * 1024**3),
        ("1GB", 1_000_000_000),
        ("1KB", 1_000),
        ("1024", 1024),
        (1024, 1024),
        ("8GB", 8 * 1000**3),
        ("2Ti", 2 * 1024**4),
    ],
)
def test_parse_bytes(value: object, expected: int) -> None:
    assert parse_bytes(value) == expected


def test_digest_is_content_keyed(tmp_path: Path) -> None:
    body = {
        "name": "t",
        "image": "im:1",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "env_step"},
    }
    a = tmp_path / "a.yaml"
    a.write_text(yaml.safe_dump(body))
    m1 = load_manifest(a)

    body["resources"] = {"cpu_limit": 2.0}
    a.write_text(yaml.safe_dump(body))
    m2 = load_manifest(a)

    assert m1.digest != m2.digest
    assert m1.digest.startswith("sha256:")


def test_instances_index_path_anchored_to_manifest_dir(tmp_path: Path) -> None:
    """Audit M3: a relative ``instances.index_path`` in the manifest
    must be resolved against the manifest's directory at load time.
    Otherwise the resolver gets a path that depends on the operator's
    CWD when ``xrlenv up`` runs, which makes the manifest-local
    ``./tasks/`` a trap."""
    plugin_dir = tmp_path / "my_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "tasks").mkdir()
    body = {
        "name": "with-instances",
        "version": "0.1",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "env_step"},
        "instances": {
            "module": "m",
            "class": "R",
            "index_path": "./tasks/",
        },
    }
    p = plugin_dir / "manifest.yaml"
    p.write_text(yaml.safe_dump(body))
    manifest = load_manifest(p)
    assert manifest.instances is not None
    assert manifest.instances.index_path == str((plugin_dir / "tasks").resolve())


def test_instances_absolute_index_path_passes_through(tmp_path: Path) -> None:
    """An absolute ``index_path`` must not be rewritten by the loader."""
    abs_path = (tmp_path / "elsewhere" / "tasks").resolve()
    body = {
        "name": "abs-instances",
        "version": "0.1",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "env_step"},
        "instances": {
            "module": "m",
            "class": "R",
            "index_path": str(abs_path),
        },
    }
    p = tmp_path / "manifest.yaml"
    p.write_text(yaml.safe_dump(body))
    manifest = load_manifest(p)
    assert manifest.instances is not None
    assert manifest.instances.index_path == str(abs_path)


def test_network_field_in_manifest_rejected_with_pointer_to_run_config(tmp_path: Path) -> None:
    """``network`` is user-side policy now — the manifest no longer
    accepts it. Loader rejects with a precise pointer at the
    run-config so an author migrating from the old shape gets a
    clear "move it here" message."""
    body = {
        "name": "bad",
        "image": "im:1",
        "network": "open",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "env_step"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(body))
    with pytest.raises(ManifestInvalid, match="run-config"):
        load_manifest(p)


def test_backend_field_in_manifest_rejected_with_pointer_to_run_config(tmp_path: Path) -> None:
    """Same as above, for ``backend``. The plug-in author doesn't
    pick the sandbox runtime — the user does."""
    body = {
        "name": "bad",
        "image": "im:1",
        "backend": "docker",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "env_step"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(body))
    with pytest.raises(ManifestInvalid, match="run-config"):
        load_manifest(p)


def test_invalid_reward_mode_rejected(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "image": "im:1",
        "env_adapter": {"module": "m", "class": "C"},
        "reward": {"mode": "ranked-by-vibes"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(body))
    with pytest.raises(ManifestInvalid):
        load_manifest(p)


def test_missing_required_field_rejected(tmp_path: Path) -> None:
    body = {"name": "x"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(body))
    with pytest.raises(ManifestInvalid):
        load_manifest(p)


# ──────────────────────────────────────────────────────────────────────────────
# Audit M2: validate_overlay must run the same security checks on a
# post-resolver Pattern-A overlay that register() runs at load time.
# Pattern-A manifests skip mount allowlist + image digest pinning at
# registration because their image/mounts come from the resolver. Once
# the resolver supplies those values, validate_overlay re-runs the
# checks so resolver-provided unpinned images / denied mounts can't slip
# through.
# ──────────────────────────────────────────────────────────────────────────────


def test_validate_overlay_rejects_denied_mount() -> None:
    from xrlenv.backends.base import MountSpec, ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        MountDenied,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    # An overlaid manifest with a mount the catalog's allowlist refuses.
    overlay = TemplateManifest(
        name="bench",
        version="0.1",
        digest="sha256:" + "00" * 32,
        image="ghcr.io/me/bench@sha256:" + "ab" * 32,
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
            mounts=(
                MountSpec(host_path="/etc/passwd", sandbox_path="/x", readonly=True),
            ),
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )
    cat = TemplateCatalog()
    with pytest.raises(MountDenied, match="/etc"):
        cat.validate_overlay(overlay)


def test_validate_overlay_logs_unpinned_resolver_image(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When a Pattern-A resolver returns an image without a digest, the
    overlay validator should warn (matching the register-time
    contract) so the operator notices."""
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    overlay = TemplateManifest(
        name="bench",
        version="0.1",
        digest="sha256:" + "00" * 32,
        image="ghcr.io/me/bench:tag-only",  # not pinned
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    audit_calls: list[tuple[str, dict]] = []
    cat = TemplateCatalog(audit_callback=lambda k, p: audit_calls.append((k, p)))
    with caplog.at_level("WARNING"):
        result = cat.validate_overlay(overlay)
    assert result.image == "ghcr.io/me/bench:tag-only"   # unchanged when no resolver
    assert any(k == "template.image_unpinned" for k, _ in audit_calls)


def test_resolver_returning_none_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the digest resolver explicitly returns ``None`` (e.g. for a
    locally-built per-node image where there's no registry-resolvable
    digest), the catalog must NOT log at WARNING level — the
    resolver did the right thing and the operator has no action to
    take. Pre-fix the catalog warned for this case despite its own
    warning text saying "or None" was acceptable; the noise polluted
    every Pattern-A rollout's startup log.

    The audit event still fires (spec-19 needs the unpinned record)
    with ``reason="resolver_returned_none"`` so post-fact analysis
    can distinguish "intentional None" from "real resolver bug".
    """
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    overlay = TemplateManifest(
        name="bench-local",
        version="0.1",
        digest="sha256:" + "00" * 32,
        image="terminal-bench-2/build-pov-ray:0.1",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    audit_calls: list[tuple[str, dict]] = []

    def resolver_returns_none(_image: str) -> str | None:
        return None

    cat = TemplateCatalog(
        digest_resolver=resolver_returns_none,
        audit_callback=lambda k, p: audit_calls.append((k, p)),
    )
    with caplog.at_level("WARNING"):
        result = cat.validate_overlay(overlay)

    # No WARNING records — None is the intentional escape hatch.
    warning_records = [
        r for r in caplog.records
        if r.levelname == "WARNING" and r.name == "xrlenv.control.template_catalog"
    ]
    assert warning_records == [], (
        f"resolver returning None should not WARN; got {warning_records}"
    )
    # Image stays tag-form since no registry digest is available.
    assert result.image == "terminal-bench-2/build-pov-ray:0.1"
    # Audit event still fires for spec-19 accounting.
    assert any(
        k == "template.image_unpinned" and p.get("reason") == "resolver_returned_none"
        for k, p in audit_calls
    )


def test_resolver_returning_malformed_digest_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A resolver bug — returning something that isn't None and
    isn't a sha256 digest — must surface at WARNING level. That's
    the actionable case the warning was originally designed for
    and is what the demote-None-to-DEBUG fix preserved.
    """
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    overlay = TemplateManifest(
        name="bench-malformed",
        version="0.1",
        digest="sha256:" + "00" * 32,
        image="ghcr.io/me/bench:0.1",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    audit_calls: list[tuple[str, dict]] = []

    def resolver_returns_garbage(_image: str) -> str | None:
        return "not-a-digest"

    cat = TemplateCatalog(
        digest_resolver=resolver_returns_garbage,
        audit_callback=lambda k, p: audit_calls.append((k, p)),
    )
    with caplog.at_level("WARNING"):
        cat.validate_overlay(overlay)

    warning_records = [
        r for r in caplog.records
        if r.levelname == "WARNING" and r.name == "xrlenv.control.template_catalog"
    ]
    assert len(warning_records) == 1, (
        f"malformed resolver return should WARN exactly once; "
        f"got {len(warning_records)}"
    )
    assert "not-a-digest" in warning_records[0].getMessage()
    # Audit event distinguishes this from the legitimate-None case.
    assert any(
        p.get("reason") == "resolver_returned_malformed_digest"
        for _k, p in audit_calls
    )


def test_catalog_register_get_list(hello_shell_manifest_path: Path) -> None:
    cat = TemplateCatalog()
    cat.register(load_manifest(hello_shell_manifest_path))

    assert cat.get("hello-shell").image == "xrlenv/hello-shell:0.1"
    assert [m.name for m in cat.list()] == ["hello-shell"]
    with pytest.raises(KeyError):
        cat.get("does-not-exist")


def test_catalog_register_dir(package_root: Path) -> None:
    cat = TemplateCatalog()
    registered = cat.register_dir(package_root / "templates")
    names = sorted(m.name for m in registered)
    assert "hello-shell" in names


# ── A1 / D20 (P1.2) — image_pin_mode ─────────────────────────────────────────


def test_manifest_default_image_pin_mode_is_registry_digest(
    hello_shell_manifest_path: Path,
) -> None:
    """A manifest without an explicit ``image_pin_mode`` declaration
    defaults to ``registry_digest`` (backward-compat with phase-0
    manifests; central pinning happens when a resolver is wired)."""
    from xrlenv.control.template_catalog import load_manifest

    manifest = load_manifest(hello_shell_manifest_path)
    assert manifest.image_pin_mode == "registry_digest"


def test_loader_rejects_invalid_image_pin_mode(tmp_path: Path) -> None:
    """The loader is the only entry point that parses raw YAML; an
    unknown ``image_pin_mode`` value must raise rather than be
    silently accepted (a typo would otherwise silently change the
    security posture for a benchmark)."""
    from xrlenv.control.template_catalog import load_manifest
    from xrlenv.errors import ManifestInvalid

    yaml_path = tmp_path / "manifest.yaml"
    yaml_path.write_text(
        "name: bad\n"
        'version: "0.1"\n'
        "image: ghcr.io/me/bad:tag\n"
        "image_pin_mode: not_a_mode\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n"
    )
    with pytest.raises(ManifestInvalid, match="image_pin_mode"):
        load_manifest(yaml_path)


def test_register_skips_central_pinning_when_per_node_local(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A manifest declaring ``image_pin_mode='per_node_local'`` must
    NOT have its tag rewritten to ``image@sha256:...`` even when a
    digest_resolver is wired. The control plane has no authoritative
    view of per-node-built bytes; pinning to whatever digest the
    control-plane-side daemon happens to produce would mismatch every
    other node's local digest. Audit response (D20) — closes the
    buildx local-only ``RepoDigests`` trap by construction.

    The audit trail still records the registration with
    ``digest_source='per_node'`` so spec 19 can account for the
    unpinned manifest honestly.
    """
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    manifest = TemplateManifest(
        name="bench",
        version="0.1",
        digest="sha256:" + "11" * 32,
        image="locally-built/bench:0.1",
        image_pin_mode="per_node_local",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    # A digest_resolver IS wired — but the per_node_local mode must
    # short-circuit before it gets called.
    resolver_calls: list[str] = []

    def fake_resolver(image: str) -> str:
        resolver_calls.append(image)
        return "sha256:" + "ff" * 32  # would-be central digest

    audit_calls: list[tuple[str, dict[str, object]]] = []
    cat = TemplateCatalog(
        digest_resolver=fake_resolver,
        audit_callback=lambda k, p: audit_calls.append((k, p)),
    )
    registered = cat.register(manifest)

    # Image stays in tag form — no @sha256 rewrite.
    assert registered.image == "locally-built/bench:0.1"
    # Resolver was not called.
    assert resolver_calls == []
    # Audit event records the per-node-local mode honestly.
    unpinned = [p for k, p in audit_calls if k == "template.image_unpinned"]
    assert len(unpinned) == 1
    assert unpinned[0]["image_pin_mode"] == "per_node_local"
    assert unpinned[0]["digest_source"] == "per_node"
    assert unpinned[0]["reason"] == "image_pin_mode"


def test_register_still_pins_when_registry_digest_default() -> None:
    """The default ``registry_digest`` mode keeps the phase-0
    central-pinning behavior — the resolver gets called, the image is
    rewritten to ``image@sha256:...``. Pin this so a future refactor
    that changes the default doesn't silently flip the security
    posture for every existing manifest.
    """
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    manifest = TemplateManifest(
        name="bench",
        version="0.1",
        digest="sha256:" + "22" * 32,
        image="ghcr.io/me/bench:tag",
        # image_pin_mode default = "registry_digest"
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    fake_digest = "sha256:" + "ab" * 32

    def fake_resolver(image: str) -> str:
        return fake_digest

    cat = TemplateCatalog(digest_resolver=fake_resolver)
    registered = cat.register(manifest)
    assert registered.image == f"ghcr.io/me/bench@{fake_digest}"


def test_register_skips_central_pinning_when_shared_storage() -> None:
    """``shared_storage`` mode is reserved for phase-2 NFS/S3 mounts;
    the catalog treats it like ``per_node_local`` for now (no central
    pinning). Pin the contract so phase-2's shared-storage layer can
    later add proper content-hash verification without re-introducing
    central digest rewrites."""
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )

    manifest = TemplateManifest(
        name="bench",
        version="0.1",
        digest="sha256:" + "33" * 32,
        image="shared/bench:0.1",
        image_pin_mode="shared_storage",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    audit_calls: list[tuple[str, dict[str, object]]] = []
    cat = TemplateCatalog(
        digest_resolver=lambda _: "sha256:" + "ff" * 32,
        audit_callback=lambda k, p: audit_calls.append((k, p)),
    )
    registered = cat.register(manifest)

    assert registered.image == "shared/bench:0.1"
    unpinned = [p for k, p in audit_calls if k == "template.image_unpinned"]
    assert len(unpinned) == 1
    assert unpinned[0]["image_pin_mode"] == "shared_storage"
    assert unpinned[0]["digest_source"] == "shared_storage"


def test_resolved_instance_image_pin_mode_overrides_manifest() -> None:
    """A1 / D20 (P1.2) — Pattern A resolver may override the outer
    manifest's image_pin_mode for a single instance (one task pulls
    from a registry while the rest are per-node-built). Pin the
    overlay-propagation contract.
    """
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.instance_resolver import (
        ResolvedInstance,
        apply_to_manifest,
    )
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateManifest,
    )

    manifest = TemplateManifest(
        name="bench",
        version="0.1",
        digest="sha256:" + "44" * 32,
        image_pin_mode="per_node_local",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    # Resolver returns an instance pointing at a registry-pinned image.
    resolved = ResolvedInstance(
        instance_id="task-x",
        image="ghcr.io/me/registry-pinned-task:1@sha256:" + "ab" * 32,
        image_pin_mode="registry_digest",
    )
    overlaid = apply_to_manifest(manifest, resolved)
    assert overlaid.image_pin_mode == "registry_digest"
    assert overlaid.image == resolved.image

    # Resolver leaves image_pin_mode unset → overlay keeps the
    # manifest-level value.
    resolved_default = ResolvedInstance(
        instance_id="task-y", image="locally-built/y:0.1",
    )
    overlaid_default = apply_to_manifest(manifest, resolved_default)
    assert overlaid_default.image_pin_mode == "per_node_local"


# ── scratch-registry build-on-demand (image_build / scratch_build) ────────────
# notes/scratch-registry-build-on-demand.md — slice 2a (control-plane manifest
# support for bring-your-own-Dockerfile).


def _write_manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(body)
    return p


def test_image_build_block_implies_scratch_build(tmp_path: Path) -> None:
    """An ``image_build`` block sets ``image_pin_mode='scratch_build'``
    without the author hand-editing the pin mode, and leaves ``image``
    unset (the scratch ref is derived at first build)."""
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image_build:\n"
        "  context: ./environment\n"
        "  build_args: { FOO: bar }\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    manifest = load_manifest(path)
    assert manifest.image_pin_mode == "scratch_build"
    assert manifest.image is None
    assert manifest.image_build is not None
    # context is anchored to the manifest dir (absolute) at load time.
    assert manifest.image_build.context == str((tmp_path / "environment").resolve())
    assert manifest.image_build.build_args == {"FOO": "bar"}


def test_image_build_git_variant(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image_build:\n"
        "  git:\n"
        "    repo: https://x/y\n"
        "    ref: abc123\n"
        "    subdir: env\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    manifest = load_manifest(path)
    assert manifest.image_pin_mode == "scratch_build"
    assert manifest.image_build is not None
    assert manifest.image_build.git is not None
    assert manifest.image_build.git.ref == "abc123"


def test_image_build_conflicting_pin_mode_rejected(tmp_path: Path) -> None:
    """An ``image_build`` block with an explicit non-scratch pin mode is a
    contradiction the loader must reject."""
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image_pin_mode: per_node_local\n"
        "image_build:\n"
        "  context: ./environment\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    with pytest.raises(ManifestInvalid, match="scratch_build"):
        load_manifest(path)


def test_scratch_build_without_image_build_rejected(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image: some/ref:0.1\n"
        "image_pin_mode: scratch_build\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    with pytest.raises(ManifestInvalid, match="image_build"):
        load_manifest(path)


def test_image_build_malformed_rejected(tmp_path: Path) -> None:
    """context + git together is invalid (ImageBuildSpec requires exactly one)."""
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image_build:\n"
        "  context: ./environment\n"
        "  git:\n"
        "    repo: https://x/y\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    with pytest.raises(ManifestInvalid, match="image_build"):
        load_manifest(path)


def test_register_skips_central_pinning_when_scratch_build() -> None:
    """A ``scratch_build`` manifest is not centrally pinned at register time
    (the manifest digest isn't known until the first build); the audit event
    records ``digest_source='scratch'``."""
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )
    from xrlenv.image_build import ImageBuildSpec

    manifest = TemplateManifest(
        name="byod",
        version="0.1",
        digest="sha256:" + "33" * 32,
        image="scratchhost:5012/scratch/deadbeef",
        image_build=ImageBuildSpec(context="./environment"),
        image_pin_mode="scratch_build",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    resolver_calls: list[str] = []
    audit_calls: list[tuple[str, dict[str, object]]] = []
    cat = TemplateCatalog(
        digest_resolver=lambda image: (resolver_calls.append(image) or "sha256:" + "ff" * 32),
        audit_callback=lambda k, p: audit_calls.append((k, p)),
    )
    registered = cat.register(manifest)

    assert registered.image == "scratchhost:5012/scratch/deadbeef"  # no @sha256 rewrite
    assert resolver_calls == []  # resolver never consulted
    unpinned = [p for k, p in audit_calls if k == "template.image_unpinned"]
    assert len(unpinned) == 1
    assert unpinned[0]["image_pin_mode"] == "scratch_build"
    assert unpinned[0]["digest_source"] == "scratch"


# ── additional scratch_build / image_build tests ──────────────────────────────


def test_image_build_explicit_scratch_build_pin_mode_accepted(tmp_path: Path) -> None:
    """An ``image_build`` block paired with an *explicit*
    ``image_pin_mode: scratch_build`` is redundant but valid: the loader
    would default to scratch_build anyway, so an explicit matching
    declaration must not be rejected as a conflict."""
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image_pin_mode: scratch_build\n"
        "image_build:\n"
        "  context: ./environment\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    manifest = load_manifest(path)
    assert manifest.image_pin_mode == "scratch_build"
    assert manifest.image_build is not None
    assert manifest.image is None


def test_scratch_build_image_none_register_does_not_call_resolver() -> None:
    """The canonical scratch_build manifest has no ``image`` set (the
    content-addressed scratch ref is unknown until the first build).
    Registering such a manifest must NOT invoke the digest resolver and
    must NOT emit ``template.image_unpinned`` — ``_maybe_pin_image`` short-
    circuits at the ``image is None`` guard before reaching either path.
    The ``template.registered`` event IS emitted with ``image=None``."""
    from xrlenv.backends.base import ResourceSpec
    from xrlenv.control.template_catalog import (
        EnvAdapterDecl,
        RewardContract,
        TemplateCatalog,
        TemplateManifest,
    )
    from xrlenv.image_build import ImageBuildSpec

    manifest = TemplateManifest(
        name="byod-no-image",
        version="0.1",
        digest="sha256:" + "55" * 32,
        image=None,
        image_build=ImageBuildSpec(context="./environment"),
        image_pin_mode="scratch_build",
        resources=ResourceSpec(
            cpu_request=1.0, cpu_limit=1.0,
            mem_request_bytes=1, mem_limit_bytes=1,
            disk_request_bytes=1,
        ),
        env_adapter=EnvAdapterDecl(module="m", class_name="C"),
        reward=RewardContract(mode="env_step"),
    )

    resolver_calls: list[str] = []
    audit_calls: list[tuple[str, dict]] = []
    cat = TemplateCatalog(
        digest_resolver=lambda image: (resolver_calls.append(image) or "sha256:" + "ff" * 32),  # type: ignore[func-returns-value]
        audit_callback=lambda k, p: audit_calls.append((k, p)),
    )
    registered = cat.register(manifest)

    assert registered.image is None
    assert resolver_calls == [], "resolver must not be called when image=None"
    unpinned_kinds = [k for k, _ in audit_calls if k == "template.image_unpinned"]
    assert unpinned_kinds == [], (
        "template.image_unpinned must not fire when image=None (nothing to pin)"
    )
    registered_events = [p for k, p in audit_calls if k == "template.registered"]
    assert len(registered_events) == 1
    assert registered_events[0]["image"] is None
    assert registered_events[0]["pinned_by_digest"] is False


def test_image_build_non_dict_in_yaml_is_rejected(tmp_path: Path) -> None:
    """``image_build: not_a_mapping`` in YAML must raise ManifestInvalid
    with a message that names ``image_build``, not crash with an internal
    Pydantic error.  The loader has an explicit isinstance check for this."""
    path = _write_manifest(
        tmp_path,
        "name: byod\n"
        'version: "0.1"\n'
        "image_build: not_a_mapping\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    with pytest.raises(ManifestInvalid, match="image_build"):
        load_manifest(path)


def test_image_build_relative_context_anchored_to_manifest_dir(tmp_path: Path) -> None:
    """A relative image_build.context is resolved to an absolute path against
    the manifest's directory at load time, so the scratch build finds it
    regardless of the process CWD."""
    sub = tmp_path / "bench"
    (sub / "environment").mkdir(parents=True)
    (sub / "environment" / "Dockerfile").write_text("FROM busybox\n")
    path = sub / "manifest.yaml"
    path.write_text(
        "name: byod\n"
        'version: "0.1"\n'
        "image_build:\n"
        "  context: ./environment\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    manifest = load_manifest(path)
    assert manifest.image_build is not None
    assert manifest.image_build.context == str((sub / "environment").resolve())


def test_image_build_absolute_context_left_unchanged(tmp_path: Path) -> None:
    abs_ctx = str((tmp_path / "already" / "abs").resolve())
    path = tmp_path / "manifest.yaml"
    path.write_text(
        "name: byod\n"
        'version: "0.1"\n'
        "image_build:\n"
        f"  context: {abs_ctx}\n"
        "env_adapter:\n"
        "  module: m\n"
        "  class: C\n"
        "reward:\n"
        "  mode: env_step\n",
    )
    manifest = load_manifest(path)
    assert manifest.image_build is not None
    assert manifest.image_build.context == abs_ctx
