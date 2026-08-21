"""build-plan.yaml schema + plan_id stability (P1.6.b + P1.7.C.2)."""

from __future__ import annotations

import pytest
import yaml
from xrlenv.control.build_plan import (
    BenchmarkBuildSpec,
    BenchmarkSelection,
    BuildBudget,
    BuildEntry,
    BuildPlan,
    EntryPlacement,
    GitSource,
    LocalSource,
    RegistrySource,
    TarballSource,
    compute_plan_id,
    load_build_plan,
)
from xrlenv.errors import ManifestInvalid


def test_selection_smoke_only() -> None:
    sel = BenchmarkSelection(smoke=True)
    assert sel.to_kwargs() == {"smoke": True}


def test_selection_instances_only() -> None:
    sel = BenchmarkSelection(instances=("a", "b"))
    assert sel.to_kwargs() == {"instances": ["a", "b"]}


def test_selection_all_only() -> None:
    sel = BenchmarkSelection(all=True)
    assert sel.to_kwargs() == {"all": True}


def test_selection_rejects_multiple() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BenchmarkSelection(smoke=True, all=True)


def test_selection_rejects_none() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BenchmarkSelection()


def test_build_plan_basic() -> None:
    plan = BuildPlan(
        replication=2,
        benchmarks=(
            BenchmarkBuildSpec(
                name="swebench-verified",
                selection=BenchmarkSelection(smoke=True),
                build_path="pull-and-retag",
            ),
        ),
    )
    assert plan.replication == 2
    assert plan.replication_for("swebench-verified") == 2


def test_build_plan_per_benchmark_replication_overrides_default() -> None:
    plan = BuildPlan(
        replication=1,
        benchmarks=(
            BenchmarkBuildSpec(
                name="a", selection=BenchmarkSelection(smoke=True),
                replication=3,
            ),
            BenchmarkBuildSpec(
                name="b", selection=BenchmarkSelection(smoke=True),
            ),
        ),
    )
    assert plan.replication_for("a") == 3
    assert plan.replication_for("b") == 1


def test_build_plan_rejects_duplicate_benchmark() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildPlan(
            benchmarks=(
                BenchmarkBuildSpec(
                    name="a", selection=BenchmarkSelection(smoke=True),
                ),
                BenchmarkBuildSpec(
                    name="a", selection=BenchmarkSelection(all=True),
                ),
            ),
        )


def test_build_plan_rejects_zero_replication() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildPlan(
            replication=0,
            benchmarks=(
                BenchmarkBuildSpec(
                    name="a", selection=BenchmarkSelection(smoke=True),
                ),
            ),
        )


def test_build_plan_rejects_empty_benchmarks() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildPlan(benchmarks=())


def test_compute_plan_id_stable_across_field_order(tmp_path) -> None:
    """Same plan content → same plan_id regardless of YAML key order."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({
        "version": 1,
        "replication": 1,
        "benchmarks": [{
            "name": "swebench-verified",
            "selection": {"smoke": True},
            "build_path": "pull-and-retag",
        }],
    }))
    b.write_text(yaml.safe_dump({
        "benchmarks": [{
            "build_path": "pull-and-retag",
            "selection": {"smoke": True},
            "name": "swebench-verified",
        }],
        "replication": 1,
        "version": 1,
    }))
    pa = load_build_plan(a)
    pb = load_build_plan(b)
    assert compute_plan_id(pa) == compute_plan_id(pb)


def test_compute_plan_id_excludes_name(tmp_path) -> None:
    """Renaming a plan must not change its plan_id — name is
    operator-facing metadata, not content."""
    p1 = BuildPlan(
        name="alpha",
        replication=1,
        benchmarks=(BenchmarkBuildSpec(
            name="a", selection=BenchmarkSelection(smoke=True),
        ),),
    )
    p2 = BuildPlan(
        name="beta",
        replication=1,
        benchmarks=(BenchmarkBuildSpec(
            name="a", selection=BenchmarkSelection(smoke=True),
        ),),
    )
    p3 = BuildPlan(
        # name omitted entirely
        replication=1,
        benchmarks=(BenchmarkBuildSpec(
            name="a", selection=BenchmarkSelection(smoke=True),
        ),),
    )
    assert compute_plan_id(p1) == compute_plan_id(p2) == compute_plan_id(p3)


def test_build_plan_name_optional() -> None:
    plan = BuildPlan(
        replication=1,
        benchmarks=(BenchmarkBuildSpec(
            name="a", selection=BenchmarkSelection(smoke=True),
        ),),
    )
    assert plan.name is None


def test_build_plan_name_round_trips_through_yaml(tmp_path) -> None:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({
        "version": 1,
        "name": "my-custom-plan",
        "replication": 1,
        "entries": [{
            "image_ref": "foo:1",
            "context_source": {"type": "registry"},
            "placement": {"size_hint_bytes": 100},
        }],
    }))
    plan = load_build_plan(p)
    assert plan.name == "my-custom-plan"


def test_compute_plan_id_changes_with_content(tmp_path) -> None:
    p1 = BuildPlan(
        replication=1,
        benchmarks=(
            BenchmarkBuildSpec(
                name="a", selection=BenchmarkSelection(smoke=True),
            ),
        ),
    )
    p2 = BuildPlan(
        replication=2,
        benchmarks=(
            BenchmarkBuildSpec(
                name="a", selection=BenchmarkSelection(smoke=True),
            ),
        ),
    )
    assert compute_plan_id(p1) != compute_plan_id(p2)


def test_load_build_plan_roundtrip(tmp_path) -> None:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({
        "version": 1,
        "replication": 2,
        "budget": {
            "reserved_runtime_gb": 40,
            "buffer_gb": 15,
            "cap_per_node_gb": 200,
        },
        "benchmarks": [
            {
                "name": "swebench-verified",
                "selection": {"instances": ["a", "b"]},
                "build_path": "pull-and-retag",
                "replication": 3,
            },
        ],
    }))
    plan = load_build_plan(p)
    assert plan.replication == 2
    assert plan.budget == BuildBudget(
        reserved_runtime_gb=40, buffer_gb=15, cap_per_node_gb=200,
    )
    assert plan.benchmarks[0].selection.instances == ("a", "b")
    assert plan.replication_for("swebench-verified") == 3


def test_load_build_plan_missing_file(tmp_path) -> None:
    with pytest.raises(ManifestInvalid, match="not found"):
        load_build_plan(tmp_path / "nope.yaml")


def test_load_build_plan_rejects_non_mapping(tmp_path) -> None:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(ManifestInvalid, match="must be a mapping"):
        load_build_plan(p)


def test_load_build_plan_rejects_malformed(tmp_path) -> None:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({
        "version": 1,
        "replication": "two",  # invalid type
        "benchmarks": [],
    }))
    with pytest.raises(ManifestInvalid, match="malformed"):
        load_build_plan(p)


# ────────────────────────────────────────────────────────────────────────────
# Per-image-ref schema (P1.7.C.2)
# ────────────────────────────────────────────────────────────────────────────


def _registry_entry(image_ref: str, *, size_hint_bytes: int = 1_000_000_000) -> BuildEntry:
    return BuildEntry(
        image_ref=image_ref,
        context_source=RegistrySource(),
        placement=EntryPlacement(
            preferred_home_count=1,
            size_hint_bytes=size_hint_bytes,
            size_hint_source="registry-probe",
        ),
    )


def test_per_image_ref_basic() -> None:
    plan = BuildPlan(entries=(_registry_entry("foo:bar"),))
    assert plan.is_per_image_ref()
    assert len(plan.entries) == 1
    assert plan.entries[0].pinned is False
    assert plan.entries[0].priority == 0
    assert plan.entries[0].labels == {}


def test_per_image_ref_discriminated_union_loads_each_source(tmp_path) -> None:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({
        "version": 1,
        "entries": [
            {
                "image_ref": "alex/fix:1",
                "context_source": {"type": "registry"},
                "placement": {"size_hint_bytes": 100, "size_hint_source": "registry-probe"},
            },
            {
                "image_ref": "xrlenv-seta-env/0:main",
                "context_source": {
                    "type": "git",
                    "repo": "https://github.com/camel-ai/seta-env",
                    "ref": "main",
                    "subdir": "Harbor-Dataset/0/environment",
                    "dockerfile": "Dockerfile",
                },
                "placement": {"size_hint_bytes": 200},
            },
            {
                "image_ref": "xrlenv-tarball/foo:1",
                "context_source": {
                    "type": "tarball",
                    "path": "/tmp/ctx.tar.gz",
                    "dockerfile": "Dockerfile",
                },
                "placement": {"size_hint_bytes": 300},
            },
            {
                "image_ref": "turing-tb2/abs-mex-service:main",
                "context_source": {
                    "type": "local",
                    "path": "/path/to/data",
                    "dockerfile": "Dockerfile",
                    "shared_fs": "hyperpod",
                },
                "placement": {"size_hint_bytes": 400},
            },
        ],
    }))
    plan = load_build_plan(p)
    assert plan.is_per_image_ref()
    assert isinstance(plan.entries[0].context_source, RegistrySource)
    assert isinstance(plan.entries[1].context_source, GitSource)
    assert isinstance(plan.entries[2].context_source, TarballSource)
    assert isinstance(plan.entries[3].context_source, LocalSource)
    assert plan.entries[1].context_source.repo == "https://github.com/camel-ai/seta-env"
    assert plan.entries[2].context_source.path == "/tmp/ctx.tar.gz"
    assert plan.entries[3].context_source.path.endswith("/environment")
    assert plan.entries[3].context_source.shared_fs == "hyperpod"


def test_local_source_requires_shared_fs() -> None:
    """A local path is non-portable, so shared_fs (the cluster-shared-fs
    assertion) is mandatory — omitting it is a validation error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LocalSource(path="/path/to/data")  # type: ignore[call-arg]


def test_local_source_rejects_empty_shared_fs() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="shared_fs"):
        LocalSource(path="/path/to/data", shared_fs="")


def test_local_source_rejects_empty_path() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="path"):
        LocalSource(path="", shared_fs="hyperpod")


def test_local_source_path_is_part_of_plan_id() -> None:
    """Unlike tarball (whose content_b64 is stripped), a local source's path is
    part of plan identity — two plans differing only in path hash differently."""
    def _plan(path: str) -> BuildPlan:
        return BuildPlan(entries=(BuildEntry(
            image_ref="ns/a:main",
            context_source=LocalSource(path=path, shared_fs="hyperpod"),
            placement=EntryPlacement(size_hint_bytes=1, size_hint_source="heuristic"),
        ),))

    assert compute_plan_id(_plan("/a/environment")) != compute_plan_id(
        _plan("/b/environment"),
    )


def test_per_image_ref_rejects_duplicate_image_ref() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildPlan(entries=(
            _registry_entry("dupe:tag"),
            _registry_entry("dupe:tag"),
        ))


def test_per_image_ref_rejects_unknown_context_type(tmp_path) -> None:
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump({
        "version": 1,
        "entries": [{
            "image_ref": "foo:1",
            "context_source": {"type": "smb"},  # not a real source
            "placement": {"size_hint_bytes": 100},
        }],
    }))
    with pytest.raises(ManifestInvalid, match="malformed"):
        load_build_plan(p)


def test_plan_rejects_mixing_benchmarks_and_entries() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildPlan(
            benchmarks=(BenchmarkBuildSpec(
                name="a", selection=BenchmarkSelection(smoke=True),
            ),),
            entries=(_registry_entry("foo:1"),),
        )


def test_plan_rejects_empty_when_neither_shape_set() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildPlan()


def test_entry_placement_rejects_zero_home_count() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EntryPlacement(preferred_home_count=0, size_hint_bytes=100)


def test_entry_placement_rejects_negative_size_hint() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EntryPlacement(preferred_home_count=1, size_hint_bytes=-1)


def test_compute_plan_id_per_image_ref_stable(tmp_path) -> None:
    """Two YAMLs with the same content but different key order have
    identical plan_ids — covers the per-image-ref shape too."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({
        "version": 1,
        "entries": [{
            "image_ref": "foo:1",
            "context_source": {"type": "registry"},
            "placement": {
                "preferred_home_count": 2,
                "size_hint_bytes": 100,
                "size_hint_source": "registry-probe",
            },
            "pinned": True,
            "priority": 5,
        }],
    }))
    b.write_text(yaml.safe_dump({
        "entries": [{
            "priority": 5,
            "pinned": True,
            "placement": {
                "size_hint_source": "registry-probe",
                "size_hint_bytes": 100,
                "preferred_home_count": 2,
            },
            "context_source": {"type": "registry"},
            "image_ref": "foo:1",
        }],
        "version": 1,
    }))
    assert compute_plan_id(load_build_plan(a)) == compute_plan_id(load_build_plan(b))


def test_replication_for_unknown_benchmark_in_per_image_ref_mode() -> None:
    plan = BuildPlan(entries=(_registry_entry("foo:1"),))
    assert plan.replication_for("anything") == 1


# ────────────────────────────────────────────────────────────────────────────
# Per-benchmark plan-gen modules (P1.7.C.2 B.1)
# ────────────────────────────────────────────────────────────────────────────


def test_terminal_bench_2_generator_no_probe() -> None:
    from xrlenv_plugins.images_build.terminal_bench_2.build_plan_gen import (
        DEFAULT_SIZE_HINT_BYTES,
        generate_plan,
    )

    plan_dict = generate_plan(
        ["fix-git", "build-pov-ray"],
        probe_sizes=False,
    )
    assert plan_dict["version"] == 1
    assert len(plan_dict["entries"]) == 2
    assert plan_dict["entries"][0]["image_ref"] == "alexgshaw/fix-git:20251031"
    assert plan_dict["entries"][0]["context_source"] == {"type": "registry"}
    assert plan_dict["entries"][0]["placement"]["size_hint_source"] == "heuristic"
    assert plan_dict["entries"][0]["placement"]["size_hint_bytes"] == DEFAULT_SIZE_HINT_BYTES
    BuildPlan.model_validate(plan_dict)


def test_swebench_verified_generator_image_ref_derivation() -> None:
    from xrlenv_plugins.images_build.swebench_verified.build_plan_gen import (
        _instance_to_image_ref,
        generate_plan,
    )

    ref = _instance_to_image_ref("astropy__astropy-7166")
    assert ref == "swebench/sweb.eval.x86_64.astropy_1776_astropy-7166:latest"

    plan_dict = generate_plan(
        ["astropy__astropy-7166", "django__django-11099"],
        probe_sizes=False,
    )
    assert len(plan_dict["entries"]) == 2
    e0 = plan_dict["entries"][0]
    assert e0["image_ref"] == ref
    assert e0["labels"]["xrlenv.benchmark"] == "swebench-verified"
    assert e0["labels"]["xrlenv.instance_id"] == "astropy__astropy-7166"
    BuildPlan.model_validate(plan_dict)


def test_seta_env_generator_emits_git_source() -> None:
    from xrlenv_plugins.benchmarks.seta.build_plan_gen import (
        IMAGE_NAMESPACE,
        _parse_range,
        generate_plan,
    )

    assert _parse_range("0-2,5,10-11") == ["0", "1", "2", "5", "10", "11"]

    plan_dict = generate_plan(["0", "1"], ref="main")
    assert len(plan_dict["entries"]) == 2
    e0 = plan_dict["entries"][0]
    assert e0["image_ref"].startswith(f"{IMAGE_NAMESPACE}/0:")
    cs = e0["context_source"]
    assert cs["type"] == "git"
    assert cs["repo"] == "https://github.com/camel-ai/seta-env"
    assert cs["ref"] == "main"
    assert cs["subdir"] == "Harbor-Dataset/0/environment"
    assert cs["dockerfile"] == "Dockerfile"
    assert e0["placement"]["size_hint_source"] == "heuristic"
    BuildPlan.model_validate(plan_dict)


# ──────────────────────────────────────────────────────────────────────────────
# Sub-slice 1.b — tarball ``content_b64`` resolution
# ──────────────────────────────────────────────────────────────────────────────


def test_tarball_content_b64_excluded_from_plan_id(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``compute_plan_id`` must produce the same id whether or not
    ``content_b64`` has been populated by the CLI helper. The
    plan_id reflects operator intent (the YAML), not which bytes
    happen to be on disk at apply time.
    """
    import base64

    plan_unresolved = BuildPlan(entries=(
        BuildEntry(
            image_ref="my/img:1",
            context_source=TarballSource(
                path="ctx.tar.gz", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    plan_resolved = BuildPlan(entries=(
        BuildEntry(
            image_ref="my/img:1",
            context_source=TarballSource(
                path="ctx.tar.gz", dockerfile="Dockerfile",
                content_b64=base64.b64encode(b"<bytes>").decode("ascii"),
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    assert compute_plan_id(plan_unresolved) == compute_plan_id(plan_resolved)


def test_resolve_tarball_sources_loads_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``resolve_tarball_sources`` reads bytes from disk and
    populates ``content_b64``."""
    import base64

    from xrlenv.control.build_plan import resolve_tarball_sources

    tarball_path = tmp_path / "ctx.tar.gz"
    payload = b"<fake-tarball-bytes>"
    tarball_path.write_bytes(payload)

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="my/img:1",
            context_source=TarballSource(
                path=str(tarball_path), dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    resolved = resolve_tarball_sources(plan, max_bytes=1024)
    assert len(resolved.entries) == 1
    src = resolved.entries[0].context_source
    assert isinstance(src, TarballSource)
    assert src.content_b64 is not None
    assert base64.b64decode(src.content_b64) == payload


def test_resolve_tarball_sources_resolves_relative_to_base_dir(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """When the YAML uses a relative ``path``, resolve_tarball_sources
    looks it up under ``base_dir`` (typically the YAML's parent
    directory)."""
    from xrlenv.control.build_plan import resolve_tarball_sources

    tarball_path = tmp_path / "contexts" / "ctx.tar"
    tarball_path.parent.mkdir()
    tarball_path.write_bytes(b"hi")

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="my/img:1",
            context_source=TarballSource(
                path="contexts/ctx.tar", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    resolved = resolve_tarball_sources(
        plan, max_bytes=1024, base_dir=tmp_path,
    )
    src = resolved.entries[0].context_source
    assert isinstance(src, TarballSource)
    assert src.content_b64 is not None


def test_resolve_tarball_sources_rejects_oversized(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A tarball larger than the cap is rejected at apply time
    with a clear operator message — fails before any wire traffic."""
    from xrlenv.control.build_plan import resolve_tarball_sources

    tarball_path = tmp_path / "huge.tar"
    tarball_path.write_bytes(b"X" * 1024)

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="huge/img:1",
            context_source=TarballSource(
                path=str(tarball_path), dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    with pytest.raises(ManifestInvalid) as excinfo:
        resolve_tarball_sources(plan, max_bytes=512)
    msg = str(excinfo.value)
    assert "huge/img:1" in msg
    assert "over the" in msg


def test_resolve_tarball_sources_idempotent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Calling resolve twice doesn't re-load the file (the second
    call sees content_b64 already set and passes through)."""
    import base64

    from xrlenv.control.build_plan import resolve_tarball_sources

    tarball_path = tmp_path / "ctx.tar"
    tarball_path.write_bytes(b"first-load")

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="my/img:1",
            context_source=TarballSource(
                path=str(tarball_path), dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    once = resolve_tarball_sources(plan, max_bytes=1024)

    # Mutate the file post-resolve. Idempotent re-run should NOT
    # pick up the change — the first resolve is the source of truth.
    tarball_path.write_bytes(b"second-load-different")
    twice = resolve_tarball_sources(once, max_bytes=1024)

    src = twice.entries[0].context_source
    assert isinstance(src, TarballSource)
    assert src.content_b64 is not None
    assert base64.b64decode(src.content_b64) == b"first-load"


def test_resolve_tarball_sources_passes_through_non_tarball() -> None:
    """Plans containing only registry / git entries pass through
    unchanged (returns the same instance — a small optimization
    that callers can rely on for identity comparison)."""
    from xrlenv.control.build_plan import resolve_tarball_sources

    plan = BuildPlan(entries=(
        BuildEntry(
            image_ref="alex/from-registry:1",
            context_source=RegistrySource(),
            placement=EntryPlacement(
                size_hint_bytes=1024, size_hint_source="registry-probe",
            ),
        ),
        BuildEntry(
            image_ref="xrlenv-seta-env/0:main",
            context_source=GitSource(
                repo="https://github.com/example/repo", ref="main",
                subdir=".", dockerfile="Dockerfile",
            ),
            placement=EntryPlacement(size_hint_bytes=1024),
        ),
    ))
    resolved = resolve_tarball_sources(plan, max_bytes=1024)
    assert resolved is plan  # identity, not just equality


def test_committed_canonical_plans_load() -> None:
    """The three committed build_plan.yaml snapshots must round-trip
    through the loader. Catches drift between the generator and the
    schema."""
    import importlib.resources as ir
    for package in (
        "xrlenv_plugins.images_build.terminal_bench_2",
        "xrlenv_plugins.images_build.swebench_verified",
        "xrlenv_plugins.benchmarks.seta",  # relocated from images_build.seta_env
    ):
        ref = ir.files(package).joinpath("build_plan.yaml")
        with ir.as_file(ref) as p:
            plan = load_build_plan(p)
        assert plan.is_per_image_ref()
        assert len(plan.entries) >= 1
        # plan_id is stable across two loads.
        with ir.as_file(ref) as p:
            plan2 = load_build_plan(p)
        assert compute_plan_id(plan) == compute_plan_id(plan2)
