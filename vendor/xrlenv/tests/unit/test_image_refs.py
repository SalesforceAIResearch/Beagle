"""Unit tests for :mod:`xrlenv.image_refs` — the registry-host
normalization shared by calibrate matching and node-cache eviction."""

from __future__ import annotations

from xrlenv.image_refs import (
    has_explicit_tag,
    manifest_digest,
    registry_agnostic_ref,
    repo_path,
    same_image,
)


def test_manifest_digest_extracts_bare_sha256() -> None:
    """The ``sha256:...`` portion of a digest ref — the content-addressed key
    calibrate cross-references between a plan ref's resolved digest and the
    ``RepoDigests`` a node reports, independent of host/repo spelling."""
    d = "sha256:" + "a" * 64
    # Registry-qualified digest ref (what the resolver returns / RepoDigests):
    assert manifest_digest(f"public.ecr.aws/d3j8x8q7/swe-bench-202605@{d}") == d
    # Bare repo digest ref:
    assert manifest_digest(f"ns/img@{d}") == d


def test_manifest_digest_none_for_non_digest_refs() -> None:
    """Tag refs, bare repos, image config ids, and empty input → ``None`` — no
    false match. Notably a Docker image *config id* (``sha256:...`` with no
    ``@``) must NOT be mistaken for a manifest digest."""
    assert manifest_digest("ns/img:v1") is None            # tag ref
    assert manifest_digest("public.ecr.aws/ns/img:main") is None
    assert manifest_digest("ns/img") is None               # bare repo
    assert manifest_digest("sha256:" + "b" * 64) is None   # config id (no @)
    assert manifest_digest("") is None
    assert manifest_digest(None) is None


def test_registry_agnostic_ref_strips_host_port() -> None:
    assert registry_agnostic_ref(
        "node-host:5011/xrlenv-webarena-infinity/substrate:1ca77813",
    ) == "xrlenv-webarena-infinity/substrate:1ca77813"


def test_registry_agnostic_ref_strips_localhost_and_dotted_host() -> None:
    assert registry_agnostic_ref("localhost:5000/foo/bar:0.1") == "foo/bar:0.1"
    assert registry_agnostic_ref("registry.example.com/foo/bar:1") == "foo/bar:1"


def test_registry_agnostic_ref_keeps_docker_hub_relative_repo() -> None:
    # First segment is NOT a registry host (no '.'/':'/localhost) → keep it.
    assert registry_agnostic_ref(
        "library/python:3.12-slim",
    ) == "library/python:3.12-slim"


def test_registry_agnostic_ref_keeps_bare_and_already_agnostic() -> None:
    assert registry_agnostic_ref("python:3.12") == "python:3.12"
    assert registry_agnostic_ref(
        "xrlenv-webarena-infinity/substrate:1ca77813",
    ) == "xrlenv-webarena-infinity/substrate:1ca77813"


def test_registry_agnostic_ref_preserves_digest_suffix() -> None:
    assert registry_agnostic_ref(
        "reg:5011/repo/name@sha256:abc123",
    ) == "repo/name@sha256:abc123"


def test_same_image_matches_across_registry_host() -> None:
    assert same_image(
        "node-host:5011/wai/substrate:1ca77813",
        "wai/substrate:1ca77813",
    )
    assert same_image("localhost:5000/a/b:1", "a/b:1")
    # Different tags never match; a tag never equals a digest.
    assert not same_image("a/b:1", "a/b:2")
    assert not same_image("a/b:1", "a/b@sha256:abc")


def test_repo_path_strips_host_tag_and_digest() -> None:
    # host + tag
    assert repo_path(
        "node-host:5011/terminalworld-verified/tw_99185:main",
    ) == "terminalworld-verified/tw_99185"
    # host + digest (the digest-pull form a node holds)
    assert repo_path(
        "node-host:5011/terminalworld-verified/tw_99185@sha256:7cd3964",
    ) == "terminalworld-verified/tw_99185"
    # bare tag / bare digest
    assert repo_path("terminalworld-verified/tw_99185:main") == (
        "terminalworld-verified/tw_99185"
    )
    assert repo_path("repo/name@sha256:abc") == "repo/name"
    # no tag/digest → unchanged (minus host)
    assert repo_path("reg:5011/ns/img") == "ns/img"
    # Docker-Hub-relative keeps its namespace component, tag stripped
    assert repo_path("library/python:3.12-slim") == "library/python"


def test_repo_path_reunites_the_digest_pull() -> None:
    # THE calibrate case: a plan's ``:tag`` ref and the node's digest-pulled ref
    # (untagged / ``@sha256``, registry-qualified) collapse to the same repo, so
    # calibrate can credit the plan ref from the on-disk image. same_image (used
    # by evict) deliberately does NOT — it never equates a tag with a digest.
    plan = "terminalworld-verified/tw_99185:main"
    node = "node-host:5011/terminalworld-verified/tw_99185@sha256:7cd3964"
    assert repo_path(plan) == repo_path(node)
    assert not same_image(plan, node)


def test_has_explicit_tag() -> None:
    # An explicit :tag (the calibrate over-credit trigger: a stale sibling tag).
    assert has_explicit_tag("ns/img:20251031")
    assert has_explicit_tag("node-host:5011/ns/img:main")
    assert has_explicit_tag("library/python:3.12-slim")
    # A digest pull carries NO explicit tag → the fallback may credit it.
    assert not has_explicit_tag("ns/img@sha256:abc")
    assert not has_explicit_tag("reg:5011/ns/img@sha256:abc")
    # Bare untagged.
    assert not has_explicit_tag("ns/img")
    assert not has_explicit_tag("reg:5011/ns/img")


def test_has_explicit_tag_gates_the_over_credit() -> None:
    # THE 2026-07-17 tb2.1 mixed-tag bug: the node holds a *different* explicit
    # tag of the same repo than the plan pins. It shares the repo path (so the
    # unguarded fallback would credit it) but it carries an explicit tag, so the
    # ``not has_explicit_tag`` guard keeps the fallback from firing — unlike the
    # genuine digest pull, which has none.
    plan = "alexgshaw/compile-compcert:20260403"
    stale = "alexgshaw/compile-compcert:20251031"
    digest = "reg:5011/alexgshaw/compile-compcert@sha256:abc"
    assert repo_path(plan) == repo_path(stale) == repo_path(digest)
    assert has_explicit_tag(stale)          # → fallback must NOT fire
    assert not has_explicit_tag(digest)     # → fallback may fire
