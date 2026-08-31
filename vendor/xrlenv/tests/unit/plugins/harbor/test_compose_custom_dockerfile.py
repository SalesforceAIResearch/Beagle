"""Tests for the custom-`dockerfile:`-at-root-context → distinct-image mapping.

A `build:` service with `context: .` but a *custom* `dockerfile:` (chess-mate's `game`
built from `Dockerfile.game`) must map to its own `<task_id>-<service>` image ref, not
be rewritten to the task's canonical main image. Existing TerminalWorld cases (main =
context `.` + default Dockerfile; sub-dir sidecars; `image:`-only services) are
unchanged.
"""
from __future__ import annotations

from xrlenv_plugins.harbor.compose import (
    build_dockerfile,
    default_image_refs,
    is_canonical_main_build,
    subdir_build_services,
)


def test_root_context_custom_dockerfile_is_distinct() -> None:
    # chess-mate: game = build: {context: ., dockerfile: Dockerfile.game}
    doc = {"services": {
        "main": {"build": {"context": "."}},
        "game": {"build": {"context": ".", "dockerfile": "Dockerfile.game"}},
    }}
    refs = default_image_refs("chess-mate", doc, namespace="ns", tag="main")
    assert refs["main"] == "ns/chess-mate:main"          # canonical main
    assert refs["game"] == "ns/chess-mate-game:main"     # DISTINCT (was the bug)
    # and it's surfaced as needing its own build entry
    assert "game" in subdir_build_services(doc)
    assert "main" not in subdir_build_services(doc)


def test_root_context_custom_dockerfile_respects_main_ref() -> None:
    doc = {"services": {
        "main": {"build": {"context": "."}},
        "game": {"build": {"context": ".", "dockerfile": "Dockerfile.game"}},
    }}
    refs = default_image_refs(
        "chess-mate", doc, namespace="ns", tag="main",
        main_ref="zli12321/lhtb-chess-mate:20260615",
    )
    assert refs["main"] == "zli12321/lhtb-chess-mate:20260615"
    assert refs["game"] == "ns/chess-mate-game:main"


def test_tw_unchanged_subdir_and_default_main() -> None:
    # TerminalWorld pattern: main (context ".") + subdir sidecar + image-only peer.
    doc = {"services": {
        "main": {"build": {"context": "."}},                 # default Dockerfile
        "solr-node": {"build": {"context": "./solr-node"}},
        "pg": {"image": "postgres:14"},                      # image-only, untouched
    }}
    refs = default_image_refs("tw_188260", doc, namespace="ns", tag="main")
    assert refs["main"] == "ns/tw_188260:main"
    assert refs["solr-node"] == "ns/tw_188260-solr-node:main"
    assert "pg" not in refs
    # subdir_build_services returns NORMALIZED contexts (./solr-node -> solr-node)
    assert subdir_build_services(doc) == {"solr-node": "solr-node"}


def test_dot_slash_dockerfile_is_default_not_custom() -> None:
    # `dockerfile: ./Dockerfile` is the DEFAULT (normalized), not a custom sidecar.
    default_svc = {"build": {"context": ".", "dockerfile": "./Dockerfile"}}
    assert build_dockerfile(default_svc) == "Dockerfile"
    assert is_canonical_main_build(default_svc) is True

    doc = {"services": {"main": {"build": {"context": ".", "dockerfile": "./Dockerfile"}}}}
    assert default_image_refs("t", doc, namespace="ns")["main"] == "ns/t:main"
    assert subdir_build_services(doc) == {}

    # a bare custom name and a plain default both classify correctly
    assert is_canonical_main_build({"build": {"context": "."}}) is True
    assert is_canonical_main_build(
        {"build": {"context": ".", "dockerfile": "Dockerfile.game"}}
    ) is False
    # image-only service is not a build at all
    assert build_dockerfile({"image": "postgres:14"}) is None
