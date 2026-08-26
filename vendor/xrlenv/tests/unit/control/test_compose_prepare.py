"""CP-side compose preparation (``xrlenv.control.compose_prepare``).

Pins the reserved-label stamping (session_kind split main=raw / sidecars=compose)
and the main-image digest pin — the transform the coordinator applies before the
compose document goes on the wire (§2.3).
"""
from __future__ import annotations

import pytest
from xrlenv.control.compose_prepare import (
    LABEL_COMPOSE_PROJECT,
    LABEL_ROLLOUT_ID,
    LABEL_SESSION_KIND,
    pin_images,
    prepare_compose,
    stamp_and_pin,
    subnet_claims,
    subnets_overlap,
)


def _svc(out: dict, name: str) -> dict:
    return out["services"][name]


def test_stamps_reserved_labels_split_by_role() -> None:
    compose = {
        "services": {
            "main": {"image": "app:tag"},
            "postgres": {"image": "postgres:14"},
        },
    }
    out = stamp_and_pin(compose, rollout_id="r1", project_name="proj")
    for name in ("main", "postgres"):
        labels = _svc(out, name)["labels"]
        assert labels[LABEL_ROLLOUT_ID] == "r1"
        assert labels[LABEL_COMPOSE_PROJECT] == "proj"
    # session_kind split: main=raw (a CP session), sidecars=compose (invisible to
    # the raw-GC sweep so they aren't force-destroyed as orphans).
    assert _svc(out, "main")["labels"][LABEL_SESSION_KIND] == "raw"
    assert _svc(out, "postgres")["labels"][LABEL_SESSION_KIND] == "compose"
    # input not mutated
    assert "labels" not in compose["services"]["main"]


def test_pins_main_image_and_drops_build() -> None:
    compose = {
        "services": {
            "main": {"build": ".", "image": "terminalworld-env-1", "pull_policy": "never"},
            "side": {"image": "busybox"},
        },
    }
    out = stamp_and_pin(
        compose, rollout_id="r1", project_name="p",
        main_image_ref="reg/ns/tw_1@sha256:abc",
    )
    main = _svc(out, "main")
    assert main["image"] == "reg/ns/tw_1@sha256:abc"
    assert "build" not in main and "pull_policy" not in main
    # a sidecar image is untouched
    assert _svc(out, "side")["image"] == "busybox"


def test_no_pin_when_ref_absent() -> None:
    compose = {"services": {"main": {"image": "app:tag"}}}
    out = stamp_and_pin(compose, rollout_id="r", project_name="p")
    assert _svc(out, "main")["image"] == "app:tag"  # unchanged


def test_reserved_keys_override_task_supplied_labels() -> None:
    # a task can't smuggle its own xrlenv.* labels — reserved keys win.
    compose = {
        "services": {
            "main": {
                "image": "app",
                "labels": {"xrlenv.rollout_id": "evil", "app.role": "web"},
            },
        },
    }
    out = stamp_and_pin(compose, rollout_id="real", project_name="p")
    labels = _svc(out, "main")["labels"]
    assert labels[LABEL_ROLLOUT_ID] == "real"  # overridden
    assert labels["app.role"] == "web"  # task's own non-reserved label preserved


def test_normalizes_list_form_labels() -> None:
    compose = {
        "services": {
            "main": {"image": "app", "labels": ["a.b=c", "flag"]},
        },
    }
    out = stamp_and_pin(compose, rollout_id="r", project_name="p")
    labels = _svc(out, "main")["labels"]
    assert labels["a.b"] == "c"
    assert labels["flag"] == ""
    assert labels[LABEL_ROLLOUT_ID] == "r"


def test_custom_main_service_name() -> None:
    compose = {"services": {"app": {"image": "x"}, "db": {"image": "y"}}}
    out = stamp_and_pin(
        compose, rollout_id="r", project_name="p", main_service="app",
        main_image_ref="reg@sha256:d",
    )
    assert _svc(out, "app")["labels"][LABEL_SESSION_KIND] == "raw"
    assert _svc(out, "app")["image"] == "reg@sha256:d"
    assert _svc(out, "db")["labels"][LABEL_SESSION_KIND] == "compose"


def test_empty_or_missing_services_is_safe() -> None:
    assert stamp_and_pin({}, rollout_id="r", project_name="p") == {}
    out = stamp_and_pin(
        {"services": {"main": None}}, rollout_id="r", project_name="p",
    )
    assert out["services"]["main"] is None  # non-dict service skipped, no crash


# ── prepare_compose: consistency between the YAML pin and the images list ──────

def test_pin_images_replaces_original_with_resolved() -> None:
    out = pin_images(
        ["app:tag", "postgres:14"],
        original_main_ref="app:tag", resolved_main_ref="app@sha256:d",
    )
    assert out == ["app@sha256:d", "postgres:14"]


def test_pin_images_appends_when_original_absent() -> None:
    # main wasn't in the list (or its ref differs) → ensure the resolved ref is
    # present so the node ensure-presents the digest.
    out = pin_images(
        ["postgres:14"], original_main_ref="app:tag",
        resolved_main_ref="app@sha256:d",
    )
    assert out == ["postgres:14", "app@sha256:d"]


def test_pin_images_noop_without_resolution() -> None:
    assert pin_images(["a:1"], original_main_ref="a:1", resolved_main_ref=None) == [
        "a:1",
    ]


def test_prepare_compose_threads_resolved_ref_through_yaml_and_images() -> None:
    # P2 audit: the digest-resolved main ref must be pinned in BOTH the compose
    # main service AND the ensure-present images list — never a tag/digest split.
    compose = {
        "services": {
            "main": {"image": "reg/ns/tw:main"},
            "postgres": {"image": "postgres:14"},
        },
    }
    prepared = prepare_compose(
        compose,
        images=["reg/ns/tw:main", "postgres:14"],
        rollout_id="r1",
        project_name="proj",
        resolved_main_ref="reg/ns/tw@sha256:abc",
    )
    # YAML main image pinned...
    assert prepared.compose["services"]["main"]["image"] == "reg/ns/tw@sha256:abc"
    # ...and the images list pinned to the SAME ref (no tag left behind)
    assert prepared.images == ["reg/ns/tw@sha256:abc", "postgres:14"]
    # labels stamped too (delegates to stamp_and_pin)
    assert prepared.compose["services"]["main"]["labels"][LABEL_SESSION_KIND] == "raw"
    assert (
        prepared.compose["services"]["postgres"]["labels"][LABEL_SESSION_KIND]
        == "compose"
    )


def test_prepare_compose_without_resolution_keeps_tags() -> None:
    compose = {"services": {"main": {"image": "app:tag"}}}
    prepared = prepare_compose(
        compose, images=["app:tag"], rollout_id="r", project_name="p",
    )
    assert prepared.compose["services"]["main"]["image"] == "app:tag"
    assert prepared.images == ["app:tag"]
    # labels still stamped
    assert prepared.compose["services"]["main"]["labels"][LABEL_ROLLOUT_ID] == "r"


# ── subnet claims + overlap (3b) ──────────────────────────────────────────────

def test_subnet_claims_extracts_pinned_cidrs() -> None:
    compose = {
        "services": {"main": {}},
        "networks": {
            "twnet": {"ipam": {"config": [{"subnet": "172.16.70.0/24"}]}},
        },
    }
    assert subnet_claims(compose) == ("172.16.70.0/24",)


def test_subnet_claims_empty_for_dns_only() -> None:
    assert subnet_claims({"services": {"main": {}}}) == ()
    assert subnet_claims(
        {"services": {"m": {}}, "networks": {"n": {"driver": "bridge"}}},
    ) == ()


@pytest.mark.parametrize(
    ("a", "b", "overlaps"),
    [
        ("172.16.70.0/24", "172.16.70.0/24", True),   # identical
        ("172.16.70.0/24", "172.16.70.128/25", True),  # subset
        ("10.0.0.0/8", "10.5.0.0/16", True),           # superset
        ("172.16.70.0/24", "172.16.71.0/24", False),   # adjacent, disjoint
        ("10.0.0.0/8", "192.168.0.0/16", False),       # far disjoint
        ("garbage", "garbage", True),                  # unparseable → str-equal
        ("garbage", "other", False),
    ],
)
def test_subnets_overlap(a: str, b: str, overlaps: bool) -> None:
    assert subnets_overlap(a, b) is overlaps
