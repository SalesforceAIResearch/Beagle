"""Storage declarations must not silently join independent Compose projects."""
from __future__ import annotations

from copy import deepcopy

import pytest
from xrlenv.control.compose_policy import vet_compose_project
from xrlenv.control.kwargs_policy import KwargsPolicy, KwargsPolicyViolation


@pytest.mark.parametrize("mount", ["cache:/cache", "cache:/cache:ro", {
    "type": "volume", "source": "cache", "target": "/cache", "read_only": True,
}])
@pytest.mark.parametrize(("definition", "field"), [
    ({"name": "shared-cache"}, "name"),
    ({"name": "${CACHE_VOLUME}"}, "name"),
    ({"external": True}, "external"),
    ({"external": {"name": "shared-cache"}}, "external"),
    ({"external": "${EXTERNAL}"}, "external"),
    ({"driver": "nfs"}, "driver"),
    ({"driver": "${DRIVER}"}, "driver"),
    ({"driver_opts": {"type": "none", "o": "bind", "device": "/shared"}}, "driver_opts"),
    ({"driver_opts": {"type": "nfs", "device": ":/shared"}}, "driver_opts"),
])
def test_shared_storage_rejected_even_for_readers(mount, definition, field) -> None:
    compose = {"services": {"main": {"volumes": [mount]}}, "volumes": {"cache": definition}}
    # An opt-in to host access is not an opt-in to cross-project volume sharing.
    policy = KwargsPolicy(allow_privileged=True, allowed_host_paths=("/shared",))
    with pytest.raises(KwargsPolicyViolation) as exc:
        vet_compose_project(compose, policy=policy)
    assert f"volumes.cache.{field}" in {r.kwarg for r in exc.value.rejections}


@pytest.mark.parametrize("definition", [None, {}, {"driver": "local"}, {
    "external": False, "driver_opts": {}, "labels": {"purpose": "cache"},
}])
def test_project_local_storage_and_sidecar_sharing_preserved(definition) -> None:
    compose = {
        "services": {
            "main": {"volumes": ["cache:/cache"]},
            "sidecar": {"volumes": [{"type": "volume", "source": "cache", "target": "/data"}]},
            "reader": {"volumes_from": ["sidecar:ro"]},
        },
        "volumes": {"cache": definition},
    }
    original = deepcopy(compose)
    vet_compose_project(compose)
    assert compose == original


@pytest.mark.parametrize("source", ["container:other", "container:other:ro", "container:other:rw", "${SOURCE}", "missing"])
def test_volumes_from_must_reference_a_declared_service(source) -> None:
    with pytest.raises(KwargsPolicyViolation, match=r"services.main.volumes_from"):
        vet_compose_project({"services": {"main": {"volumes_from": [source]}}})


@pytest.mark.parametrize("source", ["sidecar", "sidecar:ro", "sidecar:rw"])
def test_volumes_from_declared_service_allowed(source) -> None:
    vet_compose_project({"services": {"main": {"volumes_from": [source]}, "sidecar": {}}})


@pytest.mark.parametrize("mount", ["/scratch", {"type": "volume", "target": "/scratch"}, {"type": "tmpfs", "target": "/scratch"}])
def test_private_anonymous_and_tmpfs_mounts_allowed(mount) -> None:
    vet_compose_project({"services": {"main": {"volumes": [mount], "tmpfs": ["/tmp"]}}})


@pytest.mark.parametrize("mount", ["${MOUNT}", "${SOURCE}:/cache", "${SOURCE:-/shared}:/cache", {
    "type": "volume", "source": "${SOURCE}", "target": "/cache",
}, {
    "type": "${TYPE}", "source": "/shared", "target": "/cache",
}])
def test_mount_source_cannot_be_interpolated_after_vetting(mount) -> None:
    with pytest.raises(KwargsPolicyViolation, match=r"services.main.volumes"):
        vet_compose_project({"services": {"main": {"volumes": [mount]}}})


@pytest.mark.parametrize("compose", [
    {"include": ["more.yaml"]},
    {"services": {"main": {"extends": {"file": "more.yaml", "service": "base"}}}},
])
def test_deferred_documents_cannot_add_unvetted_storage(compose) -> None:
    with pytest.raises(KwargsPolicyViolation, match="resolved"):
        vet_compose_project(compose)


def test_storage_errors_aggregate_with_existing_service_errors() -> None:
    compose = {
        "volumes": {"cache": {"name": "shared", "external": True}},
        "services": {"main": {"privileged": True, "volumes_from": ["container:other"]}},
    }
    with pytest.raises(KwargsPolicyViolation) as exc:
        vet_compose_project(compose)
    assert {r.kwarg for r in exc.value.rejections} == {
        "volumes.cache.name", "volumes.cache.external", "services.main.privileged",
        "services.main.volumes_from",
    }
