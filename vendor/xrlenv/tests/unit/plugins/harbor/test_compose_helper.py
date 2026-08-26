"""Unit tests for the pure multi-service compose helper
(``xrlenv_plugins.harbor.compose``).

The helper is the single source of truth for the ``build:`` → ``image:`` mapping
and the ``<task_id>-<service>`` naming shared by the build-plan generator and the
cluster harbor plugin. These tests pin its behaviour against inline compose
fixtures shaped like the four representative corpus tasks (a service-DNS sidecar,
per-service build contexts, a static-IP peer stack, and a local-build-tag stack)
so build-time and run-time never drift. No cluster, no FSx, no harbor lib.
"""
from __future__ import annotations

import subprocess
import sys

import pytest
import xrlenv_plugins.harbor.compose as hc

# ── Fixtures: minimal compose docs mirroring the corpus shapes ────────────────

# tw_522753-shape: postgres sidecar (public image, service-DNS) + app (build .).
# No ``main`` service in the task compose (harbor's base layer supplies it), no
# pinned subnet.
POSTGRES_SIDECAR = {
    "services": {
        "postgres": {"image": "postgres:14"},
        "app": {"build": ".", "image": "terminalworld-env-522753"},
    },
}

# tw_188260-shape: per-service sub-directory build contexts + static IPs.
PER_SERVICE_BUILD = {
    "services": {
        "main": {"networks": {"ambari-net": {"ipv4_address": "10.188.74.100"}}},
        "solr-node": {
            "build": {"context": "./solr-node", "dockerfile": "Dockerfile"},
        },
        "ambari-server": {"build": {"context": "./ambari-server"}},
    },
    "networks": {
        "ambari-net": {
            "driver": "bridge",
            "ipam": {"config": [{"subnet": "10.188.74.0/24"}]},
        },
    },
}

# tw_304270-shape: peer "hosts" all building from ``.`` (== the task image),
# static IPs on a pinned subnet.
STATIC_IP_PEERS = {
    "services": {
        "main": {"privileged": True},
        "stapp02": {"build": ".", "privileged": True},
        "stapp03": {"build": ".", "privileged": True},
        "stlb01": {"build": "."},
    },
    "networks": {
        "twnet": {"ipam": {"config": [{"subnet": "172.16.70.0/24"}]}},
    },
}

# tw_299387-shape: local build-tag references (image: + pull_policy: never) and a
# public sidecar; service-DNS only.
LOCAL_BUILD_TAG = {
    "services": {
        "fake-token": {"image": "terminalworld-env-299387", "pull_policy": "never"},
        "fake-gcs": {"image": "fsouza/fake-gcs-server:latest"},
        "main": {"image": "terminalworld-env-299387"},
    },
    "networks": {"kopianet": {"driver": "bridge"}},
}


# ── load / detection ──────────────────────────────────────────────────────────

def test_import_is_harbor_free() -> None:
    # Importing the pure helper must not drag the harbor runtime lib in — the
    # build generator depends on this (lazy harbor/__init__). Assert in a FRESH
    # subprocess: sibling tests in this session import harbor, so an in-process
    # ``harbor not in sys.modules`` check is polluted (same isolation rationale
    # as test_generator_dotenv_autoload).
    code = (
        "import sys; import xrlenv_plugins.harbor.compose as hc; "
        "assert hc.is_multi_service({'services': {'a': {}, 'b': {}}}); "
        "assert 'harbor' not in sys.modules, sorted(m for m in sys.modules "
        "if m == 'harbor' or m.startswith('harbor.'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_load_compose_handles_empty_and_null() -> None:
    assert hc.load_compose("") == {}
    assert hc.load_compose("# only a comment\n") == {}
    assert hc.load_compose("services:\n  main: {}\n") == {
        "services": {"main": {}},
    }


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ({"services": {"main": {}}}, False),
        (POSTGRES_SIDECAR, True),
        (PER_SERVICE_BUILD, True),
        ({}, False),
    ],
)
def test_is_multi_service(doc: dict, expected: bool) -> None:
    assert hc.is_multi_service(doc) is expected


# ── build-service enumeration + naming ────────────────────────────────────────

def test_build_context_forms() -> None:
    assert hc.build_context({"image": "postgres:14"}) is None
    assert hc.build_context({"build": "."}) == "."
    assert hc.build_context({"build": "./solr-node"}) == "solr-node"
    assert hc.build_context({"build": {"context": "./x"}}) == "x"
    assert hc.build_context({"build": {"context": "."}}) == "."


def test_iter_and_subdir_build_services() -> None:
    # POSTGRES_SIDECAR: only ``app`` builds, from ``.`` → no sub-dir entry.
    assert hc.iter_build_services(POSTGRES_SIDECAR) == {"app": "."}
    assert hc.subdir_build_services(POSTGRES_SIDECAR) == {}
    # PER_SERVICE_BUILD: two sub-dir contexts.
    assert hc.subdir_build_services(PER_SERVICE_BUILD) == {
        "solr-node": "solr-node",
        "ambari-server": "ambari-server",
    }


def test_default_image_refs_naming() -> None:
    ns = "terminalworld-verified"
    # ``.`` context → canonical <id> ref; sub-dir → <id>-<service>.
    assert hc.default_image_refs("tw_522753", POSTGRES_SIDECAR, namespace=ns) == {
        "app": "terminalworld-verified/tw_522753:main",
    }
    assert hc.default_image_refs("tw_188260", PER_SERVICE_BUILD, namespace=ns) == {
        "solr-node": "terminalworld-verified/tw_188260-solr-node:main",
        "ambari-server": "terminalworld-verified/tw_188260-ambari-server:main",
    }
    # image-only services (no build) are absent — the rewrite leaves them alone.
    assert hc.default_image_refs("tw_299387", LOCAL_BUILD_TAG, namespace=ns) == {}


def test_default_image_refs_main_ref_override_and_tag() -> None:
    refs = hc.default_image_refs(
        "tw_304270", STATIC_IP_PEERS, namespace="ns", tag="v2",
        main_ref="registry/ns/tw_304270:v2",
    )
    # every ``.``-context peer maps to the single canonical main ref.
    assert set(refs.values()) == {"registry/ns/tw_304270:v2"}
    assert set(refs) == {"stapp02", "stapp03", "stlb01"}


# ── registry_namespace_and_tag (sidecar namespace from a repinned main ref) ────
#
# The decoupling: a multi-service task's sidecars derive their namespace from the
# already-repinned MAIN image ref. A private-registry ref splits into (namespace, tag);
# anything that isn't a private-registry ref returns (None, ...) so assemble_project
# fails loud only when the task actually has sub-dir builds needing a namespace.

@pytest.mark.parametrize(
    ("main_ref", "expected"),
    [
        # repinned private registry (host:port) with a sub-path namespace + tag →
        # the chess-mate case: sidecar becomes <ns>/<task>-game:main.
        ("ip-10-0-5-6:5011/lhtb/chess-mate:main",
         ("ip-10-0-5-6:5011/lhtb", "main")),
        # host:port, no explicit tag → tag defaults to "main".
        ("ip-10-0-5-6:5011/lhtb/chess-mate",
         ("ip-10-0-5-6:5011/lhtb", "main")),
        # dotted hostname (no port) + non-default tag.
        ("registry.example.com/ns/task:v2", ("registry.example.com/ns", "v2")),
        # localhost is a registry even without a dot or port.
        ("localhost/foo/bar:main", ("localhost/foo", "main")),
        # single-level: host is the whole namespace.
        ("reg:5011/task:main", ("reg:5011", "main")),
    ],
)
def test_registry_namespace_and_tag_private_registry(
    main_ref: str, expected: tuple[str | None, str],
) -> None:
    assert hc.registry_namespace_and_tag(main_ref) == expected


@pytest.mark.parametrize(
    "main_ref",
    [
        None,                       # no resolved main image
        "",                         # empty
        "alexgshaw/task:rev",       # docker.io-relative user/repo — NOT a registry
        "ubuntu:22.04",             # bare public image (has ':', but no namespace path)
    ],
)
def test_registry_namespace_and_tag_no_private_namespace(main_ref: str | None) -> None:
    # Nothing that isn't a private-registry ref yields a namespace; the tag is
    # irrelevant (assemble_project only consults the namespace, and fails loud if a
    # sub-dir build service needs one).
    namespace, _tag = hc.registry_namespace_and_tag(main_ref)
    assert namespace is None


# ── rewrite ───────────────────────────────────────────────────────────────────

def test_rewrite_build_to_image_and_caps() -> None:
    refs = hc.default_image_refs("tw_522753", POSTGRES_SIDECAR, namespace="ns")
    out = hc.rewrite_to_image_refs(POSTGRES_SIDECAR, refs)
    app = out["services"]["app"]
    assert "build" not in app
    assert app["image"] == "ns/tw_522753:main"
    # sidecars (both non-main) get an injected, enforceable cap.
    assert app["mem_limit"] == "1024m" and app["cpus"] == 1.0
    assert out["services"]["postgres"]["mem_limit"] == "1024m"
    # input not mutated (deep copy).
    assert "mem_limit" not in POSTGRES_SIDECAR["services"]["app"]


def test_rewrite_missing_build_ref_is_error() -> None:
    with pytest.raises(KeyError, match="solr-node"):
        # supply a ref for only one of the two build services
        hc.rewrite_to_image_refs(
            PER_SERVICE_BUILD,
            {"ambari-server": "ns/tw_188260-ambari-server:main"},
        )


def test_rewrite_callable_none_for_build_service_is_error() -> None:
    # Symmetric with the dict path: a callable returning None for a build
    # service must fail loud, not silently emit ``image: None``.
    with pytest.raises(KeyError, match="app"):
        hc.rewrite_to_image_refs(POSTGRES_SIDECAR, lambda _name: None)


def test_rewrite_repoints_local_build_tags_when_mapped() -> None:
    # A caller (the plugin) repoints local build-tag ``image:`` refs at the
    # canonical registry ref by naming them in the map — even without ``build:``.
    canonical = "ns/tw_299387:main"
    out = hc.rewrite_to_image_refs(
        LOCAL_BUILD_TAG,
        {"fake-token": canonical, "main": canonical},
    )
    ft = out["services"]["fake-token"]
    assert ft["image"] == canonical
    assert "pull_policy" not in ft  # ``never`` stripped so the ref can be pulled
    # main is the exec target → not capped; still repointed.
    assert out["services"]["main"]["image"] == canonical
    assert "cpus" not in out["services"]["main"]
    # the public sidecar, unnamed in the map and build-less, is untouched.
    assert out["services"]["fake-gcs"]["image"] == "fsouza/fake-gcs-server:latest"


def test_rewrite_respects_declared_limits() -> None:
    doc = {
        "services": {
            "main": {},
            "db": {"image": "postgres:14", "mem_limit": "512m", "cpus": 0.5},
            "cache": {
                "image": "redis:7",
                "deploy": {"resources": {"limits": {"cpus": "2", "memory": "256m"}}},
            },
        },
    }
    out = hc.rewrite_to_image_refs(doc, {})
    # author-declared caps are preserved, not overridden by the default.
    assert out["services"]["db"]["mem_limit"] == "512m"
    assert out["services"]["cache"]["deploy"]["resources"]["limits"]["memory"] == "256m"
    assert "mem_limit" not in out["services"]["cache"]


def test_rewrite_injects_only_the_missing_cap_dimension() -> None:
    # A partial declaration must have ONLY the absent dimension defaulted — else
    # the footprint reserves the missing dimension while the compose leaves it
    # uncapped (the enforcement gap the audit flagged).
    doc = {
        "services": {
            "main": {},
            "cpu_only": {"image": "busybox", "cpus": 0.5},
            "mem_only": {"image": "busybox", "mem_limit": "256m"},
            "deploy_cpu_only": {
                "image": "busybox",
                "deploy": {"resources": {"limits": {"cpus": "3"}}},
            },
        },
    }
    out = hc.rewrite_to_image_refs(doc, {}, sidecar_cpu=1.0, sidecar_mem_mb=1024)
    # declared cpu kept, memory defaulted:
    assert out["services"]["cpu_only"]["cpus"] == 0.5
    assert out["services"]["cpu_only"]["mem_limit"] == "1024m"
    # declared memory kept, cpu defaulted:
    assert out["services"]["mem_only"]["mem_limit"] == "256m"
    assert out["services"]["mem_only"]["cpus"] == 1.0
    # cpu declared via deploy.resources → only memory injected (cpus not added):
    assert "cpus" not in out["services"]["deploy_cpu_only"]
    assert out["services"]["deploy_cpu_only"]["mem_limit"] == "1024m"


def test_rewrite_callable_ref() -> None:
    out = hc.rewrite_to_image_refs(
        POSTGRES_SIDECAR,
        lambda name: "ns/img:tag" if name == "app" else None,
    )
    assert out["services"]["app"]["image"] == "ns/img:tag"
    assert out["services"]["postgres"]["image"] == "postgres:14"  # untouched


# ── footprint ─────────────────────────────────────────────────────────────────

def test_sidecar_footprint_defaults_and_declared() -> None:
    # POSTGRES_SIDECAR: 2 non-main services x default (1 cpu / 1 GiB).
    assert hc.sidecar_footprint(POSTGRES_SIDECAR) == (2.0, 2048)
    # STATIC_IP_PEERS: 3 non-main services x default.
    assert hc.sidecar_footprint(STATIC_IP_PEERS) == (3.0, 3072)


def test_sidecar_footprint_reads_declared_limits() -> None:
    doc = {
        "services": {
            "main": {},
            "db": {"image": "postgres:14", "cpus": 2.0, "mem_limit": "2g"},
            "plain": {"image": "busybox"},  # falls back to default
        },
    }
    cpu, mem = hc.sidecar_footprint(doc, default_cpu=1.0, default_mem_mb=1024)
    assert cpu == 3.0  # 2 (declared) + 1 (default)
    assert mem == 3072  # 2048 (2g) + 1024 (default)


@pytest.mark.parametrize(
    ("value", "mb"),
    [("512m", 512), ("2g", 2048), ("256M", 256), ("1073741824", 1024), ("1G", 1024)],
)
def test_mem_to_mb(value: str, mb: int) -> None:
    assert hc._mem_to_mb(value) == mb


# ── subnet claims ─────────────────────────────────────────────────────────────

def test_subnet_claims() -> None:
    assert hc.subnet_claims(PER_SERVICE_BUILD) == ["10.188.74.0/24"]
    assert hc.subnet_claims(STATIC_IP_PEERS) == ["172.16.70.0/24"]
    # service-DNS-only tasks pin no subnet → nothing to reserve exclusively.
    assert hc.subnet_claims(POSTGRES_SIDECAR) == []
    assert hc.subnet_claims(LOCAL_BUILD_TAG) == []


# ── build-context containment ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("context", "safe"),
    [
        (".", True),
        ("solr-node", True),
        ("./solr-node", True),
        ("a/b", True),
        ("..", False),
        ("../escape", False),
        ("./../escape", False),
        ("/abs", False),
        ("/etc/passwd", False),
    ],
)
def test_is_safe_relative_context(context: str, safe: bool) -> None:
    assert hc.is_safe_relative_context(context) is safe


# ── runtime assembly (step 4b-1) ──────────────────────────────────────────────

MAIN_REF = "reg:5011/terminalworld-verified/tw_x:main"
NS = "reg:5011/terminalworld-verified"


# ensure_main_service — fill-missing-only, never overwrite

def test_ensure_main_injects_when_absent() -> None:
    # tw_522753-shape: task compose has no ``main`` (harbor's base supplies it).
    out = hc.ensure_main_service(POSTGRES_SIDECAR, main_ref=MAIN_REF)
    assert out["services"]["main"] == {
        "image": MAIN_REF, "command": ["sh", "-c", "sleep infinity"],
    }
    # original is untouched (deep copy) + the sidecars are preserved
    assert "main" not in POSTGRES_SIDECAR["services"]
    assert out["services"]["postgres"] == {"image": "postgres:14"}


def test_ensure_main_fills_image_and_command_for_bare_main() -> None:
    # tw_304270-shape: ``main`` present but declares neither image nor command
    # (and an explicit ``privileged`` that must be preserved).
    out = hc.ensure_main_service(STATIC_IP_PEERS, main_ref=MAIN_REF)
    main = out["services"]["main"]
    assert main["image"] == MAIN_REF
    assert main["command"] == ["sh", "-c", "sleep infinity"]
    assert main["privileged"] is True  # explicit field preserved verbatim


def test_ensure_main_preserves_explicit_command_and_image() -> None:
    # An explicit ``main.command`` / ``main.image`` in the task compose wins —
    # fill-missing-only must not overwrite it (byte-for-byte faithful to compose
    # merge, where the task layer beats the base).
    doc = {"services": {
        "main": {"image": "task/own:tag", "command": ["/entrypoint.sh"]},
        "db": {"image": "postgres:14"},
    }}
    out = hc.ensure_main_service(doc, main_ref=MAIN_REF)
    assert out["services"]["main"] == {
        "image": "task/own:tag", "command": ["/entrypoint.sh"],
    }


def test_ensure_main_leaves_build_main_image_for_the_rewrite() -> None:
    # a ``main`` that builds from ``.`` keeps no ``image`` here — the rewrite
    # repoints it — but still gets a keepalive command filled.
    doc = {"services": {"main": {"build": "."}, "db": {"image": "redis:7"}}}
    out = hc.ensure_main_service(doc, main_ref=MAIN_REF)
    assert "image" not in out["services"]["main"]
    assert out["services"]["main"]["command"] == ["sh", "-c", "sleep infinity"]


def test_ensure_main_custom_keepalive() -> None:
    out = hc.ensure_main_service(
        {"services": {"db": {"image": "redis:7"}}},
        main_ref=MAIN_REF, keepalive=["tail", "-f", "/dev/null"],
    )
    assert out["services"]["main"]["command"] == ["tail", "-f", "/dev/null"]


# local_tag_service_names

def test_local_tag_service_names() -> None:
    # fake-token has image + pull_policy:never (local build tag → repoint);
    # fake-gcs is a public image (leave); main has a local-tag image but NO
    # pull_policy so it isn't flagged here (assemble maps main via main_ref anyway).
    assert hc.local_tag_service_names(LOCAL_BUILD_TAG) == ["fake-token"]
    # a build service is never a "local tag" (it has a build context)
    assert hc.local_tag_service_names(PER_SERVICE_BUILD) == []


# image_refs

def test_image_refs_collects_sorted_unique() -> None:
    doc = {"services": {
        "main": {"image": "b:1"}, "x": {"image": "a:1"},
        "y": {"image": "a:1"}, "z": {"build": "."},  # no image → skipped
    }}
    assert hc.image_refs(doc) == ["a:1", "b:1"]


# assemble_project — end-to-end per corpus shape

def test_assemble_no_main_sidecar_stack() -> None:
    # tw_522753: inject main; app (build .) → main_ref; postgres public untouched.
    rewritten, images = hc.assemble_project(
        POSTGRES_SIDECAR, task_id="tw_522753", main_ref=MAIN_REF,
    )
    svcs = rewritten["services"]
    assert svcs["main"]["image"] == MAIN_REF
    assert svcs["app"]["image"] == MAIN_REF and "build" not in svcs["app"]
    assert svcs["postgres"]["image"] == "postgres:14"  # public, left alone
    # sidecars capped (footprint enforced), main never capped
    assert "mem_limit" in svcs["postgres"] and "cpus" in svcs["postgres"]
    assert "mem_limit" not in svcs["main"]
    assert images == sorted({MAIN_REF, "postgres:14"})


def test_assemble_local_build_tag_stack() -> None:
    # tw_299387: fake-token (local tag) + main (local-tag image) → main_ref;
    # fake-gcs public untouched.
    rewritten, images = hc.assemble_project(
        LOCAL_BUILD_TAG, task_id="tw_299387", main_ref=MAIN_REF,
    )
    svcs = rewritten["services"]
    assert svcs["main"]["image"] == MAIN_REF
    assert svcs["fake-token"]["image"] == MAIN_REF
    assert "pull_policy" not in svcs["fake-token"]  # dropped on repoint
    assert svcs["fake-gcs"]["image"] == "fsouza/fake-gcs-server:latest"
    assert images == sorted({MAIN_REF, "fsouza/fake-gcs-server:latest"})


def test_assemble_subdir_builds_need_namespace() -> None:
    # tw_188260: sub-dir build services → <namespace>/<id>-<svc>:<tag>, the same
    # refs build_plan_gen pushed (via default_image_refs). main gets image+keepalive.
    rewritten, images = hc.assemble_project(
        PER_SERVICE_BUILD, task_id="tw_188260", main_ref=MAIN_REF,
        namespace=NS, tag="main",
    )
    svcs = rewritten["services"]
    assert svcs["main"]["image"] == MAIN_REF
    assert svcs["solr-node"]["image"] == f"{NS}/tw_188260-solr-node:main"
    assert svcs["ambari-server"]["image"] == f"{NS}/tw_188260-ambari-server:main"
    assert images == sorted({
        MAIN_REF,
        f"{NS}/tw_188260-solr-node:main",
        f"{NS}/tw_188260-ambari-server:main",
    })


def test_assemble_subdir_builds_fail_loud_without_namespace() -> None:
    # LOCKED fail-loud: sub-dir builds with no resolvable namespace (no {task_id}
    # template) raise rather than emit a ref that won't match what was pushed.
    with pytest.raises(ValueError, match="sub-dir build service"):
        hc.assemble_project(
            PER_SERVICE_BUILD, task_id="tw_188260", main_ref=MAIN_REF,
            namespace=None,
        )


def test_assemble_static_ip_peers_no_namespace_needed() -> None:
    # tw_304270: peers all build from ``.`` (== task image) → main_ref; no sub-dir
    # builds, so no namespace required. privileged preserved on peers.
    rewritten, images = hc.assemble_project(
        STATIC_IP_PEERS, task_id="tw_304270", main_ref=MAIN_REF,
    )
    svcs = rewritten["services"]
    assert svcs["stapp02"]["image"] == MAIN_REF
    assert svcs["stapp02"]["privileged"] is True
    assert svcs["stlb01"]["image"] == MAIN_REF
    assert images == [MAIN_REF]  # every peer is the task image
    # subnet claim still readable off the assembled doc (CP derives it)
    assert hc.subnet_claims(rewritten) == ["172.16.70.0/24"]


def test_assemble_preserves_explicit_main_command() -> None:
    doc = {"services": {
        "main": {"build": ".", "command": ["/run.sh"]},
        "db": {"image": "postgres:14"},
    }}
    rewritten, _ = hc.assemble_project(doc, task_id="tw_x", main_ref=MAIN_REF)
    assert rewritten["services"]["main"]["command"] == ["/run.sh"]
    assert rewritten["services"]["main"]["image"] == MAIN_REF  # build . → main_ref
