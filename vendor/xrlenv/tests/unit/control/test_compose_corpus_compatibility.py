"""A compose-policy change must not reject shapes the corpus ships in a way no cluster can undo.

## The one question this file can answer

Given a compose document and nothing else: **is it rejected no matter how the cluster is
configured?** Only that. Whether a particular cluster accepts it is a different question, and one
a document cannot answer -- see below.

``KwargsPolicy`` sorts every rejection into tiers (``kwargs_policy`` module docstring). What
matters here is only whether an operator can change the outcome:

===========================================  ======================================
 tier                                         operator can change the outcome?
===========================================  ======================================
 1  allowed, operator may restrict             yes, via ``denied_caps`` etc.
 2  rejected, operator may opt in              yes, via ``allow_privileged`` etc.
 3  never allowed (isolation escapes)          **no**
 4  rejected on architectural grounds          **no**
===========================================  ======================================

Tiers 3 and 4 are decided by code, identically on every deployment, so "is this document rejected
for one of those?" is a property of the DOCUMENT. That is what makes this file runnable with no
cluster, no cache, no Docker and no network. See :data:`NO_OPERATOR_OVERRIDE`.

## What this deliberately does NOT assert, and why

* **Tier-1 and tier-2 rejections** (``network_mode``, ``privileged``, host binds, caps, devices).
  The operator decides these, so they are not properties of the document: the same file is
  rejected under one ``nodes.yaml`` and accepted under another. Three ``terminalworld`` tasks are
  tier-2 rejected under the default policy *and run green in production*, because they are listed
  in ``SYSBOX_TASKS`` and routed to a sysbox pool whose policy permits what they need. An earlier
  version of this file asserted on tier 2, concluded those tasks were "incompatible", and recorded
  a hand-written exclusion list that silently contradicted ``SYSBOX_TASKS``. That cannot be
  patched into correctness: the answer is not in the document, because the routing that decides it
  lives in each task's ``task.toml``.

* **Whether a task actually passes.** That is the oracle sweep's job
  (``xrlenv_plugins/benchmarks/*/run_oracle_sweep.py``), which runs the real thing.

* **Shapes absent from the corpus today**: ``volumes_from``, ``extends``, ``include``, long-form
  ``{type, source, target}`` mounts, and top-level volumes carrying ``name`` / ``external`` /
  ``driver`` / ``driver_opts``. Zero occurrences at the time of writing, so a rule rejecting them
  breaks nothing today -- a statement about the corpus, not a licence to ignore them.
  :func:`test_fixtures_still_cover_the_corpus` fails if one appears later.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from xrlenv.control.compose_policy import vet_compose_project
from xrlenv.control.kwargs_policy import DEFAULT_POLICY, KwargsPolicyViolation

#: Every policy-relevant mount/service shape the corpus contains, with its occurrence count and a
#: task it came from. Committed rather than read from the cache so this file runs for anyone;
#: :func:`test_fixtures_still_cover_the_corpus` is what keeps it honest against drift.
#:
#: Vetted under ``DEFAULT_POLICY`` -- the strictest one. The policy choice cannot change what this
#: file asserts on (no operator setting reaches those tiers), which is precisely why no
#: hand-picked opt-in appears anywhere in this file.
CORPUS_SHAPES: dict[str, dict] = {
    # 27x, e.g. tw_105786. The operator decides this one today (allowed_host_paths), so it is out
    # of scope below; modelled anyway, because a NEW rule could reject it with no override.
    "short-form host bind": {"volumes": ["/var/run/docker.sock:/var/run/docker.sock"]},
    # 14x, e.g. seta-env/1133. The harness expands these; the task author never picks the value.
    "short-form interpolated source": {
        "volumes": ["${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}",
                    "${WORKSPACE_DIR}:/workspace"]},
    # 14x, e.g. tw_118507.
    "privileged service": {"privileged": True},
    # 3x, e.g. seta-env/892. Compose scopes this per project; it is the sharing a storage rule
    # cares about, and the shape most at risk from one.
    "short-form named volume, read-only": {"volumes": ["ssh-keys:/tmp/client-keys:ro"]},
    # 3x, e.g. tw_333322.
    "network_mode host": {"network_mode": "host"},
    # 1x, tw_223822. No colon: compose allocates an anonymous volume AT that container path. It is
    # not a host bind, despite looking like one.
    "target-only anonymous volume": {"volumes": ["/opt/splunk/var/lib/splunk"]},
    # 1x, tw_661946.
    "tmpfs mounts": {"tmpfs": ["/run", "/run/lock"]},
}

#: 3x, e.g. seta-env/892 — declared at the top level rather than on a service.
CORPUS_TOP_LEVEL_VOLUMES: dict = {"ssh-keys": None}

#: The rejection tiers no operator setting can lift: 3 (always-fatal isolation escapes) and 4
#: (architectural — ``platform``, ``userns_mode``). A rejection in either is decided by code and
#: lands the same way on every deployment, which is why this file can judge it from a document
#: alone. Tiers 1 and 2 are the operator's to choose and are deliberately out of scope; tier 0 is
#: always allowed, so it never appears in a rejection at all.
NO_OPERATOR_OVERRIDE = frozenset({3, 4})

#: Shapes the corpus does not currently contain. Named here so the drift check can look for them
#: by name, and so the omission is a recorded decision rather than an oversight.
UNMODELLED_SHAPES = frozenset({
    "volumes_from", "extends", "include", "long-form mount", "top-level volume attributes",
})


@pytest.mark.parametrize("label", sorted(CORPUS_SHAPES))
def test_no_corpus_shape_is_unconditionally_rejected(label: str) -> None:
    """Rejecting a corpus shape with no override available takes those tasks offline everywhere.

    No operator setting lifts it and no pool routing avoids it, so unlike a tier-1/2 rejection
    there is no cluster configuration in which the affected tasks still run.
    """
    compose = {"services": {"main": {"image": "task:latest", **CORPUS_SHAPES[label]}},
               "volumes": CORPUS_TOP_LEVEL_VOLUMES}
    try:
        vet_compose_project(compose, policy=DEFAULT_POLICY)
    except KwargsPolicyViolation as exc:
        fatal = [f"(tier {r.level}) {r.kwarg}: {r.reason}"
                 for r in exc.rejections if r.level in NO_OPERATOR_OVERRIDE]
        assert not fatal, (
            f"corpus shape {label!r} is rejected with no operator override available, so every "
            f"task using it is offline on every cluster:\n  " + "\n  ".join(fatal))


def test_fixtures_still_cover_the_corpus() -> None:
    """Keep :data:`CORPUS_SHAPES` honest: fail if the corpus grows a shape it does not model.

    Committed fixtures are what make the test above runnable without a cache; the price is that
    they can drift. This is the drift detector, and the only part that needs the cache.
    """
    cache = os.environ.get("XRLENV_BENCHMARK_CACHE")
    documents = sorted(Path(cache).rglob("*compose*.y*ml")) if cache and Path(cache).is_dir() else []
    if not documents:
        pytest.skip(
            "DRIFT CHECK DID NOT RUN: no XRLENV_BENCHMARK_CACHE, so whether CORPUS_SHAPES still "
            "mirrors the corpus was not verified. The no-override assertions above DID run, against "
            "the shapes as recorded -- the right split for a contributor, who cannot hold the "
            "corpus. Before landing a compose-policy change, a maintainer must run this with "
            "XRLENV_BENCHMARK_CACHE set and see it PASS rather than skip; a skip there means the "
            "cache is unpopulated (`build_cache.py --stage all`) and drift is still unchecked.")

    found: dict[str, str] = {}
    for path in documents:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        where = path.parent.parent.name
        if doc.get("include"):
            found.setdefault("include", where)
        top = doc.get("volumes")
        if isinstance(top, dict):
            for name, definition in top.items():
                if isinstance(definition, dict) and definition:
                    found.setdefault("top-level volume attributes",
                                     f"{where}: {name} -> {sorted(definition)}")
        for service in (doc.get("services") or {}).values():
            if not isinstance(service, dict):
                continue
            for key in ("volumes_from", "extends"):
                if service.get(key):
                    found.setdefault(key, where)
            for mount in service.get("volumes") or []:
                if isinstance(mount, dict):
                    found.setdefault("long-form mount", f"{where}: {mount}")

    unmodelled = {k: v for k, v in found.items() if k in UNMODELLED_SHAPES}
    assert not unmodelled, (
        f"the corpus now contains {len(unmodelled)} compose shape(s) CORPUS_SHAPES does not "
        "model, so the no-override assertions above no longer cover it. Add each to CORPUS_SHAPES "
        "and drop it from UNMODELLED_SHAPES:\n  "
        + "\n  ".join(f"{k} — first seen in {v}" for k, v in sorted(unmodelled.items())))
