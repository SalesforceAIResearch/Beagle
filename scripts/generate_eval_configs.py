#!/usr/bin/env python3
"""Generate 2-task smoke-gate configs from onboarded ``.beagle/agents`` manifests.

The config **shape lives here once** (:func:`build_config`) — NOT in a matrix of committed YAML files.
So there's no M×N of near-duplicate templates to maintain and no schema drift: change the shape here
and every agent×benchmark smoke config regenerates under ``tests/smoke/``. The hand-written
``examples/evaluation/`` files remain teaching material.

Flow::

    # 1. onboard each agent (records repo/ref/version in .beagle/agents/<name>.json)
    python -m beagle.tools.onboard --upstream … --ref … --version 1.18.16 --repo <you>/… …
    # 2. generate public-safe configs (mini-swe + OpenCode, local Docker, direct model API)
    python scripts/generate_eval_configs.py
    # Internal profile: also Monet, Gateway Express, and the xrlenv cluster
    python scripts/generate_eval_configs.py --internal
    # 3. run one generated smoke config
    beagle evaluate --config tests/smoke/swe-rebench/opencode-1.18.16_smoke2.yaml

**Join key = version.** Each agent below pins a ``version``; the script fills its ``source`` from the
manifest whose ``version`` matches (onboard's ``--version``). That's the ONLY thing that has to line
up — ``--profile-name`` / ``--branch-name`` / ``--dir`` are free. Edit the AGENTS / BENCHMARKS / DEFAULTS
tables to change the matrix or knobs. Public-safe defaults are deliberate; ``--internal`` opts into
Monet, LLM Gateway Express credentials, and ``xrlenv-cluster``.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import zlib
from pathlib import Path

import yaml

from beagle.agents.core.provider import provider_config, provider_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = REPO_ROOT / "tests" / "smoke"
MANIFEST_DIR = REPO_ROOT / ".beagle" / "agents"

# --- the matrix + knobs (the ONE place to edit) ------------------------------

_PUBLIC_FORWARD_ENV = ["OPENAI_API_KEY"]
_INTERNAL_FORWARD_ENV = [
    "LLM_GATEWAY_EXPRESS_API_KEY",
    "LLM_GATEWAY_EXPRESS_API_KEY_LIST",
    "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL",
]

#: Shared run/agent knobs applied to every config (a benchmark or agent can override below).
DEFAULTS = dict(
    model="gpt-5.6-sol",
    provider={"type": "direct", "name": "openai"},
    forward_env=_PUBLIC_FORWARD_ENV,
    effort="medium",
    max_turns=150,
    # LAST-RESORT agent wall clock, written into every generated config ON PURPOSE so the number is
    # visible and attributable instead of hiding in the code. It applies ONLY to a benchmark that
    # ships no agent budget of its own (the docker-drop-in path); a harbor-family task's task.toml
    # [agent] timeout_sec always wins, per task. Scale a declared budget with timeout_multiplier.
    timeout=1800,
    timeout_multiplier=1.0,
    runtime="local",
    run_dir="./results",
    parallelism=1,
    # Task-level infra retry: re-run a trial on an infra-transient (capacity / control-plane / node
    # blip) up to N times, in a fresh container. Content outcomes (verifier fail, agent timeout,
    # rate limit) are NEVER infra-retried, so eval signal is never re-rolled. `content` stays 0 (no
    # best-of-N re-roll of unresolved tasks). See notes/retry-coverage.md.
    retry_infra=2,
)

INTERNAL_DEFAULTS = {
    **DEFAULTS,
    "provider": {"type": "internal", "name": "llm-gateway-express-local-proxy"},
    "forward_env": _INTERNAL_FORWARD_ENV,
    "runtime": "xrlenv-cluster",
    "parallelism": 16,
}

#: Each agent, keyed by its beagle `harness.name` (the ADAPTER). ``versions`` lists the experiment
#: copies of it to generate for — the config schema keeps `harness.name` and `harness.version`
#: separate precisely so two copies of one harness can coexist (baseline vs candidate), and each
#: entry joins to a manifest on its version. Adapter-level facts (e.g. `extra_args`) are
#: shared by every copy. A single-version agent generates exactly the filenames it always did;
#: only a multi-version one gets the `<name>-<version>` disambiguator.
AGENTS = {
    "monet": dict(
        versions=["20260826"],
        internal=True,
        extra_args={"monet_args": [
            "--permissive-auto-approve", "--no-monet-md", "--output-format", "stream-json"]},
        note="monet_args: last two are REQUIRED by beagle's stream parser.",
    ),
    "afcode": dict(
        # Latest only. **2.3.0 is an effective floor**: earlier afcode routes gpt-5.6+ to
        # /chat/completions, where the gateway's bedrock replica rejects function tools
        # alongside reasoning_effort — measured at ~1 call in 10, which a multi-turn rollout
        # hits almost every time. 2.3.0+ routes gpt-5.6+ to /responses instead. Onboarding an
        # older copy would silently confine you to gpt-5.5.
        versions=["v2.3.0_0904"],
        internal=True,
        # No adapter-level args: the wheelhouse path defaults to the vendored location inside
        # the experiment copy, and afcode exposes no turn cap or output-format flag to pin.
        extra_args={},
        note="installs the CLONED ref (theta) against the offline wheelhouse vendored in the "
             "experiment copy; needs a gateway/internal provider route.",
    ),
    "mini-swe": dict(
        versions=["v2.4.6"],
        extra_args={"mini_swe_args": [{"config_path": "src/minisweagent/config/mini.yaml"}]},
        note="mini_swe_args.config_path is mini-swe's `-c` preset = the evolvable surface.",
    ),
    "opencode": dict(
        versions=["1.18.16"],
        extra_args={"opencode_args": ["--auto"]},
        note="opencode accepts max_turns for a uniform vocabulary but has no turn-cap flag (no-op).",
    ),
}

#: Each benchmark: a short tag for run.name, optional dataset/split, per-benchmark overrides (e.g.
#: deep-swe's lower parallelism), and the `--smoke` filename + its curated 2-task subset.
BENCHMARKS = {
    "terminal_bench_2_1": dict(short="tb21"),
    "swe-bench-verified": dict(short="swebench_verified", dataset="SWE-bench/SWE-bench_Verified", split="test",
                               # two-phase: agents GENERATE patches (parallelism), then swebench
                               # batch-EVALUATES them in test containers — fan that out wider.
                               parallelism_eval_patches=64),
    "deep-swe": dict(short="deepswe", parallelism=8,
                     note="pier / filtered-egress — needs the extra: uv pip install -e '.[deep-swe]'"),
    "swe-rebench": dict(short="swerebench",
                        note="860 SWE tasks, ~1.9 GB image each pulled on first acquire — pin a "
                             "small, STABLE task subset per campaign so image affinity helps.",
),
}

#: The 2 tasks each benchmark is gated on, SAMPLED (seeded) rather than hand-picked, and committed
#: so a gate failure re-runs identically. Regenerate with ``--reseed-smoke-tasks`` (needs whatever
#: each benchmark enumerates from: the benchmark cache, HuggingFace…). The seed is derived from the
#: benchmark name and recorded in the file, so re-sampling is an explicit, reviewable act rather
#: than something that quietly happens on every generation.
SMOKE_TASKS_PATH = REPO_ROOT / "scripts" / "smoke_tasks.json"
#: Tasks per benchmark in the gate — enough to catch plumbing breakage, few enough to stay cheap.
SMOKE_TASKS_N = 2


def smoke_tasks(bench: str) -> list[str]:
    """The committed sample for ``bench`` (empty if it has never been sampled)."""
    try:
        return list(json.loads(SMOKE_TASKS_PATH.read_text(encoding="utf-8"))[bench]["tasks"])
    except (OSError, ValueError, KeyError):
        return []


def _seed_for(bench: str) -> int:
    """Deterministic per-benchmark seed — same corpus, same sample, on any machine."""
    return zlib.crc32(bench.encode()) & 0xFFFFFFFF


def reseed_smoke_tasks(benches: list[str] | None = None) -> dict[str, dict]:
    """Re-sample the gate's tasks from each benchmark's own task list and rewrite the JSON.

    Enumerating needs the real source (benchmark cache / HF dataset), so a benchmark that can't be
    listed here keeps its existing entry rather than being silently emptied — losing coverage is
    worse than a stale sample, and the reason is printed."""
    import beagle as bgl
    from beagle.benchmarks.base import BenchmarkSpec

    try:
        book = json.loads(SMOKE_TASKS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        book = {}
    for bench in (benches or list(BENCHMARKS)):
        try:
            ids = sorted(t.task_id for t, _c in bgl.benchmarks.load_tasks(BenchmarkSpec(name=bench)))
        except Exception as e:  # noqa: BLE001 — any enumeration failure keeps the old sample
            print(f"  ! {bench}: cannot enumerate ({type(e).__name__}: {e}) — keeping "
                  f"{book.get(bench, {}).get('tasks', [])}", file=sys.stderr)
            continue
        seed = _seed_for(bench)
        picked = sorted(random.Random(seed).sample(ids, min(SMOKE_TASKS_N, len(ids))))
        book[bench] = {"seed": seed, "sampled_from": len(ids), "tasks": picked}
        print(f"  ✓ {bench}: {picked} (seed {seed}, from {len(ids)} tasks)")
    SMOKE_TASKS_PATH.write_text(json.dumps(book, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return book


#: The smoke variants `--smoke` emits per (copy × benchmark). Each takes the first ``n_tasks`` of the
#: benchmark's ``smoke_tasks`` and lands at ``tests/smoke/<benchmark>/<label>_<variant>.yaml`` —
#: the SAME shape as the eval configs (benchmark dir, copy-labelled file), so one spelling of each
#: benchmark and one rule. Both used to name their own files (a per-benchmark ``smoke_file`` string
#: and a per-variant lambda), which is why one directory held `terminal_bench_2_1_smoke2.yaml`
#: next to `tb21_smoke1_gpt56sol.yaml`.
#: ONE variant, ``smoke2``: the gate answers "does this harness × benchmark combination work at
#: all", and one 2-task config per combination answers it. (There was a second, single-task variant
#: mirroring the baseline sweep's knobs — that is the sweep's business, not the gate's.)
SMOKE_VARIANTS = {
    "smoke2": dict(
        model=DEFAULTS["model"], effort=DEFAULTS["effort"], max_turns=DEFAULTS["max_turns"],
        parallelism=2, n_tasks=2, name_suffix="smoke2"),
}


def _label(agent: str, version: str) -> str:
    """The artifact name for one experiment copy: ALWAYS ``<harness>-<version>``.

    Unconditional on purpose. Naming that depended on how many copies an agent happens to have
    would mean adding a second version silently renames the first one's configs and run dirs, and
    two agents in the same table would follow different rules. A name is a pure function of the
    cell it describes."""
    return f"{agent}-{version}"


#: A version becomes a path component (config filename, smoke filename, run.name -> results dir),
#: so it has to be safe as one. `release/1.2` would otherwise silently create a nested directory
#: instead of one artifact.
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")   # \Z, not $: `$` also
#: matches just before a trailing newline, so "1.2\n" would have become a filename.


def profile_defaults(*, internal: bool = False) -> dict:
    """Return fresh generation defaults for the public or internal profile."""
    defaults = INTERNAL_DEFAULTS if internal else DEFAULTS
    return {
        **defaults,
        "provider": dict(defaults["provider"]),
        "forward_env": list(defaults["forward_env"]),
    }


def provider_override(
    *,
    provider_type: str | None,
    name: str | None = None,
    api_base: str | None = None,
    api_key_env: str | None = None,
    auth_header: str | None = None,
) -> dict | None:
    """Build and validate a typed provider CLI override.

    Route-specific fields are accepted only for the route that consumes them. ``name`` is
    required for every generated route; ``gateway`` additionally requires ``api_base``.
    """
    secondary = (name, api_base, api_key_env, auth_header)
    if provider_type is None:
        if any(value is not None for value in secondary):
            raise ValueError("provider fields require --provider-type")
        return None
    if not name:
        raise ValueError(f"--provider-type {provider_type} requires --provider-name")
    if provider_type in {"direct", "internal"}:
        if any(value is not None for value in (api_base, api_key_env, auth_header)):
            raise ValueError(
                f"gateway fields are invalid with --provider-type {provider_type}"
            )
        raw = {"type": provider_type, "name": name}
    elif provider_type == "gateway":
        if not api_base:
            raise ValueError("--provider-type gateway requires --provider-api-base")
        if auth_header and not api_key_env:
            raise ValueError("--provider-auth-header requires --provider-api-key-env")
        extra_args = {"api_base": api_base}
        if api_key_env:
            extra_args["api_key_env"] = api_key_env
        if auth_header:
            extra_args["auth_header"] = auth_header
        raw = {"type": "gateway", "name": name, "extra_args": extra_args}
    else:
        raise ValueError(f"unknown provider type {provider_type!r}")
    return provider_dict(provider_config({"provider": raw}))


def profile_agents(*, internal: bool = False) -> dict:
    """Agents available to a profile; Monet is an explicit internal opt-in."""
    return {name: spec for name, spec in AGENTS.items() if internal or not spec.get("internal")}


def agent_cells(agents: dict | None = None, *, internal: bool = False):
    """Yield ``(harness_name, version, label)`` — one cell per experiment copy.

    ``label`` (``<harness>-<version>``) names the generated artifact, so two copies of one harness
    never overwrite each other's config or share a ``run.name`` — which would collide again in the
    results dir. See :func:`_label` for why it is unconditional.
    """
    for name, a in (agents if agents is not None else profile_agents(internal=internal)).items():
        versions = a.get("versions", [a["version"]] if "version" in a else [])
        if not versions:
            raise SystemExit(
                f"agent {name!r} lists no versions — give it `versions=[...]` with at least one "
                f"experiment copy, or drop the entry. An empty list is silently zero configs.")
        for v in [str(v) for v in versions]:
            if not _SAFE_VERSION.fullmatch(v):
                raise SystemExit(
                    f"agent {name!r} has version {v!r}, which can't be a filename component: it "
                    f"lands in the config name, the smoke name and run.name (hence the results "
                    f"dir). Use [A-Za-z0-9._-] only — e.g. 'release-1.2', not 'release/1.2'.")
            yield name, v, _label(name, v)


def build_config(agent: str, bench: str, manifest: dict, *, smoke: bool = False,
                 variant: dict | None = None, version: str | None = None,
                 internal: bool = False, runtime: str | None = None,
                 provider: dict | None = None,
                 forward_env: list[str] | None = None) -> dict:
    """The canonical eval-config shape (one place). ``source`` is filled from ``manifest`` (the
    ``token_env`` line is included only when the manifest records one).

    Pass a ``variant`` (a :data:`SMOKE_VARIANTS` entry) for a dev-smoke config — its curated
    ``n_tasks`` subset, ``./tmp``, and the variant's model/effort/turns/parallelism. ``smoke=True`` is
    a back-compat alias for the ``smoke2`` variant. Neither → the full eval config on the defaults."""
    agents = profile_agents(internal=internal)
    if agent not in agents:
        raise ValueError(f"agent {agent!r} is internal-only; pass internal=True or --internal")
    defaults = profile_defaults(internal=internal)
    if variant is None and smoke:
        variant = SMOKE_VARIANTS["smoke2"]
    a, b = AGENTS[agent], BENCHMARKS[bench]
    version = str(version if version is not None else (a.get("versions") or [a["version"]])[0])
    source = {"repo": manifest["repo"], "ref": manifest["ref"]}
    if manifest.get("token_env"):
        source["token_env"] = manifest["token_env"]
    data_entry: dict = {"benchmark": bench}
    for k in ("dataset", "split"):
        if b.get(k):
            data_entry[k] = b[k]
    if variant:
        tasks = smoke_tasks(bench)[: variant["n_tasks"]]
        if not tasks:
            raise SystemExit(
                f"no seeded smoke tasks for {bench!r}; run "
                "scripts/generate_eval_configs.py --reseed-smoke-tasks")
        data_entry["tasks"] = tasks
    agent_config = {
        "harness": {"name": agent, "version": version, "source": source},
        "model": {"name": variant["model"] if variant else defaults["model"]},
        "effort": variant["effort"] if variant else defaults["effort"],
        "max_turns": variant["max_turns"] if variant else defaults["max_turns"],
        "forward_env": list(forward_env if forward_env is not None else defaults["forward_env"]),
        "timeout": defaults["timeout"],
        "extra_args": a["extra_args"],
    }
    effective_provider = provider if provider is not None else defaults["provider"]
    if effective_provider:
        if not isinstance(effective_provider, dict):
            raise ValueError("provider override must use the typed provider mapping")
        agent_config["provider"] = provider_dict(
            provider_config({"provider": effective_provider})
        )
    run_parallelism = (
        variant["parallelism"] if variant
        else b.get("parallelism", defaults["parallelism"]) if internal
        else defaults["parallelism"]
    )
    eval_parallelism = (
        variant["parallelism"] if variant
        else b.get("parallelism_eval_patches", run_parallelism) if internal
        else run_parallelism
    )
    return {
        "run": {
            "dir": "./tmp" if variant else defaults["run_dir"],
            # smoke run-name uses the agent with its hyphen dropped (miniswe/monet/opencode)
            "name": (f"{agent.replace('-', '')}-{version}-{b['short']}-{variant['name_suffix']}"
                     if variant
                     else f"eval-{_label(agent, version)}-{b['short']}"),
            "runtime": runtime if runtime is not None else defaults["runtime"],
            "parallelism": run_parallelism,
            # A two-phase benchmark (SWE-bench) fans patch EVAL out wider than patch generation.
            **({"parallelism_eval_patches": eval_parallelism}
               if b.get("parallelism_eval_patches") else {}),
            # Scales the TASK's own declared phase budgets. RUN-level, not under `retry`: it
            # applies to the first attempt as much as to a re-run.
            "timeout_multiplier": defaults["timeout_multiplier"],
            # Retry an infra-transient trial in a fresh container; never re-rolls content.
            "retry": {"infra": defaults["retry_infra"]},
        },
        "agent": agent_config,
        "data": [data_entry],
    }


#: Comments attached to the two timeout knobs IN PLACE. A header note three lines up doesn't travel
#: with the value: `timeout: 1800` on its own reads as a per-run wall clock, which is exactly the
#: misreading the fallback design exists to prevent.
_TIMEOUT_NOTE = (
    "# LAST RESORT — used ONLY by a benchmark that ships no agent budget of its own.\n"
    "# A harbor/pier task declares its own `[agent] timeout_sec`; that always wins and this\n"
    "# value is then ignored. To make a declared budget longer/shorter, use timeout_multiplier.\n")
_MULTIPLIER_NOTE = (
    "# THE knob for run length: scales each task's OWN declared budget (agent + verifier).\n"
    "# 1.0 = exactly what the benchmark declares. Prefer this over an absolute timeout.\n")


def _annotate_timeouts(body: str) -> str:
    """Insert the notes above ``agent.timeout`` / ``run.timeout_multiplier`` in the dump."""
    out = []
    for line in body.splitlines(keepends=True):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("timeout:"):
            out += [f"{indent}{ln}\n" for ln in _TIMEOUT_NOTE.strip().splitlines()]
        elif stripped.startswith("timeout_multiplier:"):
            out += [f"{indent}{ln}\n" for ln in _MULTIPLIER_NOTE.strip().splitlines()]
        out.append(line)
    return "".join(out)


def _dump(agent: str, bench: str, cfg: dict, *, smoke: bool = False) -> str:
    kind = "smoke config" if smoke else "config"
    header = [f"# {agent} on {bench} — GENERATED {kind} by scripts/generate_eval_configs.py "
              f"(do not edit; not git-tracked).", "# Edit the matrix/knobs in that script and regenerate."]
    for src in (AGENTS[agent].get("note"), BENCHMARKS[bench].get("note")):
        if src:
            header.append(f"# {src}")
    body = _annotate_timeouts(
        yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100))
    return "\n".join(header) + "\n" + body


def _write(dest: Path, text: str, *, check: bool) -> None:
    if not check:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    try:
        shown = dest.relative_to(REPO_ROOT)
    except ValueError:
        shown = dest
    print(f"  {'would write' if check else '✓'} {shown}")


# --- driver ------------------------------------------------------------------

def _manifests_by_version(manifest_dir: Path) -> dict[str, dict]:
    """``{str(version): manifest}``; raises on a duplicate version (ambiguous join key)."""
    by_version: dict[str, dict] = {}
    for path in sorted(glob.glob(str(manifest_dir / "*.json"))):
        try:
            m = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"  ! skipping unreadable manifest {path}: {e}", file=sys.stderr)
            continue
        v = str(m.get("version") or "").strip()
        if not v:
            continue
        if v in by_version:
            raise SystemExit(f"two manifests share version {v!r} — versions must be a unique join "
                             f"key; re-onboard one with a distinct --version")
        by_version[v] = m
    return by_version


def generate(*, manifest_dir: Path, smoke_root: Path = SMOKE_ROOT,
             check: bool = False, internal: bool = False,
             runtime: str | None = None, provider: dict | None = None,
             forward_env: list[str] | None = None) -> tuple[int, list[str]]:
    """Generate the smoke GATE — every experiment copy × every benchmark × each
    :data:`SMOKE_VARIANTS` entry — into ``smoke_root/<benchmark>/<label>_<variant>.yaml``, with
    ``source`` filled from each copy's onboarded manifest.

    This is the whole output. The other two config trees are deliberately not generated here:
    ``examples/evaluation/`` is hand-written teaching material (one file per USE CASE, not per
    agent×benchmark cell), and full-benchmark runs come from
    ``experiments/scripts/generate_eval_configs.py``. Returns (written, missing copy names)."""
    manifests = _manifests_by_version(manifest_dir)
    cells = list(agent_cells(internal=internal))
    missing = [f"{label} ({v})" for _n, v, label in cells if manifests.get(v) is None]
    resolved = [(name, v, label, manifests[v]) for name, v, label in cells if v in manifests]
    written = 0
    print(f"[gen] smoke gate → {smoke_root}")
    for agent, version, label, manifest in resolved:
        for bench in BENCHMARKS:
            for key, variant in SMOKE_VARIANTS.items():
                _write(smoke_root / bench / f"{label}_{key}.yaml",
                       _dump(agent, bench, build_config(agent, bench, manifest, variant=variant,
                                                        version=version, internal=internal,
                                                        runtime=runtime, provider=provider,
                                                        forward_env=forward_env), smoke=True),
                       check=check)
                written += 1
    return written, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate runnable eval configs from onboarded manifests")
    ap.add_argument("--out", default=str(SMOKE_ROOT), metavar="DIR",
                    help=f"output root for the gate (default: {SMOKE_ROOT.name}/)")
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR), metavar="DIR",
                    help="where onboard filed the manifests (default: .beagle/agents)")
    ap.add_argument("--internal", action="store_true",
                    help="include Monet and default to Gateway Express on xrlenv-cluster")
    ap.add_argument("--runtime", choices=["local", "xrlenv-cluster"],
                    help="override the profile runtime (public: local; internal: xrlenv-cluster)")
    ap.add_argument("--provider-type", choices=["direct", "gateway", "internal"],
                    help="override the profile provider route type")
    ap.add_argument("--provider-name",
                    help="provider name (required with every --provider-type)")
    ap.add_argument("--provider-api-base",
                    help="gateway API base URL (required for provider type gateway)")
    ap.add_argument("--provider-api-key-env",
                    help="gateway API-key environment variable name")
    ap.add_argument("--provider-auth-header",
                    help="optional gateway authentication header")
    ap.add_argument("--forward-env", nargs="+", metavar="NAME",
                    help="override credential variables forwarded to the agent container")
    ap.add_argument("--reseed-smoke-tasks", action="store_true", dest="reseed_smoke_tasks",
                    help="re-sample the smoke gate's tasks from each benchmark's own task list "
                         "(needs the benchmark cache / dataset access) and rewrite "
                         "scripts/smoke_tasks.json, then exit")
    ap.add_argument("--check", action="store_true",
                    help="report what WOULD generate (and which agents aren't onboarded); write "
                         "nothing; exit non-zero if any agent in the matrix has no matching manifest")
    args = ap.parse_args(argv)

    try:
        provider = provider_override(
            provider_type=args.provider_type,
            name=args.provider_name,
            api_base=args.provider_api_base,
            api_key_env=args.provider_api_key_env,
            auth_header=args.provider_auth_header,
        )
    except ValueError as exc:
        ap.error(str(exc))

    if args.reseed_smoke_tasks:
        print(f"[gen] re-sampling smoke tasks → {SMOKE_TASKS_PATH}")
        reseed_smoke_tasks()
        return 0

    written, missing = generate(
        manifest_dir=Path(args.manifest_dir),
        smoke_root=Path(args.out),
        check=args.check,
        internal=args.internal,
        runtime=args.runtime,
        provider=provider,
        forward_env=args.forward_env,
    )
    print(f"[gen] {'would generate' if args.check else 'generated'} {written} config(s)")
    if missing:
        print(f"[gen] not onboarded (skipped): {', '.join(missing)} — onboard with the matching "
              f"--version, then re-run", file=sys.stderr)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
