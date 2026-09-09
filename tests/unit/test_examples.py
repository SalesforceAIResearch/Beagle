"""The tracked example configs under ``examples/evaluation/``.

These are TEACHING material — one file per use case (choose an agent + benchmark, take a task
subset, mix benchmarks, pass@k, timeouts, retries, harness args), not one per agent×benchmark
cell. That job belongs to the smoke gate. They are hand-written and committed, so their `source` is a
placeholder: nobody's private experiment copy belongs in git.

The risk with hand-written examples is silent rot — a renamed field leaves a file that only fails
when a user copies it. So every example is loaded through the same seam `beagle evaluate` uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from beagle.cli._canonical import build_evaluation, build_evolution

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples" / "evaluation").glob("*.yaml"))
EVOLUTION_EXAMPLES = sorted((ROOT / "examples" / "evolution").glob("config*.yaml"))


def test_there_are_examples() -> None:
    assert EXAMPLES, "examples/evaluation/*.yaml is empty — the use-case examples are tracked"


def test_evolution_examples_load_through_the_evolve_seam() -> None:
    assert EVOLUTION_EXAMPLES, "examples/evolution/config*.yaml is empty"
    for path in EVOLUTION_EXAMPLES:
        cfg, _run_dir, _run_name = build_evolution(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        assert cfg.evolvee.name == "opencode"
        assert cfg.evolver.name == "cursor"
        assert cfg.benchmark is not None
        assert cfg.benchmark.name == "terminal_bench_2_1"
        assert cfg.benchmark.task_ids == ["gcode-to-text"]


def test_oss_evolution_example_is_local_and_portable() -> None:
    path = ROOT / "examples" / "evolution" / "config.oss.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["run"]["runtime"] == "local"
    assert raw["run"]["parallelism"] == 1
    assert "<your-org>" in raw["evolvee"]["harness"]["source"]["repo"]
    assert raw["evolvee"]["provider"] == {"type": "direct", "name": "openai"}
    assert "OPENAI_API_KEY" in raw["evolvee"]["forward_env"]
    assert raw["evolver"]["model"]["name"] == "auto"
    assert "version" not in raw["evolver"]["harness"]


def test_evolution_docs_do_not_reference_the_removed_quick_start_path() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "advanced.md",
        ROOT / "examples" / "evolution" / "README.md",
        ROOT / "examples" / "evolution" / "README.oss.md",
        ROOT / "examples" / "evolution" / "quick_start_inline.py",
        ROOT / "examples" / "evolution" / "quick_start_inline.oss.py",
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in paths
        if "examples/quick-start" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"stale examples/quick-start links: {offenders}"


def test_darwinx_vendor_readme_describes_the_current_packages() -> None:
    path = ROOT / "beagle" / "algorithms" / "darwinx" / "vendor" / "README.md"
    text = path.read_text(encoding="utf-8")
    for current in ("evolve/", "gate/", "dx_trace/", "DarwinX.evolve"):
        assert current in text
    for stale in ("`atelier`", "`self_evolve`", "`trace_analyzer`", "still to wire"):
        assert stale not in text


def test_root_quick_start_launches_evolution_through_the_cli() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    quick_start = text.split("## Quick Start", 1)[1].split('<a name="evaluate">', 1)[0]

    assert "scripts/generate_evolution_config.py" in quick_start
    assert (
        "beagle evolve --config "
        "tests/smoke/terminal_bench_2_1/opencode-1.18.16_darwinx_evolution_smoke2.yaml "
        "--dry-run"
    ) in quick_start
    assert "quick_start_inline.py" not in quick_start


def test_documented_darwinx_yaml_hparams_are_typed() -> None:
    """Every algorithm snippet in the concept guide must load through DarwinXConfig."""
    import re

    from beagle.algorithms.darwinx.config import DarwinXConfig

    text = (ROOT / "docs" / "darwinx-configuration.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    seen = 0
    for block in blocks:
        raw = yaml.safe_load(block)
        if not isinstance(raw, dict) or not isinstance(raw.get("algorithm"), dict):
            continue
        DarwinXConfig(**(raw["algorithm"].get("hparams") or {}))
        seen += 1
    assert seen >= 5, "the DarwinX guide lost its typed configuration examples"


def test_darwinx_guidance_relative_links_resolve() -> None:
    """The quick-start and concept guide should not ship links to renamed/missing files."""
    import re

    paths = [
        ROOT / "docs" / "darwinx-configuration.md",
        ROOT / "examples" / "evolution" / "README.md",
        ROOT / "examples" / "evolution" / "README.oss.md",
        ROOT / "beagle" / "algorithms" / "darwinx" / "vendor" / "README.md",
        ROOT / "scripts" / "README.md",
    ]
    broken = []
    for path in paths:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, f"broken DarwinX documentation links: {broken}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_loads_through_the_evaluate_seam(path: Path) -> None:
    cfg, _run_dir = build_evaluation(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert cfg.benchmark.name, f"{path.name}: no benchmark resolved"
    # a placeholder source, never a real private repo
    src = (yaml.safe_load(path.read_text(encoding="utf-8"))["agent"]["harness"].get("source") or {})
    assert "<your-org>" in src.get("repo", ""), (
        f"{path.name}: `source.repo` must stay a <your-org> placeholder — an example is committed, "
        f"and someone's private experiment copy is not ours to publish")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_names_registered_benchmarks_and_agents(path: Path) -> None:
    import beagle as bgl

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["agent"]["harness"]["name"] in bgl.agents.available()
    for group in raw["data"]:
        assert group["benchmark"] in bgl.benchmarks.available()


def test_the_mixture_example_actually_mixes() -> None:
    """The loader used to read only `data[0]`, so a mixture config silently scored its FIRST
    benchmark and looked like it worked. Pin the behaviour the example teaches."""
    path = next(p for p in EXAMPLES if "mixture" in p.name)
    cfg, _ = build_evaluation(yaml.safe_load(path.read_text(encoding="utf-8")))
    names = [b.name for b in cfg.all_benchmarks()]
    assert len(names) > 1 and cfg.is_mixture()
    assert cfg.benchmark.name == names[0]        # the primary is a real member of the mixture
    # each entry keeps its OWN selection — that is why a mixture isn't one longer task list
    assert all(b.task_ids for b in cfg.all_benchmarks())


def test_harness_extra_args_example_folds_named_knobs() -> None:
    from beagle.cli._canonical import agent_dict

    path = next(p for p in EXAMPLES if "harness-extra-args" in p.name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = agent_dict(raw["agent"])["config"]
    assert config["config_path"] == "src/minisweagent/config/mini.yaml"
    assert config["responses_api"] is False


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_is_publicly_reproducible(path: Path) -> None:
    """No internal deployment detail in a committed example.

    These files ship in the public mirror, and an internal gateway id / URL there is both a leak
    and unusable advice: a reader cannot route through infrastructure they have no access to.
    Direct provider access (an explicit provider name + that provider's key) is the portable form.
    """
    text = path.read_text(encoding="utf-8").lower()
    for internal in ("llm-gateway-express", "llm_gateway_express", "gateway_proxy"):
        assert internal not in text, f"{path.name}: names internal infrastructure ({internal})"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_provider_maps_to_a_known_host(path: Path) -> None:
    """Every direct example names its provider explicitly, and beagle resolves that route to an API
    host to allowlist on a network-restricted benchmark.

    A gateway-routed example is exempt: the host to allowlist is then the gateway's own `api_base`
    (which serves whatever model ids that gateway routes), and its key comes from
    `provider.extra_args.api_key_env`, not the provider's own variable.
    """
    from beagle.agents.core.litellm_gateway import provider_api_host
    from beagle.agents.core.provider import DirectProvider, GatewayProvider, provider_config

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    route = provider_config({"provider": raw["agent"].get("provider")})
    if isinstance(route, GatewayProvider):
        return
    assert isinstance(route, DirectProvider)
    assert route.name, f"{path.name}: teaching examples must name a direct provider explicitly"
    model = raw["agent"]["model"]["name"]
    host = provider_api_host(f"{route.name}/{model}")
    assert host, f"{path.name}: provider {route.name!r} maps to no known API host"
    # ...and the forwarded key belongs to that provider
    expected = {"api.openai.com": "OPENAI_API_KEY", "api.anthropic.com": "ANTHROPIC_API_KEY",
                "generativelanguage.googleapis.com": "GEMINI_API_KEY",
                "api.mistral.ai": "MISTRAL_API_KEY", "api.groq.com": "GROQ_API_KEY",
                "api.x.ai": "XAI_API_KEY"}[host]
    assert expected in (raw["agent"].get("forward_env") or []), (
        f"{path.name}: this {route.name!r} example needs {expected} in forward_env")


def test_benchmark_remarks_cover_every_registered_benchmark() -> None:
    """docs/benchmark-remarks.md is advisory — beagle never drops tasks for you — so it is only
    useful if it stays complete and matches the kits it mirrors."""
    import re

    import beagle as bgl

    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "benchmark-remarks.md").read_text(encoding="utf-8")
    for name in bgl.benchmarks.available():
        assert re.search(rf"^## `{re.escape(name)}`", doc, re.MULTILINE), (
            f"{name} is registered but has no section in docs/benchmark-remarks.md — a reader "
            f"cannot tell 'nothing to exclude' from 'nobody checked'")

    # every id the vendored kits gate on must be listed, or the advice is already stale
    for kit in ("swe_rebench", "terminal_bench_2_1"):
        sweep = root / "vendor" / "xrlenv" / "xrlenv_plugins" / "benchmarks" / kit / "run_full_sweep.sh"
        if not sweep.exists():          # vendored submodule not checked out
            continue
        block = re.search(r"^EXCLUDE=\((.*?)^\)", sweep.read_text(encoding="utf-8"), re.S | re.M)
        if not block:
            continue
        for line in block.group(1).splitlines():
            task = line.strip()
            if task and not task.startswith("#"):
                assert task in doc, f"{kit} gates on excluding {task!r}, which the doc doesn't list"


def test_user_facing_docs_do_not_name_an_unshared_harness() -> None:
    """monet is never published, so it cannot be the example agent in a file that ships.

    The port's own guard does not cover it (`INTERNAL_REPO_RE` is
    `coding-bench|self_evolve|atelier`), and the sanitizer only rewrites the internal ORG out of a
    URL — the name survives. Files with an `.oss.` sibling are exempt: the port promotes that
    variant over the internal one before anything is published.
    """
    root = Path(__file__).resolve().parents[2]
    surface = [root / "README.md", *(root / "docs").glob("*.md"),
               *(root / "examples").rglob("*.yaml"), *(root / "examples").rglob("*.md"),
               *(root / "examples").rglob("*.py")]
    offenders = []
    for path in surface:
        if ".oss." in path.name or path.with_name(
                path.stem + ".oss" + path.suffix).exists():
            continue                                   # the port publishes the .oss. variant
        if "monet" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        f"{offenders} name monet and ship as-is — use a public harness (mini-swe / opencode) in "
        f"user-facing examples, or add an .oss. variant")


def test_the_gateway_example_seals_egress_to_the_gateway() -> None:
    """The org-gateway example promises that a filtered-egress benchmark (deep-swe seals the RUN
    phase to `network_hosts`) still reaches the gateway. Pin that: the built agent must advertise
    the declared `api_base` and NOTHING else — advertising the model provider's public host instead
    would both break the run and defeat the point of the example."""
    import beagle as bgl
    from beagle.config import AgentConfig

    from beagle.cli._canonical import agent_dict

    path = next(p for p in EXAMPLES if "gateway" in p.name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["data"][0]["benchmark"] == "deep-swe", (
        f"{path.name} teaches the airgapped case — it must run on a filtered-egress benchmark")
    agent = bgl.agents.build(AgentConfig.model_validate(agent_dict(raw["agent"])))
    assert agent.network_hosts() == [raw["agent"]["provider"]["extra_args"]["api_base"]]
