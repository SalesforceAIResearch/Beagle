"""The tracked example configs under ``examples/evaluation/``.

These are TEACHING material — one file per use case (choose an agent + benchmark, take a task
subset, mix benchmarks, pass@k, timeouts, retries), not one per agent×benchmark cell. That job
belongs to the smoke gate. They are hand-written and committed, so their `source` is a
placeholder: nobody's private experiment copy belongs in git.

The risk with hand-written examples is silent rot — a renamed field leaves a file that only fails
when a user copies it. So every example is loaded through the same seam `beagle evaluate` uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from beagle.cli._canonical import build_evaluation

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples" / "evaluation").glob("*.yaml"))


def test_there_are_examples() -> None:
    assert EXAMPLES, "examples/evaluation/*.yaml is empty — the use-case examples are tracked"


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


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_is_publicly_reproducible(path: Path) -> None:
    """No internal deployment detail in a committed example.

    These files ship in the public mirror, and an internal gateway id / URL there is both a leak
    and unusable advice: a reader cannot route through infrastructure they have no access to.
    Direct provider access (model name + that provider's key) is the portable form.
    """
    text = path.read_text(encoding="utf-8").lower()
    for internal in ("llm-gateway-express", "llm_gateway_express", "gateway_proxy"):
        assert internal not in text, f"{path.name}: names internal infrastructure ({internal})"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_model_maps_to_a_known_provider(path: Path) -> None:
    """The model name is what selects the provider, and beagle resolves it to an API host to
    allowlist on a network-restricted benchmark. A name outside that map would run but could not
    be allowlisted — so an example must not teach one."""
    from beagle.agents.core.litellm_gateway import provider_api_host

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = raw["agent"]["model"]["name"]
    host = provider_api_host(model)
    assert host, f"{path.name}: model {model!r} maps to no known provider host"
    # ...and the forwarded key belongs to that provider
    expected = {"api.openai.com": "OPENAI_API_KEY", "api.anthropic.com": "ANTHROPIC_API_KEY",
                "generativelanguage.googleapis.com": "GEMINI_API_KEY",
                "api.mistral.ai": "MISTRAL_API_KEY", "api.groq.com": "GROQ_API_KEY",
                "api.x.ai": "XAI_API_KEY"}[host]
    assert expected in (raw["agent"].get("forward_env") or []), (
        f"{path.name}: model {model!r} needs {expected} in forward_env")


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
