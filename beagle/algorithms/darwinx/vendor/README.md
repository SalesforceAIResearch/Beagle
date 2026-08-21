# DarwinX — vendored source (`self_evolve` + `atelier` + `trace_analyzer`)

These three packages are a **verbatim copy** of the DarwinX self-evolution algorithm, dropped
in as-is (not a submodule/subtree). beagle *hosts* DarwinX; it does not fork or refactor it —
**the DarwinX authors own and evolve this directory in-repo.** The clean-room adapter that
wires it into beagle is the beagle-authored code one level up
(`beagle/algorithms/darwinx/{eval.py, meta_agent.py, algorithm.py}`).

## Why they live here (and stay top-level)

`atelier` and `trace_analyzer` use a few **absolute self-imports** (`from atelier.x`,
`from trace_analyzer.y`) and cross-import each other (`atelier → self_evolve`, both →
`trace_analyzer`). So they must remain importable by their **original top-level names**. This
`vendor/` directory has **no `__init__.py`** on purpose — it's not a subpackage; the launch
path (`DarwinX.evolve`) prepends it to `sys.path` so `import self_evolve` / `atelier` /
`trace_analyzer` resolve to these copies without rewriting a single import.

## Convention exemption (important)

beagle's rules — **no new env vars, no internal-repo names** — bind *beagle-authored* code.
This vendored subtree is **exempt**: it keeps its own names, its ~40 env vars, its internal
paths, its imports. Do **not** scrub or lint it against those rules; the authors refactor it on
their own schedule. (Analogous to `vendor/xrlenv`.)

## The seams the authors wire to (the whole integration surface)

DarwinX calls *out* at exactly these points; beagle already backs each — point the vendored
code at them instead of its hardcoded originals:

| DarwinX seam (in this vendored code) | Back it with (beagle, one level up) |
|---|---|
| **A · eval** — `codingbench_eval.py` shells `python -m runner.run <cfg>` and reads `run.json` | `python -m beagle.algorithms.darwinx.eval` (config→Runner→the `per_task_results` run.json shape it reads). See `../eval.py`. |
| **B · evolver** — the `meta_agent` dispatcher (`meta_agent.py`, `META_AGENT` env) | `../meta_agent.py` — `run()` → the injected beagle `Editor.edit` (`set_editor`); swaps the evolver by config. |
| **C · config/env** — `run_config.py` + ~40 env vars + repo paths | set from `BeagleConfig` at launch (`AgentSource` → repo/ref/entrypoint). |
| **D · trace QC** — `trace_analyzer/llm.py` hand-rolled LLM client | route through beagle's gateway config (don't add `TRACE_ANALYZER_*` env). |

## Launch (still to wire — §6.4)

`DarwinX.evolve()` (in `../algorithm.py`) is the seam that will: add this `vendor/` to
`sys.path`, prepare the host env (seam C), inject the evolver (seam B), and launch the driver
(`self_evolve`'s pipeline). It currently fails loud until that's wired + live-validated.
See `notes/darwinx-dropin-contract.md`.
