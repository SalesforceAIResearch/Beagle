# `atelier` — verification + test-time-scaling layer for `self_evolve`

Atelier is a parallel module to `self_evolve` that adds structural,
behavioral, and adversarial defenses on top of self_evolve's existing
Layer-1 (prompt), Layer-2 (content scan), and Layer-3 (canary task) guards.

## Modules

| File | Purpose | Status |
|---|---|---|
| `scope_filter.py` | Layer-4: apples-to-apples path allowlist on monet_code mutations | week 1 |
| `honeypot.py` | Layer-5: reward-hacking honeypot (Terminal Wrench corpus) | week 1 |
| `best_of_n.py` | Test-time Best-of-N trajectory selection using `verifier` | week 1 (skeleton) → week 2 (impl) |
| `verifier.py` | LLM-as-a-Verifier (criteria decomposition + repeated verification + tournament) | week 2 |
| `verifier_fitness.py` | Use `verifier` scores as parent-selection signal in `self_evolve` | week 2 |
| `transfer.py` | Layer-6+7: cross-model + cross-benchmark transfer gates | week 2 |
| `gate.py` | Orchestrator: ties all layers into a promote-or-reject decision per node | week 1 (skeleton) → week 2 (full) |
| `verifier_backend.py` | OpenAI-compatible `VerifierBackend` w/ `core.credentials` adapter | week 2 |
| `runners/batch_harbor.py` | Shared batched-Harbor invocation used by honeypot + transfer | week 3 |
| `runners/honeypot_harbor.py` | `HarborHoneypotRunner` implementing `HoneypotRunner` against Harbor | week 3 |
| `runners/transfer_harbor.py` | `HarborTransferEvaluator` implementing `TransferEvaluator` against Harbor | week 3 |
| `trajectory_loader.py` | Harbor trial dir → `verifier.TrajectoryInput` (powers `ATELIER_FITNESS_ALPHA`) | week 3 |
| `proposer.py` | (Optional stretch) Monet-as-proposer replacement for cursor-agent | week 4 |

## Where each defense slots in

```
self_evolve campaign
   ├── Layer 1 (prompts)         ── existing
   ├── Layer 2 (diff content)    ── existing
   ├── Layer 3 (canary task)     ── existing
   └── mini-eval / final-eval

   ↓ candidate node lands in archive

Atelier verification
   ├── Layer 4 (path scope)      ── atelier/scope_filter.py
   ├── Layer 5 (honeypot)        ── atelier/honeypot.py
   ├── Layer 6 (cross-model)     ── atelier/transfer.py
   ├── Layer 7 (cross-benchmark) ── atelier/transfer.py
   └── verifier trajectory check ── atelier/verifier.py

   ↓ promote-or-reject

Atelier test-time
   └── Best-of-N (verifier-tournament)  ── atelier/best_of_n.py
```

The orchestrator in `gate.py` runs Layers 4–7 + verifier in order, with
cheap layers first and expensive ones gated on prior passes.

## Apples-to-apples surface (the "harness")

Layer 4's allowlist mirrors the harness slots every production coding
agent (Claude Code, Codex, Gemini CLI) ships in. Within `monet_code/`:

**Allowed**:
- `src/core/bundled-skills.js` — bundled skills (macro)
- `src/core/agents.js` — agent registry (macro)
- `src/core/context.js` — `basePrompt()` template (macro)
- `src/core/hooks.js` — bundled hooks (micro)
- `src/tools/*.js` — tool descriptions (semantic; content-level check still applies)
- `.monet/agents/**/*.md` — default agent templates (macro)

**Denied**:
- `src/query/**` — control loop, compaction, streaming
- `src/api/**` — provider clients
- `src/core/{permissions,sandbox}.js` — security
- `src/mcp/**`, `src/swarm/**`, `src/tui/**`, `src/bridge/`, `src/buddy/`

A path neither allowed nor denied (e.g., `src/cli/`, `bin/`) is "outside
allowlist" — a softer flag than "denied" but still a violation.

## Modes

`scope_filter.ScopeMode`:
- `SOFT_FLAG` (default): record violations but don't reject. Used for
  measurement — answers "what fraction of self_evolve's discoveries
  actually fit a typed harness surface?"
- `STRICT_REJECT`: reject the commit. Used in production after we've
  confirmed the search behaves reasonably under the constraint.

`honeypot.HoneypotMode`:
- `MEASURE` (default): score candidates on Terminal Wrench, record gap
  vs baseline.
- `GATE`: reject if pass-rate gap exceeds threshold.

## Tests

All modules ship with unit tests under `tests/atelier/`. Tests use
fixtures (no network calls); integration tests requiring Harbor /
cursor-agent live in `tests/integration/atelier/`.

## Apples-to-apples vs zeyuan's existing layers

| Layer | Where | What it catches |
|---|---|---|
| L1 | `prompts/{analyze,implement,review}.md` | "Don't hardcode task names" via prompt |
| L2 | `self_evolve/generalization.scan_diff` | Task-name strings, narrowing conditionals, copied verifier output (content patterns) |
| L3 | `self_evolve/generalization.pick_canary_tasks` + mini-eval | Mutation regresses a previously-passing task |
| **L4 (new)** | `atelier/scope_filter` | Mutation touches files outside the apples-to-apples harness surface |
| **L5 (new)** | `atelier/honeypot` | Candidate's pass rate on reward-hackable environments rises (learned to game) |
| **L6 (new)** | `atelier/transfer` | Cross-model regression |
| **L7 (new)** | `atelier/transfer` | Cross-benchmark regression |

L4 is complementary to L2 — they catch different overfitting shapes. L2
might pass a beautifully general patch that happens to rewire the query
loop; L4 catches it because the query loop isn't apples-to-apples.

## `self_evolve` integration (opt-in)

Atelier is wired into `self_evolve.pipeline._finalize` via the
`self_evolve.atelier_hook` shim. The integration is opt-in via env vars
so default `self_evolve` behavior is unchanged:

| Env var | Default | Effect |
|---|---|---|
| `ATELIER_GATE_ENABLED` | `0` | When `1`, after each node's final eval the gate runs and a per-node decision JSON is written to `reports/<campaign>/atelier/<node_id>.json` |
| `ATELIER_FITNESS_ALPHA` | `0` | When > 0, blends pass-rate with verifier score before writing `Node.score` (affects parent selection). Live: trajectory loader walks the final-eval Harbor job dir, scores per-task with `Verifier`, blends per `FitnessRunner` |
| `ATELIER_VERIFIER_MODEL` | `gpt-5.4-mini` | Model used for verifier scoring |
| `ATELIER_VERIFIER_PROVIDER` | `sfr_gateway` | Credential profile passed to `core.credentials.resolve()` |
| `ATELIER_SCOPE_MODE` | `soft_flag` | `soft_flag` records scope violations without rejecting; `strict_reject` rejects |
| `ATELIER_REPORTS_SUBDIR` | `atelier` | Override the subdirectory under `reports/<campaign>/` where decision sidecars are written |

The hook lives in `self_evolve/atelier_hook.py` and is
the **only** place `self_evolve` imports `atelier`. Errors inside the
hook are logged but never raised — Atelier can never crash a
self_evolve pipeline run.

## Terminal Wrench honeypot corpus

The honeypot needs the Terminal Wrench corpus locally. The repository
is cloned into `external/terminal-wrench/` (which is `.gitignore`d).
We use a partial / sparse-checkout layout so the working tree carries
only what the corpus loader needs:

- `index/tasks.json` — task index (used by `HoneypotCorpus.default_task_ids`)
- `dataset_manifest.json` — corpus manifest
- `tasks/<task_id>/<model>/task.json` — per-task metadata
- `tasks/<task_id>/<model>/original_task/` — the actual task spec
  (instruction.md + environment/tests/solution + task.toml) used
  for deployment

Trajectory dumps (`baseline_trajectories/`, `hack_trajectories/`,
`sanitized_trajectories/`, `stripped_trajectories/`) are **not**
needed — Atelier doesn't read prior trajectories, it generates fresh
ones by running candidate monet_code against the task.

## Harbor runner adapters

The `atelier.runners` subpackage provides the production wiring from
Atelier's protocols to Harbor:

- `HarborHoneypotRunner` (implements `HoneypotRunner`)
- `HarborTransferEvaluator` (implements `TransferEvaluator`)

Both share `BatchHarborRunner`, which runs Harbor once for the whole
task batch and caches per-task results. This keeps the protocol-per-task
ergonomics while amortizing Harbor's startup cost.

The Harbor config files are caller-supplied — Atelier's adapters don't
know whether a config points at TB-2, TW, or SWE-bench, so the same
adapter is reused for honeypot vs cross-model vs cross-benchmark by
swapping `HarborInvocation.config_path` / `extra_env`.

Production configs (e.g. `configs/terminal_wrench_honeypot.yaml`,
`configs/terminal_bench_2_xfer_gpt55.yaml`) are added per-campaign;
they are simple sibling YAMLs of `configs/terminal_bench_2.yaml` with
different `dataset:` / `monet.model:` values.

## Campaign speedup levers (live in `self_evolve`, not `atelier`)

The campaign wall-clock optimization pass introduced two opt-in knobs.
They live under `self_evolve/` because they're general — every
campaign benefits whether or not Atelier is active — but they're
documented here because they compose with the verifier-fitness work
to make end-to-end Atelier iterations affordable.

| Knob | What it does | Default |
|---|---|---|
| `harbor.n_concurrent_self_evolve` (YAML) | Per-eval container concurrency for mini-eval + final-eval | unset → 2 (historical); `6` in `configs/terminal_bench_2.yaml` after the speedup pass |
| `MONET_EVAL_CACHE_ENABLED` (env) | Skip Harbor for `(task, monet_sha, config)` combos already evaluated. Cache invalidates on commit / config / lock-file / runtime env changes. | `0` |
| `MONET_EVAL_CACHE_TTL_DAYS` (env) | Cache entry max age before forced miss | `14` |
| `MONET_EVAL_CACHE_VALIDATE_EVERY_N` (env) | Force a fresh re-eval for 1-in-N lookups (cache drift detection) | `0` (off) |
| `MONET_EVAL_CACHE_ROOT` (env) | Override the cache directory | `<repo>/.eval_cache` |

Per-pipeline timing sidecars land at
`reports/<campaign>/timing/<pipeline_id>.json`.

> **Not yet ported into coding-bench.** The eval cache module
> (`eval_cache.py` — content-addressed, one JSON file per `(task,
> monet_sha, monet_lock_sha, config_fp, dataset_id, runtime_env_fp)`
> tuple) and the timing-report aggregator
> (`scripts/self_evolve_timing_report.py`) live in `monet_code_eval` but
> have not been ported to this branch. The `MONET_EVAL_CACHE_*` knobs
> above are documented for parity only and are inert here until the cache
> module is ported.

## Smoke driver (`scripts/atelier_smoke.py`) — not yet ported

> **Not yet ported into coding-bench.** The `scripts/atelier_smoke.py`
> driver lives in `monet_code_eval` but has not been ported to this
> branch (its test suite, `tests/atelier/test_atelier_smoke.py`, skips
> accordingly). The description below documents the intended driver for
> parity; the commands will not run until the script is ported.

End-to-end sanity check for the trajectory loader + verifier + fitness
blending stack. Runs against any existing Harbor job dir — does **not**
require a self_evolve campaign.

```
# Dry-run: load trajectories + show what would be scored.
python scripts/atelier_smoke.py jobs/<job_dir> --dry-run

# Live: score 5 trajectories via SFR Gateway gpt-5.4-mini.
python scripts/atelier_smoke.py jobs/<job_dir> --max-tasks 5 --alpha 0.3

# Write full per-task JSON report.
python scripts/atelier_smoke.py jobs/<job_dir> --out reports/smoke.json
```

Prints a per-task table (pass-rate from `result.json` + verifier
aggregated score) and the blended fitness summary. Cost ≈ <$0.01 per
task on `gpt-5.4-mini` at default config.

Use this to validate Atelier's verifier on real trajectories before
enabling `ATELIER_FITNESS_ALPHA` in a live campaign.
