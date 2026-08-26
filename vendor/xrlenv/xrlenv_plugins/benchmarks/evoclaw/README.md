# EvoClaw — xrlenv onboarding

EvoClaw (per-repo milestone SWE tasks) run on the xrlenv cluster. xrlenv manages only
EvoClaw's **Docker containers + images**; EvoClaw's own harness (orchestrator, milestone
DAG, agents, evaluator) runs **unchanged**. Its xrlenv shape is the **interceptor
outlier** (GUIDELINE §7.1): an in-process shim reroutes EvoClaw's raw `docker` CLI calls
to the cluster, so the whole benchmark runs from the **EvoClaw checkout** (the "call
site"), not from xrlenv. The **oracle** agent (no LLM) applies each milestone's golden
solution end-to-end (a `resolved` verdict proves the path); swap `--agent`/`--model` for
a real LLM.

## What's here

Outlier layout — top level is docs + the one image xrlenv builds; **everything executable
is in `copy_to_call_site/`**, staged into the EvoClaw checkout as `xrlenv_onboard/` and run
from there.

| path | runs where | role |
|---|---|---|
| `go-zero-gitfix.Dockerfile` | xrlenv host | the ONE image xrlenv builds (§2) |
| `README.md` · `STATUS.md` | — | docs |
| `copy_to_call_site/run_full_sweep.sh` | call site | the sweep entrypoint (§3) |
| `copy_to_call_site/run_all_xrlenv.py` | call site | batch sweep driver (EvoClaw's `run_all`, xrlenv-adapted) |
| `copy_to_call_site/run_e2e_xrlenv.py` | call site | per-trial worker (wraps EvoClaw's `run_e2e`) |
| `copy_to_call_site/oracle.py` | call site + container | golden-solution oracle (also the golden-tar cache builder) |
| `copy_to_call_site/image_resolution.py` | call site | maps EvoClaw image names → pullable Docker Hub refs (§2) |
| `copy_to_call_site/docker_shim.py` | call site | the `docker` CLI interceptor |
| `copy_to_call_site/{env_loader,workspace,yd_fixes,cpu_pinning}.py`, `_yd_bootstrap/` | call site | env load / workspace / correctness fixes / cpu pinning |
| `copy_to_call_site/tests/` | — | offline unit tests |

Canonical phases (some are embedded / N/A — documented, not stubbed): **build-cache** =
golden-tar cache in `oracle.py`; **image-plan** = in-process `image_resolution.py` (no
`build_plan.yaml` — resolved at run time); **run-sweep** = `run_all_xrlenv.py`; **no
`patches/`** (corrections are runtime monkeypatches in `yd_fixes.py`).

```
stage copy_to_call_site/ → EvoClaw checkout xrlenv_onboard/ ─▶ run_full_sweep.sh (run_all_xrlenv.py, oracle) ─▶ resolved/98
```

## 1. Set up + stage (once)

1. **EvoClaw itself** — follow EvoClaw's own README §Setup 0–2 (creates the `.venv`,
   clones the `EvoClaw-data` dataset via `git lfs`; note its absolute path).
2. **Install xrlenv into EvoClaw's venv:** `cd <EvoClaw-Repo> && uv pip install -e <path-to-xrlenv>`.
3. **Stage the payload** into the checkout (re-run after editing payload files):

   ```bash
   rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' --exclude 'tmp/' \
     <path-to-xrlenv>/xrlenv_plugins/benchmarks/evoclaw/copy_to_call_site/ \
     <EvoClaw-Repo>/xrlenv_onboard/
   ```

4. **`.env_private`** at the EvoClaw repo root — deployment/data only (every *behavioural*
   knob is a flag). Precedence: shell var > `.env_private` > EvoClaw's `.env`.

   ```dotenv
   EVOCLAW_DATA_ROOT=/abs/path/to/EvoClaw-data          # REQUIRED — shared dataset
   XRLENV_GRPC_HOST=<control-plane-host>                # REQUIRED (+ XRLENV_GRPC_PORT=50051)
   XRLENV_CONSUMER_TOKEN=...                            # REQUIRED — cluster auth
   EVOCLAW_GOZERO_BASE_IMAGE=<registry>:5011/go-zero:base-v0.9-gitfix   # go-zero base (§2)
   ```

Run every command below **from the EvoClaw checkout** — the wrappers derive the project
root as the parent of `xrlenv_onboard/`, import EvoClaw's `harness.*` there, and default
results/ + golden_cache/ under it.

## 2. Prepare the images

**You almost never build an image.** EvoClaw's milestone + agent-base images are its own
(`hyd2apse/<short>:<milestone>-v0.9`, `…:base-offline-v0.9`), published by EvoClaw's
`scripts/pull_images.sh` and **pulled** via the cluster's Docker Hub mirror. At acquire,
`image_resolution.py` (an in-process patch of EvoClaw's `resolve_image` — it imports
`harness.e2e.*`, so it lives at the call site) maps EvoClaw's internal retag name to the
pullable ref. Nothing is serialized to a `build_plan.yaml`.

**The one image xrlenv builds** is the go-zero corrected base (upstream's `base-v0.9`
ships without `.git` → runs stall at `Completed: 0`):

```bash
set -a; source ./.env; set +a 
docker build -f xrlenv_plugins/benchmarks/evoclaw/go-zero-gitfix.Dockerfile \
  -t "$XRLENV_PRIVATE_REGISTRY_HOST:$XRLENV_PRIVATE_REGISTRY_PORT/go-zero:base-v0.9-gitfix" .
docker push "$XRLENV_PRIVATE_REGISTRY_HOST:$XRLENV_PRIVATE_REGISTRY_PORT/go-zero:base-v0.9-gitfix"
```

Then set `EVOCLAW_GOZERO_BASE_IMAGE` to that ref (§1); `image_resolution.py` redirects
**only** go-zero's `base` ref. Build details: `go-zero-gitfix.Dockerfile`. (Known drift:
that redirect still *defaults* to a baked host `<registry-host>:5011/…` — set the var.)

## 3. Run the oracle sweep

The headline correctness sweep (**92/98**), from the EvoClaw checkout after staging:

```bash
bash xrlenv_onboard/run_full_sweep.sh                 # WORKERS=64, RUN_NAME=... override via env
```

It runs `run_all_xrlenv.py --parallelization-level milestone --apply-yd-fixes`. Spot-check one milestone / one repo:

```bash
.venv/bin/python xrlenv_onboard/run_e2e_xrlenv.py --agent oracle --model none --force \
  --repo-name navidrome_navidrome_v0.57.0_v0.58.0 --milestones 1
```

A real LLM uses the **same** driver: `--agent claude-code --model <m>` + `UNIFIED_API_KEY`
/ `UNIFIED_BASE_URL`. Results land in `<checkout>/results/<run-name>__<ts>/`; start at
`xrlenv_summary.json` (real `resolved` counts, worst-first).

Facts:
- **`--apply-yd-fixes`** (off by default) — **required for a correct result set** (runtime
  monkeypatches for known upstream eval-protocol bugs; harness files stay pristine). Off =
  byte-faithful / leaderboard-comparable, and a loud warning prints. 92/98 with it.
- **`--fleet`** (on by default) — reserves each task's agent+eval footprint on one node so
  evals aren't starved; `--workers` above capacity is safe (tasks queue as whole fleets).
- **`--parallelization-level milestone`** — one task per milestone; use to saturate the
  cluster (`repo` caps at `#repos × 4`).
- **quarantine** (anti-cheat) is auto-on per repo; the **go-zero base fix** is on by default.
- **`--copy-testbed`** (off) — opt into EvoClaw's whole-`/testbed` debug copy (a 128 MiB
  control-plane relay cap contains it; `ArchiveTooLarge` in a log is the cap working).
- **cannot-solve without sysbox:** dubbo `M001.1`/`M001.2` and nushell `G01` are **content**
  hard-fails (no DinD involved). element-web `maintenance_ui_ux` is the one that needs
  **Docker-in-Docker** — the headline 92/98 doesn't route it to sysbox, so it fails there.
- **recover `maintenance_ui_ux` with sysbox:** the cluster runs a working `sysbox-runc` pool
  (seta/TW DinD tasks use it), so add `--sysbox-milestone <element-web-repo>/maintenance_ui_ux`
  to run it under an unprivileged inner `dockerd` → up to ~93–95/98. Unvalidated for evoclaw
  (the onboarding predates the cluster's sysbox), but the mechanism is proven. See DESIGN.md
  → *element-web … DinD*.

Most-used flags (full set: `run_all_xrlenv.py --help` / `run_e2e_xrlenv.py --help`):

| flag | default | what |
|---|---|---|
| `--run-name` | *(required)* | run label → `results/<name>__<ts>/` |
| `--parallelization-level` | `repo` | `repo` (per-repo) or `milestone` (per-milestone; saturates) |
| `--workers` | `#tasks`/8 | tasks in flight at once |
| `--repos` / `--milestones` | *(all)* | restrict the run |
| `--apply-yd-fixes` | off | all correctness fixes (see above) |
| `--cpu-pinning-milestone <id>` | *(none)* | dedicated cores for a Table-A contention milestone (repeatable) |
| `--sysbox-milestone <id>` | *(none)* | route a DinD milestone (element-web `maintenance_ui_ux`) to the cluster's `sysbox-runc` pool (repeatable) |
| `--mem-per-cpu-gb` | 2 | per-container mem cap + fleet reservation rate |
| `--agent` / `--model` | `oracle`/`none` | agent + model (real LLM needs a real `--model` + `UNIFIED_*`) |
| `--dry-run` | off | print the plan + fleet footprint, launch nothing |

## 4. Warm / calibrate

**N/A.** There is no committed build plan to warm — EvoClaw's images are pulled on acquire
(LRU-managed) and the one built image (§2) is pushed directly to the registry.

## Troubleshooting

- **`get_agent_framework` / registration error** — not running through the wrapper; use
  `xrlenv_onboard/run_e2e_xrlenv.py` (it registers `oracle` + relaxes `--agent`).
- **`XRLENV_GRPC_HOST is required`** — run from the EvoClaw checkout so `.env_private` is found.
- **`run_e2e` demands `UNIFIED_API_KEY`** even for the oracle — set a dummy (`UNIFIED_API_KEY=none`).
- **`ControlPlaneLost` / `Connection refused` mid-run** — CP restarted / node dropped; the shim
  retries transient blips (`--cluster-retries`) then re-raises so EvoClaw re-runs the eval. Don't
  `xrlenv up` during a run.
- **A milestone stalls at `Completed: 0`** — raise `--oracle-tag-settle-s` (default 12 s); for
  go-zero specifically, check `EVOCLAW_GOZERO_BASE_IMAGE` (§2).
- **`ArchiveTooLarge` / `transfer refused`** under `--copy-testbed` — the 128 MiB relay cap
  working, not a failure (the eval still grades). See DESIGN.md.

## See also

- [`DESIGN.md`](DESIGN.md) — the *why* (fleet-reservation math, `--apply-yd-fixes`, quarantine,
  go-zero base, element-web memory/DinD + sysbox, the whole-`/testbed` relay cap). The exact
  intercepted `docker` surface is in `copy_to_call_site/docker_shim.py`.
- `GUIDELINE_onboard_benchmarks.md` §7.1 — the interceptor pattern (reference for other raw-CLI harnesses).
- [`STATUS.md`](STATUS.md) — the milestone disposition (the 94/98 tiers) + reproduce command.
- `docs/supported_benchmarks_and_harnesses/evoclaw.md` — the Sphinx user page.
