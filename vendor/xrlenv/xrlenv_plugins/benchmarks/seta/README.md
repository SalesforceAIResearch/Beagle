# seta-env

`seta-env` ([camel-ai/seta-env](https://github.com/camel-ai/seta-env)) is a
harbor-format task corpus namespaced under the `seta-env/` shard of the shared
harbor cache. Its xrlenv shape is the **harbor golden path** — the same one-line
`import_path` swap terminal-bench-2 uses:

```diff
 environment:
-  import_path: harbor.environments.docker.docker:DockerEnvironment
+  import_path: xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster
```

Its **one distinctive trait**: seta-env tasks do **not** ship a prebuilt
`docker_image`. Each ships an `environment/Dockerfile` that is *built* — so the
image-prep step is a build (`context_source: type: git`), not a warm-only pull.

## What's here

| File | Role |
|---|---|
| `build_cache.py` | Populate the cache (clone `camel-ai/seta-env`) **and** write the DinD sysbox routing markers — `--stage all` (default), idempotent |
| `build_plan_gen.py` | Emit the `type: git` image build plan (one entry per `Harbor-Dataset/<id>/environment`) |
| `build_plan.yaml`, `build_plan_1376_full.yaml` | Committed build plans (16-task starter; full 1376-task set) |
| `black_list.txt` | Excluded task ids — **build-unbuildable** (5) **+ runtime-excluded** (86, reasons inline); the single exclusion source honored by the generator, the sweep, and `run_full_sweep.sh`. Root-cause + fix-status: `STATUS.md` |
| `patches/` | Curated migration-repair overlays applied by `--stage all` (§1.1); `patches/README.md` |
| `run_oracle_sweep.py` | Oracle sweep on local Docker (`--local`) or the xrlenv cluster (default); `reward > 0` PASS/FAIL gate |
| `run_full_sweep.sh` | The entrypoint / CI gate — (re)build cache + run the green set (present − `black_list.txt`), content-retrying reward-0 flakes |
| `STATUS.md` | Current oracle-sweep disposition + reproduce command |
| `tests/` | Offline unit tests for the pure generator / sweep / build_cache logic |

```
camel-ai/seta-env (git) ──①build_cache──▶ <cache>/seta-env/<id>/  (Dockerfile, no prebuilt image)
                                              │
        ②build_plan_gen ──▶ type: git plan ──▶ build+push ──▶ <registry>/seta-env/<id>:main
                                              │
                        ③run_full_sweep ──▶ per-task reward>0 PASS/FAIL on the cluster
```

## 1. Build the cache

`build_cache.py` (default **`--stage all`**) yields a correct cache in one command:
**populate** — clone `camel-ai/seta-env` and land each `Harbor-Dataset/<id>/` under
`<cache>/seta-env/<id>/`; then apply two kinds of cache-level repair (no separate
stage for either — they're just part of building a correct cache):
**migration-repair overlays** (`patches/`, §1.1) and **DinD sysbox markers** (§1.2).
All idempotent (the clone is skipped once the shard holds tasks; overlays + markers
re-apply), so it's safe to re-run — seta lives in its own `seta-env/` shard of the
same `XRLENV_BENCHMARK_CACHE`, so it never collides with terminal-bench-2.

```bash
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache
.venv/bin/python xrlenv_plugins/benchmarks/seta/build_cache.py      # --stage all (default)
```

`run_full_sweep.sh` runs this same `--stage all` as its step 1, so an operator who
runs the standard entrypoint (§3) gets the fixes automatically — **there is no
separate step to remember.** `--stage populate` clones only (no overlays/markers);
`--stage sysbox` (re)writes only the markers. Exclusions (build-unbuildable +
runtime-excluded) stay in `black_list.txt`, handled at build-plan + sweep time
(§2 / §3); root-cause + fix-status for the runtime set is in `STATUS.md`.
`--dest` overrides the cache root (defaults to `$XRLENV_BENCHMARK_CACHE`, else
`~/.cache/harbor/tasks`); `--repo` / `--ref` override the upstream git URL / ref.

### 1.1 Migration-repair overlays (`patches/`)

The `camel-ai/seta-env` **Harbor-Dataset** conversion dropped runtime-critical
config the original **`Dataset/<id>/`** (pre-Harbor Terminal-Bench format) still
has. `patches/<id>/<rel>` files are full-file overlays copied over the populated
task by `--stage all`; each restores the smallest faithful piece. Harbor reads
`solution/solve.sh` from this cache, so a solve.sh overlay needs no rebuild.
Current: **309** — solve.sh guarded on the pre-Harbor `/oracle` run path, but harbor
runs the oracle from `/solution`; one-line repair, validated reward 1.0. Details +
the running list: [`patches/README.md`](patches/README.md).

### 1.2 Task-level cache fixes — DinD sysbox markers

A few seta tasks are **Docker-in-Docker**: their oracle runs `docker …` against a
daemon that plain `runc` can't host (`Cannot connect to the Docker daemon`). The
`sysbox` step of `--stage all` writes `[environment.env] XRLENV_CONTAINER_RUNTIME
= "sysbox-runc"` (+ `XRLENV_INNER_DOCKERD`, plus `XRLENV_INSTALL_DOCKERD` for a
CLI-only image) into their `task.toml`, so the cluster plug-in routes them to a
sysbox node and brings up an unprivileged nested dockerd. This is task-level
*routing*, not content — the same markers terminalworld uses. The set
(`SYSBOX_TASKS` in `build_cache.py`, grown one **validated** task at a time):

| task | markers | why |
|---|---|---|
| `8` | `sysbox-runc` + inner dockerd | image ships full `docker-ce`; `docker ps` needs a live daemon |
| `1004` | `sysbox-runc` + inner dockerd + install dockerd | multi-network DinD; image is `docker-ce-cli` only (no daemon) |

Both validated end-to-end on the dev sysbox cluster (oracle reward 1.0). A marked
task **hard-fails on a cluster with no sysbox node** — for a runc-only cache use
`--stage populate`, not `all`. To (re)apply the markers by hand — e.g. after a
manual re-populate — `--stage sysbox` runs just this step; but the normal path is
never to touch it directly: `run_full_sweep.sh` / `--stage all` already do.

### 1.3 Task-level cache fixes — base-image restore

The Harbor migration swapped every task's original
`FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:<date>` base (which bakes
python3/curl/uv/wget/tmux/…) for bare `FROM ubuntu:24.04`, so tasks whose
**identical** `solve.sh` assumes one of those tools fail `command not found`.
`--stage all` rewrites the `FROM` back to the t-bench base in the
`BASE_IMAGE_FIX_TASKS` (in `build_cache.py`) cache Dockerfiles, and **`build_plan_gen`
builds those tasks `context_source: type: local`** from the restored cache Dockerfile
(a `type: git` build would use upstream's broken `FROM`). **10 tasks** (python3 ×6,
curl ×2, uv, tmux) — rebuilt + validated 10/10 green (367 → python3 3.12.3). This
needs an **image rebuild** — see §2. The one legacy case (`197`, `FROM ubuntu:14.04`
EOL) can't be fixed by a base swap and stays in `black_list.txt`.

### 1.4 Task-level cache fixes — dropped-command restore

The migration dropped a non-trivial single-service compose `command:` (a service,
spawned workers, seeded state the task premise needs) from some tasks, so harbor boots
the bare image and the oracle fails. Harbor's raw-container path sets docker
`CMD = sleep infinity` (**overriding** any baked `CMD`) but preserves the image
`ENTRYPOINT`, so `--stage all` bakes a boot wrapper as the Dockerfile `ENTRYPOINT` —
`( <cmd> ) & exec "$@"` (recovered command runs in the background, then execs harbor's
`sleep infinity` as PID 1). Like §1.3, `build_plan_gen` builds these
`context_source: type: local` and this needs an **image rebuild** (§2). **8 tasks**
(`DROPPED_COMMAND_TASKS` in `build_cache.py`, oracle-validated 8/11); commands are verbatim
from `git show HEAD:Dataset/<id>/docker-compose.yaml`. Three candidates were pulled after the
oracle (`775` content/setup gap, `1246` timing race, `1309` needs systemd). Dropped-command
tasks that ALSO need a capability/systemd (`144`/`164`/`189`/`960`/`1309`) stay blacklisted —
they need a sysbox marker (§1.2) too.

### 1.5 Task-level cache fixes — verifier-as-root

A task whose Dockerfile sets a non-root `USER` (`[metadata] sets_custom_user = true`) has
harbor 0.20 run its **verifier** as that non-root user (harbor `single_step.py` runs the
verifier as `task.config.verifier.user`, which defaults to the image USER). But
terminal-bench's verifier contract is **root**: the stock `tests/test.sh` does
`apt-get install curl` (the uv bootstrap) and the tests do `su -l <user>` — both need root.
As non-root the bootstrap dies (`curl: command not found`) → reward 0. `--stage all` writes
`[verifier] user = "root"` into these tasks' task.toml (`VERIFIER_ROOT_TASKS` in
`build_cache.py`) — surgical (only the verifier phase; the agent/solve still runs as the
task user) and, since task.toml is read at trial time, **no image rebuild**. **4 tasks**
(`15 304 729 1092`), oracle-validated 4/4 green. Scoped to the failing custom-user tasks;
the passing ones are left alone (forcing root risks a regression).

## 2. Prepare the images

Unlike prebuilt-image benchmarks (deep_swe, tb2.1), seta tasks have **no registry
image to pull** — each has to be built from its `environment/Dockerfile`.
`build_plan_gen.py` emits a plan whose entries are mostly `context_source: type:
git` (build from upstream's GitHub repo); the exceptions are the base-restore set
(§1.3) and the dropped-command set (§1.4), which are `type: local` (build from the
cache Dockerfile `build_cache --stage all` patched — a git build would use upstream's
unpatched Dockerfile). Pass `--cache-root` so those local entries get the right path:

```yaml
- image_ref: seta-env/0:main            # <namespace>/<task_id>:<git_ref>  (type: git)
  context_source:
    type: git
    repo: https://github.com/camel-ai/seta-env
    ref: main
    subdir: Harbor-Dataset/0/environment
    dockerfile: Dockerfile
- image_ref: seta-env/367:main          # a base-restore task → type: local
  context_source:
    type: local
    path: <cache-root>/seta-env/367/environment   # the t-bench-base Dockerfile
    dockerfile: Dockerfile
    shared_fs: hyperpod
```

Image refs are tagged `seta-env/<task_id>:<git_ref>` so a plan rebuild on a new
commit yields fresh refs. The bare `seta-env` namespace can't collide with a
public registry because the **private-registry host prefixes every ref at push**
(`<host>:5011/seta-env/<id>:<ref>`). Sizes are `size_hint_source: heuristic` (a
Dockerfile build can't be probed before it runs — see §4).

Excluded ids are in `black_list.txt` — both **build-unbuildable** (upstream
Dockerfile bugs) and **runtime-excluded** (built + ran under harbor 0.20's oracle
but scored 0; root-caused + categorized in `STATUS.md`, some fixable-but-deferred).
The generator skips them all (`--no-blacklist` keeps them):

```bash
# Generate the plan. --starter = committed 16-task set; --range '0-1375' / --remote
# (all Harbor-Dataset tasks via the GitHub Trees API) for more. The full 1376-task
# plan is committed as build_plan_1376_full.yaml. --cache-root is REQUIRED for the
# base-restore tasks (§1.3) — without it they emit type: git (unrestored, still fail).
.venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \
    --range 0-1375 --cache-root "$XRLENV_BENCHMARK_CACHE" \
    --output xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml
```

**Build + push.** The shared `scripts/build_and_push_images.py` consumes the plan:
`type: git` entries clone `{repo, ref}` once (shared FSx `build-context-cache`) and
build `subdir/Dockerfile`; `type: local` entries (the base-restore set) build the
cache Dockerfile in place; both push to `<registry>/seta-env/<id>:main`. It HEADs
each manifest first and **skips refs already present**, so a base-restore rebuild
needs **`--force`** (the old broken `:main` images already exist). Run on a build
host whose docker daemon has the private (HTTP) registry in `insecure-registries` —
the one-time setup (needs sudo; restarts docker):

```bash
source .env
# pre-requisite:  private registry to insecure-registries has been setup; should be taken care of if the cluster has been set up correctly
# on a worker node
#  build + push (--force to overwrite the old base-broken images):
.venv/bin/python scripts/build_and_push_images.py \
    --plan xrlenv_plugins/benchmarks/seta/build_plan_1376_full.yaml \
    --registry "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}" \
    --registry-scheme http --force
```

Builds are embarrassingly parallel — shard the plan across CPU hosts with
`--shard-index` / `--num-shards` (auto-read from `$SLURM_PROCID` / `$SLURM_NTASKS`
under Slurm; 1000 Dockerfiles on one node is slow). The reference build+push is
recorded in the committed `build-push-report.shard*.json`.

Refs land at `<registry>/seta-env/<id>:main` — exactly what the sweep resolves.
Image resolution is redirected via the **sweep-injected `xrlenv_image_template`
kwarg** (no adapter subclass): `run_oracle_sweep.py` composes
`<registry>/seta-env/{task_id}:main` from `XRLENV_PRIVATE_REGISTRY_HOST/PORT` and
passes it via `EnvironmentConfig(kwargs={"xrlenv_image_template": ...})`.
Adapter precedence: `xrlenv_image_template` kwarg > task `docker_image` >
`hb__<env>`. **This is the seta reference pattern** for private-registry image
redirection (GUIDELINE §6).

If you want to build a selected number of tasks, you can use the following command as an example:

```bash
# Generate a 15-task plan, then --force just that:
.venv/bin/python -m xrlenv_plugins.benchmarks.seta.build_plan_gen \
    --tasks 240,367,390,617,906,953,15,60,304,729,827,1092,1203,172,723 \
    --cache-root "$XRLENV_BENCHMARK_CACHE" \
    --output ./tmp/seta_baseimg_plan.yaml

# Build and push the selected tasks:
# on a build host
source .env
# pre-requisite:  private registry to insecure-registries has been setup; should be taken care of if the cluster has been set up correctly
.venv/bin/python scripts/build_and_push_images.py \
    --plan ./tmp/seta_baseimg_plan.yaml \
    --registry "${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}" \
    --registry-scheme http --force
```

## 3. Run the oracle sweep (validate the cache)

`run_full_sweep.sh` is the gate. Under harbor's stock `OracleAgent` (copies each
task's `solution/solve.sh` into the container + runs the verifier), a non-passing
task is a **plumbing/content bug** (reward ceiling 0 → poison for RL), not a model
signal. **Pass gate: seta requires ALL rewards `> 0`** (`_trial_passes`).

```bash
# THE FULL GREEN SWEEP — (re)builds the cache, runs the green set = present −
# black_list.txt (via --all), content-retrying reward-0 flakes.
bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh
bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh --max-workers 32   # cluster concurrency

# print the green set (present − black_list.txt) and exit:
bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh --list-green
# cache already built this session — skip step 1:
bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh --skip-build-cache

# --- targeted subsets go through run_oracle_sweep.py directly ---
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py                       # default tasks 0..7
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --tasks 0,42,100 --max-workers 4
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --all --max-workers 8 # --retries defaults to 6 (infra-only)
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --local               # local build baseline
```

**The two retry layers** — orthogonal, and eval signal is never re-rolled:

- `--retries` (default 6) retries **infra-transient errors ONLY**
  (`CapacityExhausted`, `ControlPlaneLost`, `NodeLost`, `NodeCommandTimeout` — via
  harbor's `RetryConfig.include_exceptions`). A capacity-capped acquire fails fast and
  re-queues. **The final stats record one result per task** — a retried task that then
  passes counts once, never double-counted. In the common case the infra failure is a
  fail-fast **acquire** (before `solve.sh`); a **post-acquire** infra error (e.g.
  `NodeCommandTimeout` on an exec) re-runs the whole attempt in a **fresh container**, so
  `solve.sh` can *execute* more than once — this only matters for **external** side
  effects, not the recorded result. A content failure is **never** re-rolled into a fluke
  pass.
- `--content-retries` (default 2 in the wrapper) re-runs only the tasks that came
  back non-passing (`reward = 0`), up to N more rounds; a task is solved if **any**
  attempt passes. This catches nondeterministic reward-0 flakes (a transient
  DNS/verifier blip) that `--retries` deliberately never re-rolls.

**Timeouts are native-budget** (no artificial cap). **`black_list.txt` exclusion:**
`--all` skips the blacklisted ids automatically; requesting one via `--tasks` warns
and then fails at acquire (no image in the registry). **Image resolution** is the
sweep-injected `xrlenv_image_template` kwarg composed from
`XRLENV_PRIVATE_REGISTRY_HOST/PORT` — no per-task config, no subclass.

seta's green set is **dynamic by design** (present − `black_list.txt`, no pinned
catalog size), so there is no fixed-count completeness gate — only a **nonzero
floor** (an empty green set fails `--list-green` rather than reporting a 0-task
no-op). See `STATUS.md` for the current disposition. Exit code is `0` only if every
oracle solved, so the sweep is CI-usable.

## 4. Warm the image and Calibrate the image size (optional)

`xrlenv build apply --plan <plan> --fill-missing` warms/builds the plan ahead of a
run. This is **optional**: the cluster's dynamic image cache (lazy build/pull on
first acquire + LRU + image-affinity) means the sweep works with no pre-warm —
pre-warm only to amortize the first-acquire build across a big run.

Plan sizes are `size_hint_source: heuristic` because a Dockerfile build can't be
probed before it runs. After the first build, `xrlenv build calibrate` queries each
node for the **actual on-disk (uncompressed) size**, takes the per-`image_ref` max,
and writes `<plan>.calibrated.yaml` with `size_hint_source: cluster-reported`:

```bash
xrlenv build calibrate --plan xrlenv_plugins/benchmarks/seta/build_plan.yaml \
    --output xrlenv_plugins/benchmarks/seta/build_plan.calibrated.yaml \
    --connect-host <cp-host>
```

Those true sizes feed the FFD bin-packer for tighter placement on the next apply.
It writes a **separate** file (diff before promoting) and is optional because the
heuristic hint plus the bin-packer's safety margin already place images correctly —
calibration only tightens packing density.

## See also

- `xrlenv_plugins/benchmarks/GUIDELINE_onboard_benchmarks.md` — §6 image-template
  reference (seta is the `xrlenv_image_template` reference pattern), §5 image &
  registry mechanics, §4 the onboarding workflow.
- `xrlenv_plugins/harbor/README.md` — the harbor cluster plug-in that runs these
  tasks (image resolution via the sweep-injected `xrlenv_image_template` kwarg).
- `docs/supported_benchmarks_and_harnesses/harbor_framework.md` — the `seta-env`
  section of the harbor framework Sphinx page.
- [`STATUS.md`](STATUS.md) — current oracle-sweep disposition + reproduce command.
