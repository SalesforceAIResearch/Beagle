# seta — oracle sweep status

Corpus-quality gate: run harbor's stock `OracleAgent` (copies each task's
`solution/solve.sh` into the container + runs the verifier) per task **on the xrlenv
cluster** via the harbor plug-in (`xrlenv_plugins.harbor:XrlenvHarborEnvironmentCluster`)
and confirm each earns a positive `reward`. Under the oracle a non-passing task is a
plumbing/content bug (reward ceiling 0 → poison for RL), not a model signal.

seta-env ships a `Dockerfile` (no prebuilt `docker_image`), so images are built once and
pushed to the private registry as `<registry>/seta-env/<id>:main`; the sweep resolves each
task via the `xrlenv_image_template` kwarg (README §"images").

## Gate config

`run_full_sweep.sh` (the entrypoint): green set = **present cached tasks − `black_list.txt`**
(exclusion, so a re-populate auto-picks up new tasks); concurrency via `--max-workers`;
the oracle gate is `reward > 0`; content-retries reward-0 flakes. Integration-gate
contract is DYNAMIC (no fixed present/green count) with only a **nonzero floor** — an
empty green set fails `--list-green` rather than passing a 0-task no-op.

## Results — harbor 0.20 full sweep (2026-08-03/04)

The first full 1371-task oracle sweep on harbor 0.20 scored **1272/1371** — **99
non-blacklisted failures** (blacklist was only the 5 build-unbuildable). Each was
root-caused empirically on the dev cluster. **None is a harbor-0.20 regression**
(seta sets no network policy; the seta sweep is byte-identical main↔branch;
`custom_docker_compose` is a no-op in harbor 0.8 AND 0.20). The causes are
container-privilege semantics + upstream **Harbor-migration damage**: the
`camel-ai/seta-env` conversion from the pre-Harbor `Dataset/<id>/` format to
`Harbor-Dataset/<id>/` dropped runtime-critical config the original still has.

Of the 99: **13 fixed pure-cache** (no rebuild); **10 base-image restored** (codified +
rebuilt+validated 10/15 on 2026-08-04 — `build_cache --stage all` rewrites the base,
`build_plan_gen` builds them `type: local`, §1.3); **8 dropped-command restored**
(§1.4; oracle-validated **8/11**); **4 verifier-as-root** (§1.5; oracle-validated **4/4**);
**64 excluded** in `black_list.txt` (some fixable-but-deferred, some permanent).
13+10+8+4+64 = 99.

| Bucket | Count | Meaning |
|---|---:|---|
| ✅ **Fixed pure-cache (live)** | **13** | validated reward 1.0; markers/overlay applied |
| 🔁 **Base-image restored** | **10** | rebuilt on the t-bench base → reward 1.0 (rebuild+push, §1.3) |
| 🔁 **Dropped-command restored** | **8** | ENTRYPOINT boot wrapper + `type: local`; oracle 8/11 (§1.4) |
| ✅ **Verifier-as-root (live)** | **4** | `[verifier] user=root` task.toml marker; oracle 4/4, no rebuild (§1.5) |
| ⛔ **Excluded — runtime** | **64** | built + ran but scored 0; see categories below |
| ⛔ **Excluded — build** | **5** | upstream Dockerfiles that can't build |
| ✅ Already passing | ~1272 | earn `reward > 0` as-is |

### ✅ The 13 pure-cache fixes (validated reward 1.0)

Applied by `build_cache.py --stage all` (idempotent; no separate stage).

| mechanism | tasks | what it restores |
|---|---|---|
| sysbox DinD | `8`, `1004` | nested dockerd (1004 CLI-only → +install) |
| sysbox iptables | `1117`, `1347` | unprivileged NET_ADMIN |
| sysbox mount | `311`, `119`, `1225`, `830` | unprivileged SYS_ADMIN |
| sysbox netns | `1059`, `484` | unprivileged NET_ADMIN (veth/netns) |
| sysbox systemd | `345` | unprivileged systemd PID 1 |
| sysbox cap_add | `846` | `NET_RAW`+`NET_ADMIN` recovered from `Dataset/846` compose |
| solve.sh overlay (`patches/`) | `309` | `/oracle`→`/solution` run-path (harbor runs the oracle from `/solution`) |

`SYSBOX_TASKS` + `patches/` live in `build_cache.py` / `patches/`. See README §1.1–1.2.

### 🔁 The 10 base-image restores (`BASE_IMAGE_FIX_TASKS`, validated reward 1.0)

The migration swapped these tasks' `FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:<date>`
(bakes python3/curl/uv/wget/tmux) for bare `FROM ubuntu:24.04`, so their identical
`solve.sh` failed `command not found`. `build_cache.py --stage all` rewrites the FROM
back, and `build_plan_gen` builds them `type: local` from the restored cache
Dockerfile. **NOT blacklisted** — they take effect after a rebuild+push (§1.3 / README
§2). All 10 rebuilt + validated green (367 → `python3` 3.12.3).

| tool assumed | tasks |
|---|---|
| python3 | `240 367 390 617 906 953` |
| curl (→ uv bootstrap) | `60 827` |
| uv | `1203` |
| tmux | `723` |

The 15→10: the base-restore fixes tasks that *assume* the tool. Five candidates
still failed even rebuilt (→ base-image HARD, below): `15 304 729 1092` (solve's
apt/dpkg/permission edits break the verifier `curl`→`uv` bootstrap), `172` (solve
pins an unavailable `wget=<ver>`).

### 🔁 The 8 dropped-command restores (`DROPPED_COMMAND_TASKS`, oracle-validated 8/11)

The migration dropped a non-trivial single-service compose `command:` (a service /
workers / seeded state the task premise needs), so harbor boots the bare image and the
oracle fails. **Mechanism:** harbor's raw-container path sets docker `CMD = sleep infinity`
(OVERRIDING any baked `CMD`) but does NOT pass an entrypoint (PRESERVING the image
`ENTRYPOINT`). `build_cache --stage all` bakes a boot wrapper as the Dockerfile
`ENTRYPOINT` — `( <cmd> ) & exec "$@"` — so the recovered command runs in the background
and the wrapper execs harbor's `sleep infinity` as PID 1. `build_plan_gen` builds these
`type: local`. Commands are **verbatim from `git show HEAD:Dataset/<id>/docker-compose.yaml`**
(the repo's `Dataset/` working tree is a partial checkout — read from git).

| task | restored command |
|---|---|
| `227` | `/server/start_server.sh && sleep infinity` |
| `26` | `/usr/local/bin/start_rsync.sh && sleep infinity` |
| `412` | `/start-services.sh` |
| `475` | `myserver infinity & sleep infinity` |
| `669` | `/usr/sbin/sshd && sleep infinity` |
| `946` | `/app/startup.sh` |
| `1287` | `/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf` |
| `1349` | `/opt/start_services.sh sleep infinity` |

**Scope discipline (faithful derivation).** From `git show`, 52 tasks carry a non-trivial
compose command/entrypoint — but **24 of them PASS** (their `solve.sh` or a build-time
`RUN` does the setup, or the test doesn't need it). A task is restored ONLY if it BOTH
(a) failed the oracle AND (b) has no surviving Harbor `ENTRYPOINT`. That gave 11 candidates;
the operator's oracle rebuild (same loop as base-image 15→10) validated **8 green, 3 pulled**:

- **8 green** (above) — reward 1.0 under the oracle.
- **3 needed more than the command** (re-blacklisted with the real reason): `775` (the
  on-disk config-drift the premise needs is not set up — the oracle reload correctly yields
  the unchanged `LOG_LEVEL=INFO`; content/setup gap), `1246` (oracle timing race:
  stopped-worker log-freshness vs a 10 s test threshold), `1309` (dropped-command **+ SYSTEMD**
  — grafana via `service grafana-server start`; needs a sysbox systemd marker too).

**Deferred (still blacklisted):** dropped-command **+ a capability** — need the ENTRYPOINT
wrapper AND a sysbox marker: `144` SYS_ADMIN, `164`/`189` NET_ADMIN, `960` SYS_NICE, `1309`
systemd; the two genuine ssh/key-exchange **multi-service** tasks (`892`, `1198`).

### ✅ The 4 verifier-as-root fixes (`VERIFIER_ROOT_TASKS`, oracle-validated 4/4)

Not a base-image issue (that earlier label was a wrong guess). A task whose Dockerfile sets
a non-root `USER` (`[metadata] sets_custom_user = true`) has harbor 0.20 run **both** its
agent AND its verifier as that user — because `[verifier] user` is unset (`None` → the image
USER; harbor `single_step.py` runs the verifier as `task.config.verifier.user`). But
terminal-bench's verifier contract is **root**: the stock `tests/test.sh` does
`apt-get install curl` (→ the uv bootstrap) and the tests do `su -l <user>` — both need
root. As the non-root image USER the bootstrap dies (`curl: command not found`) and `su`
fails (`Authentication failure`) → reward 0. **Verified from the actual trial logs.**

Fix: `[verifier] user = "root"` in task.toml (`build_cache.VERIFIER_ROOT_TASKS`, applied by
`--stage all`). Surgical — only the verifier phase; the agent/solve still runs as the task
user. **task.toml is read at trial time → NO image rebuild.** Oracle-validated 4/4 green
(reward 1.0, `run_oracle_sweep --local`): `15 304 729 1092`. Scoped to the FAILING
custom-user tasks; the 9 passing custom-user tasks are left alone (their tests are fine as
the task user — forcing root risks a regression). `407` (host-sysctl) / `999`
(build-unbuildable) also set a custom user but fail for a different primary reason.

### ⛔ The 64 runtime exclusions (`black_list.txt`)

**Fixable — deferred** (un-blacklist + apply when ready):
| category | count | ids / fix |
|---|---:|---|
| dropped-command **+ cap/systemd** (needs the ENTRYPOINT wrapper AND a sysbox marker) | 5 | `144` SYS_ADMIN, `164`/`189` NET_ADMIN, `960` SYS_NICE, `1309` systemd (grafana) |
| base-image legacy (`FROM ubuntu:14.04` EOL — apt mirrors gone + pip too old; legacy rehab, not a base swap) | 1 | `197` |
| cap-only (sysbox-candidate: single-svc, no command, just a cap/privileged) | 4 | `19` privileged, `971` NET_ADMIN+SYS_MODULE+priv, `1103` NET_ADMIN+NET_RAW, `1121` NET_ADMIN |
| init:true (tini PID-1 reaper) | 1 | `1184` |
| multi-service (GENUINELY >1 svc — needs sidecar hosts + the `<id>-main` image) | 5 | `890` (4), `892` (2), `973` (2), `1133` (7), `1198` (3) |

**Permanent — no faithful fix:**
| category | count | why |
|---|---:|---|
| upstream solve/content — the oracle solution (under the fixed harbor path) does NOT produce the state its own tests require | 6 | `172` pins `wget=1.21.4-1ubuntu4.1` (gone from the 24.04 repo), `409` pyenv builds no python versions, `414` adb not runnable, `535` no `gm-custom` binary, `962` no python 3.8/3.11 from source, `308` GTK2 source rebuild scores 0 (needs deeper triage). Verified from trial logs — the earlier "base-image HARD" labels were wrong guesses. |
| upstream content-logic (solve ran clean, reference wrong) | 14 | reference doesn't satisfy its own test |
| upstream shell-context (writes `~/.bashrc`; verifier non-login) | 6 | `Dataset/277/solution.sh` == cache, both incomplete |
| harness / infra (no-reward, container race, mount-src, reward-0) | 10 | not an oracle result |
| non-hermetic external fetch (yt-dlp/YouTube, downloads) | 5 | external dependency |
| host-global sysctl (`vm.swappiness`, non-namespaced) | 2 | un-containerizable (runc RO, sysbox denies) |
| loop device (`losetup`) | 2 | needs host `/dev/loop` |
| dropped-command ran but insufficient (verified via oracle) | 2 | `775` on-disk config-drift not set up (content/setup gap), `1246` log-freshness timing race |
| uncategorized (vanilla compose; real failure not root-caused) | 1 | `193` — was mislabeled multi-service; re-triage |

Full per-task categories + reasons are inline in `black_list.txt`.

**Blacklist re-categorization (2026-08-04):** the earlier dropped-command/multi-service
labels were corrected against **git-verified** compose evidence. `19`/`1184` had no
dropped command (privileged / init:true); `971`/`1103`/`1121` are single-service cap-only
(not multi-service); `164`/`189` are single-service dropped-command+cap (not multi-service);
`193` is vanilla (real reason unknown). Genuine multi-service is only 5 tasks.

### ⛔ The 5 build-unbuildable (upstream `Dockerfile` defects)

`25` (COPY uncommitted file) · `305` (wget dead URL) · `387`/`683` (`python3` not
installed) · `999` (COPY uncommitted file). Details in `black_list.txt`.

## Key root-cause evidence

- **Base-image swap (confirmed).** `Dataset/367/Dockerfile` = `FROM
  ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624` (bakes python3/curl/…);
  `Harbor-Dataset/367` = bare `FROM ubuntu:24.04`. ALL seta tasks are `ubuntu:24.04`;
  the failing ones just need tools the t-bench base had. Failure signature:
  `python3: command not found` etc. *in the oracle*.
- **Dropped compose command (confirmed + fixed).** `Dataset/227/docker-compose.yaml` has
  `command: [sh,-c,/server/start_server.sh && sleep infinity]`; `Harbor-Dataset/227` has
  none, so harbor boots `sleep infinity` and the task premise (a server on :8080) is
  absent → oracle fails. Restoring it as an ENTRYPOINT boot wrapper → **oracle reward 1.0**.
- **Read Dataset from GIT, not disk.** The staging clone's `Dataset/` working tree is a
  partial checkout (538/1376 dirs on disk; `git ls-tree HEAD` has all 1376). Reading
  compose from disk gives false "absent" conclusions — always
  `git show HEAD:Dataset/<id>/docker-compose.yaml`.
- **Harbor run-path (confirmed + fixed).** Harbor 0.20 runs the oracle from
  `/solution`; 309's solve.sh guarded on the pre-Harbor `/oracle`.
- **`Dataset/` is the faithful reference** for what the migration dropped (command,
  cap_add, init, privileged, base image). Recovering from it is a faithful cache fix
  where the mechanism allows (846 cap_add, 309 path); base-image/command need a rebuild.

## Rebuild the 10 base-image tasks (codified — follow README §2)

The base-restore is **codified**, so the standard build flow applies it:
1. **build host** (one-time): `sudo PRIVATE_REGISTRY=<host>:5011 bash
   scripts/configure_docker_registry.sh --restart` (adds the HTTP registry to
   `insecure-registries`).
2. `build_cache.py --stage all` — rewrites the 10 tasks' cache Dockerfiles to the
   t-bench base (`BASE_IMAGE_FIX_TASKS`).
3. `build_plan_gen --range 0-1375 --cache-root "$XRLENV_BENCHMARK_CACHE" --output
   build_plan_1376_full.yaml` — emits the 10 as `type: local` (from the restored
   cache), the rest `type: git`.
4. Rebuild ONLY the 10 (avoids re-pushing the ~1285 unchanged `type: git` images):
   `build_plan_gen --tasks 240,367,390,617,906,953,60,827,1203,723 --cache-root
   "$XRLENV_BENCHMARK_CACHE" --output /tmp/baseimg.yaml` then
   `build_and_push_images.py --plan /tmp/baseimg.yaml --registry-scheme http --force`
   (`--force` overwrites the old base-broken `:main`; on the FULL plan `--force`
   would rebuild all ~1295).
5. `run_oracle_sweep.py --tasks 240,367,390,617,906,953,60,827,1203,723` — validated
   10/10 green (2026-08-04).

Proven: all 10 rebuilt on the t-bench base pass (367 → `python3` 3.12.3). The
**base-image HARD 11** (java/py-ver/adb/gm/gcc + `15`/`304`/`729`/`1092`/`172`)
stay blacklisted — after a green
rebuild proves the base restore + their solve's own install works, add them to
`BASE_IMAGE_FIX_TASKS` and un-blacklist.

## Rebuild the 8 dropped-command tasks (codified — same flow as base-image)

`build_cache --stage all` bakes the ENTRYPOINT boot wrapper; `build_plan_gen` emits them
`type: local`. Identical rebuild loop — only the id list differs:

```bash
# after configure_docker_registry.sh (§ above) + build_cache --stage all:
IDS=26,227,412,475,669,946,1287,1349
build_plan_gen --tasks $IDS --cache-root "$XRLENV_BENCHMARK_CACHE" --output /tmp/dropcmd.yaml
build_and_push_images.py --plan /tmp/dropcmd.yaml --registry-scheme http --force
run_oracle_sweep.py --tasks $IDS          # the gate — reward>0 per task
```

**Validated 8/8 green under the oracle (2026-08-04).** The pass loop went 11→8: three
candidates (`775`, `1246`, `1309`) needed a *further* fix beyond the command (content/setup
gap, timing race, systemd) so they were pulled from `DROPPED_COMMAND_TASKS` and
re-blacklisted with the real reason — exactly as base-image went 15→10.

## Reproduce

```bash
# .env auto-loads (import xrlenv) — needs XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN
# + XRLENV_PRIVATE_REGISTRY_HOST/_PORT. Use the .env consumer token (the
# ~/.xrlenv/secrets/consumer.token file is stale).
export XRLENV_BENCHMARK_CACHE=/path/to/benchmark-cache

# THE GATE — build_cache --stage all (populate + patches/ overlays + sysbox markers),
# then the green set (present − black_list.txt), content-retrying reward-0 flakes:
bash xrlenv_plugins/benchmarks/seta/run_full_sweep.sh --max-workers 32

# targeted: a fixed task / subset
.venv/bin/python xrlenv_plugins/benchmarks/seta/run_oracle_sweep.py --tasks 8,309,846 --max-workers 3
```

Exit code is `0` only if every oracle solved. Per-trial artifacts land under
`<jobs_dir>/<job_name>/<trial>/` (default `tmp/`).

## Notes

- **Pass criterion:** every trial has `verifier_result.rewards` fully populated with
  positive values. Under the oracle a non-passing trial is a plumbing bug.
- **Image resolution** is the sweep-injected `xrlenv_image_template` kwarg.
- **Adding exclusions:** put the id in `black_list.txt` (the single exclusion source,
  read by the generator + sweep + `run_full_sweep.sh`) with a one-line reason.
