# EvoClaw onboarding — design notes (the *why*)

Rationale behind the interceptor onboarding. The **how** is in `README.md`; this is the
background a maintainer needs. Interceptor pattern reference: `GUIDELINE_onboard_benchmarks.md`
§7.1. The exact intercepted `docker` surface is in `copy_to_call_site/docker_shim.py`.

## How it works

- **Container/image management only.** EvoClaw drives Docker through raw
  `subprocess.run(["docker", …])`, so there is no `import_path` seam. `docker_shim.install()`
  monkeypatches `subprocess.run`/`Popen` in-process: a call whose argv[0] basename is `docker`
  is rerouted to the xrlenv docker-py-compat client (`run`/`exec`/`cp`/`rm`/`stop`/…);
  everything else falls through. Uncovered surface (`build`/`rmi`/`pull`/`tag`) fails loud (rc
  127). Cross-process workers (`spawn`/`forkserver`) don't inherit the patch → `_yd_bootstrap/
  sitecustomize.py` re-applies it at interpreter startup via a `PYTHONPATH`-prepended dir.
- **Images** come from the Docker Hub pull-through mirror (`hyd2apse/*`), resolved to pullable
  refs by `image_resolution.py` (README §2).
- **The oracle injects the golden solution from the host.** EvoClaw's agent container is
  answer-free by design (it strips tags + `git gc`s), so `oracle.py` extracts the golden END
  source from the milestone image into a cached tar and applies it in-container.

## Fleet reservation (on by default)

Each task runs **two** containers — a cheap long-lived **agent** (`cpu_request=2`) and an
expensive **eval** (`cpu_request=16`). Under plain per-container admission, at high `--workers`
the cheap agents get admitted first and hold node capacity, starving the evals → throughput
*drops* as `--workers` climbs. Fleet reservation reserves each task's whole fleet (agent +
eval slots) on one node up front, so an agent is admitted only when its eval slot(s) can be
too. With it on, `--workers` far above capacity is fine (excess tasks queue as whole fleets).

Footprint (printed at launch as a `FLEET RESERVATION: ON` box; every term overridable):

```
fan-out N     = 1 (--parallelization-level milestone)  or  --fleet-eval-pool (repo, default 4)
footprint cpu = --fleet-agent-cpu (2)  +  N × --fleet-eval-cpu (16)
footprint mem = footprint cpu × --mem-per-cpu-gb (2)
```

→ a `milestone` task reserves 2 + 16 = **18 cpu / 36 GiB**; a `repo` task 2 + 4×16 = **66 cpu
/ 132 GiB**. A standalone `run_e2e_xrlenv.py` worker needs an explicit
`--fleet-footprint-cpu`/`--fleet-footprint-mem-gb`.

## Correctness fixes (`--apply-yd-fixes`)

All corrections for known **upstream** eval-protocol bugs live behind this one flag as runtime
monkeypatches, so the vendored harness stays byte-for-byte pristine (off = leaderboard-faithful).
Fixes: (1) stage untracked GT test files before the evaluator's `git clean` (element-web); (2)
reassemble a `go test -json` benchmark line split across events (go-zero); (3) DNS-poison a
quarantine-denied CDN host to `0.0.0.0` (scikit-learn); (4) re-run the eval on a small `none_to_pass`
DB-race flake (navidrome); (5) drop genuinely-flaky bystander timing tests from graded `pass_to_pass`
only. `_yd_bootstrap/sitecustomize.py` re-applies them in spawned eval children.

Residual conc-64 flakes: a few go-zero/dubbo timing bystanders rotate which milestone they hit,
so a single full-98 sweep lands 92–93/98 with the misses rotating (`no_verdict` stays 0 — content
flake, not infra). The faithful fix for a bystander is the drop-list (fix 5), not another pin; a
genuine CPU-contention flake gets `--cpu-pinning-milestone`. Simplest deterministic green: re-run
only the non-resolving milestones and merge (the summary ORs `resolved` across attempts).

## Quarantine (anti-cheat, auto-on)

EvoClaw ships a per-repo `quarantine_configs/<repo>.yaml` and fail-closes any run without it. It
forces the eval deterministic-offline (`GOPROXY=off` / pip-offline against the closure baked into
`base-offline-<v>`, plus `/etc/hosts`-poisoning the repo's proxy/registry domains). The batch
driver applies it exactly as EvoClaw's `scripts/run_all.py` does — `load_quarantine_env(repo, root)`
merged into each worker's env — so xrlenv only carries the docker job; the unmodified harness
enforces isolation. This determinism also recovers network-flaky milestones.

## go-zero base image

EvoClaw's `hyd2apse/go-zero:base-v0.9` ships without `.git` (its `.dockerignore` excludes it;
only the *milestone* Dockerfiles `COPY .git` back), so EvoClaw's setup git-truncation + any agent
`git tag` fail → the run stalls at `Completed: 0`. Reproduces on stock EvoClaw + real Docker (not
an onboarding bug). Fix: build the corrected base (`go-zero-gitfix.Dockerfile`, README §2) and
point `EVOCLAW_GOZERO_BASE_IMAGE` at it; the redirect touches only go-zero's `base` ref.

## element-web: memory + Docker-in-Docker

- **Memory.** EvoClaw's evaluator sets no `--memory`; xrlenv caps an undeclared container at
  ~4 GiB, which element-web's 16-worker `jest` OOM-kills (exit 137). The shim declares a limit on
  acquire, scaled by `--mem-per-cpu-gb` (default 2 GiB/cpu → 32 GiB for a 16-cpu eval).
- **DinD.** Some element-web regression suites are Playwright E2E using `testcontainers`, which
  needs a Docker runtime *inside* the eval container. Route those with `--sysbox-milestone
  <repo>/<mid>`: the driver sets `EVOCLAW_CONTAINER_RUNTIME=sysbox-runc` for those workers only
  (unprivileged inner `dockerd`); the operator must have a Sysbox pool with `sysbox-runc` in
  `nodes.yaml` `policy.allowed_runtimes`. **Why sysbox, not a host-socket bind:** an earlier
  `EVOCLAW_DOCKER_SOCK` stop-gap bind-mounted the node's `/var/run/docker.sock` — a container-escape
  / cross-tenant vector (spec 19). Sysbox gives each container its own userns + inner dockerd,
  nothing on the node shared in. **The cluster runs a working `sysbox-runc` pool** (seta's DinD
  oracle tasks and TerminalWorld use it — a patched Sysbox build; the packaged sysbox-ce ↔
  Docker-29 incompatibility is a de-risk detail, not the running state). So `maintenance_ui_ux`
  is recoverable via `--sysbox-milestone` — the 92/98 headline just doesn't route it there
  (unvalidated for evoclaw, whose onboarding predates the cluster's sysbox); a validation run
  would confirm ~93–95/98.

## Whole-`/testbed` copy + relay cap (`--copy-testbed`)

EvoClaw's `cleanup()` does `docker cp {c}:/testbed .` — a whole-repo debug exfil the grader doesn't
need. **Off by default.** xrlenv caps a single `get_archive` relay through the control plane
(`XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES`, default 128 MiB — the CP is a metadata channel, not a bulk
pipe); an over-cap `/testbed` is refused (`ArchiveTooLarge`) and the shim turns it into a non-zero
`docker cp` so the copy fails but grading continues. A `--copy-testbed` sweep is the guardrail
stress test (pass: no node-lost, `ArchiveTooLarge` in logs = the cap working).

## Trials + results

`--workspace-root` is where EvoClaw both reads a repo's data and writes trials — pointing it at the
shared dataset would write into it, so the wrapper builds `<results-root>/<repo>/` with read-only
**symlinks** into `EVOCLAW_DATA_ROOT` and a local `e2e_trial/`. `--copy-testbed` off, `--keep-container`
off (a kept container isn't released), and `--fast-oracle` on are the shim defaults; the golden cache
covers `repo_src_dirs` only (not root build files like `go.mod`).
