# TerminalWorld task patches

Curated, durable fixes for individual `terminalworld-verified` tasks whose
**upstream packaging is broken** (a deliberately-partial reference `solve.sh`, a
missing verifier user, a non-hermetic dependency pin, …). The `patch` stage of
`build_cache.py` overlays these onto the extracted task dir **after** extraction
+ task.toml normalization, on every run, so the fixes survive re-populate (the
cache dir itself is overwritten whenever a task is re-extracted).

## Layout

```
patches/<task_id>/<relative_path_within_task_dir>
```

Each file is a **full-file replacement** (preserving exec bits), e.g.:

```
patches/tw_655577/task.toml            # overrides <cache>/terminalworld-verified/tw_655577/task.toml
patches/<id>/solution/solve.sh         # overrides that task's reference solution
```

A patch file with no upstream counterpart is still copied (lets a patch add a
missing file), though **every** current overlay replaces an *existing* upstream
file — the paths in use today are `solution/solve.sh` (27 tasks), `task.toml`
(1 task), and `environment/Dockerfile` (1 task). Overrides are logged per task on
`patch` (`[patch] <id>: overrode […]`).

## Applying

```bash
.venv/bin/python xrlenv_plugins/benchmarks/terminalworld/build_cache.py --stage patch
```

`--stage patch` applies the overlays (+ the cpu-pinning markers) and is safe on
any cluster — it is the runc-only entry point. `--stage all` additionally runs
`populate` (HF download) and `sysbox` (the `SYSBOX_TASKS` routing markers). It
applies to every **already-present** task (a task absent from the shard is
skipped with a SKIP note — populate it first). Idempotent.

## Caveat

Full-file overlay means an upstream change to that file is silently overridden.
Keep the patch set small and re-review overlays if the dataset is bumped. The
`patch` log lists exactly which files each patch overrode.

**Minimality principle.** *Complete the partial; don't re-author the task.* Keep
each overlay the smallest diff that lifts the oracle's reward ceiling to 1 (or
restores loadability). The per-patch table below records the diff size and flags
(⚠) any overlay that exceeds a minimal change, so drift stays visible on review.
Some ⚠ diffs are *intrinsic* — a fake-ELF stub can't be turned into a valid
compiled binary "minimally" — and are noted as such; those are still the smallest
faithful fix, not a re-authoring of the task's intent.

## Current patches

All 29 overlays, sorted by id. Δ = pristine-upstream lines → patched lines
(diffs are against the shipped HF artifact, not the cache copy, which `patch`
overwrites). See **What each patch changes** for the exact edit.

| task | file | Δ | shipped defect (why the overlay exists) |
|---|---|---|---|
| `tw_11696`  | `solution/solve.sh` | 7→21  | partial: starts the MariaDB container but omits the nsenter + docker-enter install the verifier checks |
| `tw_118507` | `solution/solve.sh` | 4→13  | singularity: only stdout captured (missed the stderr "runscript" msg) + SIF build truncates under plain runc (needs userns → sysbox) |
| `tw_132673` | `solution/solve.sh` | 42→61 | partial: creates bucket + uploads but no website config and no public ACL |
| `tw_147241` | `solution/solve.sh` | 14→40 | partial (adds/syncs but never renames the config address); ALSO mulle-sourcetree stash-layout drift — it checks out to `stash/<name>`, not the literal `external/zlib`, so the shipped `mv external/zlib` fails |
| `tw_15324`  | `solution/solve.sh` | 59→70 | dep drift: unpinned `thrift` pulls setuptools>=61, which drops the task's Python 2.7 |
| `tw_179356` | `solution/solve.sh` | 19→80 | partial: writes a fake-ELF stub instead of a real compiled Denarius binary |
| `tw_18948`  | `solution/solve.sh` | 41→54 | lldb `process launch` fails `'A' packet error 8` — lldb disables ASLR via personality(), blocked by seccomp; breakpoint never hits so no variables evaluate |
| `tw_245032` | `environment/Dockerfile` | — | image drift: `git clone --depth 1` of master builds openal `1.25.2` but the verifier wants `1.25.1`; pin `--branch 1.25.1` + base ubuntu:24.04 (g++-13 ships `<format>`) |
| `tw_245733` | `solution/solve.sh` | 7→15  | partial: pulls `ubuntu:latest` but never writes `/app/result.txt` |
| `tw_305897` | `solution/solve.sh` | 20→25 | partial: iptables rules in the wrong order (REJECT before the subnet ACCEPT) |
| `tw_312373` | `solution/solve.sh` | 22→58 | partial: adds the 8098 port rule but skips the 8084 rich rule; also assumes a live firewalld |
| `tw_313581` | `solution/solve.sh` | 21→30 | partial: writes the zone XML but never starts firewalld (`test_firewalld_process_running` fails) |
| `tw_347571` | `solution/solve.sh` | 32→37 | partial: runs both TACT steps but never copies the tree to `/app/result.txt` |
| `tw_354080` | `solution/solve.sh` | 30→36 | partial: no WHOIS expiry-line filter and no ping exit codes |
| `tw_435744` | `solution/solve.sh` | 34→54 | stub: self-declared "Partial solve … signature is skipped" + writes an empty result.txt (verifier wants both oras artifacts AND the image digest) |
| `tw_454252` | `solution/solve.sh` | 13→26 | partial: writes `pega.json` but never creates the required `/app/deploy.sh` |
| `tw_474864` | `solution/solve.sh` | 66→64 | broken oracle: wrong demo args + a `shift`-loop quoting bug in `cobafungsi.sh` |
| `tw_523250` | `solution/solve.sh` | 26→60 | stub: self-declared "Partial solution 2" — loads the lib but never calls `texlistsymbols` or writes the result ("Tests should reject") |
| `tw_569867` | `solution/solve.sh` | 24→42 | partial: retrieves only protein info, skips the task indices + dummy_output |
| `tw_570064` | `solution/solve.sh` | 20→28 | partial: writes only the filename-match section, skips the content-match section |
| `tw_655577` | `task.toml`          | 30→37 | image is `USER delicate`; `test.sh` needs root for `apt`/`uv` |
| `tw_668448` | `solution/solve.sh` | 21→43 | harness: depends on a uv-installed `~/.local/bin/env` + misses `mpi4py`/`torchvision` |
| `tw_686647` | `solution/solve.sh` | 82→88 | dep drift: coniferest imports `sklearn.tree._tree.DTYPE`, removed in scikit-learn ≥1.6 |
| `tw_690306` | `solution/solve.sh` | 17→30 | partial: extracts + installs deps but never runs the cmake configure/build |
| `tw_709166` | `solution/solve.sh` | 12→18 | partial: builds + dumps the image but writes an empty `result.txt` (`touch`) |
| `tw_717308` | `solution/solve.sh` | 13→43 | partial: pushes branches but skips `git pull cslab cs400` (never merges the upstream) |
| `tw_739272` | `solution/solve.sh` | 47→60 | dep drift: rustup self-update to 1.96.1 fails with a cross-device rename → no `rustc` |
| `tw_7829`   | `solution/solve.sh` | 19→36 | partial: runs gdb but feeds NO stdin, so the program EOFs + exits normally (no SIGSEGV/backtrace); also gdb ASLR-disable via personality() |
| `tw_99185`  | `solution/solve.sh` | 19→41 | partial: generates cf.yml + patches warden but never creates the BOSH dev release |

## What each patch changes

Grouped by fix class. Each entry states the *exact* edit; ⚠ marks an overlay that
goes beyond a one-liner, with the reason.

### A. Partial-oracle completions

The shipped `solution/solve.sh` is a deliberate **"Partial solution N" distractor**
whose reward ceiling is 0 (it stops short of the state the verifier checks). Each
overlay completes it to reach reward 1 and nothing more.

- **`tw_11696`** — add `install -m 755 nsenter` + a `docker-enter` download before
  the (byte-identical) MariaDB `testing` container step. *(Also a `SYSBOX_TASKS`
  CLI-only DinD task — marker set in `build_cache.py`, not here.)*
- **`tw_132673`** — add static-website config (`configure_website` + a
  `website_config.json` state file the verifier reads) and `blob.make_public()`
  per uploaded object; upload loop otherwise unchanged.
- **`tw_147241`** — add the omitted `mulle-sourcetree rename external/zlib
  external/zlib.old` + second `update`, AND (2026-07-07) fix for stash-layout
  drift: this mulle-sourcetree checks the node out to `stash/zlib`, not the
  literal `external/zlib`, so the shipped `mv external/zlib external/zlib.old`
  failed under `set -e` before any rename ran and the task never reached reward 1.
  The overlay now moves the on-disk **stash** dir (`stash/zlib → stash/zlib.old`)
  so the checkout path contains "zlib.old" as the verifier requires (guarded for a
  version that still uses `external/`).
- **`tw_179356`** — ⚠ **large but intrinsic (19→80).** Replace the fake-ELF stub
  with a real qmake/`make` Qt5 build (USE_QRCODE/USE_UPNP), pre-generating
  `build/build.h` so the parallel make can't race on it. A stub cannot be made a
  valid ELF minimally. *(Pairs with the `XRLENV_CPU_PINNING` marker so
  `make -j$(nproc)` sizes to the declared cpus.)*
- **`tw_245733`** — append `docker image history ubuntu:latest > /app/result.txt`
  (with `mkdir -p /app`). *(Also the decisive `SYSBOX_TASKS` probe.)*
- **`tw_305897`** — core fix: swap the two INPUT lines so the subnet ACCEPT
  precedes the REJECT. ⚠ **also drops `set -e`** (cosmetic — the script only
  writes `/etc/sysconfig/iptables`, which is all the verifier reads).
- **`tw_312373`** — ⚠ **strategy rewrite (22→58).** Complete the zone with the
  8084 rich rule *and* switch from a live firewalld/dbus to writing
  `zones/public.xml` directly + creating the systemd enable-symlinks. Faithful to
  the on-disk end state the verifier inspects; unlike `tw_313581` this needs no
  running daemon (runc-safe).
- **`tw_313581`** — start `dbus` + `firewalld --nofork` in the foreground (as the
  image's own Dockerfile documents) under sysbox NET_ADMIN, then
  `firewall-cmd --permanent --add-port=3306/tcp`. *(A `SYSBOX_TASKS` task —
  CentOS 7 systemd v219 can't bring up D-Bus under sysbox, hence the foreground
  start.)*
- **`tw_347571`** — append the one omitted line:
  `cp .../Carangaria.tacted.tre /app/result.txt`. *(Also a `SYSBOX_TASKS`
  CLI-only DinD task.)*
- **`tw_354080`** — add the WHOIS `"Registered until expiry date."` filter and
  proper ping exit codes (0 on success, 1 on failure) to the generated
  `/app/check`.
- **`tw_435744`** — the shipped solve self-declares "Partial solve … signature is
  skipped" and writes an empty `result.txt`. Complete it: attach the omitted oras
  `signature/example` artifact (mirroring the shipped SBOM attach — the verifier
  checks `artifactType`, same fidelity) AND write the real pushed-image manifest
  digest to `result.txt` (the empty `touch` failed test_result_format's sha256:/71
  chars). *(Also a `SYSBOX_TASKS` CLI-only DinD task.)*
- **`tw_454252`** — append the missing `/app/deploy.sh` (`liara env:set
  ACCEPT_EULA=Y` + `liara deploy --detach`) and `chmod +x` it.
- **`tw_474864`** — change the demo args `alpha beta gamma` →
  `ken bianka joni tono tini` (**verifier-required**, confirmed in
  `test_state.py`) and fix the `shift`-loop quoting bug in `cobafungsi.sh` so the
  `u`/`l`/`p` behaviors pass. Every changed line maps to a verifier assertion.
- **`tw_523250`** — the shipped "Partial solution 2" loads LibTeXPrintf but never
  calls `texlistsymbols` or writes the result. Complete it: write a Julia script to
  `/app/solve.jl` (so `test_script_contents` finds Suppressor/CBinding/@capture_out/
  texlistsymbols), genuinely call `texlistsymbols()` under `@capture_out`, and write
  the line count. The answer (215-217) is a C-stdout-buffer artifact — the reference
  env's `@capture_out` didn't flush the 8192-byte buffer (~216 lines); newer
  Julia/Suppressor captures the full 646-line output, so we reproduce the documented
  8192-byte truncation and count its lines (215). Computed from the real output, not
  hardcoded.
- **`tw_569867`** — add the EnzymeClassTask train/test split indices and a
  dummy_output sample to `result.json` (partial wrote only the protein fields).
- **`tw_570064`** — append the content-match section (`grep -rHI '666' /tmp`)
  after the filename-match section, `|| true` to survive grep's no-match exit.
- **`tw_690306`** — add the omitted `cmake -H. -Bbuild/` configure (with the MPI
  compilers) and `cmake --build build/` steps.
- **`tw_709166`** — replace `touch /app/result.txt` with
  `find .../docker_dump/test-ubuntu -type f | sort > /app/result.txt`. *(A
  `SYSBOX_TASKS` task — needs the legacy image store; dockdiver is schema2-only.)*
- **`tw_717308`** — ⚠ **fuller workflow (13→43, justified).** Run
  `init-subtree`/`fetch-heads`, then `git pull --no-ff cslab cs400` (the skipped
  merge) and `push --all origin` + `push origin cs400`. Maps 1:1 to the three
  `test_state.py` assertions.
- **`tw_99185`** — init the rbenv PATH, then add step 3:
  `printf 'cf\n' | bosh create release --force --with-tarball` (the CLI prompts
  interactively for the dev-release name; empty stdin raised `EOFError`).

### B. Non-hermetic dependency pins

The task froze without pinning a dependency that has since drifted; the overlay
pins it back to the era version. **Only the dependency line changes** (plus a
header comment explaining the drift) — everything else is byte-identical.

- **`tw_15324`** — `thrift` → `thrift==0.13.0`. The solve runs on Python 2.7;
  unpinned `thrift` now needs `setuptools>=61`, which dropped 2.7.
- **`tw_686647`** — add `"scikit-learn<1.6"` to the pip line. coniferest 0.0.15
  does `from sklearn.tree._tree import DTYPE`, removed in scikit-learn ≥1.6.
- **`tw_739272`** — pin the Rust step: `--default-toolchain 1.83.0 --profile
  minimal` and keep `TMPDIR` on rustup's own fs. The image ships rustup, so the
  unpinned installer self-updates to 1.96.1, which fails with a cross-device
  rename over the overlay fs → `rustc` never installs.

### C. Single-container / privilege adaptations

Upstream assumes services or an environment the one unprivileged harness
container doesn't provide; the overlay supplies the equivalent in-container.

- **`tw_299387`** — **REMOVED 2026-07-17 (step 5, multi-service compose).** This
  was a `+50` overlay that bootstrapped the two docker-compose sidecars
  (`fake-gcs` + `fake-token`) *inside* the single container, compensating for the
  single-acquire path's lack of compose. The cluster-compose path now brings the
  real sidecars up on a project network, so the task runs its **original,
  unpatched** `solve.sh` faithfully — the overlay is deleted (`patches/tw_299387/`
  gone). See `notes/multi-service-compose-step5-runbook.md`.
- **`tw_668448`** — drop the `~/.local/bin/env` uv dependency (use the image's
  system pip), add the deps ezpz needs but doesn't pull (`mpi4py`,
  `torchvision`), cap the run at `--train_iters 20`, and stop masking the test
  exit code so a real failure surfaces.
- **`tw_118507`** — capture stderr (`2>&1`); singularity writes its progress + the
  runscript info to stderr, so the shipped stdout-only redirect missed "runscript".
  Also `SYSBOX_TASKS`-marked (build_cache.py) — the rootless SIF build from
  `docker://ubuntu` needs userns, else it truncates at "Exploding layer" under
  plain runc.

*(`tw_312373` and `tw_313581` also belong to this class — both bridge the
missing systemd/live-daemon — but are listed under A since they primarily
complete a shipped partial.)*

### D. Debugger / seccomp (personality) adaptations

Debuggers disable ASLR by default via the `personality(ADDR_NO_RANDOMIZE)`
syscall, which the container's seccomp profile blocks — so the target won't launch
and the verifier (which needs the post-crash / post-breakpoint state) fails. The
fix is runtime-independent (no sysbox needed) — tell the debugger not to disable
ASLR (the checks look for variable names / crash addresses, not specific values).

- **`tw_18948`** — add `settings set target.disable-aslr false` to the lldb batch
  script before `process launch`; the launch then succeeds, the breakpoint hits,
  and the `p <var>` commands evaluate (test_lldb_output_contains_variables /
  _tracks_len_multiple_times pass).
- **`tw_7829`** — a partial completion + the same seccomp guard: feed the
  crash-triggering stdin (`2\n<name>\n1\n` — cadastrar a `Cliente` with an
  uninitialized `compras` pointer, then ler_clientes dereferences it → SIGSEGV) so
  gdb catches the segfault + backtrace, and add `set disable-randomization off`.

### E. task.toml — verifier user

- **`tw_655577`** — add `[verifier] user = "root"`. The image bakes
  `USER delicate` (non-root), but `tests/test.sh` runs `apt-get install` +
  `curl … uv | sh` + `uvx`, which need root. harbor runs the verifier as
  `task.config.verifier.user`, so root (the Terminal-Bench convention) lets
  `test.sh` bootstrap uv and run pytest.

## Cross-references

Some patched tasks also carry markers set by **other** `build_cache.py` stages —
those live in the task.toml, not under `patches/`:

- **`CPU_PINNING_TASKS`** (applied in the `patch` stage): `tw_179356` (solve.sh)
  and `tw_245032` (Dockerfile) are *both* patched and cpu-pinned. The full
  `CPU_PINNING_TASKS` set also includes `tw_528959` + `tw_234227`, which carry no
  `patches/` overlay — see `build_cache.py` for the authoritative list.
- **`COMPOSE_DROP_PRIVILEGED`** (applied in the `patch` stage): `tw_304270`,
  `tw_304271`, `tw_305044` — the multi-service compose stacks whose redundant
  `privileged: true` is stripped (keeping `cap_add: [NET_ADMIN, NET_RAW]`) so they
  run under runc without a cluster-wide `allow_privileged`. This edits the task's
  own `environment/docker-compose.yaml` in the cache, not a `patches/` file — see
  `build_cache.py`.
- **`SYSBOX_TASKS`** (applied in the `sysbox` stage): only these patched tasks are
  also sysbox-marked: `tw_245733`, `tw_709166`, `tw_313581`, `tw_347571`,
  `tw_11696`, `tw_435744`, `tw_118507`. The full `SYSBOX_TASKS` set (23) lives in
  `build_cache.py`.
