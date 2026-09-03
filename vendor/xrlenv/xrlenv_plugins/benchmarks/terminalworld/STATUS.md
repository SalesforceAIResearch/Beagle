# TerminalWorld — current task status (all 200)

Status of the full 200-task `verified` split under the oracle, after this
onboarding's fixes. Baseline: the **2026-07-17 full sweep** (after the
multi-service-compose landing), superseding the 2026-07-06 baseline; the **green set
re-confirmed clean 192/192** on a 2026-07-17 re-run (`tmp/tw-oracle-sweep__192_2`, 3
infra-retries recovered, 0 content failures — the flakes below did not recur). Update
after each full sweep.

## Gate config (current)

`run_full_sweep.sh` (the entrypoint): green set = **present 200 − EXCLUDE (9)** (asserts
**191**); concurrency via `--max-workers` (8 = stable on a single sysbox node, 32 = platform
proof); the oracle gate is `reward > 0`. Two retry layers, both reported: `--retries`
(per-*trial*, infra-transient exceptions only, default 6) + content-retries (per-*task*,
reward-0 outcome, `--content-retries`, default 2). Multi-service compose tasks run under
**runc**; sysbox-marked tasks (`SYSBOX_TASKS`) route to a `sysbox-runc` node. README §3 has
the full walkthrough. Launch via the [Reproduce](#reproduce) command.

## Section 1 — buckets

| Bucket | Count | Meaning |
|---|---:|---|
| ✅ **Consistently passed** | **188** | reliably reward 1.0. Includes the **6 multi-service compose** tasks (see the compose note below). |
| 🎲 **Flaky** | **3** | reward-0 occasionally but retry-/content-retry green: `tw_241711` (Gutenberg SSL), `tw_11696`, `tw_425218` — all **passed the clean 192 re-run**; the latter two failed once each on 2026-07-17 and were investigated (see the flaky note below): **not reproducible**, rare non-deterministic transients under peak sysbox/DinD contention, not degradations. |
| 🌐 **External-dep — EXCLUDED** | **1** | `tw_507605` — its verifier resolves external domains' MX records and asserts on `nasa.org:`, but nasa.org resolution intermittently fails/times out from the container (`nasa.org: ERROR`), reward-0 on every content-retry (redhat.com resolves fine). An external-DNS flake, **not** retry-recoverable and **not** xrlenv/model — EXCLUDEd (2026-08-01) so it can't surprise-fail the gate. Re-include if rewritten to not depend on live nasa.org DNS. |
| ❌ **Failed** | **7** | need a substrate feature we don't provide (multi-host / k8s / loop-device / Splunk-fs / ~10-service Harbor), a broken oracle we can't faithfully complete, OR an **unreasonable task config** (`tw_528959`: its own 2700s ceiling is too tight for a CPython from-source `make -j2` build even uncontended). |
| 🔍 **Need investigation** | **1** | `tw_222108` (netns-DNS, deep) — see below. |
| **Total** | **200** | 188 + 3 + 1 + 7 + 1 = 200, fully accounted. **Green set = 191** (`run_full_sweep.sh` EXCLUDEs the 9 non-green: the 7 Failed + `tw_222108` + `tw_507605`). |

> **✅ Step 5 — multi-service compose LANDED (2026-07-17).** The cluster now supports
> multi-service `docker-compose` tasks (compose-on-the-node, under runc). All **6/6**
> compose tasks pass (validated in a targeted 6-task smoke AND the full sweep):
> - **Recovered** (were ❌ Failed "needs cluster-compose", each a 2-4 service stack on
>   a private network): `tw_522753` (postgres), `tw_188260` (solr + ambari, with
>   `solr-node`/`ambari-server` sub-builds + subnet `10.188.74.0/24`), `tw_304270`
>   (`172.16.70.0/24`), `tw_304271` (`10.71.238.0/24`), `tw_305044` (`192.168.20.0/24`;
>   the old "multi-host" label was really 4 services on one private net). These +5
>   took green 187 → 192. `tw_488034` stays ❌ (macOS-hardcoded paths + ~10-service Harbor).
> - **Now faithful** — `tw_299387` was already ✅ via an in-container sidecar-bootstrap
>   workaround (`patches/tw_299387`, now deleted); it runs its original solve.sh against
>   the real `fake-gcs`/`fake-token` compose sidecars.
> - The 3 privileged stacks (`tw_304270/271/305044`) run under runc **without**
>   `allow_privileged`: `build_cache.py`'s `COMPOSE_DROP_PRIVILEGED` strips their
>   redundant `privileged: true` (they only need `NET_ADMIN`/`NET_RAW`, already allowed).

> **🎲 Flakes seen 2026-07-17 — investigated, not reproducible, retry-recovered:**
> - **`tw_11696`** (sysbox CLI-only DinD, MariaDB): once failed `docker run -d -p
>   33306:3306` → `bind host port 0.0.0.0:33306: address already in use` (agent exit
>   125) — a **content/task** failure (not `CapacityExhausted`), correctly NOT
>   infra-retried; `run_full_sweep.sh`'s content-retry recovers it. **Investigation
>   (2026-07-17): NOT reproducible.** 20× via the real `run_oracle_sweep` path @ conc-32
>   under 4-way sysbox contention (node load 20+, D-state spiked & cleared) → 0 failures;
>   200 node-level iterations (tight re-bind loop + 4 containers churning the *same*
>   33306 in their own netns) → 0; sysbox netns isolation verified intact (33306 does
>   not leak across containers). Ruled out: compose-rerouting (it's single-service),
>   host-port leak, leftover container, image-prebind. Conclusion: a **rare,
>   non-deterministic docker port-setup transient** at peak DinD pressure on the single
>   4-slot sysbox node — **not a logic bug** (xrlenv can't touch the task's own nested
>   `docker run`). Lever = pace sysbox concurrency (lower the `sysbox-runc` cap or add a
>   2nd sysbox node); content-retry handles the residual.
> - **`tw_425218`** (oVirt hosted-engine, 1800s ceiling): reward 0 once in a conc-64 raw
>   sweep (oracle exit 1, `hosted-engine --vm-status` never wrote `/app/vm_status.txt`)
>   but PASSED at conc-32 **and** the clean 192 re-run — a heavy, timing/resource-sensitive
>   setup that flakes under contention. Content-retry / lower concurrency recovers it.

> **Stub-oracle completions (2026-07-07).** Where a stub oracle can be completed
> with a *genuine* golden solution (does the real work, doesn't game the verifier),
> we do: **tw_435744** now attaches the omitted oras `signature/example` artifact +
> writes the real image digest; **tw_523250** genuinely calls `texlistsymbols` and
> writes the (documented 8192-byte-buffer-truncated) line count. **tw_488034** was
> considered but left Failed — a faithful golden solution needs a full Harbor
> server stood up in-container (~10-service compose) + the `hkjc-demo` project;
> too heavy to be worth it, and the shortcut (faking the Harbor API) would be
> gaming the verifier.

> **Stub-oracle pattern (2026-07-07):** several tasks in the "verified" split ship
> an oracle `solution/solve.sh` that DELIBERATELY doesn't complete the task
> (comments openly say so). Where a *genuine* golden completion is feasible we
> supply it (same class as the existing "partial-oracle completion" patches):
> tw_523250 (now calls texlistsymbols + writes the buffer-truncated count) and
> tw_435744 ("signature is skipped" → now attaches it) are ✅. tw_488034's stub is
> left Failed — completing it faithfully needs a full Harbor service (see below).

> **Dep/solve fixes (2026-07-07):** `tw_15324` — pyload solve installs unpinned
> `thrift` (→ 0.23.0) whose PEP-517 build needs `setuptools>=61` (dropped py2.7);
> pinned `thrift==0.13.0` → PASS. `tw_99185` — `bosh create release` prompts for the
> dev-release name and EOFs on empty stdin; feed its default (`cf`) → PASS. Both are
> curated `patches/<task>/solution/solve.sh` overlays.

> **Investigation update (2026-07-07, Round A-1 @ `--max-workers 4`).** Ran the 7
> sysbox investigation candidates concurrently. **5/7 recovered:** `tw_586787`,
> `tw_583114` (re-confirmed), plus **`tw_305688` + `tw_333762`** (iptables — these
> previously died on `pre-register with sysbox-fs: DeadlineExceeded` under
> concurrency; now pass, recovered by the node-saturation create-cap+retry fix in
> xrlenv-core) and **`tw_526185`** (CLI-only DinD redis). **Infra validation:** the
> node stayed healthy under 4 concurrent sysbox creates (D-state peaked 3, drained
> to 0; sysbox-fs never wedged) — the live proof the cap+retry prevents the
> pre-register storm. **Still failing (task-specific, NOT infra):**
> `tw_222108` (netns — the *verifier* has no DNS: `apt`/`curl`/`uvx` all fail to
> resolve, likely the netns solve or egress leaves it network-less) and
> `tw_435744` (DinD — solve builds+pushes+attaches the SBOM, but the verifier wants
> an oras `signature/example` artifact the solve never produces). Plus **tw_234227**
> flaky → passed via cpuset pinning; **tw_528959** (CPython timeout) + **tw_686647**
> (`scikit-learn<1.6` pin) validating; **tw_304270/tw_304271** reclassified Failed
> (5-service compose). Untouched: the 10 below.

The green set includes the **15 tasks recovered under sysbox** (were excluded → now pass;
`SYSBOX_TASKS` in `build_cache.py`) and the **2 repaired** dep-drift regressions
(`tw_739272` rustup pin, `tw_179356` Denarius `build.h` + cpu-pinning). The recovered ids
are listed in the [Reproduce](#reproduce) command below.

## Section 2 — per-task notes (flaky / failed / need-investigation)

### 🎲 Flaky (3)

The other two flaky tasks — `tw_11696` (sysbox DinD MariaDB port re-bind) and `tw_425218`
(oVirt hosted-engine timing) — are detailed in the "🎲 Flakes seen 2026-07-17" note near the
top (both investigated → not reproducible, retry-recovered). The residual network flake:

| task | signature | note |
|---|---|---|
| tw_241711 | `GnuTLS: Unable to establish SSL connection` (Project Gutenberg) | downloads Ulysses at solve time; transient network. 3/3 on re-run. |

> Moved out: tw_234227 (gdb backtrace) → ✅ passed via cpuset pinning; tw_528959
> (CPython timeout) → 🔍 investigation (pin validating).

### ❌ Failed — need an out-of-scope feature/privilege, a broken oracle, or an unreasonable config (7)

> **2026-07-17:** the 5 multi-service compose tasks (`tw_304270`, `tw_304271`,
> `tw_188260`, `tw_522753`, `tw_305044`) moved OUT of this bucket → ✅ via the
> cluster-compose path (see the "Step 5 — compose LANDED" note at the top).

| task | signature | why |
|---|---|---|
| tw_528959 | `AgentTimeoutError` after 2700 s (CPython from-source) | **Unreasonable task config**, not infra. The task's *own* `timeout_sec=2700` (45 min) is too tight for its solve (`wget Python-3.9.1.tgz; ./configure; make -j2; make altinstall`) — a `make -j2` CPython build from source doesn't fit 45 min even **uncontended** (cpu-pinned to 2 cores, isolated conc-1 run). Excluded 2026-07-08: 0 verifier signal, purely a build-speed vs task-ceiling mismatch. Would pass with a larger `timeout_sec` / `--timeout-multiplier` or more cores; not worth carrying a task whose shipped budget can't complete its own oracle. |
| tw_488034 | `Error: exit status 255` (tanzu harbor) | oracle hardcodes the author's macOS paths (`/Users/hinl/GolandProjects/…`) AND needs a `harbor` server sidecar + VMware registry network. **Considered a golden solution** (rewrite paths + stand up Harbor + create `hkjc-demo`), but a faithful one needs a full ~10-service Harbor deploy in-container — too heavy; the shortcut would game the verifier. Left Failed by decision. |
| tw_223822 | `splunkd validatedb … unusable filesystem` | Splunk 7.2.3 rejects overlay **and** tmpfs; needs an ext4/xfs index dir or a fs-check bypass. |
| tw_661946 | `error validating "kata-rbac.yaml"` (kubectl) | needs a real Kubernetes cluster (kata-deploy daemonset). |
| tw_513637 | `test_02_clusterrolebinding` / `test_03_configmap_ws1` / `test_04_crd_ws2` fail (kubectl) | needs a real Kubernetes cluster (clusterrolebinding + CRD + configmap). |
| tw_230695 | `losetup: cannot find an unused loop device` | needs real host loop devices. |
| tw_291556 | `losetup: … Operation not permitted` | LVM-on-loop; needs real loop-device privilege. |

### 🔍 Need investigation (2)

> Recovered in the final push (now ✅): tw_18948 (lldb `disable-aslr false` — the
> `'A' packet error 8` was seccomp blocking personality(), NOT a runtime issue),
> tw_7829 (gdb crash-input completion + `disable-randomization off`), tw_118507
> (sysbox userns SIF build + `2>&1` capture), tw_528959 (cpuset-pin validated).

> Reclassified ❌ Failed since last update: tw_488034 (author macOS paths + harbor
> sidecar), tw_523250 + tw_435744 (stub oracles that deliberately don't finish).

Each has a concrete candidate path, but NONE is a confirmed fix yet. Signatures
below are from Round A-1 (sysbox candidates) and Round A-2 (the 10 untouched, run
as-is at `--max-workers 10` — all 10 reproduced their failures; infra stayed clean,
no wedge). Most are **task-content issues (oracle solve / dep drift / verifier),
not substrate** — consistent with CLAUDE.md's "don't reinvent benchmark-side
wheels": the fix is usually a solve/dep pin, not an xrlenv change.

> Recovered under sysbox in Round A-1 (now ✅): tw_586787, tw_583114, tw_305688,
> tw_333762, tw_526185. Reclassified ❌ Failed: tw_304270, tw_304271 (5-svc compose),
> tw_513637 (k8s clusterrolebinding/CRD/configmap — needs a real cluster, like tw_661946).

| task | signature (verified) | candidate path |
|---|---|---|
| tw_222108 | verifier can't resolve DNS (`apt`/`curl`/`uvx` fail) — but only AFTER the netns solve | **DEFERRED (deep).** NOT an allow_internet gap (harbor defaults it True; other sysbox tasks resolve fine). The netns/veth solve breaks the *sysbox* container's DNS in-place, and this verifier uniquely needs internet (bootstraps `uv`+`curl` at verify time — the standard TW verifier pattern). Confirmed offline: `solve.sh` adds a veth 10.0.0.2/24 to the MAIN ns; the resolver breakage is a sysbox+nested-netns interaction (127.0.0.11 embedded-DNS should be unaffected by a 10.x route, so the mechanism is sysbox-fs-level, not routing). Needs live in-container repro; the only real fixes are restoring DNS in the solve or a hermetic verifier (upstream contract). |
| tw_245032 | verifier wants `libopenal.so.1.25.1`; Dockerfile `git clone --depth 1` of master builds `1.25.2` | **✅ FIXED + validated 1.0 (2026-07-08).** Two coupled root causes: (1) the Dockerfile clones master (drifted to 1.25.2) `--depth 1` (no tags); (2) openal 1.25.1's `alformat.hpp` uses C++20 `#include <format>`, absent from ubuntu:22.04's g++-12 (`fatal error: format`), so master only builds via its fmt fallback. Fix = `patches/tw_245032/environment/Dockerfile` overlay: pin `--branch 1.25.1` AND base **ubuntu:24.04** (default g++-13 ships `<format>`). Rebuilt+repushed via `benchmarks/terminalworld/build_plan_gen.py` → `build_and_push_images.py --force`. Re-verified 1.0 on the fresh image (the CP registry digest-resolve served the re-pushed `:main` automatically — no node eviction needed). Removed from `run_full_sweep.sh` EXCLUDE → green set 187→188. cpu-pin (OOM fix) retained. |

> **2026-07-08 — infra-fix validation (fresh cluster, cap=8 + re-admit + teardown
> deadlines live).** Re-ran the previously-failing bucket-A sysbox tasks at conc 4:
> **tw_709166 ✅ 1.0, tw_650591 ✅ 1.0** — both had died at conc 32 on
> `pre-register with sysbox-fs: DeadlineExceeded`; now start clean ("nested dockerd
> ready after 2s"), 0 errored trials, 0 retries, no wedge, no teardown hang. The
> re-admit + per-node sysbox cap resolve **bucket A**. The 2026-07-07 full-sweep
> failures dispositioned: **A** (tw_709166/tw_650591) → fixed (validated); **B**
> contention timeouts — tw_582345/tw_526185 (sysbox DinD) relieved by the cap,
> tw_593620 (vcpkg from-source) is pure-runc oversubscription that clears at low
> concurrency (passed at cap=4/conc-8, not a task bug); **C** teardown hang
> (tw_526185) → fixed by the client-side gRPC teardown deadlines. tw_245032 →
> fixed (image rebuilt). tw_222108 → deferred.
>
> **2026-07-08 update — cap=4/conc-8 full sweep: 187/188 reward-1.0, 0 verifier
> failures, 0 infra errors.** Dropping the per-node sysbox cap 8→4 + running at
> conc 8 made the single sysbox node completely stable (no wedge, no queue
> timeout, no teardown hang the whole run). The lone non-pass was **tw_528959**,
> which — unlike tw_593620 — does NOT clear at low concurrency: it busts its own
> 2700s ceiling even uncontended (isolated conc-1, cpu-pinned). Reclassified ❌
> Failed (unreasonable config) and removed from the green set (188 → 187).
>
> **2026-07-08 — conc-32 infra hardening: 187/187 (0 infra exceptions).** Pushing
> the single-sysbox-node cluster to **conc-32** (the downstream shouldn't have to
> hand-lower it) surfaced two infra bugs the conc-8 run masked, both now fixed:
> **(1) the CP re-admit budget starved the committed container-create deadline** —
> after a long sysbox queue-wait a `docker create` was left ~30 s and timed out
> mid-create (`NodeCommandTimeout`; tw_650591/tw_709166). Fix: floor the committed
> create deadline (`_MIN_CREATE_DEADLINE_S`), + preserve `NodeCommandTimeout` across
> gRPC rehydration. **(2) sysbox DESTROYS weren't serialized like creates** — 4
> concurrent FUSE unmounts wedged sysbox-fs, so `docker rm` hung in D-state and the
> container LEAKED (Up ~57 min, holding a cap slot + dragging the whole sysbox
> layer, which starved *other* tasks past their agent/verifier timeouts:
> tw_526185/586787/582345/583114). Fix: `raw_sysbox_destroy_concurrency=1` (the
> symmetric analog of the create gate). Result: **181/187 → 186/187 with 0 infra
> exceptions, no wedge, no leak** across repeated full sweeps. The lone residual is
> a **rotating ~1-task CONTENT flake** on the most nondeterministic tasks —
> gdb-backtrace unwind (tw_234227, ASLR-disable EPERM → non-deterministic
> backtrace) and DinD-verifier timing (tw_650591) occasionally reward 0 even
> cpu-pinned (the oracle ran fine; the verifier saw a nondeterministic result).
> `run_full_sweep.sh` content-retried reward-0 tasks (`--content-retries 2`, a
> task is solved if ANY attempt rewards 1.0) → **reliable 187/187**; a persistent
> failure across all attempts still fails the run.
>
> ⚠️ **The gate default is now `--content-retries 0`** (a task that only passes on
> a re-run is a finding, not a pass). This 187/187 is therefore **retry-dependent
> and pending re-verification**: the nondeterministic tasks named above will
> surface as failures at the new default. Reproduce the recorded number with an
> explicit `--content-retries 2`; re-run at 0 and triage those tasks to make it
> hold at the default. (Node leaks from the pre-fix
> runs were cleaned reboot-free via FUSE-abort + `docker rm -f`, no reboot.)

## Reproduce

```bash
set -a; source ./.env; set +a            # XRLENV_GRPC_HOST + XRLENV_CONSUMER_TOKEN + registry
# XRLENV_BENCHMARK_CACHE (the shared cache ROOT) is read from .env — see .env.example

# THE GATE — (re)builds the cache, computes the green set (present − EXCLUDE, asserts 192),
# runs it, then content-retries reward-0 flakes:
bash xrlenv_plugins/benchmarks/terminalworld/run_full_sweep.sh --max-workers 32

# --- or drive the cache + a targeted subset directly ---
python xrlenv_plugins/benchmarks/terminalworld/build_cache.py --stage all   # populate + patch + sysbox
# the 15 recovered + the 2 repaired:
python xrlenv_plugins/benchmarks/terminalworld/run_oracle_sweep.py --tasks \
  tw_245733,tw_247958,tw_650591,tw_709166,tw_27806,tw_333322,tw_16553,tw_347571,tw_11696,tw_582345,tw_105786,tw_268653,tw_420790,tw_27037,tw_313581,tw_179356,tw_739272
```

Mechanism deep-dives + full ladder: `tmp/sysbox-terminalworld-recovery-plan.md`.
