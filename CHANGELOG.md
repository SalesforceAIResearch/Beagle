# Changelog

> Notable changes to beagle, newest first. Entries describe **what changed for a user and why**. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). 

## [Unreleased]

_Nothing yet._

## [v0.0.2] — 2026-09-17

### Fixed

- **Rollouts no longer fail on task images whose Linux distribution has reached end of life.**
  A retired suite can keep serving a package index after its pool has been emptied, so the
  in-container git bootstrap resolved package versions whose files no longer exist and every
  affected trial died with `git bootstrap failed after 3 attempts` before the agent ever ran.
  Debian 11 entered this state in September 2026 — its security index still advertised
  `deb11u5` builds that had been purged, and had not yet been republished to
  `archive.debian.org` — which broke two terminal-bench tasks (`qemu-alpine-ssh`,
  `qemu-startup`) for *every* agent. The bootstrap now falls back to the distribution's `main` archive, which is still served,
  after the normal install has already failed. Three properties are deliberate:

  - it is **inert** unless the normal path fails, so images on a supported release are
    unaffected;
  - the suite codename and the package version to downgrade to are both **read from the
    container** at run time, so nothing is pinned to one release;
  - apt is redirected at temporary files, so the container's own `/etc/apt` is **left
    untouched** — an agent that shells out to `apt` later still sees its image's real sources.

  This is a workaround for a live archive inconsistency and is marked for removal in
  `beagle/benchmarks/harness/_common.py`; drop it once the affected packages are reachable
  again.

- **`beagle.tools.onboard` recovers from an interrupted seed.** Creating the experiment-copy
  repo and seeding it are separate network operations. When creation succeeded but the upstream
  fetch did not, the copy was left existing-but-empty, and re-running the same command failed
  with `ref 'baseline' not found on upstream` — a confusing message, since the missing ref is on
  the *copy*, not upstream. Recovery needed the destructive `--reseed`. An empty copy is now
  treated as a fresh one, so simply re-running the command repairs it. A non-empty copy still
  follows the existing skip/reseed rules, so nothing can be silently overwritten.

- **`onboard` no longer offers a GitHub token to other hosts.** Token injection was applied to
  any `https://` URL, which meant an upstream hosted somewhere other than github.com would be
  handed a GitHub PAT that cannot authenticate there. Injection is now limited to `github.com`;
  other hosts authenticate by their own means (an SSH remote, for instance).

- **An unreachable LLM gateway now fails before anything is spent.** Previously a wrong or dead
  endpoint produced a *silent* full-sweep failure: every trial installed cleanly, started work,
  exhausted its own retries and recorded a clean completion with zero tokens, so the run scored
  0.000 and looked like a capability result rather than broken plumbing. `beagle evaluate` now
  TCP-connects to the resolved endpoint before acquiring any container, and refuses to start if
  nothing is listening. `--dry-run` reports the same check.

  The check is by **route type, not vendor**: a `gateway` route (any OpenAI-compatible
  `api_base`) is probed, as is a deployment-native `internal` route; a `direct` first-party
  provider is deliberately not, since it is not an endpoint beagle operates and probing it would
  false-fail behind an egress proxy.

  When the failing endpoint came from the environment and `.env` disagrees, the error says so
  explicitly. That case is otherwise very hard to diagnose: `.env` reads correct, yet a variable
  already exported in the shell takes precedence over it, so re-running whatever writes `.env`
  cannot help.

### Added

- **Optional per-solved-task latency and cost columns in the results dashboard.** A checkbox —
  *"Also show latency & cost per SOLVED task"*, off by default — adds `[solved]Latency/task (s)`
  and `[solved]Cost/task ($)` beside the existing all-task columns.

  They are **additional**, never a replacement: the existing columns answer "what does attempting
  a task cost", which mixes in failures that can be cheap (died early) or expensive (burned the
  whole budget getting nowhere); the new ones answer "what does solving one cost", which is the
  figure worth comparing across harnesses. Both populations are visible side by side. Cells are
  blank rather than `0` where a benchmark solved nothing, since `0` would read as "free".

- **A `Version` column in the dashboard, immediately after `Harness`**, labelled
  `<version>_<ref6>` — for example `20260826_f0d15a`.

  Both halves earn their place: the declared version alone stops distinguishing runs once
  evolution starts, because every candidate inherits its baseline's version, while the commit ref
  alone is unreadable and loses the name the harness was onboarded under. Together they stay
  meaningful in both modes — several benchmark rows sharing one `<version>_<ref6>` are visibly
  the same candidate.

  The version is read from the canonical config the run recorded, falling back to the onboarding
  manifest and finally to a short ref. It is deliberately *not* parsed out of the run directory
  name, which only carries a version for generated configs and would be wrong silently otherwise.

- **An end-to-end runbook for onboarding a new agent harness**, `docs/onboarding-an-agent.md`:
  the up-front questions worth answering before writing code, the `install`/`run_in` split for
  network-phased harnesses, the contracts an adapter must honour (timeout resolution, token
  normalisation, ATIF trajectories, patch capture, egress declaration, provider routing), how to
  wire an agent into config generation, and the traps that have cost real runs.
  `docs/advanced.md` keeps its short conceptual introduction and links to it.

