# XRLEnv

Agentic-RL sandbox infrastructure: sandboxing + orchestration for long-horizon
rollouts across cloud VMs.

> **Status:** phase-0 platform feature-complete; under rapid development with
> large refactors expected. This README is for **internal developers**;
> end-user / external-developer docs live under [`docs/`](docs/).

## Quick start

```bash
git clone <this-repo>
cd XRLEnv
uv sync --all-extras              # .venv with runtime + dev + docs + benchmark deps
.venv/bin/python -m pytest tests/ -q
```

Use `--all-extras` (not a bare `uv sync`): the dev tools (`pytest`/`mypy`/`ruff`),
docs (`sphinx`), observability, and the pinned benchmark harnesses (`harbor`/`pier`/
`swebench`) are `[project.optional-dependencies]` **extras**, so a bare `uv sync`
installs only the runtime deps — and would *remove* the extras from an existing
`.venv`. `uv.lock` pins the exact version of every package (extras included), so
`uv sync --all-extras` is fully reproducible. A git worktree needs its **own**
`.venv` (`uv sync --all-extras` inside it) — a shared editable install binds to the
main checkout's path.

That's it for a working dev env. The full architectural picture is in
[CLAUDE.md](CLAUDE.md) (project status, slice progression, what runs today)
and [docs/getting_started/architecture.md](docs/getting_started/architecture.md) (three-plane split, the 10
load-bearing invariants, slice history).

## Common dev commands

| What | Command |
|---|---|
| Unit tests | `.venv/bin/python -m pytest tests/ -q` |
| Lint | `.venv/bin/python -m ruff check xrlenv/ tests/` |
| Type check (strict) | `.venv/bin/python -m mypy xrlenv/` |
| Build Sphinx docs | `.venv/bin/sphinx-build -W -E -b html docs docs/_build/html` (sphinx ships with `--all-extras`) |
| Live-preview Sphinx docs | `uv run sphinx-autobuild docs docs/_build/html --open-browser --port 0` |
| End-to-end laptop smoke | `python examples/single_rollout.py` (needs Docker) |
| Regenerate gRPC stubs | `bash scripts/gen_protos.sh` |

Sphinx site source lives at [`docs/`](docs/) — see [docs/README.md](docs/README.md)
for build details. Open `docs/_build/html/index.html` after building.

## Where to find things

- **Specs (design source of truth):** [`specs/`](specs/) — read
  [`specs/00-overview.md`](specs/00-overview.md) first; it has the
  authoritative phase matrix + invariants. Per-spec phase ladders defer
  to it.
- **Project status:** [`CLAUDE.md`](CLAUDE.md) — what runs today, slice
  progression, what's missing.
- **User-facing docs:** [`docs/`](docs/) — architecture, operator guide,
  trainer SDK, template / EnvAdapter authoring, security, observability.
- **Code:** [`xrlenv/`](xrlenv/) — package layout summarized at the top of
  CLAUDE.md.
- **Deployment scripts:** [`deploy/`](deploy/) — bootstrap / refresh /
  node bring-up. The optional cluster-wide pull-through registry mirror
  (server + worker client config + cache warming) is in
  [`deploy/registry/README.md`](deploy/registry/README.md); the
  operator-facing walkthrough is in the Sphinx site under
  *Deploy → Multi-node → Registry mirror*.
- **Internal release-cycle notes:** [`notes/`](notes/) — design docs +
  phase-gate acceptance records (tracked in git). Only the audit ↔ rebuttal
  scratch (`notes/audit.md`, `notes/rebuttal.md`) is gitignored. Not part of
  the Sphinx site by design.

## Contributing / development workflow

The project iterates in **slices**: a self-contained chunk of platform work
that lands as one commit, then runs through an audit → rebuttal cycle in
[`notes/`](notes/) before the next slice begins. The slice progression is
recorded in [CLAUDE.md](CLAUDE.md) ("Slice progression" section).

Before opening a PR:

1. **Tests + lint + types must all be clean.** No exceptions.
2. **Touch only what the slice needs.** Don't bundle drive-by refactors;
   they bury the intent of the change.
3. **Don't auto-update [CLAUDE.md](CLAUDE.md).** Status updates land
   when the user asks for them — not as part of slice commits.
4. **Specs drive code, not the other way around.** If a slice changes
   the design, update the relevant spec under [`specs/`](specs/) in the
   same commit.
5. **Use the design-first workflow** — for non-trivial choices, surface
   the fork (e.g. via discussion) before coding.

Two specialized agents live under [`.claude/agents/`](.claude/agents/) and
should be invoked proactively when their description matches:

- **qa-test-engineer** — after a logical chunk of new code lands.
- **sphinx-docs-writer** — when a new public API or module is added /
  changed.

## Bug reports

Open a GitHub issue with:

1. **Repro steps** — minimal example that triggers the bug. Reference
   the slice / commit you saw it on (`git log --oneline -1` is fine).
2. **Expected vs actual** — what the spec or invariant promises vs what
   you observed.
3. **Logs + relevant state** — JSON-structured stdout (the platform
   logs that way by default — see
   [docs/observability/index.md](docs/observability/index.md)) plus the rollout's
   `coordinator.log` from `~/.xrlenv/runs/<date>/<rollout_id>/`.
4. **Spec reference if applicable** — if the bug violates a documented
   invariant, cite spec 00 §"Critical design rules" or the specific
   per-spec section.

Bugs against pre-phase-1 features should be tagged `phase-0`.

## License

Apache-2.0 (see `pyproject.toml`).
