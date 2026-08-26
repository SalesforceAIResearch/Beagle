# XRLEnv

Infrastructure for **agentic RL training** — sandboxing (Docker today, microVM
coming) plus orchestration for thousands of concurrent long-horizon rollouts
across cloud VMs and a local laptop. It is **not** a trainer or a model server

## Documentation

**The full documentation is the Sphinx site under [`docs/`](docs/).** Start there:

- **[Getting Started → Quickstart](docs/getting_started/quickstart.md)** — install,
  bring up a control plane, and run your first rollout.
- **[Architecture](docs/getting_started/architecture.md)** — the trainer / control
  / data three-plane split and the core design.
- Guides for the trainer SDK, writing an EnvAdapter, onboarding a benchmark,
  multi-node deployment, observability, and the security model all live under
  [`docs/`](docs/).

Build and browse the rendered site locally (`sphinx-build` command below), or
read the Markdown sources directly.

## Development

Working *on* xrlenv? Here's the dev loop.

```bash
git clone <this-repo> && cd xrlenv
uv sync --all-extras        # .venv with runtime + dev + docs + benchmark deps
.venv/bin/python -m pytest tests/ -q
```

Use `--all-extras`, not a bare `uv sync`: the dev tools (`pytest` / `mypy` /
`ruff`), `sphinx`, observability, and the benchmark harnesses (`harbor` / `pier` /
`swebench`) are optional-dependency **extras** — a bare `uv sync` installs only
runtime deps and would *remove* the extras from an existing `.venv`. `uv.lock`
pins every version, so `uv sync --all-extras` is fully reproducible. (A git
worktree needs its own `.venv`; a shared editable install binds to the main
checkout's path.)

| Task | Command |
|---|---|
| Unit tests | `.venv/bin/python -m pytest tests/ -q` |
| Lint | `.venv/bin/python -m ruff check xrlenv/ tests/` |
| Type check (strict) | `.venv/bin/python -m mypy xrlenv/` |
| Build docs | `.venv/bin/sphinx-build -W -E -b html docs docs/_build/html` |
| Live-preview docs | `uv run sphinx-autobuild docs docs/_build/html --open-browser --port 0` |
| End-to-end laptop smoke | `python tests/smoke/cluster_bringup/single_rollout.py` (needs Docker) |
| Regenerate gRPC stubs | `bash scripts/gen_protos.sh` |

### Repo layout

| Path | What |
|---|---|
| [`xrlenv/`](xrlenv/) | Core platform — control plane, node agent, trainer SDK, sandbox backends, observability, admin. |
| [`xrlenv_plugins/`](xrlenv_plugins/) | Benchmark + EnvAdapter plug-ins (a PEP-420 namespace package). |
| [`docs/`](docs/) | The Sphinx documentation site. |
| [`deploy/`](deploy/) | Stand up and operate a fleet by hand — bootstrap, the registry servers + ops scripts (`registry/`), node provisioning (`node/`), systemd units. See [`deploy/README.md`](deploy/README.md). |
| [`scripts/`](scripts/) | Developer scripts (e.g. gRPC stub generation). |
| [`tests/`](tests/) | Unit tests plus Docker-gated end-to-end smokes. |
| [`examples/`](examples/) | Runnable examples — build plans, deployment runbook. |

### Contributing

- **Tests, lint, and types must be clean** before a PR
  (`pytest` / `ruff check` / `mypy`).
- **Keep changes focused** — don't bundle unrelated drive-by refactors.
- **Design-first for non-trivial choices** — surface the trade-off in the issue
  or PR before coding.
- **Report bugs** via GitHub issues: a minimal repro, expected vs. actual
  behavior, and the JSON-structured logs (xrlenv logs that way by default; the
  per-rollout run dir is under `~/.xrlenv/runs/<date>/<rollout_id>/`).

## License

[Apache-2.0](LICENSE).
