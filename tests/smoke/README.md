# beagle smoke tests

Manually-run smokes that exercise beagle end-to-end against a real xrlenv cluster
(agent rollouts through the native harness, real task-image pulls, real cost).
**Excluded from the default `pytest -q`** via `addopts = "--ignore=tests/smoke"` in
`pyproject.toml` — run them deliberately, not on every push.

## Tiers

| Tier | Path | What it is | Runs when |
|---|---|---|---|
| **Unit** | `tests/unit/` | Mechanical, hermetic, fast. No cluster/Docker/money. | Every `pytest -q`. |
| **Smoke** | `tests/smoke/` | Real beagle runs on a live cluster; cost minutes/money. Manual. | Deliberately, per-release. |
| **Integration** | `tests/integration/` | CI/CD orchestration — runs the smokes back-to-back, fails loudly. | Post-release gate. |

## Layout

Each smoke group lives in its own subdirectory with its tests and a `README.md`
runbook beside them (mirrors `vendor/xrlenv/tests/smoke/`):

```
tests/smoke/
├── README.md              ← you are here
└── <group>/
    ├── README.md          ← operator runbook for this group's smokes
    └── test_*.py
```

## Conventions

**Dual-mode** — every smoke runs as a pytest module *and* a standalone script:

```bash
python -m pytest -v -s tests/smoke/<group>/<smoke>.py   # CI-shaped
python tests/smoke/<group>/<smoke>.py [--flags ...]     # ad-hoc
```

`-s` matters — smokes print progress + a summary to stdout.

**Standard env, no smoke-specific vars.** Gate on the existing xrlenv knobs — never
invent new ones (see `CLAUDE.md` §1): `XRLENV_GRPC_HOST` / `XRLENV_GRPC_PORT` /
`XRLENV_CONSUMER_TOKEN` / `XRLENV_BENCHMARK_CACHE`. Missing required env → **skip**
(pytest) or exit-with-guidance (script). The unit tier strips these for
hermeticity (`tests/unit/conftest.py`), so smokes are the only place they take
effect.

**Trust the artifact, not the exit code.** Decide pass/fail by re-reading the reward
from disk, not the returned scalar — a run can exit "clean" while every trial
silently scored 0.

**Artifact output.** Route durable output to `<repo>/tmp/smoke-<label>-<utc-ts>/`
(gitignored).
