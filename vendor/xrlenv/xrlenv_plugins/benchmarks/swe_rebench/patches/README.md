# swe-rebench — curated task-content patches

**This directory is currently EMPTY.** It is the hook for curated content fixes
if — and only if — the oracle sweep surfaces broken upstream content.

Each `patches/<task_id>/<relative_path>` is a **full-file overlay** applied on
top of the faithfully-populated cache by `build_cache.py --stage patch`
(idempotent, and re-applied on every `--stage all` so it survives a
re-populate). Overlays touch only benchmark content; xrlenv core is never
changed. Every overlay must be logged here **and** in `STATUS.md`.

## The faithfulness rule

Keep each overlay the **smallest** diff that lifts a task's reward ceiling to
passing — *complete the partial, don't re-author the task*. A dependency that
drifted → pin it back to the era version, changing only that line. Record the
line-delta so drift stays visible.

If a fix would amount to rewriting the task's premise, its tests, or its
grading rule, it is **not** a patch — exclude the task instead (add it to
`EXCLUDE` in `run_full_sweep.sh` with its evidence in `STATUS.md`).

## The other half of `--stage patch`

`--stage patch` applies **two kinds** of curated fix — the same split
terminal-bench-2-1 uses (`PATCHES` + `ENV_PATCHES`):

1. the full-file overlays in this directory, and
2. the programmatic `task.toml` **resource routing** —
   `XRLENV_CPU_PINNING` markers and memory overrides, from
   `CPU_PINNING_TASKS` / `MEMORY_OVERRIDES` in `build_cache.py`.

Keep the routing there rather than expressing it as an overlay here: a full-file
`task.toml` overlay would also freeze the `docker_image` pin (which must stay
programmatic so it tracks a re-populate), and a memory value written as an
overlay would bypass `_assert_memory_override_is_fair` — the guard that refuses
to override a resource upstream declared for itself.

## What does NOT belong here

- **Anything that rewrites a verifier.** `tests/test.sh` and `tests/parser.py`
  decide reward; editing them to make a task pass is scoring your own homework,
  not fixing content. If a task only passes with a changed verifier, exclude it.
- **The `docker_image` pin.** That is written programmatically by
  `--stage repin` from each task's own `tests/config.json`, so it tracks a
  re-populate automatically. A frozen overlay would silently drift.
- **Resource routing.** The `XRLENV_CPU_PINNING` markers and the memory
  overrides are written programmatically by `--stage patch` from
  `CPU_PINNING_TASKS` / `MEMORY_OVERRIDES` in `build_cache.py`. Keep them there,
  not here: a full-file `task.toml` overlay would also freeze the
  `docker_image` pin, and a memory value written as an overlay would bypass
  `_assert_memory_override_is_fair` — the guard that refuses to override a
  resource upstream declared for itself.
- **A registry host.** Nothing under this directory may contain a literal
  `<host>:5011/…` ref — see GUIDELINE §5.3.1. Use the
  `${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}/…`
  placeholder, which the harbor plug-in expands from `.env` at acquire.
