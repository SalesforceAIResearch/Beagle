# seta curated patches

`patches/<task_id>/<rel_path>` files are **full-file overlays** copied over the
populated cache task (`<cache>/seta-env/<task_id>/<rel_path>`) by
`build_cache.py --stage patch` (which `--stage all` runs after populate, before
sysbox). They exist to repair **Harbor-migration damage** — cases where the
`camel-ai/seta-env` Harbor-Dataset conversion dropped or changed runtime-critical
config that the original `Dataset/<id>/` (pre-Harbor Terminal-Bench format) still
has. The overlay is the smallest faithful change that restores the original
behavior; the fix and its reason are logged here.

Harbor's OracleAgent reads `solution/solve.sh` **from this cache** (not from git),
so a `solve.sh` overlay takes effect with no image rebuild. (Overlays under
`environment/` only affect the built image if that task's build uses a local
context — see `build_plan_gen.py`.)

## Current patches

| task | file | why (migration damage) |
|---|---|---|
| `309` | `solution/solve.sh` | The solve.sh installs itself to `/app/secure_delete.sh` only when it detects it is the oracle run, guarding on `SCRIPT_DIR == "/oracle"`. Harbor 0.20 runs the oracle from **`/solution`** (its `paths.py`: "copied to container @ /solution"), so the guard never fires → the test invokes a missing `/app/secure_delete.sh` (exit 127). One-line repair `"/oracle"` → `"/solution"`; validated reward 1.0. |
