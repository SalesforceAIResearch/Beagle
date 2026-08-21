# Structured logs

`xrlenv up` emits one of two output styles, picked from `--log-format`:

- `pretty` — short ANSI-colorized line per record (`HH:MM:SS LEVEL  event: message`)
  with red=ERROR, yellow=WARN, green=INFO, dim=DEBUG. Designed for an
  operator watching a terminal.
- `json` — one JSON object per line that
  `journalctl`/`docker logs` capture and `jq` filters.

The default is `auto`: `pretty` when stdout is a TTY, `json` when it's
piped or redirected — so running `xrlenv up` interactively gets you
colorized output, while piping to a file or running under `systemd`
keeps producing structured records. Force a format with
`--log-format pretty` or `--log-format json`. Set `NO_COLOR=1` in the
environment to suppress ANSI escapes (the layout is unchanged so log
lines remain grep-friendly). The same flag is wired into
`xrlenv-node serve` and the in-sandbox stub.

In `json` mode, every record carries:

```json
{"ts": "2026-04-26T12:34:56.789Z", "level": "INFO", "event": "rollout.start",
 "rollout_id": "abc123", "template": "terminal-bench-2", "node_id": "gcp-1"}
```

**Common fields:** `ts`, `level`, `event`, `rollout_id?`, `sandbox_id?`,
`node_id?`, `template?`.

**Key events** in the JSON log stream:

| Event | When |
|-------|------|
| `sandbox.create` | Sandbox created on a node |
| `sandbox.create.failed` | Sandbox failed to start (includes `reason`) |
| `sandbox.destroy` | Sandbox destroyed |
| `rollout.start` | Rollout admitted and sandbox creation begun |
| `rollout.step` | Step completed (sampled at INFO to avoid volume) |
| `rollout.finish` | Rollout sealed as `finished` |
| `rollout.truncate` | Hard deadline reached |
| `rollout.fail` | Rollout sealed as `failed` |
| `node.connected` | Node registered |
| `node.disconnected` | Node stream dropped; open rollouts sealed as `node_lost` |
| `plugin_root.mounted` | External plugin root bind-mounted into the sandbox `PYTHONPATH` |

Audit events (`auth.token_used`, `auth.denied`, `mount.denied`,
`template.registered`, `placement.image_check`) live in the separate
`audit` table. Query them with `xrlenv audit` rather than
`xrlenv events`; see {doc}`/developer_guide/cli_reference`.

Configure log level via `--log-level`:

```bash
xrlenv up --log-level DEBUG ...
```

Pipe to `jq` for filtering (use `--log-format json` so a TTY still emits structured records):

```bash
xrlenv up --log-format json ... 2>&1 | jq -r 'select(.event | startswith("rollout.fail")) | [.ts, .rollout_id, .reason] | @tsv'
```

## Rotating file sink

By default the full log firehose goes to stdout. When the control plane
runs for days under a process supervisor (Slurm `#SBATCH --output=…`,
`nohup`, `tee`) that captures stdout, that capture file grows without
bound. Pass `--log-file` to move the firehose to a size-rotating file
instead:

```bash
xrlenv up \
    --grpc-host 0.0.0.0 \
    --grpc-port 50051 \
    --log-file ~/.xrlenv/xrlenv-up.log \
    --log-max-bytes 52428800 \
    --log-backup-count 10
```

`--log-file` (and the other logging flags) live on the `up` subcommand,
so they go **after** `up` alongside the other `up` flags.

**What changes with `--log-file`:**

- A `RotatingFileHandler` writes every record ≥ `--log-level` to PATH,
  one JSON envelope per line. The file format is always JSON regardless
  of `--log-format` — `--log-format` only affects the console formatter.
- Stdout is floored at `WARNING`, so a Slurm `.out` file or `nohup.out`
  stays small but still captures the boot banner and any crashes.
- The path is **stable across restarts**: `tail -f` the same file every
  time instead of hunting for a per-job output file.
- Rotated backups get `.1` / `.2` / … suffixes in the same directory.

**Disk ceiling:** `--log-max-bytes × (--log-backup-count + 1)` — with
the defaults (50 MiB × 11) that is ≈ 550 MiB total.

**Override the stdout floor** with `--stdout-log-level`. The effective
default is `--log-level` when `--log-file` is absent (full firehose on
stdout), or `WARNING` when `--log-file` is set. To mirror the full
firehose to stdout while also writing to the rotating file:

```bash
xrlenv up --log-file ~/.xrlenv/xrlenv-up.log --stdout-log-level INFO ...
```

**Tail the rotating file** with `jq` for live filtering:

```bash
tail -f ~/.xrlenv/xrlenv-up.log \
  | jq -r 'select(.level == "ERROR") | [.ts, .event, .rollout_id] | @tsv'
```

**Node daemon:** the node daemon runs under systemd with
`StandardOutput=journal`. journald already rotates its log, so
`--log-file` is a control-plane flag only and has no effect on node
agents.

See {doc}`/developer_guide/cli_reference` for the full flag reference
(`--log-file`, `--log-max-bytes`, `--log-backup-count`,
`--stdout-log-level`).

## See also

- {doc}`metrics` — Prometheus `/metrics` for the same lifecycle data in scraped form.
- {doc}`/observability/admin_panel` — `coordinator.log` for per-rollout debug output.
- {doc}`/developer_guide/cli_reference` — `xrlenv events` and `xrlenv audit`.
