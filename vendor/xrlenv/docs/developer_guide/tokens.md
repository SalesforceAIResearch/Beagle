# Tokens

xrlenv uses bearer tokens for inter-plane authentication. Four
roles, each issued separately:

| Role | Issued by | Used by | Purpose |
|---|---|---|---|
| **node** | `xrlenv tokens issue node` | `xrlenv-node serve` | Authenticates an `xrlenv-node` daemon to the control plane on bidi-stream registration. |
| **consumer** | `xrlenv tokens issue consumer` | `Client.grpc(token=...)` / `xrlenv.from_env()` | Authenticates SDK and Docker SDK drop-in workflows. |
| **viewer** | `xrlenv tokens issue viewer` | Admin panel GET routes; browser or HTTP client. | Read-only admin panel access. Token carries a `read_` prefix. |
| **operator** | `xrlenv tokens issue operator` | Admin panel write routes; CLI mutate-shaped commands. | Full admin access: viewer scope plus `POST /api/build/*` and future destructive actions. Token carries a `write_` prefix. |

`node` and `consumer` tokens are unprefixed. The `read_` / `write_` prefixes
on viewer and operator tokens are decorative — they make the privilege level
visible when a token is shared in a chat message or runbook, but the server
validates role by the token's stored identity, not by the prefix string.

Tokens are stored as per-role files under `~/.xrlenv/secrets/` on
the control-plane host (one file per role, mode `0600`):
`node.token`, `consumer.token`, `viewer.token`, `operator.token`. `tokens issue`
refuses to overwrite an existing file — use `xrlenv tokens rotate
<role>` to replace a token in place (see [Rotating a
token](#rotating-a-token) below). Override the directory with
`--secrets-root <path>` on any `tokens` subcommand.

## Issuance

```bash
# On the control-plane host:
.venv/bin/xrlenv tokens issue node
# → prints the token string to stdout. Copy this into the data-plane
#   node's systemd unit at [Service] Environment=XRLENV_NODE_TOKEN=<paste>.

.venv/bin/xrlenv tokens issue consumer
# → prints the token. Set as XRLENV_CONSUMER_TOKEN in the
#   workflow shell, or pass to Client.grpc(token=...) directly.

.venv/bin/xrlenv tokens issue viewer
# → prints a read_<...> token. Share with teammates who need
#   read-only access to the admin panel.

.venv/bin/xrlenv tokens issue operator
# → prints a write_<...> token. Use for CLI admin commands or
#   browser sessions that need write access.
```

See {doc}`/observability/admin_auth` for how to distribute viewer tokens
to teammates and configure the browser login flow.

## Audit

Every token use is logged in XRLEnv's audit log:

```bash
.venv/bin/xrlenv audit | grep auth.token_used
.venv/bin/xrlenv audit | grep auth.denied
```

Failed authentications appear under `auth.denied` with the
attempted role, the remote address, and the reason
(`unknown_token` / `expired` / `wrong_role` / `revoked`).

## Auth-off (loopback dev)

`xrlenv up` defaults auth **off** when binding to `127.0.0.1` with no
TokenStore configured. This is the {doc}`/getting_started/quickstart`
shape and the single-node dev loop — no token setup required.

For any non-loopback admin bind, credentials are required. Passing
`--admin-host 0.0.0.0 --admin-allow-public` without any issued tokens
raises `AdminBindError` at startup. Issue at least one viewer or
operator token before binding publicly. See {doc}`/observability/admin_auth`
for the full bind-guard rules.

## Rotating a token

Rotation replaces the active token for a role with a new one. Use this
after a suspected leak, during node re-provisioning, or as part of
regular credential hygiene.

```bash
# Immediate cutover — the prior token is invalid on the next RPC:
xrlenv tokens rotate node

# Grace-window cutover — the prior token remains valid for 24 hours.
# Use only during a deployment rollover where nodes cannot all be
# updated at once:
xrlenv tokens rotate node --grace 24h
```

**Immediate cutover** is the security default. The new token is written
to `<secrets-root>/<role>.token`; any previous token is rejected by the
control plane from the next RPC onwards. In-flight RPCs that arrive
before the hot-reload window are not affected.

**Grace-window cutover** (`--grace <duration>`) keeps the prior token
valid for the specified period alongside the new one. Accepted formats:
plain seconds (`3600`), minutes (`5m`), hours (`2h`), or days (`1d`).
Use this only when a fleet of nodes cannot be updated atomically. Once
the window closes, only the new token is accepted.

The new token is written to `<secrets-root>/<role>.token` (mode
`0600`). With `--grace`, the prior token is also written to
`<role>.token.previous.json` (mode `0600`). Without `--grace`, any
leftover `.previous.json` sidecar is cleaned up automatically.

A running `xrlenv up` picks up the change without restart — the
control plane's mtime watcher reloads token files on the next RPC.

## Revoking a token

Revocation permanently invalidates a specific token by its ID. Use this
when a token has been distributed to a host that is decommissioned or
compromised.

```bash
# Revoke by the full 12-character token_id:
xrlenv tokens revoke a3f9c1d82b47

# Revoke by digest_hint — the 6-character prefix visible in audit logs.
# The prefix must be at least 6 characters and unambiguous:
xrlenv tokens revoke a3f9c1
```

The `token_id` is the first 12 characters of the bearer's SHA-256
digest. The `digest_hint` (first 6 characters) appears in
`auth.token_used` and `auth.denied` audit log entries, so you can
paste directly from an audit query to the revoke command.

Revocation appends a record to `<secrets-root>/revoked.json`. The
control plane reloads this file on the next RPC, so no restart is
needed. The command is idempotent — revoking an already-revoked token
is a no-op.

**`revoked.json` is not secret.** It contains only SHA-256 prefixes,
not bearer bytes. It can be treated like any other operator config file
when backing up or auditing the secrets directory.

Exit codes: `0` on success, `1` if no matching token was found, `2` if
the prefix is shorter than 6 characters or matches more than one token.

## Listing tokens

`xrlenv tokens list` shows the active token state per role without
printing raw bearer bytes.

```bash
xrlenv tokens list
```

Example output:

```
ROLE      TOKEN_ID      DIGEST_HINT  GRACE_REMAINING  STATUS
node      a3f9c1d82b47  a3f9c1       —                active
consumer  b72e440f9a11  b72e44       —                active
operator  c01d8e3f5512  c01d8e       1h 23m           grace (prior)
```

The `TOKEN_ID` column is the 12-character SHA-256 prefix. `DIGEST_HINT`
is the 6-character prefix that appears in audit log lines. `STATUS` is
`active` for the current token; `grace (prior)` for a prior token still
within its grace window. Raw token bytes are never printed by this
command.

## On-disk layout

All token files live under `~/.xrlenv/secrets/` by default. Override
with `--secrets-root PATH` on any `xrlenv tokens` subcommand.

| File | Format | Mode | Lifecycle |
|---|---|---|---|
| `<role>.token` | Plain text — the raw bearer string, no trailing newline. | `0600` | Written by `tokens issue` and `tokens rotate`. Overwritten in place. |
| `<role>.token.previous.json` | JSON object: `{"token": "<bearer>", "grace_until": "<ISO-8601 UTC>"}` | `0600` | Written only when `tokens rotate --grace` is used. Deleted automatically by the next `tokens rotate` without `--grace`. |
| `revoked.json` | JSON array of objects: `[{"token_id": "<12-char>", "revoked_at": "<ISO-8601 UTC>"}, ...]` | umask-default (typically `0644`) | Appended to by `tokens revoke`. Never shrinks. Not secret — contains only SHA-256 prefixes, so the file can be world-readable without leaking bearer material. |

A running `xrlenv up` hot-reloads all three files via an mtime watcher
on the secrets directory. Changes take effect on the next RPC; no
restart required.

## See also

- {doc}`/developer_guide/security` — security model and audit behavior.
- {doc}`/deploy/multi_node_deployment/runbook` — where tokens fit in
  a multi-node deployment.
- {doc}`cli_reference` — `xrlenv tokens` subcommand flag reference.
