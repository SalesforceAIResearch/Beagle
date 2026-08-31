# Admin authentication

The admin panel (`http://control-plane:8080/`) has two transports for the same
{doc}`/developer_guide/tokens`: a browser **sign-in page** that establishes a
logout-able **cookie session**, and **bearer-auth** for the CLI / HTTP clients.

```{note}
The browser path used to be HTTP **basic auth**. It was replaced (B7.4) by the
cookie session because browsers cache basic-auth credentials per realm and
replay them on every request with no application-controllable logout — so an
operator who signed in as one consumer could never switch to another without
clearing browser state. The sign-in page + **`log out`** button fix that. A
cached basic-auth header from the old flow is now ignored, so logout is
authoritative.
```

## Roles

| Role | What it can do |
|------|----------------|
| **consumer** | The per-user token a user already keeps in their `.env` to submit jobs. It also opens the admin **read-only**, **scoped to that user's own jobs** (their rollouts / sessions / sandboxes), plus the read-only cluster-infra tabs (nodes / capacity / health / images). No writes; the Fair-share tab is operator-only. So a user needs just **one token**. |
| **viewer** | Watch-only identity for people who don't submit jobs. Same read access as a consumer; owner-scoped if it carries an `owner_id`. Cannot trigger writes. |
| **operator** | Everything a viewer can do, but **un-scoped** (sees every owner), plus the write routes: `POST /api/build/apply`, `POST /api/build/cancel`, `POST /api/build/calibrate`, and any future destructive admin actions. |

`node` tokens are for the data plane and are not accepted by the admin panel.

Owner scoping: a `consumer` or `viewer` token sees only its own `owner_id`'s
jobs; an `operator` token sees all owners. (See {doc}`/deploy/multi_tenancy`.)

## Issuing tokens

On the control-plane host:

```bash
# Viewer token — share with teammates who need read access:
xrlenv tokens issue viewer
# prints: read_<urlsafe-32-char-string>

# Operator token — keep restricted; full write access:
xrlenv tokens issue operator
# prints: write_<urlsafe-32-char-string>
```

The `read_` and `write_` prefixes are visible signals only — the server
validates role by the token's stored identity, not by the prefix string.
When you paste a token into a chat message or a runbook, the prefix makes
the privilege level visible at a glance.

Tokens are stored in `~/.xrlenv/secrets/` on the control-plane host.
See {doc}`/developer_guide/tokens` for rotation, revocation, and the
on-disk layout.

## When auth engages

The admin server gates requests **only on non-loopback binds**. The matrix:

| Bind | Tokens issued | Auth engages? |
|---|---|---|
| `127.0.0.1` (default) | doesn't matter | **No.** SSH tunnel is the protection boundary. |
| `0.0.0.0` (`--admin-allow-public`) | no admin tokens | Server refuses to start — bind guard catches it. |
| `0.0.0.0` (`--admin-allow-public`) | viewer or operator tokens | **Yes** — every non-open route needs a session cookie (browser) or bearer token (CLI). |

The "loopback bypasses auth" rule reflects the actual security model:
adding auth on top of an SSH-tunnel adds zero security uplift (anyone
reaching the loopback already passed SSH) and inflicts real UX cost —
modern browsers auto-upgrade `http://localhost` to `https://`, which the
HTTP-only admin server can't speak; the failure surfaces as a mysterious
"Internal Server Error" with nothing in the server log.

For the **loopback workflow** (cluster VM + `ssh -L 8080:127.0.0.1:8080`
to your laptop, then a browser on the laptop): no tokens needed, no
sign-in page. **Use `http://127.0.0.1:8080/` in the address bar**,
not `http://localhost:8080/` — Chrome / Safari upgrade `localhost` to
`https://localhost` silently, and the connection never reaches the
server. The `xrlenv up` startup log calls this out on every boot.

For a **public bind** workflow (rare; typically a status-page-style
deployment with a real domain behind a TLS-terminating proxy): issue
viewer / operator tokens up front, then the browser flow below applies.

## Browser flow (public binds only)

Navigate to `http://control-plane:8080/`. Because you have no session yet, the
panel redirects you to **`/login`** (the target you asked for is preserved in
`?next=`, so you land back on it after signing in).

On the sign-in page, paste your **token**:

- a per-user **consumer** token (the one in your `.env`) for a read-only view
  scoped to your own jobs, or
- the **operator** token for the full, un-scoped view plus write actions.

There is no username field — identity (role + owner) is resolved from the token
alone. On success the server stores the token in an `HttpOnly`,
`SameSite=Lax` cookie (`xrlenv_admin_session`) and you're in. The token is
re-verified against the TokenStore on every request, so revoking or rotating it
takes effect immediately regardless of cookie lifetime.

### Switching tokens / signing out

Every page's nav shows the signed-in identity (`owner · role`) and a **`log
out`** button. Clicking it clears the session cookie and returns you to the
sign-in page, where you can paste a **different** token — e.g. an operator
inspecting one consumer's scoped view, then another's. This is the workflow
HTTP basic auth made impossible; a credential cached by the old basic-auth flow
is ignored, so logout always wins.

## CLI and programmatic flow

HTTP clients (the CLI, `curl`, scripts) authenticate with a **bearer** header —
unchanged:

```bash
# Apply a build plan from the CLI:
xrlenv build apply \
    --connect-host control-plane:8080 \
    --operator-token write_EXAMPLE_OPERATOR_TOKEN

# Or pass it as an Authorization header in any HTTP client:
curl -H "Authorization: Bearer write_EXAMPLE_OPERATOR_TOKEN" \
    http://control-plane:8080/api/build/plans/<plan_id>
```

A credential-less API request (no `Accept: text/html`) gets a JSON `401` with a
`WWW-Authenticate: Bearer` challenge rather than a redirect, so scripts see a
clean error instead of the sign-in HTML.

## Sharing a viewer token with teammates

1. Issue a viewer token once:

   ```bash
   xrlenv tokens issue viewer
   # read_EXAMPLE_VIEWER_TOKEN_NOT_REAL
   ```

2. Share the printed string. The `read_` prefix makes the privilege
   obvious without additional explanation.

3. Each teammate navigates to `http://control-plane:8080/`, pastes
   `read_EXAMPLE_VIEWER_TOKEN_NOT_REAL` on the sign-in page, and gains
   read-only panel access.

To revoke access later, rotate or revoke the token:

```bash
# Option A: rotate — issues a new viewer token; the old one is immediately
# invalid. Use when you want to keep viewer access alive under a new bearer
# (e.g. teammate left, redistribute to remaining viewers).
xrlenv tokens rotate viewer

# Option B: revoke — kills the specific token by its 12-char token_id (the
# SHA-256-derived identifier, NOT the raw bearer bytes). Get the id from
# `tokens list`, then revoke by full id or by any ≥6-char unique prefix.
xrlenv tokens list
# → viewer    active   token_id=aaaaaaaaaaaa digest_hint=aaaaaa ...
xrlenv tokens revoke aaaaaa                 # revoke by the 6-char digest_hint
# or:
xrlenv tokens revoke aaaaaaaaaaaa           # revoke by the full 12-char token_id
```

See {doc}`/developer_guide/tokens` for the full rotation and revocation
workflow.

## Bind-guard: public binds require admin-capable credentials

```bash
# Requires --admin-allow-public AND a TokenStore holding at least one
# admin-capable (viewer or operator) token:
xrlenv up \
    --admin-host 0.0.0.0 \
    --admin-port 8080 \
    --admin-allow-public

# A public bind raises AdminBindError at startup when either condition fails:
# - no TokenStore wired / store is empty, OR
# - no shared `viewer` / `operator` role-token is present. A per-user
#   `consumer` token now grants read-only, owner-scoped access, but the guard
#   still wants an explicit viewer/operator before public exposure so the panel
#   has a management-capable identity. The operator normally issues an operator
#   token anyway, so this passes in the common case.
```

The bind-guard enforces two conditions together. Passing
`--admin-allow-public` without an issued `viewer` or `operator` token is
rejected: the error message names the roles currently present and tells the
operator to run `xrlenv tokens issue viewer` or
`xrlenv tokens issue operator` before retrying. `node` and `consumer` tokens
do not satisfy the guard — those identities authenticate the gRPC data
plane, not the admin HTTP surface.

### SSH-tunnel alternative (no credentials needed)

If you prefer not to manage viewer tokens, the SSH-tunnel approach requires
no TokenStore at all. The loopback bind stays the protection boundary:

```bash
# On the control-plane host — keep --admin-host at its loopback default:
xrlenv up --admin-port 8080

# On your local machine:
ssh -L 8080:127.0.0.1:8080 user@control-plane
# Then open http://127.0.0.1:8080 locally.
```

The tunnel encrypts the connection and uses SSH keys for access control, so
no HTTP credentials are required.

## Loopback dev escape hatch

When the TokenStore is absent or empty and the admin server binds to
`127.0.0.1`, auth is bypassed entirely. This is the default for local
development and single-host smoke runs — `xrlenv up` with no token setup
just works:

```bash
xrlenv up          # no tokens issued, no --admin-allow-public needed
# Open http://127.0.0.1:8080 — no login prompt.
```

The escape hatch is strictly loopback-only. Any non-loopback bind triggers
the bind-guard regardless of TokenStore state.

## Routes that are always open

`/healthz`, `/static/*`, `/login`, and `/logout` bypass authentication.
Load-balancer health probes and the sign-in / sign-out flow need these
without credentials.

## See also

- {doc}`/developer_guide/tokens` — issuance, rotation, revocation, on-disk
  layout, and audit log integration.
- {doc}`admin_panel` — page reference for every admin URL.
- {doc}`/deploy/multi_tenancy` — per-user tokens and per-user revocation for
  a shared control plane.
- {doc}`/deploy/multi_node_deployment/runbook` — where admin auth fits in a
  multi-node deployment.
- {doc}`/developer_guide/security` — full security model.
