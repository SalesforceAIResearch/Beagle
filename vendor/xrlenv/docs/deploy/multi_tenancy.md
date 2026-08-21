# Multi-tenancy

A single XRLEnv control plane can serve a small team. **Multi-tenancy**
gives each person their own bearer token bound to a stable tenant id —
their `owner_id` — instead of everyone sharing one consumer token. The
operator mints these tokens, hands them out, and can cut off any one
person without disturbing the others.

This page is for the **operator** running the shared control plane. It
covers what works today — per-user identity, per-user revocation,
server-stamped ownership, owner-scoped admin views, and per-owner
fair-share scheduling — the issue/distribute/revoke workflow, where the
records live on disk, and how to bound how much capacity any one tenant
can claim.

## Why mint per-user tokens

With the legacy single shared token, everyone authenticates as the same
identity. Revoking it — after a laptop is lost, or a teammate leaves —
forces a rotation that locks *everyone* out until they pick up the new
token.

Per-user tokens remove that coupling. Each token carries its own
`owner_id` and its own `token_id`, so:

- You can **revoke one person's access** and leave everyone else's
  tokens valid.
- Each person's token has its own `digest_hint`, so the audit log
  (`auth.token_used` / `auth.denied`) distinguishes one user's calls
  from another's — under a single shared token every call looks
  identical.
- Issuing a new person's token does **not** require restarting
  `xrlenv up` — the running control plane hot-reloads the change.

These are concrete benefits available now, and reason enough to start
minting per-user tokens for your team today. The `owner_id` each token
carries also drives **owner-scoped admin views** (described in
[Ownership and owner-scoped views](#ownership-and-owner-scoped-views)
below): the control plane stamps every job with the authenticated
user's owner, and the admin panel shows each user only their own work.
That same `owner_id` is the key **fair-share scheduling** (described in
[Per-owner fair-share scheduling](#per-owner-fair-share-scheduling)
below) uses to bound how much of the cluster any one tenant can claim.

## Issuing a per-user token

On the control-plane host, add `--owner <id>` to `xrlenv tokens issue`.
The `--owner` flag is accepted for the **consumer**, **viewer**, and
**operator** roles:

```bash
# A consumer token for Alice — she uses it from the SDK / Docker drop-in:
xrlenv tokens issue consumer --owner alice
# issued consumer token for owner=alice (token_id=...)
#   recorded at: ~/.xrlenv/secrets/users.json (hashed; plaintext not stored)
#   raw token:   <urlsafe-string>
#   Copy the token now — it will not be shown again.

# A viewer token for Bob, with a friendly label:
xrlenv tokens issue viewer --owner bob --name "Bob (read-only)"
```

The `--name "<label>"` is an optional display name shown in
`xrlenv tokens list`. It is for your convenience as the operator; it
does not change the token's privileges.

The **raw token is printed exactly once** at issue time and is never
written to disk. Copy it immediately and hand it to the user (see
[Distributing](#distributing-a-token-to-a-user) below). What gets stored
is the token's SHA-256 digest plus its `owner_id` — not the bearer bytes
— so the file is safe to back up alongside your other operator config.

Many per-user tokens of the **same role** coexist. You can issue a
consumer token for every member of the team; they do not collide and do
not overwrite each other.

:::{note}
`--owner` is rejected with `role=node`. Nodes are infrastructure, not
tenants — a node authenticates the data plane, not a person. Mint node
tokens without `--owner`, as described in
{doc}`/developer_guide/tokens`.
:::

### Backward compatibility

Omitting `--owner` keeps the existing behaviour exactly: a single
shared role-token written to `<secrets-root>/<role>.token`, which maps
to `owner_id="default"`. Existing single-tenant deployments are
unaffected — you do not have to adopt per-user tokens, and mixing one
shared consumer token with several per-user consumer tokens is fine.

## Listing tokens

`xrlenv tokens list` shows shared role-tokens first, then a dedicated
**per-user tokens** section below them:

```bash
xrlenv tokens list
```

```text
tokens loaded from ~/.xrlenv/secrets:
  consumer  active   token_id=b72e440f9a11 digest_hint=b72e44 owner=default
  operator  active   token_id=c01d8e3f5512 digest_hint=c01d8e owner=default
per-user tokens (multi-user):
  consumer  user     token_id=a3f9c1d82b47 digest_hint=a3f9c1 owner=alice
  viewer    user     token_id=f5b210e9c884 digest_hint=f5b210 owner=bob (Bob (read-only))
```

Each per-user row shows the role, the 12-character `token_id`, the
6-character `digest_hint` that appears in audit logs, the `owner`
(tenant id), and the optional display name in parentheses. Raw bearer
bytes are never printed.

## Distributing a token to a user

1. Issue the token with the user's tenant id:

   ```bash
   xrlenv tokens issue consumer --owner alice --name "Alice"
   ```

2. Copy the `raw token:` line from the output. This is the only time it
   is shown.

3. Hand it to the user over a private channel. They set it as their
   consumer credential — `XRLENV_CONSUMER_TOKEN` in their `.env` /
   workflow shell, or pass it to `Client.grpc(token=...)`. **That same
   one token also opens the admin panel** — read-only and scoped to
   their own jobs: they paste it on the panel's sign-in page (and use
   **`log out`** to switch tokens). So a user needs just the one consumer
   token. A separate `viewer` token is only for watch-only people who
   don't submit jobs; `operator` is the un-scoped admin. See
   {doc}`/observability/admin_auth` for the browser flow.

No restart is needed on your side. The running control plane watches
`users.json` by mtime and hot-reloads the new record, so Alice can
authenticate as soon as she has the token.

```{warning}
**Don't let `XRLENV_CONSUMER_TOKEN` reach the control-plane process.**
That variable is a *client* credential (it tells the SDK which token to
send). If the control plane is launched with it set — e.g. `xrlenv up`
run in a shell that `source`d a client `.env`, or a systemd
`EnvironmentFile=` / `--env-file` pointing at one — the control plane
registers that value as the **shared `consumer` role-token**, which always
carries `owner_id="default"`. A user whose per-user token *is* that value
would then silently authenticate as the shared **default** tenant (their
jobs stamped `owner_id="default"`, their admin view un-scoped to "default")
instead of as themselves.

XRLEnv now resolves this collision in the user's favor — the more-specific
per-user identity wins — and logs a `WARNING` at token-load time naming the
owner and role. If you see that warning, scrub `XRLENV_CONSUMER_TOKEN`
(and `XRLENV_NODE_TOKEN`) from the control-plane environment: the control
plane only needs its **operator** token (admin writes) and **node** token
(to authenticate nodes); per-user consumers come from `users.json`.
```

## Revoking one user's access

Per-user tokens revoke **exactly like** shared role-tokens — by
`token_id`. Revoking one tenant's token leaves every other token,
shared or per-user, untouched.

```bash
# Find the token_id (or read it from an audit log line):
xrlenv tokens list
#   consumer  user  token_id=a3f9c1d82b47 digest_hint=a3f9c1 owner=alice

# Revoke by the full 12-character token_id:
xrlenv tokens revoke a3f9c1d82b47

# …or by any unambiguous prefix of at least 6 characters
# (the digest_hint works directly):
xrlenv tokens revoke a3f9c1
```

The running control plane picks up the revocation on its next
hot-reload — **no restart**. From that point, the revoked bearer is
rejected and the attempt is logged under `auth.denied`. The command is
idempotent; revoking an already-revoked token is a no-op.

This is the workflow you reach for when a teammate leaves or a laptop is
lost: revoke that one token, and everyone else keeps working under their
own unchanged tokens.

## Ownership and owner-scoped views

Each per-user token carries an `owner_id`, and the control plane now uses
it to **stamp ownership** and **scope what each user sees** in the admin
panel. This is what lets a whole team share one admin URL: each person
logs in with their own token — the **same consumer token they keep in
`.env`** — and sees only their own work.

### Ownership is server-stamped

When a consumer authenticates — whether they run a rollout or open a raw
container session via the docker-py drop-in / `acquire_container` path —
the control plane reads the `owner_id` off the **verified bearer token**
and stamps it onto the resulting rollout or session server-side.

The owner is taken from the token the server validated, **not** from
anything the client sends. A client cannot claim to be another owner:
there is no request field a caller can set to spoof ownership. Runs made
under an embedded/in-process runtime, with no auth, or under a single
shared (non-`--owner`) token are stamped `owner_id="default"`, exactly as
before.

### Who sees what

The admin panel scopes the per-user data tabs — **Rollouts**, raw
**Sessions**, **Sandboxes**, and their artifact / log / download files —
by the viewer's role:

| Who is logged in | What they see |
|---|---|
| **Per-user `consumer` token** (`--owner <id>`, the one in their `.env`) | Read-only, **scoped to their own `owner_id`**: only their rollouts, raw sessions, sandboxes, and files. Another owner's id returns **404** (including direct artifact/log/download URLs). The one token a user already has. |
| **Per-user `viewer` token** (`--owner <id>`) | Same scoped read-only view — for watch-only people who don't submit jobs. |
| **Operator token** | **All** owners, plus writes. The operator is the admin and sees the whole cluster's jobs. |
| **Loopback / SSH-tunnel dev flow** (auth bypassed) | **All** owners — the loopback bind bypasses auth, so it has the same see-all view as an operator. |

On the panel's **sign-in page** a user just pastes their token — identity
(role + owner) comes from the token alone (operator token for the full
view, their consumer/viewer token for the scoped view). **`log out`** in
the nav clears the session so they can switch tokens.

The cluster-infrastructure tabs — **Nodes**, **Capacity**, **Cluster
health**, **Images** — are **global for everyone**. They carry no
per-user data (they describe the cluster itself, not any one tenant's
jobs), so they are not scoped and every authenticated viewer sees the
same cluster-wide view.

In short: an **operator** token is the admin view (see-all); a **per-user
consumer or viewer** token is read-only and scoped to its own `owner_id`;
the **loopback dev flow** sees all because it bypasses auth entirely.
Because the owner is stamped from the verified bearer, the scoping cannot
be bypassed by a crafted request.

### One admin URL for the whole team

This removes the need for per-user SSH tunnels just to give teammates a
filtered view. Expose **one** admin URL behind token auth (a non-loopback
bind with an operator token issued), and each teammate logs in with their
own token — the consumer token they already have — to get a view scoped
to their jobs. The bind mechanics, the browser login flow, and the
bind-guard rules are unchanged — see {doc}`/observability/admin_auth` for
those; this page only adds the
per-owner scoping layer on top.

The loopback + `ssh -L` flow still works as the zero-token operator
escape hatch, and (because it bypasses auth) still shows every owner's
jobs — that is the operator/admin view, not a per-user one.

## Where the records live

Per-user token records live in a single file on the control-plane host:

| File | Format | Mode | Holds |
|---|---|---|---|
| `~/.xrlenv/secrets/users.json` | JSON array of records: `token_sha`, `token_id`, `role`, `owner_id`, `display_name`, `created_at`. | `0600` | **Hashes and metadata only.** The raw token is never stored — it is printed once at issue and is unrecoverable afterwards. |

Because the file holds SHA-256 digests rather than bearer bytes, losing
read access to it does not leak any usable credential. It still lives
under `~/.xrlenv/secrets/` and inherits the tight `0600` perms by
convention. Override the directory with `--secrets-root <path>` on any
`xrlenv tokens` subcommand.

Revocations are recorded in the shared `~/.xrlenv/secrets/revoked.json`
alongside role-token revocations — there is no separate per-user
revocation file. See {doc}`/developer_guide/tokens` for that file's
format and the full on-disk layout.

The running control plane hot-reloads `users.json` and `revoked.json`
via an mtime watcher on the secrets directory, so issuing and revoking
per-user tokens both take effect without restarting `xrlenv up`.

## Per-owner fair-share scheduling

The pieces above give each tenant an identity and scope what they *see*.
Fair-share scheduling bounds what each tenant can *consume*: it caps how
much of the cluster's concurrent capacity any one `owner_id` may hold, so
one user's large sweep cannot starve everyone else.

It is **off by default** — until you opt in, all tenants share one
scheduling pool and no per-owner cap applies. You enable it, and tune it,
live from the control-plane host.

### The model

When fair-share is enabled, the control plane applies a **per-owner
concurrency cap** at admission. The cap for one owner is:

> **cap = the global default owner cap, overridden by any owner-specific cap;
> uncapped owners bypass fair-share; blocked owners get cap 0.**

Concretely, with a default owner cap of `N`:

```text
cap(owner) = N                              by default
cap(owner) = owner_cap                      if you set --owner ... --cap
cap(owner) = None / uncapped                if you set --owner ... --uncap
cap(owner) = 0                              if you set --owner ... --block
```

Two properties fall out of this:

- **No manual cluster cap.** `--default-cap N` is not an overall cluster
  limit and does not reserve resources. It means each owner can receive up
  to `N` concurrent containers when the scheduler has real resources.
- **No head-of-line block.** When a tenant is at its cap, only *that
  tenant's* new rollout/session requests park in the admission queue
  until they drain below the cap. A different owner's requests are
  admitted ahead of them — one tenant filling its share never blocks
  another.

**Fair-share is soft and never destructive.** Lowering a cap or blocking a
tenant only stops that tenant's *new* admissions; **running jobs are
never killed**. To forcibly reclaim in-flight work you use cancel, which
is a separate operation.

#### Worked examples

Take a default owner cap of `4`:

| Situation | Result |
|---|---|
| Two owners, both active | cap **4** each, subject to real scheduler/node capacity. |
| Block one of them | the blocked owner gets **0** (new admissions stop); the other still has cap **4**. |
| `bob` has `--cap 2` | bob's cap is **2**, while owners without an override stay at **4**. |
| `alice` has `--cap 32` | alice's cap is **32**, while owners without an override stay at **4**. |
| `carol` has `--uncap` | carol bypasses fair-share caps; scheduler/node resources still apply. |

### Enabling and tuning it live

Fair-share is configured with `xrlenv fairshare` on the **control-plane
host** — the same host that runs `xrlenv up`. There is **no restart**:
the command writes the policy to the control plane's state, and the
control plane re-reads it on its next admission drain, so a change applies
within a few seconds.

**Inspect the current policy:**

```bash
xrlenv fairshare show
```

```text
fair-share: ENABLED  default_cap=4
per-owner:
  alice                running=2  effective_cap=4
  bob                  running=1  effective_cap=2  owner_cap=2
  carol                running=8  effective_cap=uncapped  UNCAPPED
```

Each per-owner row shows the owner's current `running` count, the
effective cap computed for it right now, plus any owner cap, uncapped state,
or blocked state. When fair-share is disabled the command says so and reminds
you that all owners run uncapped.

**Enable fairness with a default per-owner cap:**

```bash
# Let every owner reach 4 concurrent sandboxes when resources exist:
xrlenv fairshare set --default-cap 4
```

`--default-cap <N>` is the default owner concurrent-sandbox cap. It is
not a cluster-wide cap; the scheduler still decides whether resources are
available. To turn fairness back off entirely:

```bash
xrlenv fairshare set --disable
```

**Give one tenant a larger or smaller owner-specific cap:**

```bash
# Let alice reach 32 concurrent sandboxes when resources exist:
xrlenv fairshare set --owner alice --cap 32

# Keep bob at 2 concurrent sandboxes:
xrlenv fairshare set --owner bob --cap 2

# Return bob to the default cap later:
xrlenv fairshare set --owner bob --recap
```

**Uncap or recap a tenant** (bypass / reapply fair-share caps for that owner):

```bash
xrlenv fairshare set --owner carol --uncap
xrlenv fairshare set --owner carol --recap
```

**Block or unblock a tenant** (stop / resume only its *new* admissions —
running jobs keep going):

```bash
xrlenv fairshare set --owner carol --block
xrlenv fairshare set --owner carol --unblock
```

**Remove a tenant's override entirely** (back to default cap, not uncapped,
not blocked):

```bash
xrlenv fairshare set --clear-owner alice
```

Every `xrlenv fairshare set` prints the resulting policy and a one-line
confirmation that the control plane will pick it up on its next admission
drain.

### Fair-share in the admin panel

The admin panel has a read-only **Fair share** tab that shows the live
policy and a per-owner running / effective cap / owner cap / uncapped /
blocked table — the same view as `xrlenv fairshare show`, in the browser. It is **operator-only**:
because it lists every tenant's usage, it is not part of the owner-scoped
per-user view. Tuning is done from the CLI above (the tab links to it);
the panel itself does not write policy.

## See also

- {doc}`/observability/admin_auth` — admin-panel roles, the browser login flow, and the
  bind-guard rules.
- {doc}`/developer_guide/tokens` — issuance, rotation, revocation, and
  the full on-disk layout for all token files.
- {doc}`/developer_guide/security` — the full security model and audit
  behavior.
