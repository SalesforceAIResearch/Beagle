# Multi-tenancy smoke — two users, real traffic

Validates the multi-user feature set **end to end against a live `xrlenv up`**,
with no faked records: two users (alice, bob) submit *real* jobs over separate
authenticated gRPC connections, and you confirm by eye in the admin panel that
each user sees only their own work while the operator sees all.

It drives **both** consumer use-cases:

- **Raw containers** (the primary path) — each user acquires long-lived
  containers (`acquire_container` → `exec` → kept alive). Visible as live,
  owner-scoped sessions; destroyed when you stop the smoke.
- **Gym/step rollouts** (optional) — each user runs short rollouts to
  completion, leaving owner-scoped sealed trajectories.

The control plane stamps every rollout/session with the `owner_id` read off the
caller's verified token (never client-supplied), so this exercises the actual
production owner boundary, not a simulation.

## Prerequisites

1. A running `xrlenv up` with **≥1 node attached**, gRPC reachable (default
   `127.0.0.1:50051`).
2. **Auth on** — owner-stamping only engages when the control plane has a
   `TokenStore` (i.e. at least one token issued). The smoke issues per-user
   tokens into the secrets root for you.
3. The admin panel bound **non-loopback** (e.g. `xrlenv up --admin-host 0.0.0.0
   --admin-allow-public`). A loopback bind is treated as the trusted SSH-tunnel
   boundary and **bypasses auth** — every page would render as the admin view,
   hiding the per-user scoping. Browse with `127.0.0.1` in the URL (not
   `localhost`, which browsers silently upgrade to https).
4. A small image present/pullable on nodes (default `busybox:latest`).
5. For the rollout path: a template registered on the cluster (e.g.
   `hello-shell`).

## Run

```bash
# Raw-container traffic only (most portable):
.venv/bin/python tests/smoke/multi_tenancy/two_user_cluster_smoke.py \
    --connect-host 127.0.0.1 --connect-port 50051 \
    --admin-url http://127.0.0.1:8080 \
    --jobs-per-user 2

# Add real gym/step rollout traffic too:
.venv/bin/python tests/smoke/multi_tenancy/two_user_cluster_smoke.py \
    --rollout-template hello-shell --rollouts-per-user 2
```

The script issues one **consumer** token per user (alice, bob) — that single
token both submits jobs *and* opens the admin panel (read-only, scoped to the
user's own jobs) — plus ensures an **operator** token exists for the see-all
view. They're written into `--secrets-root` (default `~/.xrlenv/secrets`) so
your running `xrlenv up` hot-reloads them with no restart. It prints every token
+ a checklist, then holds the raw sessions alive until you Ctrl-C.

**Re-running is idempotent.** Each user's consumer token is **reused** across
runs (stable `token_id`, so a browser login keeps working) rather than minting a
fresh one each time, and stale duplicates from earlier runs are pruned — so
`users.json` doesn't pile up. Reuse needs the plaintext (which the hashed
`users.json` can't return), so minted tokens are cached `0600` at
`<secrets-root>/.smoke_multi_tenancy_tokens.json`. A cached token is reused only
if it still verifies as that owner's consumer token, so revoking/rotating/
deleting it self-heals to a fresh mint on the next run.

## What to verify (printed as a checklist at runtime)

Open the admin URL; you land on a **sign-in page**. Paste a token — each user
signs in with the same consumer token they submit with. Use **`log out`** in the
nav to switch to a different token (e.g. operator → alice → bob):

| Sign in with … | Expect |
|---|---|
| operator token | `/rollouts/raw` lists **both** alice's and bob's sessions |
| alice's consumer token | `/rollouts/raw` lists **only** `alice-task-*` |
| bob's consumer token | `/rollouts/raw` lists **only** `bob-task-*` |
| alice's consumer token | `/sandboxes` shows only alice's; operator shows all |
| alice's consumer token | opening bob's `/raw-rollouts/<bob-id>` by URL → **404** |
| operator token | `/fairshare` shows the live policy; a scoped user gets **404** |

If you ran the rollout path, the same scoping holds on `/rollouts/template` and
the per-rollout detail pages.

## Fair-share (live, no restart)

In a separate terminal, against the **same** `state.db` your cluster uses:

```bash
.venv/bin/python -m xrlenv.cli fairshare show
.venv/bin/python -m xrlenv.cli fairshare set --default-cap 2  # cap each owner at 2
.venv/bin/python -m xrlenv.cli fairshare set --owner bob --block   # stop bob's NEW admissions
```

Re-run the smoke with a larger `--jobs-per-user` after capping and watch one
user's acquires park in the admission queue while the other proceeds — and
note that pausing/capping never kills already-running jobs (soft throttle).

## Cleanup

Ctrl-C destroys the live raw sessions. Re-runs reuse the alice/bob tokens (no
pile-up). To fully reset the smoke's tokens, delete the cache + let the next run
re-mint, or revoke them:

```bash
rm <secrets-root>/.smoke_multi_tenancy_tokens.json   # forget the cached plaintexts
xrlenv tokens revoke <token_id>                       # the smoke prints each token_id
```

The operator token is never touched (the smoke reuses your existing
`operator.token`).
