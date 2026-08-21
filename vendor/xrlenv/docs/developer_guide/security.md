# Security model

XRLEnv is designed for trusted operators running their own workloads
on their own Docker-capable hosts. The control plane supports multiple
authenticated users on the same cluster; each user's identity is
scoped and enforced server-side. It is not a hardened public
sandbox service — container images and workloads are treated as
trusted within the operator's environment.

## Network shape

Node daemons initiate the connection to the control plane. The
control plane does not open inbound connections to nodes. In a
multi-node deployment, expose the control-plane gRPC port only to the
workflow hosts and node VMs that need it.

The admin panel binds to `127.0.0.1` by default. For remote access,
either use an SSH tunnel (no credentials needed):

```bash
ssh -L 8080:127.0.0.1:8080 user@control-plane
```

or bind publicly with credentials — see
{doc}`/observability/admin_auth` for the two-tier role model, browser
login flow, and bind-guard rules.

## Per-user identity and owner scoping

Consumer tokens carry a server-stamped `owner_id`. The control plane
writes this field at token-issue time; the consumer cannot set or
spoof it — the field is ignored on the wire and overwritten by the
server on every rollout and raw-container acquire.

Key behaviors:

- **Tokens hashed at rest.** Per-user tokens are stored as SHA-256
  digests in `users.json`; the plaintext is never persisted.
- **Per-user revocation.** `xrlenv tokens revoke <token-id>` adds the
  token's id to the revocation set. Future requests carrying that
  token are rejected without touching other tokens for the same owner.
- **Owner-scoped views.** The admin panel's `/rollouts/raw` and
  `/sandboxes` endpoints filter results to the requesting token's
  `owner_id`. An `operator` token sees all owners.
- **Fair-share gate.** Concurrent-container admission applies a
  per-owner cap so one owner cannot starve others on the same cluster.

Issue a per-user token on the control-plane host:

```bash
xrlenv tokens issue consumer --owner alice
```

See {doc}`/deploy/multi_tenancy` for the full operator workflow
(issuing, rotating, and revoking per-user tokens; setting the
per-owner cap).

## Tokens and roles

XRLEnv uses bearer tokens for authenticated deployments.

| Role | Command | Used by |
|---|---|---|
| `consumer` | `xrlenv tokens issue consumer` | SDK callers and `xrlenv.from_env()` workflows. |
| `node` | `xrlenv tokens issue node` | `xrlenv-node serve`. |
| `viewer` | `xrlenv tokens issue viewer` | Read-only admin panel access (browser or HTTP client). |
| `operator` | `xrlenv tokens issue operator` | Full admin panel write access; CLI admin commands. |

Use the narrowest token role that can perform the action. Do not put
operator tokens into benchmark jobs or long-running harness
processes.

## Container isolation

The shipped backend is Docker. XRLEnv schedules containers, applies
resource limits where configured, tracks lifecycle, and destroys
containers when sessions end. It does not make untrusted code safe to
run on shared hosts. Treat container images and workloads as trusted
within the operator's environment.

## Egress restriction for running containers

`ClusterContainerSession.apply_egress(allowlist)` installs an iptables
policy in the container's network namespace after the container is
running. Build the allowlist from `EgressRule` objects:

```python
from xrlenv.backends.egress import EgressAllowlist, EgressRule

allowlist = EgressAllowlist(rules=(
    EgressRule(cidr="internal-ip/8"),           # all internal traffic
    EgressRule(cidr="93.184.216.34/32", ports=(443,)),  # one external host, HTTPS only
))
await session.apply_egress(allowlist)
```

An **empty allowlist blocks all external egress** (loopback remains
up). This is the right default for evaluation tasks that should not
reach external endpoints:

```python
await session.apply_egress(EgressAllowlist())  # block all
```

Implementation details and guarantees:

- **Enforcement mechanism.** Rules are compiled host-side and
  installed via `nsenter` into the container's netns. The workload
  holds no `CAP_NET_ADMIN` so it cannot flush or modify the chain.
- **Fail-closed.** If the enforcer fails partway through applying
  rules, the container is destroyed immediately. A task can never
  run on partial (under-restricted) egress.
- **Refused containers.** `apply_egress` raises `XRLEnvError` for
  containers with `network_mode=host` or `container:<id>` (shared
  netns — entering would rewrite the node host's chain), for
  privileged containers or containers with `CAP_NET_ADMIN` (the
  workload could flush the rules), and for containers running under
  `sysbox-runc` (the inner root controls its own netns and can flush
  the OUTPUT chain). Acquire restricted tasks without those flags.
- **Optional DNS resolver.** Pass `dns_resolver="8.8.8.8"` to pin
  DNS traffic to a specific resolver; omit it to leave DNS
  unrestricted by the egress policy.

This is the sole anti-cheat / network containment primitive exposed
to consumers. Policy (what CIDRs to allow for a given benchmark) is
decided by the benchmark adapter, not by XRLEnv core.

## Metadata services

Cloud metadata endpoints can leak credentials if a workload can reach
them. Use closed networks or explicit egress controls for workloads
that should not reach provider metadata services.

## Audit log

Authentication decisions are written to a separate `audit` table in `state.db`.
Two event kinds exist:

- **`auth.denied`** — always recorded. A rejected authentication attempt
  (bad or revoked token, wrong scope). These are the high-value security
  events; a non-zero count warrants investigation.
- **`auth.token_used`** — off by default. Per-RPC successful-authentication
  records. At scale these were ~99.9% of the audit table, churning the SQLite
  WAL on every call. Set `XRLENV_AUDIT_AUTH_SUCCESS=1` in the control-plane
  environment to restore the full spec-19 success trail.

Query the audit log:

```bash
# Check for auth denials (should be 0 in a healthy cluster):
xrlenv audit --kind auth.denied --since 1h

# Full trail (requires XRLENV_AUDIT_AUTH_SUCCESS=1):
xrlenv audit --since 1h
```

See {doc}`tokens` for token issuance and storage details. See {doc}`cli_reference` for the `xrlenv audit` filter reference and the `xrlenv db prune` / `xrlenv db vacuum` commands for managing audit table size.
