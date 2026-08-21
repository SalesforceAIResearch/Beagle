# `scripts/gateway/` — reach LLM Gateway Express from the cluster

Cluster/HyperPod nodes usually can't reach LLM Gateway Express directly; your laptop
can. This bridges the gap so any agent that speaks the gateway's OpenAI-compatible API
via `LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL` works — **no agent code change**, agent-agnostic.

```
container → http://<node-ip>:18088/  →  node forwarder  →  127.0.0.1:<tunnel>
          →  ssh -R tunnel  →  laptop relay  →  https://<gateway>/…
```

## The two-script workflow

Two hops, two scripts. Run each on the machine named after it.

**1. On your laptop** (which can reach the gateway):

```bash
set -a; source .env; set +a           # LLM_GATEWAY_EXPRESS_API_KEY_LIST (laptop.sh also sources it)
scripts/gateway/laptop.sh w1          # <ssh alias | host | user@ip>  [+ extra ssh options]
```

Starts the relay + an auto-reconnecting `ssh -R` tunnel to the node. Add jump hosts /
keys as trailing args: `scripts/gateway/laptop.sh w1 -J bastion -i ~/.ssh/id`.

**2. On the cluster login node** (the ssh target — where the tunnel landed):

```bash
scripts/gateway/login-node.sh         # optional: login-node.sh <forward-port>  (default 18088)
```

It **auto-detects** the reverse-tunnel port (which drifts if the laptop's 18080 was
busy) and the node IP, verifies the tunnel is live, and — this is the point — **writes
`LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL` into your `.env` for you** (replace-or-append; every
other line and the file permissions untouched):

```
LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL=http://<node-ip>:18088/
```

Because it derives the node IP at runtime and rewrites that one line every run, the
value can't go stale when you land on a different node — **no drift, no copy-paste.**
(Escape hatches: `ENV_FILE=<path>` to target a different file, `GATEWAY_ENV_WRITE=0` to
only print.) beagle then forwards `LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL` +
`LLM_GATEWAY_EXPRESS_API_KEY_LIST` into the trial container via the run config's `model`
block (`provider: llm-gateway-express-local-proxy`).

**3. Verify**, from any shell or container that will host the agent:

```bash
python3 scripts/gateway/gateway_proxy.py check --url http://<node-ip>:18088/
# any HTTP response ⇒ forwarder + tunnel + relay are all up
```

`check` proves the **tunnel** is up, but its health probe is answered by the relay
itself and never touches the gateway — so a real completion is the only true test:

```bash
curl -s -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5.5","messages":[{"role":"user","content":"ping"}],"max_tokens":5}' \
  http://<node-ip>:18088/chat/completions
```

### TLS to the gateway (`CERTIFICATE_VERIFY_FAILED`)

The gateway's cert is signed by a **corporate CA** that lives in the macOS keychains but
not in Python's trust store — so a raw Python client fails with "unable to get local
issuer certificate" (monet's Node client works only because Node ships its own CA
bundle). `laptop.sh` fixes this **automatically on macOS**: it builds a combined bundle
(`certifi` + the `SystemRootCertificates` and `System` keychains) at
`~/.cache/beagle/gateway-ca-bundle.pem` and points Python + curl at it
(`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` + `serve --cafile`). So `bash
laptop.sh w1` just works — no manual cert steps.

Overrides: `GATEWAY_CAFILE=/path/to/ca.pem` to supply your own bundle, or
`GATEWAY_INSECURE=1` to skip verification (the ssh tunnel still protects laptop↔node;
verify-on is otherwise the default).

## Why two hops (and not just the tunnel)

`ssh -R` binds the node's **loopback**. That's reachable if the agent runs *on the node*,
but a harbor trial **container** has its own `127.0.0.1` and can't see it. `login-node.sh`
(the forwarder) re-exposes the loopback tunnel on the node's routable IP
(`0.0.0.0:18088`) — no `socat`, no sshd `GatewayPorts` — so containers on any node reach
it, and the address is stable enough to allowlist (see below).

## Internet-off tasks

xrlenv acquires the container with the network **open** (agent install needs it); an
`allow_internet=false` task is *meant* to have egress restricted to an allowlist
afterwards, which must include the proxy's `<node-ip>/32:18088` so the model hop
survives while general internet is blocked. **Caveat:** on the native `harbor.Job` path
that restriction is not yet enforced — it's infra, tracked for the xrlenv side in
`notes/triage/xrlenv-harbor-offline-egress.md`. All 89 tb2.1 tasks are
`allow_internet=true`, so the baseline is unaffected.

## `gateway_proxy.py` — the underlying tool

`laptop.sh` / `login-node.sh` are thin wrappers; the stdlib-only (no monet, no beagle
import — `scp` it and run `python3`) engine is `gateway_proxy.py`, with three
subcommands:

| subcommand | where | what |
|---|---|---|
| `serve --remote <target>` | laptop | relay + `ssh -R` tunnel (what `laptop.sh` runs) |
| `forward --listen H:P --to H:P` | node | expose the loopback tunnel routably (what `login-node.sh` runs) |
| `check [--url …]` | cluster | one HTTP probe; any response ⇒ the chain is up |

The relay serves only the gateway's two paths (`/chat/completions`, `/responses`, no
`/v1`), streams SSE, and round-robins the key list for requests with no `Authorization`
header. `serve --help` lists the knobs (`--ssh-option`, `--local-port`, `--upstream`,
`--max-concurrent`, `--remote-bind`, reconnect flags).
