# GCP node bootstrap

## One-shot bring-up (recommended)

On the freshly provisioned GCP VM:

```bash
# 1. Get the code onto the VM.
git clone <your-fork-of-xrlenv>.git
cd xrlenv

# 2. Set the two required env vars.
export XRLENV_CONTROL_PLANE="<control-plane-host>:50051"
export XRLENV_NODE_TOKEN="<token from `xrlenv tokens issue node` on the control plane>"

# 3. Run the one-shot bring-up — auto-detects GCP, runs the right
#    bootstrap, wires the token into a systemd drop-in, restarts the
#    daemon, and tails the first few seconds of journalctl so you see
#    the connect (or a clear failure) before the script returns.
sudo -E bash deploy/bring-up-node.sh
```

After it returns, on the control plane:

```bash
xrlenv nodes    # the new node should show STATUS=connected
```

That's it.

`sudo -E` is load-bearing: bare `sudo bash deploy/bring-up-node.sh`
strips your environment and the script aborts in red with
`XRLENV_CONTROL_PLANE must be set` *before* it installs any packages.
The validation runs up front so you can fix the call and re-run
without rolling back a half-finished install.

The bootstrap script installs Docker, configures cgroup v2 limits, sets
up the `xrlenv-node` systemd unit, and runs the cloud-metadata
block self-test that refuses to register the node if the metadata IP
is reachable from sandboxes. Output is colorized: green `OK`, yellow
`WARN`, red `ERROR`. Set `NO_COLOR=1` to suppress the ANSI codes (CI
/ piping to a file).

## Required pre-reqs

| Env var | Required | What it is |
|---------|----------|------------|
| `XRLENV_CONTROL_PLANE` | yes | `host:port` the node will dial outbound. |
| `XRLENV_NODE_TOKEN` | recommended | Bearer token issued by `xrlenv tokens issue node`. Omit only for trusted loopback development. |
| `XRLENV_NODE_ID` | usually auto | Stable id; auto-filled from GCP metadata when unset. |
| `XRLENV_REPO` *or* `XRLENV_WHEEL` | one of | Where to install xrlenv from (no public PyPI release yet). Defaults to the directory `bring-up-node.sh` lives in. The repo path is installed *non-editable* — see the {doc}`/deploy/multi_node_deployment/runbook` for why and how to iterate. |
| `XRLENV_FORCE_CLOUD` | no | Override cloud auto-detect (`gcp` or `aws`). Rarely needed. |
| `XRLENV_LOG_TAIL_S` | no | Seconds of journalctl to tail at the end (default 8). |

Issue the node token on the control-plane host before bringing up
the VM:

```bash
xrlenv tokens issue node
```

See {doc}`/developer_guide/cli_reference` — `xrlenv tokens issue` for role and scope details.

## Manual flow (if you need to debug a step)

The one-shot script delegates to two pieces — both can be run
directly when you want to drive each step yourself:

```bash
# Just install + start the daemon (no token wiring):
sudo -E bash deploy/bootstrap-gcp.sh

# Add the token afterward (writes a systemd drop-in at
# /etc/systemd/system/xrlenv-node.service.d/10-token.conf):
echo -e "[Service]\nEnvironment=\"XRLENV_NODE_TOKEN=<token>\"" \
    | sudo tee /etc/systemd/system/xrlenv-node.service.d/10-token.conf
sudo chmod 0600 /etc/systemd/system/xrlenv-node.service.d/10-token.conf
sudo systemctl daemon-reload
sudo systemctl restart xrlenv-node
```

The drop-in approach (vs editing `/etc/xrlenv/node.env` directly)
keeps the secret out of the world-readable EnvironmentFile and
survives bootstrap re-runs without clobbering the token.

## See also

- {doc}`/deploy/multi_node_deployment/index` — topology overview.
- {doc}`/deploy/multi_node_deployment/cloud_VM_providers/aws` — AWS bootstrap.
- {doc}`/deploy/multi_node_deployment/inventory` — declare this node in `nodes.yaml`.
- {doc}`/deploy/multi_node_deployment/runbook` — full deployment runbook.
