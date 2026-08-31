# AWS node bootstrap

## One-shot bring-up (recommended)

On the freshly provisioned AWS EC2 instance (Amazon Linux 2023 or Ubuntu 22.04):

```bash
# 1. Get the code onto the VM.
git clone <your-fork-of-xrlenv>.git
cd xrlenv

# 2. Set the two required env vars.
export XRLENV_CONTROL_PLANE="<control-plane-host>:50051"
export XRLENV_NODE_TOKEN="<token from `xrlenv tokens issue node` on the control plane>"

# 3. Run the one-shot bring-up — auto-detects AWS, runs the right
#    bootstrap (branches on /etc/os-release for Amazon Linux 2023 vs
#    Ubuntu 22.04), wires the token into a systemd drop-in, restarts
#    the daemon, and tails the first few seconds of journalctl so you
#    see the connect (or a clear failure) before the script returns.
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

The bootstrap script installs Docker, configures cgroup v2
limits, sets up the `xrlenv-node` systemd unit, and runs the
cloud-metadata block self-test that refuses to register the node if
the metadata IP is reachable from sandboxes. The script branches
automatically on `/etc/os-release` to support both Amazon Linux 2023
and Ubuntu 22.04. Output is colorized: green `OK`, yellow `WARN`,
red `ERROR`. Set `NO_COLOR=1` to suppress the ANSI codes.

## Required pre-reqs

| Env var | Required | What it is |
|---------|----------|------------|
| `XRLENV_CONTROL_PLANE` | yes | `host:port` the node will dial outbound. |
| `XRLENV_NODE_TOKEN` | recommended | Bearer token issued by `xrlenv tokens issue node`. Omit only for trusted loopback development. |
| `XRLENV_NODE_ID` | usually auto | Stable id; auto-filled from IMDSv2 when unset. |
| `XRLENV_REPO` *or* `XRLENV_WHEEL` | one of | Where to install xrlenv from (no public PyPI release yet). Defaults to the directory `bring-up-node.sh` lives in. |
| `XRLENV_FORCE_CLOUD` | no | Override cloud auto-detect (`gcp` or `aws`). Rarely needed. |
| `XRLENV_LOG_TAIL_S` | no | Seconds of journalctl to tail at the end (default 8). |

Issue the node token on the control-plane host before bringing up
the VM:

```bash
xrlenv tokens issue node
```

See {doc}`/developer_guide/cli_reference` — `xrlenv tokens issue` for role and scope details.

## Manual flow (if you need to debug a step)

```bash
# Just install + start the daemon (no token wiring):
sudo -E bash deploy/bootstrap-aws.sh

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
- {doc}`/deploy/multi_node_deployment/cloud_VM_providers/gcp` — GCP bootstrap.
- {doc}`/deploy/multi_node_deployment/inventory` — declare this node in `nodes.yaml`.
- {doc}`/deploy/multi_node_deployment/runbook` — full deployment runbook.
