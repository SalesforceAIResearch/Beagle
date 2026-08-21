# Cloud VM providers

Bootstrap scripts are available for the two cloud providers XRLEnv
currently targets:

```{toctree}
:maxdepth: 1

gcp
aws
```

Both scripts assume the user has VM-only access (no admin, no
Terraform, no managed instance groups). Each script is run by hand
on a freshly provisioned VM and:

1. Installs Docker Engine + Python 3.12.
2. Clones the xrlenv repo + `uv sync`'s the runtime dependencies.
3. Drops a systemd unit (`xrlenv-node.service`) that runs
   `xrlenv-node serve` against the configured control-plane host.
4. Starts the unit and verifies the node attaches to the control
   plane.

The two scripts share the same shape — they branch on
`/etc/os-release` to handle Amazon Linux 2023 vs Ubuntu 22.04
package naming differences.

After running the script on each VM, declare them in `nodes.yaml`
({doc}`../inventory`) and run the {doc}`../runbook`.
