#!/bin/bash
#
#SBATCH --job-name=example-xrlenv-nodes                # Job name
#SBATCH --nodes=2                            # Number of nodes
#SBATCH --nodelist=node-host,node-host
#SBATCH --output=/path/to/xrlenv/slurm_logs/%x_%j.out         # Standard output log
#SBATCH --error=/path/to/xrlenv/slurm_logs/%x_%j.err          # Standard error log
#SBATCH --partition=your-slurm-partition
#SBATCH --account=your-slurm-account

echo "Batch host: $(hostname)"
echo "Batch host IPs: $(hostname -I)"

# Node count + nodelist come from the SBATCH header above (the single
# place to edit when adding workers); srun derives its task count from
# the allocation so it never drifts out of sync with --nodes.
#
# The remote script is fed via a QUOTED heredoc (<<'REMOTE_SCRIPT') captured
# into the bash -lc argument, NOT inlined as bash -lc '...'. This is
# load-bearing: the body below contains single quotes (the `docker ps
# --format '{{.ID}} {{.Names}}'` and `awk '...'` filter). Inside a
# bash -lc '...' single-quoted string those inner quotes terminate the
# string early, and the unquoted space in `{{.ID}} {{.Names}}` then splits
# the argument — silently truncating the remote command to `docker ps
# --format {{.ID}}` and dropping the entire bootstrap (cd/source/sudo) into
# unused positional params (this is exactly what happened on the 2026-06-29
# run: nodes kept stale /opt/xrlenv code). The quoted heredoc keeps all inner
# quotes literal, and the quoted delimiter defers $(hostname)/$XRLENV_*
# expansion to the REMOTE shell — same semantics as the old single-quoted form.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "$(cat <<'REMOTE_SCRIPT'
    echo "Bootstrapping on $(hostname), IPs: $(hostname -I)"
    echo "removing stale containers"
    docker ps --format '{{.ID}} {{.Names}}' | awk '$2 != "efa-node-exporter" {print $1}' | xargs -r docker kill
    echo "done"
    cd /path/to/xrlenv

    # Load the gitignored project-root .env. It carries SECRETS
    # (XRLENV_NODE_TOKEN / DOCKERHUB_USER / DOCKERHUB_TOKEN) AND the topology
    # host/port values (XRLENV_GRPC_HOST + registry hosts/ports) — but the
    # topology values are GENERATED from clusters.yaml by
    # `generate_deployment_script.py --env-cluster`, so do NOT hand-edit them
    # (edit clusters.yaml + regenerate). The `:=` defaults below are only a
    # fallback for a .env that predates a key.
    set -a; source ./.env; set +a

    # Control-plane + registry-mirror topology — the single place to
    # edit addresses. No IP/port is hard-coded in the bootstrap command
    # below: both the control-plane address and the registry-mirror URL
    # are composed from these values. XRLENV_MIRROR_REGISTRY_HOST defaults
    # to the control-plane host (the mirror usually runs there); set it
    # explicitly to point at a dedicated registry node. Override any by
    # setting it in .env or exporting it before sbatch.
    : "${XRLENV_GRPC_HOST:=node-host}"
    : "${XRLENV_GRPC_PORT:=50051}"
    # Registry hosts default to the per-cluster control plane (XRLENV_GRPC_HOST).
    # For the shared-registry model — one mirror + private pair on a single box
    # (e.g. node-host) serving BOTH clusters — set
    # XRLENV_MIRROR_REGISTRY_HOST / XRLENV_PRIVATE_REGISTRY_HOST in .env to that
    # box. The .env values win over these per-cluster defaults, so the node
    # daemon.json points at the shared registry, not the local control plane.
    : "${XRLENV_MIRROR_REGISTRY_HOST:=${XRLENV_GRPC_HOST}}"
    : "${XRLENV_MIRROR_REGISTRY_PORT:=5010}"
    # Private (writable) registry — built benchmark images (e.g. seta-env) live
    # here; usually the same box as the mirror. Port 5011 (mirror uses 5010).
    : "${XRLENV_PRIVATE_REGISTRY_HOST:=${XRLENV_MIRROR_REGISTRY_HOST}}"
    : "${XRLENV_PRIVATE_REGISTRY_PORT:=5011}"

    source .venv/bin/activate
    which python

    # XRLENV_REGISTRY_MIRROR routes docker.io pulls on this node through the
    # pull-through mirror; XRLENV_PRIVATE_REGISTRY adds the private registry to
    # insecure-registries so the node can pull our built images over plain HTTP.
    # Both are merged into /etc/docker/daemon.json by the bootstrap, so a freshly
    # provisioned worker gets them automatically — not just nodes configured by
    # hand. The final positional arg is the stable XRLENV_NODE_ID. Keep it in
    # lock-step with ``xrlenv nodes-from-slurm``s default id template so the
    # generated roster matches the node-agent IDs shown by admin/CLI.
    sudo \
        XRLENV_NODE_TOKEN="$XRLENV_NODE_TOKEN" \
        DOCKERHUB_USER="$DOCKERHUB_USER" \
        DOCKERHUB_TOKEN="$DOCKERHUB_TOKEN" \
        XRLENV_REGISTRY_MIRROR="http://${XRLENV_MIRROR_REGISTRY_HOST}:${XRLENV_MIRROR_REGISTRY_PORT}" \
        XRLENV_PRIVATE_REGISTRY="${XRLENV_PRIVATE_REGISTRY_HOST}:${XRLENV_PRIVATE_REGISTRY_PORT}" \
        bash deploy/bootstrap-aws.sh --hyperpod "${XRLENV_GRPC_HOST}:${XRLENV_GRPC_PORT}" "aws-$(hostname -s)"
REMOTE_SCRIPT
)"

sleep 26280h

# scancel --name=example-xrlenv-nodes && sbatch slurm_scripts/example_xrlenv_node.sh