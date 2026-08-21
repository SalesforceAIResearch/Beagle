set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
echo "REPO_ROOT: ${REPO_ROOT}"
source "${REPO_ROOT}/.env"

EC2_VM_PUBLIC_IP="${1:-${EC2_VM_PUBLIC_IP}}"
echo "EC2_VM_PUBLIC_IP: ${EC2_VM_PUBLIC_IP}"

# error out if the EC2_VM_PUBLIC_IP is not set
if [ -z "${EC2_VM_PUBLIC_IP}" ]; then
    echo "Usage: $0 <EC2_VM_PUBLIC_IP>"
    exit 1
fi

if command -v autossh >/dev/null 2>&1; then
    SSH_CMD=(autossh -M 0)
    export AUTOSSH_GATETIME="${AUTOSSH_GATETIME:-0}"
else
    echo "autossh not found; falling back to plain ssh (no auto-reconnect if the link drops; brew install autossh to get reconnects)" >&2
    SSH_CMD=(ssh)
fi

# run on the local machine
exec "${SSH_CMD[@]}" -N \
       -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=30 \
       -o ServerAliveCountMax=3 \
       -R 50051:127.0.0.1:50051 \
       -R 9090:127.0.0.1:9090   \
       -R 8080:127.0.0.1:8080   \
       ec2-user@"${EC2_VM_PUBLIC_IP}"