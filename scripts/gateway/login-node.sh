#!/usr/bin/env bash
# Run on the CLUSTER LOGIN NODE that laptop.sh tunnelled to (the ssh target).
#
# The reverse tunnel binds the node's LOOPBACK, which harbor trial *containers*
# (their own 127.0.0.1) can't see. This exposes it on the node's routable IP, so a
# container reaches the gateway at  http://<node-ip>:<forward-port>/ .
#
#   ./login-node.sh [forward-port]        (default 18088)
#
# It auto-detects the reverse-tunnel port (which may have drifted if the laptop's
# preferred port was busy) and the node IP, verifies the tunnel is live, then
# **writes LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL into .env for you** (replace-or-append,
# every other line untouched) so the config always matches the current node — no
# drift, no copy-paste. Finally it runs the forwarder.
#
# Knobs:  GATEWAY_TUNNEL_PORT=<n>  pin the tunnel port (if you run several)
#         ENV_FILE=<path>          which .env to update (default: repo-root .env)
#         GATEWAY_ENV_WRITE=0      don't touch .env, just print the line
set -euo pipefail

fwd="${1:-18088}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
env_file="${ENV_FILE:-$root/.env}"

# replace-or-append `KEY=VAL` in a file in one awk pass, preserving all other lines
# and the file's own permissions/inode (cat > file, not mv).
update_env() {
  local file="$1" key="$2" val="$3" line tmp
  line="${key}=${val}"
  if [[ ! -e "$file" ]]; then
    printf '%s\n' "$line" > "$file"; echo "[login-node] created ${file} with ${key}"; return
  fi
  if [[ ! -w "$file" ]]; then
    echo "[login-node] WARNING: ${file} not writable — set it yourself: ${line}" >&2; return
  fi
  tmp="$(mktemp)"
  awk -v repl="$line" -v k="^${key}=" \
    'BEGIN{d=0} $0~k{if(!d){print repl;d=1}next} {print} END{if(!d)print repl}' "$file" > "$tmp"
  cat "$tmp" > "$file"; rm -f "$tmp"
  echo "[login-node] updated ${file}: ${key}"
}

# 1) find the reverse-tunnel port: a loopback listener in the 1808x range.
tun="${GATEWAY_TUNNEL_PORT:-}"
if [[ -z "$tun" ]]; then
  tun="$(ss -ltn 2>/dev/null | grep -oE '127\.0\.0\.1:1808[0-9]' | grep -oE '[0-9]+$' | head -1 || true)"
fi
if [[ -z "$tun" ]]; then
  echo "[login-node] no reverse tunnel found on 127.0.0.1:1808x." >&2
  echo "             run  scripts/gateway/laptop.sh <this-node>  on your laptop first." >&2
  exit 1
fi

# 2) verify the tunnel/relay is actually live (health GET → any HTTP code but 000).
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${tun}/" || true)"
if [[ -z "$code" || "$code" == "000" ]]; then
  echo "[login-node] tunnel port ${tun} found but not responding — is laptop.sh still connected?" >&2
  exit 1
fi

# 3) the node IP a container will use, and the proxy URL derived from it.
node_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -z "$node_ip" ]] && node_ip="$(hostname)"
url="http://${node_ip}:${fwd}/"

echo "[login-node] tunnel 127.0.0.1:${tun} is live (HTTP ${code}); forwarding 0.0.0.0:${fwd} -> 127.0.0.1:${tun}"

# 4) keep .env in sync (the node IP is the thing that drifts) — this is the point.
if [[ "${GATEWAY_ENV_WRITE:-1}" == "0" ]]; then
  echo "[login-node] set this on the cluster side: LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL=${url}"
else
  update_env "$env_file" "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL" "$url"
fi
echo "[login-node]   -> LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL=${url}"
echo "[login-node] verify:  python3 ${here}/gateway_proxy.py check --url ${url}"
echo "[login-node] running the forwarder — Ctrl-C to stop."

exec python3 "$here/gateway_proxy.py" forward --listen "0.0.0.0:${fwd}" --to "127.0.0.1:${tun}"
