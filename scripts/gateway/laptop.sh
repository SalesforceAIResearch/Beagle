#!/usr/bin/env bash
# Run on your LAPTOP (the machine that CAN reach LLM Gateway Express).
#
# Starts the streaming relay + an `ssh -R` reverse tunnel to a cluster node, so the
# node can reach the gateway. Pair it with login-node.sh (run that ON the node).
#
#   ./laptop.sh <ssh-target> [extra ssh options ...]
#     <ssh-target>   an ssh alias, host, or user@ip   (e.g. w1, ubuntu@192.0.2.10)
#     extra args     passed through as ssh options     (e.g. -J bastion -i ~/.ssh/id)
#
# The gateway key list is read from $LLM_GATEWAY_EXPRESS_API_KEY_LIST; this script
# sources the repo-root .env if present so you don't have to export it by hand. On
# macOS it also builds the CA bundle the gateway needs (see the TLS block below), so
# `bash laptop.sh w1` just works.
# Optional: GATEWAY_PORT=<n> pins the laptop relay port (default: 18080, auto-bumps
#           if busy — login-node.sh auto-detects it, so drift is harmless).
#           GATEWAY_CAFILE=<pem> use your own CA bundle; GATEWAY_INSECURE=1 skip verify.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ssh-target> [extra ssh options ...]" >&2
  exit 2
fi
remote="$1"; shift

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

# credentials legitimately come from the environment — source .env if it's there.
if [[ -f "$root/.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$root/.env"; set +a
fi

# each extra arg becomes one --ssh-option (they land before -R, in order).
# exception: --debug is a relay flag (per-request [relay-diag] logging), not an ssh option.
ssh_opts=()
serve_flags=()
for opt in "$@"; do
  if [[ "$opt" == "--debug" ]]; then serve_flags+=(--debug); else ssh_opts+=(--ssh-option "$opt"); fi
done

port_args=()
[[ -n "${GATEWAY_PORT:-}" ]] && port_args=(--local-port "$GATEWAY_PORT")

# upstream TLS. The gateway's cert is signed by a corporate CA that lives in the macOS
# keychains but NOT in Python's trust store → "unable to get local issuer certificate".
# On macOS we build a combined bundle (certifi + system roots + system keychain) and
# point Python + curl at it, so `bash laptop.sh w1` just works. Linux's default store
# already trusts it. Overrides: GATEWAY_CAFILE=<pem> use your own; GATEWAY_INSECURE=1
# skip verification (the ssh tunnel still protects laptop↔node).
tls_args=()
use_cafile() {  # export for python/curl/requests + pass --cafile to the relay
  export SSL_CERT_FILE="$1" REQUESTS_CA_BUNDLE="$1" CURL_CA_BUNDLE="$1"
  tls_args=(--cafile "$1")
}
if [[ "${GATEWAY_INSECURE:-0}" == "1" ]]; then
  tls_args=(--insecure)
elif [[ -n "${GATEWAY_CAFILE:-}" ]]; then
  use_cafile "$GATEWAY_CAFILE"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  bundle="$HOME/.cache/beagle/gateway-ca-bundle.pem"
  mkdir -p "$(dirname "$bundle")"
  certifi_pem="$(python3 -m certifi 2>/dev/null || true)"   # same python3 the relay uses
  roots="$(mktemp)"; keych="$(mktemp)"
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > "$roots" 2>/dev/null || true
  security find-certificate -a -p /Library/Keychains/System.keychain        > "$keych" 2>/dev/null || true
  cat ${certifi_pem:+"$certifi_pem"} "$roots" "$keych" > "$bundle"
  rm -f "$roots" "$keych"
  if [[ -s "$bundle" ]]; then
    use_cafile "$bundle"
    echo "[laptop] built macOS CA bundle (certifi + keychains) → $bundle"
  else
    echo "[laptop] WARNING: CA bundle came out empty; falling back to --insecure" >&2
    tls_args=(--insecure)
  fi
fi

echo "[laptop] relaying to the gateway and tunnelling to '$remote' (Ctrl-C to stop)…"
# ${arr[@]+"${arr[@]}"} — expand safely even when empty (bash 3.2 + set -u, e.g. macOS).
exec python3 "$here/gateway_proxy.py" serve --remote "$remote" \
  ${serve_flags[@]+"${serve_flags[@]}"} \
  ${port_args[@]+"${port_args[@]}"} ${tls_args[@]+"${tls_args[@]}"} ${ssh_opts[@]+"${ssh_opts[@]}"}
