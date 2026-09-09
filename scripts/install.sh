#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="$(command -v uv || true)"

if [[ -z "$uv_bin" ]]; then
  printf 'uv was not found; installing it from astral.sh...\n'
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    printf 'error: installing uv requires curl or wget\n' >&2
    exit 1
  fi

  # The installer does not update PATH in this already-running shell.
  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      uv_bin="$candidate"
      break
    fi
  done
  if [[ -z "$uv_bin" ]]; then
    printf 'error: uv was installed but its executable could not be found\n' >&2
    exit 1
  fi
fi

printf 'Installing Beagle dependencies with %s...\n' "$uv_bin"
cd "$repo_root"
"$uv_bin" sync --extra all