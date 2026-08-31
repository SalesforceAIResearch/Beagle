"""Framework-free helpers shared by the installed-agent shims (harbor + pier).

These carry **no** harbor/pier import, so either shim can use them without pulling in the other
framework — ``beagle[terminal-bench]`` (harbor) and ``beagle[deep-swe]`` (pier) install
independently. The shims themselves (:mod:`beagle.benchmarks._harbor_agent`,
:mod:`beagle.benchmarks._pier_agent`) only add the framework-specific base class.
"""

from __future__ import annotations

from typing import Any

# Distro-agnostic git bootstrap: harbor/pier task images are minimal and often lack git (and
# ca-certificates), which the agent needs to clone its own source. Runs as root in install().
# Retries the network-touching steps — a single CDN blip shouldn't kill a long trial before the
# agent ever runs.
_GIT_BOOTSTRAP = r"""
set -e
if ! command -v git >/dev/null 2>&1; then
  attempt=1
  while :; do
    if command -v apk >/dev/null 2>&1; then
      apk add --no-cache git ca-certificates && break
    elif command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y --no-install-recommends git ca-certificates && break
    elif command -v microdnf >/dev/null 2>&1; then
      microdnf install -y git ca-certificates && break
    elif command -v dnf >/dev/null 2>&1; then
      dnf install -y git ca-certificates && break
    elif command -v yum >/dev/null 2>&1; then
      yum install -y git ca-certificates && break
    elif command -v zypper >/dev/null 2>&1; then
      zypper --non-interactive install git ca-certificates && break
    elif command -v pacman >/dev/null 2>&1; then
      pacman -Sy --noconfirm git ca-certificates && break
    else
      echo "no supported package manager (apk/apt/dnf/yum/zypper/pacman)" >&2; exit 1
    fi
    if [ "$attempt" -ge 3 ]; then echo "git bootstrap failed after 3 attempts" >&2; exit 1; fi
    sleep $((attempt * 5)); attempt=$((attempt + 1))
  done
fi
"""


def _rebuild_agent(identity: dict[str, Any]):
    """Reconstruct the beagle agent from its serializable identity descriptor."""
    from beagle.agents.core.registry import build
    from beagle.agents.core.spec import AgentSource, AgentSpec, ModelSpec

    src = identity.get("source")
    spec = AgentSpec(
        name=identity["agent"],
        model=ModelSpec(name=identity["model"]) if identity.get("model") else None,
        config=dict(identity.get("config") or {}),
        source=AgentSource(**src) if src else None,
    )
    return build(spec)


__all__ = ["_GIT_BOOTSTRAP", "_rebuild_agent"]
