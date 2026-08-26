"""Centralized xrlenv cluster config — the single source of cluster coordinates
and credentials, all sourced from the project-root ``.env``.

This is the ONLY module that reads ``XRLENV_*`` / LLM-key environment variables.
The orchestrator and the smoke tests import from here, so there are no scattered
``os.environ.get(...)`` calls and config can't drift between files.

``<repo>/.env`` is loaded with ``override=True`` at import, so the .env wins over
stale shell exports and a run is reproducible regardless of where it's launched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)

# --- xrlenv import (guarded; it lives in a separate repo/venv) ---------------
# Static checkers won't see xrlenv here — that's expected.
try:
    from xrlenv import Client  # type: ignore[import-not-found]
    from xrlenv.client.dotenv import parse_dotenv  # type: ignore[import-not-found]
except ImportError:
    _repo = os.environ.get("XRLENV_REPO", "")
    if _repo and os.path.isdir(_repo):
        sys.path.insert(0, _repo)
    try:
        from xrlenv import Client  # type: ignore[import-not-found]
        from xrlenv.client.dotenv import parse_dotenv  # type: ignore[import-not-found]
    except ImportError:
        Client = None
        parse_dotenv = None

# --- cluster coordinates (read once, here only) -----------------------------
GRPC_HOST = os.environ.get("XRLENV_GRPC_HOST")
GRPC_PORT = int(os.environ.get("XRLENV_GRPC_PORT", "50051"))
CONSUMER_TOKEN = os.environ.get("XRLENV_CONSUMER_TOKEN")
REGISTRY_HOST = os.environ.get("XRLENV_PRIVATE_REGISTRY_HOST")
REGISTRY_PORT = os.environ.get("XRLENV_PRIVATE_REGISTRY_PORT", "5011")

# The substrate image ref (namespace:tag); the registry host:port is prepended.
# `:dev` is a stable CHANNEL tag (prod uses `:stable`), decoupled from the WAI
# commit: the xrlenv control plane resolves it to the current registry digest
# per acquire, so a rebuild re-pushes `:dev` and this never changes.
IMAGE_REF = "xrlenv-webarena-infinity/substrate:dev"

# LLM credentials forwarded into every container.
LLM_ENV_KEYS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)

# --- Monet agent (--model monet): cloned into the container from GitHub --------
# The substrate image ships Node 20.x + Chromium but NOT Monet; the orchestrator
# clones it into MONET_CONTAINER_REPO per container. The repo is private, so the
# in-container `git clone` authenticates with $GH_TOKEN, expanded INSIDE the
# container (never on the host argv). MONET_GIT_URL is REQUIRED (no default —
# set it in .env, e.g. github.com/<org>/monet_code.git); MONET_GIT_REF is optional.
MONET_CONTAINER_REPO = "/opt/agent"
MONET_GIT_URL = os.environ.get("MONET_GIT_URL", "")
MONET_GIT_REF = os.environ.get("MONET_GIT_REF", "main")

# Forwarded into the container only for --model monet: gateway creds + Monet knobs.
# GH_TOKEN rides along solely for the clone step. Empty values are dropped.
MONET_ENV_KEYS = (
    "LLM_GATEWAY_EXPRESS_API_KEY",
    "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL",
    "MONET_PROVIDER",
    "MONET_MODEL",
    "MONET_MAX_IMAGES",
    "MONET_CHROME_PATH",
    "GH_TOKEN",
)


def default_image() -> str | None:
    """Full image ref from the registry in .env, or None if the registry is unset."""
    if not REGISTRY_HOST:
        return None
    return f"{REGISTRY_HOST}:{REGISTRY_PORT}/{IMAGE_REF}"


def llm_env(extra_env_file: str | None = None) -> dict:
    """LLM keys present in the environment (from .env), plus an optional extra
    .env file. This is the dict forwarded into each container."""
    env = {k: os.environ[k] for k in LLM_ENV_KEYS if os.environ.get(k)}
    if extra_env_file:
        if parse_dotenv is None:
            print("Warning: xrlenv not importable; extra --env-file ignored.")
        else:
            env.update(parse_dotenv(extra_env_file))
    return env


def monet_env() -> dict:
    """Extra container env for --model monet: gateway creds + Monet config, plus
    MONET_REPO pointing at the in-container clone path. GH_TOKEN (if set) rides
    along for the clone. Missing keys are omitted — monet_preflight() validates the
    required ones up front, before any container is acquired."""
    env = {k: os.environ[k] for k in MONET_ENV_KEYS if os.environ.get(k)}
    env["MONET_REPO"] = MONET_CONTAINER_REPO
    return env


def monet_preflight() -> list[str]:
    """Hard prerequisites for --model monet. Returns a list of human-readable
    missing-requirement messages (empty list = good to go)."""
    missing = []
    if not MONET_GIT_URL:
        missing.append(
            "MONET_GIT_URL is unset (the Monet agent repo to clone, "
            "e.g. github.com/<org>/monet_code.git; set it in .env)."
        )
    if not os.environ.get("LLM_GATEWAY_EXPRESS_API_KEY"):
        missing.append("LLM_GATEWAY_EXPRESS_API_KEY is unset (Express gateway auth).")
    if not os.environ.get("LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL"):
        missing.append(
            "LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL is unset "
            "(should be http://<node-ip>:18088/, reachable from containers)."
        )
    if not os.environ.get("GH_TOKEN"):
        missing.append("GH_TOKEN is unset (needed to clone the private monet_code repo).")
    return missing


def make_client(host: str | None = None, port: int | None = None):
    """Build an xrlenv Client from .env coordinates (host/port optional overrides)."""
    if Client is None:
        raise RuntimeError("xrlenv is not importable. Set XRLENV_REPO or pip install it.")
    resolved_host = host or GRPC_HOST
    if not resolved_host:
        raise RuntimeError(
            "No control-plane host: set XRLENV_GRPC_HOST in .env or pass --xrlenv-host."
        )
    return Client.grpc(host=resolved_host, port=port or GRPC_PORT, token=CONSUMER_TOKEN)
