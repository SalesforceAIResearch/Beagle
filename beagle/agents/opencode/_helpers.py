"""Pure, harness-agnostic helpers for the opencode agent.

Everything here is a plain function over strings/config — no container, no harbor,
no benchmark. It builds the bash that drives opencode inside a container and parses
opencode's ``--format json`` event stream back into tokens / turns / errors.
:class:`OpenCodeAgent` composes these with a runtime; the docker, harbor, and pier
harnesses all reuse them unchanged (that reuse is the whole point of the capability
seam).

Deliberately **absent**: any prompt templating. The benchmark supplies the task
instruction (harbor hands the agent its ``instruction.md``); the agent runs it. No
per-benchmark ``.j2`` keyed on a benchmark name — that coupling is the N×M trap this
framework exists to avoid.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

from beagle.agents.core.usage import Usage
from beagle.agents.core.usage import add as usage_add

#: opencode's headless entry inside its checkout — a Bun-executed TypeScript file
#: (``bun <container_path>/packages/opencode/src/index.ts run …``). opencode is a Bun
#: monorepo; running the source (not a prebuilt binary) is what makes an evolved ref
#: take effect, so we execute ``src/index.ts`` directly, exactly like its ``dev`` script.
OPENCODE_ENTRYPOINT = "packages/opencode/src/index.ts"
#: Where opencode's source is cloned + built inside the task container. World-writable so
#: the clone/`bun install` run as the container's (often non-root) agent user.
DEFAULT_CONTAINER_PATH = "/tmp/beagle-opencode"
#: opencode is pinned to this Bun (root ``package.json`` ``packageManager``); install the
#: same version for reproducibility.
DEFAULT_BUN_VERSION = "1.3.14"
#: The **canonical** opencode install — defined here ONCE so configs never repeat it.
#: opencode is a Bun monorepo that shells out to ripgrep (its search/grep tools) and builds
#: native modules (node-pty, tree-sitter grammars) in ``bun install`` via node-gyp, so ensure:
#: ripgrep + git (search/clone), a pinned Bun (the runtime), and a C/Python toolchain. Crucially,
#: node-gyp needs **Python >= 3.8** (its ``gyp_main.py`` uses the walrus operator) but a benchmark
#: task image often puts an old conda Python (3.6) first on PATH — so pin node-gyp to a detected
#: modern interpreter via ``PYTHON``/``npm_config_python``. Best-effort across apt (debian/ubuntu)
#: and apk (alpine); needs root (benchmark task images run as root). Runs in ``container_path`` (the
#: caller ``cd``s there). Override wholesale via ``config.install_cmd``.
DEFAULT_INSTALL_CMD = """\
. /etc/os-release 2>/dev/null || true
if [ "${ID:-}" = "alpine" ]; then
  apk add --no-cache ripgrep git curl unzip bash libgcc libstdc++ python3 make g++ || true
else
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || true
  apt-get install -y --no-install-recommends \
    ripgrep git curl unzip ca-certificates python3 build-essential || true
fi
# Bun (opencode's runtime + package manager). BUN_INSTALL=/usr/local puts `bun` on PATH.
if ! command -v bun >/dev/null 2>&1; then
  export BUN_INSTALL=/usr/local
  curl -fsSL https://bun.sh/install | bash -s "bun-v%(bun_version)s"
fi
export PATH="/usr/local/bin:$PATH"
# Pin node-gyp to a Python >= 3.8 (a task image's conda python3 on PATH may be 3.6, which can't
# parse node-gyp's gyp_main.py). Prefer an explicitly-versioned interpreter, else the OS python3.
GYP_PY=""
for p in python3.13 python3.12 python3.11 python3.10 python3.9 python3.8 /usr/bin/python3 python3; do
  cp=$(command -v "$p" 2>/dev/null) || continue
  if "$cp" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,8) else 1)' 2>/dev/null; then
    GYP_PY="$cp"; break
  fi
done
if [ -n "$GYP_PY" ]; then export PYTHON="$GYP_PY"; export npm_config_python="$GYP_PY"; fi
bun install"""
DEFAULT_TIMEOUT_SEC = 1800.0
#: opencode has no turn/step cap flag (unlike monet's ``--max-turns`` or mini's
#: ``agent.step_limit``); the config knob is accepted for a uniform vocabulary but is a
#: best-effort no-op here. Kept so a config need not special-case opencode.
DEFAULT_MAX_TURNS = 0
#: Harbor bind-mounts ``/logs/agent`` to the host trial dir, so opencode's event stream
#: written there survives an agent-timeout cancel and lands as a native artifact.
DEFAULT_OUTPUT_DIR = "/logs/agent"
#: opencode-idiosyncratic *behavior* flags only — ``--format json`` (the machine stream),
#: ``--model`` / ``--variant`` / ``--dir`` are composed by :func:`build_inner_script`.
#: ``--auto`` auto-approves non-denied permissions (opencode's documented headless mode) so
#: a task doesn't stall on an approval prompt.
DEFAULT_OPENCODE_ARGS: tuple[str, ...] = ("--auto",)
#: opencode provider id we register the gateway under (and prefix ``--model`` with) when the
#: run config names no ``provider``. A config that sets ``provider`` uses that string instead,
#: so ``--model <provider>/<model>`` and the injected provider block agree.
DEFAULT_PROVIDER_ID = "beagle"

# Sentinel fences the opencode event stream inside the combined stdout so a single exec
# round-trip returns both the exit code and the stream. Pure plumbing — never real output.
_STREAM_SENTINEL_START = "<<<BEAGLE_OPENCODE_OUT_START>>>"
_STREAM_SENTINEL_END = "<<<BEAGLE_OPENCODE_OUT_END>>>"

_OPENCODE_STREAM_FILENAME = "opencode.stream.jsonl"
_OPENCODE_STDERR_FILENAME = "opencode.stderr.log"


@dataclass(frozen=True)
class OpenCodeConfig:
    """Fully-resolved, runnable opencode config. Strictly agent-side.

    ``image`` / ``repo_path`` are benchmark concerns supplied at run time, not here.
    """

    model: str
    container_path: str = DEFAULT_CONTAINER_PATH
    entrypoint: str = OPENCODE_ENTRYPOINT
    bun_version: str = DEFAULT_BUN_VERSION
    install_cmd: str = DEFAULT_INSTALL_CMD
    opencode_args: tuple[str, ...] = DEFAULT_OPENCODE_ARGS
    #: opencode provider id the gateway is registered under; also the ``--model`` prefix.
    provider_id: str = DEFAULT_PROVIDER_ID
    #: reasoning effort → opencode's ``--variant`` (provider-specific reasoning level); ``""`` = unset.
    variant: str = ""
    #: ``(container_name, host_name)`` pairs forwarded into the container.
    forward_env: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    max_turns: int = DEFAULT_MAX_TURNS
    timeout: float = DEFAULT_TIMEOUT_SEC
    output_dir: str = DEFAULT_OUTPUT_DIR

    @property
    def opencode_entry(self) -> str:
        """Absolute path to opencode's Bun entry inside the checkout."""
        return f"{self.container_path.rstrip('/')}/{self.entrypoint}"

    @property
    def stream_path(self) -> str:
        """In-container path opencode's json event stream is written to (a native artifact
        under harbor's ``/logs/agent`` mount by default)."""
        return f"{self.output_dir.rstrip('/')}/{_OPENCODE_STREAM_FILENAME}"


def build_install_script(cfg: OpenCodeConfig) -> str:
    """The bash that builds opencode from its checkout (run after the clone).

    ``cd`` is its own guarded statement so a multi-line ``install_cmd`` (the toolchain +
    Bun bootstrap) runs cleanly in ``container_path`` rather than being ``&&``-chained onto
    only the first line.
    """
    install_cmd = cfg.install_cmd
    if "%(bun_version)s" in install_cmd:
        install_cmd = install_cmd % {"bun_version": cfg.bun_version}
    return f"cd {shlex.quote(cfg.container_path)} || exit 1\n{install_cmd}"


def build_provider_config(cfg: OpenCodeConfig, gateway: dict[str, str]) -> str:
    """opencode config JSON declaring the LLM gateway as an OpenAI-compatible provider.

    Injected via opencode's native ``OPENCODE_CONFIG_CONTENT`` env (no file, no workspace
    pollution). ``@ai-sdk/openai-compatible`` POSTs ``baseURL + /chat/completions`` — the
    same OpenAI wire shape :func:`gateway_litellm_kwargs` speaks — so any model the gateway
    routes (gpt or claude) works unchanged. The model is declared under the provider so
    opencode needn't reach models.dev to resolve it.
    """
    # Strip a trailing slash: @ai-sdk/openai-compatible POSTs ``baseURL + "/chat/completions"``, so a
    # gateway URL ending in ``/`` would yield ``//chat/completions`` (many servers 404 the double slash).
    base_url = gateway["api_base"].rstrip("/")
    doc = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            cfg.provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "beagle gateway",
                "options": {
                    "baseURL": base_url,
                    "apiKey": gateway.get("api_key") or "sk-noauth",
                },
                "models": {cfg.model: {"name": cfg.model}},
            }
        },
    }
    return json.dumps(doc)


def build_inner_script(cfg: OpenCodeConfig, *, repo_path: str, shell_preamble: str = "") -> str:
    """Build the bash that runs opencode once inside the container.

    The script: runs ``shell_preamble`` (tool PATH) + puts Bun on PATH, cd's into
    ``repo_path`` and resets the worktree (reproducible diff), pipes the prompt via stdin
    (``$OPENCODE_PROMPT`` — dodges argv limits/quoting), invokes opencode with ``--format
    json`` capturing the event stream + stderr to files under ``cfg.output_dir``, then echoes
    the stream fenced with sentinels so one exec returns rc + stream.
    """
    quoted_args = " ".join(shlex.quote(a) for a in cfg.opencode_args)
    variant_flag = f"--variant {shlex.quote(cfg.variant)} " if cfg.variant else ""
    model_ref = f"{cfg.provider_id}/{cfg.model}"

    out_dir = cfg.output_dir.rstrip("/")
    out_dir_q = shlex.quote(out_dir)
    stream_out = f"{out_dir}/{_OPENCODE_STREAM_FILENAME}"
    stream_err = f"{out_dir}/{_OPENCODE_STDERR_FILENAME}"

    return f"""\
set -uo pipefail
{shell_preamble}
export PATH="/usr/local/bin:$PATH"
mkdir -p {out_dir_q}
cd {shlex.quote(repo_path)}

# Clean worktree so the eventual diff is reproducible. Empty repos / missing config
# shouldn't fail the run.
git config --global user.email "agent@beagle.local" >/dev/null 2>&1 || true
git config --global user.name  "beagle agent"       >/dev/null 2>&1 || true
git reset --hard HEAD >/dev/null 2>&1 || true
git clean -fd         >/dev/null 2>&1 || true

set +e
printf '%s' "$OPENCODE_PROMPT" | bun {shlex.quote(cfg.opencode_entry)} run \\
  --format json \\
  --model {shlex.quote(model_ref)} \\
  {variant_flag}--dir {shlex.quote(repo_path)} \\
  {quoted_args} \\
  > {shlex.quote(stream_out)} 2> {shlex.quote(stream_err)}
OPENCODE_RC=$?
set -e

echo "OPENCODE_RC=$OPENCODE_RC"
echo {shlex.quote(_STREAM_SENTINEL_START)}
cat {shlex.quote(stream_out)} || true
echo
echo {shlex.quote(_STREAM_SENTINEL_END)}

# Replay opencode's stderr on the script's stderr so the runtime captures it.
cat {shlex.quote(stream_err)} >&2 || true
"""


def parse_combined_output(combined: str) -> tuple[int | None, str]:
    """Split the inner script's fenced stdout into ``(opencode_rc, stream)``.

    The first line is ``OPENCODE_RC=<n>`` (opencode's exit code), then the json event
    stream fenced by sentinels. ``opencode_rc`` is ``None`` if the sentinel line never
    arrived (e.g. the script was killed before its epilogue).
    """
    first, _, rest = combined.partition("\n")
    opencode_rc: int | None = None
    if first.startswith("OPENCODE_RC="):
        try:
            opencode_rc = int(first[len("OPENCODE_RC=") :].strip())
        except ValueError:
            opencode_rc = None
        combined = rest
    stream = slice_between(combined, _STREAM_SENTINEL_START, _STREAM_SENTINEL_END)
    return opencode_rc, stream


def slice_between(text: str, start: str, end: str) -> str:
    """Extract the payload between two sentinel lines; ``""`` if either is missing."""
    s = text.find(start)
    if s == -1:
        return ""
    s = text.find("\n", s)
    if s == -1:
        return ""
    e = text.find(end, s + 1)
    if e == -1:
        return ""
    return text[s + 1 : e].rstrip("\n")


def _iter_json_lines(stream: str):
    """Yield each parseable JSON object (dict) in opencode's NDJSON stream, tolerating
    blank / non-JSON lines (stderr leaks, partial writes)."""
    for line in stream.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _step_tokens(part: dict) -> Usage:
    """The cache-split :class:`Usage` on one ``step-finish`` part. opencode reports
    ``tokens: {input, output, reasoning, cache: {read, write}}`` where ``input`` is FRESH
    (cache is counted *in addition*), so the buckets map directly; reasoning folds into output."""
    tok = part.get("tokens")
    if not isinstance(tok, dict):
        return Usage()
    raw_cache = tok.get("cache")
    cache = raw_cache if isinstance(raw_cache, dict) else {}
    return Usage(
        input_uncached=int(tok.get("input") or 0),
        cache_read=int(cache.get("read") or 0),
        cache_write=int(cache.get("write") or 0),
        output=int(tok.get("output") or 0) + int(tok.get("reasoning") or 0),
    )


def parse_opencode_usage(stream: str) -> Usage:
    """Cache-split :class:`Usage` summed across every ``step_finish`` event.

    opencode emits one ``step-finish`` part per LLM step carrying that step's token counts;
    sum them. Returns an all-zero :class:`Usage` when no step-finish exists (json lost / crash)."""
    total = Usage()
    for obj in _iter_json_lines(stream):
        if obj.get("type") == "step_finish" and isinstance(obj.get("part"), dict):
            total = usage_add(total, _step_tokens(obj["part"]))
    return total


def count_opencode_turns(stream: str) -> int:
    """Count ``step_finish`` events — one per completed LLM step. 0 for lost stdout."""
    return sum(1 for o in _iter_json_lines(stream) if o.get("type") == "step_finish")


def last_stream_error(stream: str) -> str | None:
    """Last ``error`` event's message, if any. opencode emits ``{type:"error", error:…}`` on
    a session failure (and exits non-zero); surfacing it marks the run failed."""
    last: str | None = None
    for obj in _iter_json_lines(stream):
        if obj.get("type") == "error":
            err = obj.get("error")
            if isinstance(err, str):
                last = err
            elif isinstance(err, dict):
                last = err.get("message") or err.get("name") or json.dumps(err)
    return last


def summarize_opencode_failure(opencode_rc: int, opencode_stderr: str) -> str:
    """One-line ``error`` summary for a non-zero opencode exit (first stderr line + rc)."""
    first_line = ""
    for raw in (opencode_stderr or "").splitlines():
        s = raw.strip()
        if s:
            first_line = s
            break
    if not first_line:
        return f"opencode exited rc={opencode_rc} (no stderr)"
    if len(first_line) > 200:
        first_line = first_line[:197] + "..."
    return f"opencode exited rc={opencode_rc}: {first_line}"


__all__ = [
    "OpenCodeConfig",
    "OPENCODE_ENTRYPOINT",
    "DEFAULT_CONTAINER_PATH",
    "DEFAULT_BUN_VERSION",
    "DEFAULT_INSTALL_CMD",
    "DEFAULT_OPENCODE_ARGS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TIMEOUT_SEC",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PROVIDER_ID",
    "build_install_script",
    "build_provider_config",
    "build_inner_script",
    "parse_combined_output",
    "slice_between",
    "parse_opencode_usage",
    "count_opencode_turns",
    "last_stream_error",
    "summarize_opencode_failure",
]
