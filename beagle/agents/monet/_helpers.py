"""Pure, harness-agnostic helpers for the monet agent.

Everything here is a plain function over strings/config — no container, no harbor,
no benchmark. It builds the bash that drives monet inside a container and parses
monet's stream-json stdout back into tokens / turns / errors. :class:`MonetAgent`
composes these with a runtime; both the docker and harbor harnesses reuse them
unchanged (that reuse is the whole point of the capability seam).

Deliberately **absent**: any prompt templating. The benchmark supplies the task
instruction (harbor hands the agent its ``instruction.md``); the agent runs it. No
per-benchmark ``.j2`` keyed on a benchmark name — that coupling is the N×M trap
this framework exists to avoid.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

from beagle.agents.core.usage import Usage
from beagle.agents.core.usage import add as usage_add

# monet's entry script inside its checkout (``<container_path>/bin/monet.js``).
MONET_BIN_PATH = "bin/monet.js"
#: Where monet's source is cloned + built inside the task container. Defaults to a
#: world-writable path so the clone/install run as the container's (often non-root)
#: agent user without a chown dance.
DEFAULT_CONTAINER_PATH = "/tmp/beagle-monet"
#: The **canonical** monet install — defined here ONCE so configs never repeat it
#: (the ~25-line node bootstrap was otherwise copy-pasted across 100+ configs).
#: monet is a Node agent and TB task images frequently lack node or ship one too old,
#: so ensure node >= 20.5 (nodesource on debian, ``nodejs-current`` on alpine — plain
#: ``nodejs`` is < 20.5 there), repoint any shadowing ``/usr/local/bin`` node from an
#: old base image, then install deps. Needs root (TB task images run as root). Runs in
#: ``container_path`` (the caller ``cd``s there). No ``npm link`` — the inner script
#: invokes monet by absolute path. Override wholesale via ``config.install_cmd``.
DEFAULT_INSTALL_CMD = """\
need_install=1
if command -v node >/dev/null 2>&1; then
  # Need Node >= 20.5 for execa's addAbortListener import.
  if node -e 'const [maj,min]=process.versions.node.split(".").map(Number); process.exit((maj>20||(maj===20&&min>=5))?0:1)'; then
    need_install=0
  fi
fi
if [ "$need_install" = "1" ]; then
  . /etc/os-release 2>/dev/null || true
  if [ "${ID:-}" = "alpine" ]; then
    apk add --no-cache nodejs-current npm || apk add --no-cache nodejs npm
  else
    if ! command -v curl >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y --no-install-recommends curl ca-certificates gnupg
    fi
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
  fi
  # Repoint any older /usr/local/bin node (from a node:16 base) at the fresh install
  # so PATH resolution doesn't shadow it.
  for bin in node npm npx; do
    if [ -x "/usr/bin/$bin" ] && [ -e "/usr/local/bin/$bin" ]; then
      ln -sf "/usr/bin/$bin" "/usr/local/bin/$bin"
    fi
  done
fi
npm ci --omit=dev"""
#: monet-idiosyncratic *behavior* flags only — the gateway (``--provider``) comes
#: from ``ModelSpec.provider``, not here. stream-json so per-turn events reach stdout
#: (one NDJSON line per turn, which the parsers below consume).
#: ``--permissive-auto-approve`` is monet's documented mode for *ephemeral containers
#: / benchmarks*: it auto-approves but keeps the narrow catastrophic-only classifier,
#: so it won't false-positive-prompt and stall a task (unlike
#: ``--dangerously-skip-permissions``, which uses the broad classifier).
DEFAULT_MONET_ARGS: tuple[str, ...] = (
    "--permissive-auto-approve",
    "--no-monet-md",
    "--output-format", "stream-json",
)
DEFAULT_TIMEOUT_SEC = 1800.0
#: monet's own headless fallback is 150 turns; mirror it so an unset cap yields
#: numbers comparable to monet's reference runs. Always paired with
#: ``--strict-max-turns`` so monet doesn't auto-continue past the cap.
DEFAULT_MAX_TURNS = 150
#: Harbor bind-mounts ``/logs/agent`` to the host trial dir, so monet's stream
#: written there survives an agent-timeout cancel and lands as a native artifact.
DEFAULT_OUTPUT_DIR = "/logs/agent"

# Sentinels fence the patch + monet-stdout payloads in one combined stdout so a
# single exec round-trip suffices. Pure plumbing — never appear in real output.
_PATCH_SENTINEL_START = "<<<BEAGLE_PATCH_START>>>"
_PATCH_SENTINEL_END = "<<<BEAGLE_PATCH_END>>>"
_MONET_OUT_SENTINEL_START = "<<<BEAGLE_MONET_OUT_START>>>"
_MONET_OUT_SENTINEL_END = "<<<BEAGLE_MONET_OUT_END>>>"

_MONET_STREAM_FILENAME = "monet.stream.jsonl"
_MONET_STDERR_FILENAME = "monet.stderr.log"


@dataclass(frozen=True)
class MonetConfig:
    """Fully-resolved, runnable monet config. Strictly agent-side.

    ``image`` / ``repo_path`` are benchmark concerns supplied at run time, not here.
    """

    model: str
    container_path: str = DEFAULT_CONTAINER_PATH
    entrypoint: str = MONET_BIN_PATH
    install_cmd: str = DEFAULT_INSTALL_CMD
    monet_args: tuple[str, ...] = DEFAULT_MONET_ARGS
    #: ``(container_name, host_name)`` pairs forwarded into the container.
    forward_env: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    max_turns: int = DEFAULT_MAX_TURNS
    timeout: float = DEFAULT_TIMEOUT_SEC
    max_tokens: int | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR

    @property
    def monet_bin(self) -> str:
        return f"{self.container_path.rstrip('/')}/{self.entrypoint}"

    @property
    def stream_path(self) -> str:
        """In-container path monet's stream-json is written to (a native artifact
        under harbor's ``/logs/agent`` mount by default)."""
        return f"{self.output_dir.rstrip('/')}/{_MONET_STREAM_FILENAME}"


def build_install_script(cfg: MonetConfig) -> str:
    """The bash that builds monet from its checkout (run after the clone).

    ``cd`` is its own guarded statement so a multi-line ``install_cmd`` (the node
    bootstrap) runs cleanly in ``container_path`` rather than being ``&&``-chained
    onto only the first line.
    """
    return f"cd {shlex.quote(cfg.container_path)} || exit 1\n{cfg.install_cmd}"


def build_inner_script(cfg: MonetConfig, *, repo_path: str, shell_preamble: str = "") -> str:
    """Build the bash that runs monet once inside the container.

    The script: runs ``shell_preamble`` (tool PATH), cd's into ``repo_path`` and
    resets the worktree, invokes monet with ``--print="$MONET_PROMPT"`` (prompt via
    env to dodge argv limits/quoting), captures stdout/stderr/patch to files under
    ``cfg.output_dir``, then echoes them fenced with sentinels so one exec suffices.
    """
    quoted_args = " ".join(shlex.quote(a) for a in cfg.monet_args)
    # --provider (the gateway) comes from monet_args (agent.config), not the model block.
    max_tokens_flag = f"--max-tokens {cfg.max_tokens}" if cfg.max_tokens else ""
    # Always pair --max-turns with --strict-max-turns; otherwise monet auto-continues
    # past the cap in headless mode and silently leaks the bound.
    max_turns_flag = f"--max-turns {cfg.max_turns} --strict-max-turns"

    out_dir = cfg.output_dir.rstrip("/")
    out_dir_q = shlex.quote(out_dir)
    monet_out = f"{out_dir}/{_MONET_STREAM_FILENAME}"
    monet_err = f"{out_dir}/{_MONET_STDERR_FILENAME}"
    patch_path = f"{out_dir}/monet.patch.diff"

    return f"""\
set -uo pipefail
{shell_preamble}
mkdir -p {out_dir_q}
cd {shlex.quote(repo_path)}

# Clean worktree so the eventual diff is reproducible. Empty repos / missing
# config shouldn't fail the run.
git config --global user.email "agent@beagle.local" >/dev/null 2>&1 || true
git config --global user.name  "beagle agent"       >/dev/null 2>&1 || true
git reset --hard HEAD >/dev/null 2>&1 || true
git clean -fd         >/dev/null 2>&1 || true

set +e
node {shlex.quote(cfg.monet_bin)} \\
  --print="$MONET_PROMPT" \\
  --cwd {shlex.quote(repo_path)} \\
  --model {shlex.quote(cfg.model)} \\
  {max_tokens_flag + ' ' if max_tokens_flag else ''}{max_turns_flag} {quoted_args} \\
  > {shlex.quote(monet_out)} 2> {shlex.quote(monet_err)}
MONET_RC=$?
set -e

git -C {shlex.quote(repo_path)} add -A >/dev/null 2>&1 || true
# Text diff only — --binary would base85-encode stray binary artifacts and bloat
# the payload; source fixes are never binary.
git -C {shlex.quote(repo_path)} diff --cached > {shlex.quote(patch_path)} 2>/dev/null || true

echo "MONET_RC=$MONET_RC"
echo {shlex.quote(_PATCH_SENTINEL_START)}
cat {shlex.quote(patch_path)} || true
echo
echo {shlex.quote(_PATCH_SENTINEL_END)}
echo {shlex.quote(_MONET_OUT_SENTINEL_START)}
cat {shlex.quote(monet_out)} || true
echo
echo {shlex.quote(_MONET_OUT_SENTINEL_END)}

# Replay monet's stderr on the script's stderr so the runtime captures it.
cat {shlex.quote(monet_err)} >&2 || true
"""


def parse_combined_output(combined: str) -> tuple[int | None, str, str]:
    """Split the inner script's fenced stdout into ``(monet_rc, patch, monet_stdout)``.

    The first line is ``MONET_RC=<n>`` (monet's exit code), then the patch and
    monet-stdout payloads fenced by sentinels. ``monet_rc`` is ``None`` if the
    sentinel line never arrived (e.g. the script was killed before its epilogue).
    """
    first, _, rest = combined.partition("\n")
    monet_rc: int | None = None
    if first.startswith("MONET_RC="):
        try:
            monet_rc = int(first[len("MONET_RC=") :].strip())
        except ValueError:
            monet_rc = None
        combined = rest
    patch = slice_between(combined, _PATCH_SENTINEL_START, _PATCH_SENTINEL_END)
    monet_stdout = slice_between(combined, _MONET_OUT_SENTINEL_START, _MONET_OUT_SENTINEL_END)
    return monet_rc, patch, monet_stdout


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


def _iter_json_lines(monet_stdout: str):
    """Yield each parseable JSON object (dict) in monet's NDJSON stdout, tolerating
    blank / non-JSON lines (stderr leaks, partial writes)."""
    for line in monet_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _monet_event_usage(usage: dict) -> Usage:
    """One monet ``usage`` event → cache-split :class:`Usage`, robust to BOTH provider shapes.

    The cache-token field's meaning differs by provider, and monet passes it through verbatim, so we
    must not assume one universal semantics (assuming one double-counts the other):

    * **OpenAI-compatible** (LLM Gateway Express, ``wireApi='chat_completions'``): ``cacheTokens``
      comes from ``prompt_tokens_details.cached_tokens`` and is a cached **SUBSET** of ``inputTokens``
      (which is the TOTAL prompt). So fresh = ``inputTokens − cacheTokens`` and the billed prompt is
      exactly ``inputTokens`` — never add the cache on top.
    * **Anthropic-shaped**: ``inputTokens`` is FRESH input and ``cacheReadTokens`` /
      ``cacheCreationTokens`` are billed **IN ADDITION**, mapping straight to cache_read / cache_write.

    Shape is detected by which field is present. The invariant ``prompt = input_uncached + cache_read
    + cache_write`` holds either way."""
    inp = int(usage.get("inputTokens") or 0)
    output = int(usage.get("outputTokens") or 0)
    cache_write = int(usage.get("cacheCreationTokens") or 0)     # Anthropic cache-write (additive)
    cache_subset = int(usage.get("cacheTokens") or 0)            # OpenAI: cached SUBSET of inputTokens
    if cache_subset > 0:
        cache_read = min(cache_subset, inp)                     # clamp: a subset can't exceed the total
        return Usage(input_uncached=inp - cache_read, cache_read=cache_read,
                     cache_write=cache_write, output=output)
    # Anthropic-shaped (or no cache hit — then inputTokens is already the whole prompt): read is additive.
    return Usage(input_uncached=inp, cache_read=int(usage.get("cacheReadTokens") or 0),
                 cache_write=cache_write, output=output)


def parse_monet_usage(monet_stdout: str) -> Usage:
    """Cache-split :class:`Usage` summed across every ``usage`` event (:func:`_monet_event_usage`).

    stream-json emits one ``usage`` per turn carrying that turn's own counts (not a running total),
    so we sum — each turn re-sends the growing context, so the billed total is ``sum(inputTokens)`` /
    ``sum(outputTokens)``. Returns an all-zero :class:`Usage` when no ``usage`` line exists."""
    total = Usage()
    for obj in _iter_json_lines(monet_stdout):
        usage = obj.get("usage")
        if isinstance(usage, dict):
            total = usage_add(total, _monet_event_usage(usage))
    return total


def count_monet_turns(monet_stdout: str) -> int:
    """Count ``turn_complete`` events — one per LLM round-trip. 0 for json-mode /
    lost stdout."""
    return sum(1 for o in _iter_json_lines(monet_stdout) if o.get("type") == "turn_complete")


def hit_max_turns(monet_stdout: str) -> bool:
    """True iff monet emitted a ``max_turns_reached`` event (budget exhaustion — treat
    like a wall-clock timeout, not a retry-worthy crash)."""
    return any(o.get("type") == "max_turns_reached" for o in _iter_json_lines(monet_stdout))


def last_stream_error(monet_stdout: str) -> str | None:
    """Last ``stream_error.error`` string, if any. Transient provider failures monet
    retried internally; surfacing it lets a caller mark the run infra-ish."""
    last: str | None = None
    for obj in _iter_json_lines(monet_stdout):
        if obj.get("type") == "stream_error":
            err = obj.get("error")
            if isinstance(err, str):
                last = err
    return last


#: A benign Node process-warning line — ``(node:PID) [CODE] WarningType: …`` (e.g. undici's
#: EnvHttpProxyAgent ExperimentalWarning, printed whenever HTTP(S)_PROXY is set) — plus the
#: ``(Use `node --trace-warnings …`)`` hint. These print BEFORE any real error, so taking the first
#: stderr line as the failure summary would always report the warning and mask the actual cause.
_NODE_WARNING_RE = re.compile(r"^\(node:\d+\)\s")
#: Lines that read like the actual failure — preferred over a bare file path / stack frame.
_ERROR_SIGNAL_RE = re.compile(r"(?i)(error|exception|fatal|econnreset|etimedout|epipe|rate.?limit|\brc=)")


def summarize_monet_failure(monet_rc: int, monet_stderr: str) -> str:
    """One-line ``error`` summary for a non-zero monet exit: ``rc`` + the first MEANINGFUL stderr
    line. Benign Node warnings (undici EnvHttpProxyAgent / ExperimentalWarning + the trace-warnings
    hint) are skipped so they don't mask the real cause; among the rest an error-looking line
    (``Error:``/``Exception``/``ECONNRESET``/…) wins over a bare stack frame. The FULL stderr is
    persisted to ``agent/stderr.log`` for the cases this one line can't capture."""
    lines = [s for s in (raw.strip() for raw in (monet_stderr or "").splitlines()) if s]
    signal = [s for s in lines if not (_NODE_WARNING_RE.match(s) or s.startswith("(Use `node"))]
    pick = next((s for s in signal if _ERROR_SIGNAL_RE.search(s)),
                signal[0] if signal else (lines[0] if lines else ""))
    if not pick:
        return f"monet exited rc={monet_rc} (no stderr)"
    if len(pick) > 200:
        pick = pick[:197] + "..."
    return f"monet exited rc={monet_rc}: {pick}"


__all__ = [
    "MonetConfig",
    "MONET_BIN_PATH",
    "DEFAULT_CONTAINER_PATH",
    "DEFAULT_INSTALL_CMD",
    "DEFAULT_MONET_ARGS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TIMEOUT_SEC",
    "DEFAULT_OUTPUT_DIR",
    "build_install_script",
    "build_inner_script",
    "parse_combined_output",
    "slice_between",
    "parse_monet_usage",
    "count_monet_turns",
    "hit_max_turns",
    "last_stream_error",
    "summarize_monet_failure",
]
