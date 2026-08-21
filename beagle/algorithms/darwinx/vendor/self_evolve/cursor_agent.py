"""Wrapper around the `cursor-agent` CLI.

Validated against `cursor-agent` 2026.05.09-0afadcc:
    cursor-agent -p --force \\
        --output-format stream-json \\
        --workspace <wt_dir> \\
        --model gpt-5.5 \\
        [--mode plan]                            # only on the analyze call
        "<prompt>"

Stream-json events seen during testing:
  - system/init        (session_id, model, cwd, permissionMode)
  - user
  - thinking/delta
  - tool_call/started, tool_call/completed   (full args + diff in completed)
  - assistant          (final text)
  - result             (duration_ms, usage.{input/output/cacheRead/cacheWrite}Tokens,
                        result = final assistant text)

Plan-mode caveat: under `--mode plan`, Cursor offers a built-in `CreatePlan`
tool that the model often calls *instead of* emitting the plan body in a final
`assistant` text message. The plan body then lives in
`tool_call.createPlanToolCall.args.plan`, not in any `assistant` event. The
wrapper therefore extracts the plan body from that tool-call payload and
prefers it over the assistant/result text when it looks like a real plan
(see `_is_trivial_plan_body`). Falls back to assistant text and finally to the
`result` event's `result` field — both of which are also captured.
"""

from __future__ import annotations

import json
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TIMEOUT_S = 30 * 60  # 30 min wall clock per call
_RESULT_EXIT_GRACE_S = 3.0

# Minimum length (chars) below which a CreatePlan body is considered a stub.
# Real plans rendered from `prompts/analyze.md` always exceed this comfortably.
_MIN_PLAN_BODY_CHARS = 200

# Substrings whose presence in a CreatePlan body marks it as a stub that points
# the reader at a separate assistant message (which the model then often forgets
# to emit). Matched case-insensitively against the body's stripped text. The
# list is intentionally short and specific: anything ambiguous (e.g. "see
# above") would risk false-positives on a real long plan that happens to use
# the phrase in normal prose. The 200-char minimum length above already screens
# out short placeholders on its own; these markers exist purely to catch verbose
# placeholders that pad themselves over the length floor.
_PLAN_STUB_MARKERS = (
    "see assistant message",
    "see the assistant message",
    "see assistant text",
    "final plan is rendered",
    "in the final assistant message",
)


def _is_trivial_plan_body(body: str | None) -> bool:
    """Return True for an empty / placeholder CreatePlan body.

    We only accept the CreatePlan plan field as the canonical plan when it
    looks like a real markdown plan: long enough, and without any of the known
    "see other channel" stub markers.
    """
    if not body:
        return True
    s = body.strip()
    if len(s) < _MIN_PLAN_BODY_CHARS:
        return True
    low = s.lower()
    return any(marker in low for marker in _PLAN_STUB_MARKERS)


def _extract_plan_body(evt: dict[str, Any]) -> str | None:
    """If `evt` is a `tool_call` event for `createPlanToolCall`, return its
    `args.plan` string (which may be empty). Otherwise return None.
    """
    if evt.get("type") != "tool_call":
        return None
    tc = evt.get("tool_call") or {}
    cp = tc.get("createPlanToolCall") if isinstance(tc, dict) else None
    if not isinstance(cp, dict):
        return None
    args = cp.get("args") or {}
    if not isinstance(args, dict):
        return None
    plan = args.get("plan")
    return plan if isinstance(plan, str) else None


@dataclass
class _StreamState:
    """Mutable accumulator updated as stream-json events arrive.

    Used by both `run()` (live subprocess events) and `parse_log()` (events
    re-read from disk) so the two callers can never drift in how they classify
    or count events. `final_text()` collapses the accumulated state into the
    canonical "best plan body" via `_select_final_text`.
    """

    assistant_text: str = ""
    result_text: str = ""
    plan_body_completed: str | None = None
    plan_body_started: str | None = None
    duration_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    is_error_flag: bool = False
    saw_result_event: bool = False

    def final_text(self) -> str:
        return _select_final_text(
            plan_body_completed=self.plan_body_completed,
            plan_body_started=self.plan_body_started,
            assistant_text=self.assistant_text,
            result_text=self.result_text,
        )


def _consume_event(state: _StreamState, evt: dict[str, Any]) -> None:
    """Apply one parsed stream-json event to `state`.

    The shared classifier for both `run()` and `parse_log()`. Unknown event
    types are silently ignored — we only look for the four shapes documented in
    the module docstring.
    """
    etype = evt.get("type")
    if etype == "tool_call":
        subtype = evt.get("subtype")
        if subtype == "completed":
            state.tool_calls.append(evt.get("tool_call") or {})
            state.tool_call_count += 1
        plan_body = _extract_plan_body(evt)
        if plan_body is not None:
            if subtype == "completed":
                state.plan_body_completed = plan_body
            elif subtype == "started":
                state.plan_body_started = plan_body
    elif etype == "assistant":
        msg = evt.get("message") or {}
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "text":
                # Last non-empty assistant text wins (preserves behavior:
                # an empty `text` field never overwrites a previous one).
                state.assistant_text = part.get("text") or state.assistant_text
    elif etype == "result":
        state.saw_result_event = True
        if evt.get("result") is not None:
            state.result_text = evt.get("result") or state.result_text
        if evt.get("duration_ms") is not None:
            state.duration_ms = int(evt["duration_ms"])
        u = evt.get("usage") or {}
        if isinstance(u, dict):
            state.usage = {k: int(v) for k, v in u.items() if isinstance(v, int)}
        if evt.get("is_error"):
            state.is_error_flag = True


@dataclass
class CursorResult:
    """What cursor_agent.run() hands back."""

    text: str                              # final assistant message
    exit_code: int
    duration_ms: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_log_path: Path | None = None
    error: str | None = None               # populated when something went wrong


# ─── Template rendering ──────────────────────────────────────────────────


def render_prompt(template_path: Path, context: dict[str, Any]) -> str:
    """Render a Jinja2 template file with context. Strict undefined → loud failure."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    tpl = env.get_template(template_path.name)
    return tpl.render(**context)


# ─── Subprocess invocation ───────────────────────────────────────────────


# Transient cursor-agent failures worth retrying. The dominant one observed in
# the a015afdb campaign: under gateway/account rate-limiting the meta-agent's
# model resolution flaps and the CLI exits 1 with "AI Model Not Found: gpt-5.5"
# (preceded by hundreds of 429s in the stream). These are NOT real model-config
# errors — a backoff+retry rides them out. ~63% of analyze stages were dying
# this way with no retry.
_RETRYABLE_RE = re.compile(
    r"Model name is not valid|AI Model Not Found|rate.?limit|ratelimit|"
    r"\b429\b|too many requests|quota|overloaded|temporarily unavailable|"
    r"timed out|ECONNRESET|ETIMEDOUT|service unavailable|\b50[234]\b",
    re.IGNORECASE,
)


def _looks_retryable(res: "CursorResult") -> bool:
    blob = f"{res.error or ''}\n{res.text or ''}"
    if _RETRYABLE_RE.search(blob):
        return True
    try:
        tail = res.raw_log_path.read_text(errors="ignore")[-8000:]
    except Exception:  # noqa: BLE001
        tail = ""
    return bool(_RETRYABLE_RE.search(tail))


def run(
    prompt: str,
    *,
    workspace: Path,
    log_path: Path,
    model: str = DEFAULT_MODEL,
    plan_mode: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    extra_args: list[str] | None = None,
    cursor_bin: str | None = None,
    max_attempts: int = 4,
    backoff_base_s: float = 8.0,
) -> CursorResult:
    """Resilient wrapper around ``_run_once``.

    Retries on TRANSIENT cursor-agent failures (rate-limit 429s that surface as
    ``AI Model Not Found: "gpt-5.5"``, quota/overload, transient network) with
    exponential backoff + jitter. Successes and non-transient failures (incl.
    timeouts, code 127 binary-missing) return immediately so we never double a
    legitimate long run. Tune via ``CURSOR_AGENT_MAX_ATTEMPTS`` /
    ``CURSOR_AGENT_BACKOFF_BASE_S`` env. This recovers the proposer-crash loop
    that left the a015afdb campaign with ~63% ``analyze_failed`` iterations.
    """
    try:
        max_attempts = max(1, int(os.environ.get("CURSOR_AGENT_MAX_ATTEMPTS", max_attempts)))
    except (TypeError, ValueError):
        pass
    try:
        backoff_base_s = float(os.environ.get("CURSOR_AGENT_BACKOFF_BASE_S", backoff_base_s))
    except (TypeError, ValueError):
        pass

    res: CursorResult | None = None
    for attempt in range(1, max_attempts + 1):
        res = _run_once(
            prompt, workspace=workspace, log_path=log_path, model=model,
            plan_mode=plan_mode, timeout_s=timeout_s, extra_args=extra_args,
            cursor_bin=cursor_bin,
        )
        # Success, timeout (124), or binary-missing (127) → don't retry.
        if res.exit_code in (0, 124, 127):
            return res
        if attempt >= max_attempts or not _looks_retryable(res):
            return res
        # Preserve the failed attempt's log for debugging before the next
        # attempt overwrites it.
        try:
            log_path.replace(log_path.with_name(
                f"{log_path.stem}.attempt{attempt}{log_path.suffix}"))
        except Exception:  # noqa: BLE001
            pass
        delay = backoff_base_s * (2 ** (attempt - 1)) + random.uniform(0, backoff_base_s)
        time.sleep(delay)
    return res  # type: ignore[return-value]


def _run_once(
    prompt: str,
    *,
    workspace: Path,
    log_path: Path,
    model: str = DEFAULT_MODEL,
    plan_mode: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    extra_args: list[str] | None = None,
    cursor_bin: str | None = None,
) -> CursorResult:
    """Invoke cursor-agent in non-interactive print mode and stream output to log_path.

    Args:
        prompt: full prompt text (typically rendered from a template).
        workspace: directory cursor-agent runs in (its --workspace).
        log_path: where to tee the stream-json output (one JSON object per line).
        model: e.g. "gpt-5.5" (the bare family slug — reasoning level
            and context length come from the user's account-level
            preference, set in the Cursor IDE/terminal).
        plan_mode: True → pass `--mode plan` (read-only). Use for analyze.
        timeout_s: wall-clock cap; subprocess is killed if it hits this.
        extra_args: extra CLI flags appended to cursor-agent (rarely needed).
        cursor_bin: override path to the cursor-agent executable.

    Returns a CursorResult. On timeout / non-zero exit / parse failure, the
    `error` field is populated; the caller decides whether to abort or
    continue.
    """
    cursor_bin = cursor_bin or _resolve_cursor_bin()
    if cursor_bin is None:
        return CursorResult(
            text="",
            exit_code=127,
            error="cursor-agent binary not found on PATH",
            raw_log_path=log_path,
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    cmd = [
        cursor_bin, "-p", "--force",
        "--output-format", "stream-json",
        "--workspace", str(workspace),
        "--model", model,
    ]
    if plan_mode:
        cmd += ["--mode", "plan"]
    if extra_args:
        cmd += extra_args
    # Defensive last-line guard: a command-line argument CANNOT contain a NUL
    # byte — subprocess rejects it at fork/exec with `ValueError: embedded null
    # byte`, which crashes the whole pipeline (this is exactly what killed
    # ~40% of the express3 run via a binary task-verifier file injected into
    # the analyze prompt). The prompt is assembled from many sources (failing
    # -trial logs, diffs, task files, sibling traces), any of which may carry
    # binary/NUL bytes, so strip them here so NO prompt path can crash the
    # proposer. Upstream sources should also avoid injecting binary; this is
    # the boundary backstop.
    if "\x00" in prompt:
        prompt = prompt.replace("\x00", "")
    # Pass the prompt via STDIN, not argv. A single argv entry is capped at
    # MAX_ARG_STRLEN (128KB on Linux); large WAI proposer prompts (300-task
    # PRESERVE/EXTEND lists + trajectories) exceed it → OSError [Errno 7]
    # "Argument list too long" at fork/exec, which crashed the supervisor.
    # cursor-agent reads the prompt from stdin when no positional prompt is given
    # (verified: `echo <prompt> | cursor-agent -p --force ...` responds). The
    # prompt is written to proc.stdin in a thread below (see stdin=PIPE).

    env = {**os.environ}
    # Force unbuffered stdout from any child processes the agent spawns.
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Per-call managed HOME so background tasks spawned by Cursor read a
    # `cli-config.json` we control instead of the user's stray IDE state.
    # See `_materialize_cursor_home` for the leak this guards against.
    managed_home = Path(tempfile.mkdtemp(prefix="monet_cursor_home_"))
    try:
        _materialize_cursor_home(model, managed_home)
        env["HOME"] = str(managed_home)

        state = _StreamState()
        error: str | None = None
        exit_code = -1

        def consume_line(line: str, logf) -> None:
            logf.write(line)
            logf.flush()
            line_s = line.rstrip("\n")
            if not line_s:
                return
            try:
                evt = json.loads(line_s)
            except json.JSONDecodeError:
                # Some lines may be plain text (errors, banners).
                return
            _consume_event(state, evt)

        try:
            with log_path.open("w") as logf, subprocess.Popen(
                cmd,
                cwd=str(workspace),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            ) as proc:
                assert proc.stdout is not None
                # Feed the prompt via stdin in a writer thread: large prompts can
                # exceed the ~64KB pipe buffer, and a blocking write before we start
                # draining stdout would deadlock. The thread writes + closes stdin.
                def _write_stdin() -> None:
                    try:
                        if proc.stdin is not None:
                            proc.stdin.write(prompt)
                            proc.stdin.close()
                    except (OSError, ValueError, BrokenPipeError):
                        return
                threading.Thread(target=_write_stdin, daemon=True).start()
                lines: "queue.Queue[str]" = queue.Queue()

                def read_stdout() -> None:
                    try:
                        assert proc.stdout is not None
                        for line in proc.stdout:
                            lines.put(line)
                    except (OSError, ValueError):
                        # The main thread may close stdout while cleaning up a
                        # leaked descendant that inherited the pipe.
                        return

                reader = threading.Thread(target=read_stdout, daemon=True)
                reader.start()
                deadline = time.monotonic() + timeout_s
                result_seen_at: float | None = None

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        error = f"cursor-agent timed out after {timeout_s}s"
                        exit_code = 124  # GNU coreutils convention for timeout
                        _terminate_process_group(proc)
                        break

                    try:
                        line = lines.get(timeout=min(0.2, remaining))
                    except queue.Empty:
                        line = None
                    if line is not None:
                        consume_line(line, logf)
                        if state.saw_result_event and result_seen_at is None:
                            result_seen_at = time.monotonic()
                        continue

                    if proc.poll() is not None:
                        exit_code = int(proc.returncode)
                        # cursor-agent can exit after emitting a final result while
                        # tool-spawned descendants keep stdout inherited. Kill the
                        # private process group so the reader can observe EOF.
                        _terminate_process_group(proc)
                        break

                    if (
                        result_seen_at is not None
                        and time.monotonic() - result_seen_at >= _RESULT_EXIT_GRACE_S
                    ):
                        # stream-json `result` is the terminal event. If the CLI is
                        # still alive after a short grace period, a tool-spawned
                        # descendant is usually keeping the session open.
                        exit_code = 0
                        _terminate_process_group(proc)
                        break

                reader.join(timeout=2.0)
                if reader.is_alive():
                    try:
                        proc.stdout.close()
                    except OSError:
                        pass
                    reader.join(timeout=1.0)

                while True:
                    try:
                        consume_line(lines.get_nowait(), logf)
                    except queue.Empty:
                        break
        except FileNotFoundError as e:
            return CursorResult(
                text="",
                exit_code=127,
                error=f"failed to spawn cursor-agent: {e}",
                raw_log_path=log_path,
            )

        if state.is_error_flag and not error:
            error = "cursor-agent reported is_error=true"
        if exit_code != 0 and not error:
            error = f"cursor-agent exited with code {exit_code}"

        return CursorResult(
            text=state.final_text(),
            exit_code=exit_code,
            duration_ms=state.duration_ms,
            usage=state.usage,
            tool_calls=state.tool_calls,
            raw_log_path=log_path,
            error=error,
        )
    finally:
        shutil.rmtree(managed_home, ignore_errors=True)


def probe_init_model(
    model: str,
    *,
    workspace: Path,
    plan_mode: bool = False,
    timeout_s: int = 15,
    cursor_bin: str | None = None,
) -> str:
    """Return the model display string from cursor-agent's system/init event.

    `cursor-agent models` can advertise a context window that differs from the
    runtime selected by `--model`. This probe launches the CLI just far enough
    to read the init event, then terminates the process before doing real work.
    """
    cursor_bin = cursor_bin or _resolve_cursor_bin()
    if cursor_bin is None:
        raise RuntimeError("cursor-agent binary not found on PATH")

    workspace.mkdir(parents=True, exist_ok=True)
    cmd = [
        cursor_bin, "-p", "--force",
        "--output-format", "stream-json",
        "--workspace", str(workspace),
        "--model", model,
    ]
    if plan_mode:
        cmd += ["--mode", "plan"]
    cmd.append("Reply with ok only.")

    env = {**os.environ}
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc: subprocess.Popen[str] | None = None
    reader: threading.Thread | None = None
    lines: "queue.Queue[str]" = queue.Queue()
    first_plain_line: str | None = None

    # Same managed-HOME shim as `run()` — cursor-agent's init probe also
    # reads `cli-config.json`, and the validator that calls this probe runs
    # at supervisor startup before any worker. Without isolation here, a
    # mis-pinned host config can fail the preflight in confusing ways.
    managed_home = Path(tempfile.mkdtemp(prefix="monet_cursor_home_"))
    try:
        _materialize_cursor_home(model, managed_home)
        env["HOME"] = str(managed_home)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert proc.stdout is not None

            def read_stdout() -> None:
                try:
                    assert proc is not None and proc.stdout is not None
                    for line in proc.stdout:
                        lines.put(line)
                except (OSError, ValueError):
                    return

            reader = threading.Thread(target=read_stdout, daemon=True)
            reader.start()
            deadline = time.monotonic() + timeout_s

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"cursor-agent did not emit system init within {timeout_s}s"
                    )

                try:
                    line = lines.get(timeout=min(0.2, remaining))
                except queue.Empty:
                    if proc.poll() is not None:
                        break
                    continue

                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    evt = json.loads(line_s)
                except json.JSONDecodeError:
                    first_plain_line = first_plain_line or line_s
                    continue
                if evt.get("type") == "system" and evt.get("subtype") == "init":
                    actual = evt.get("model")
                    if isinstance(actual, str) and actual:
                        return actual
                    raise RuntimeError("cursor-agent system init omitted model")
                if evt.get("is_error"):
                    raise RuntimeError(line_s)

            detail = f": {first_plain_line}" if first_plain_line else ""
            raise RuntimeError(
                f"cursor-agent exited before system init (code {proc.returncode}){detail}"
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"failed to spawn cursor-agent: {e}") from e
        finally:
            if proc is not None and proc.poll() is None:
                _terminate_process_group(proc, grace_s=1.0)
            if reader is not None and reader.is_alive():
                if proc is not None and proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except OSError:
                        pass
                reader.join(timeout=1.0)
    finally:
        shutil.rmtree(managed_home, ignore_errors=True)


def _terminate_process_group(proc: subprocess.Popen, *, grace_s: float = 5.0) -> None:
    """Terminate the cursor-agent process group and any inherited children."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        proc.terminate()

    if proc.poll() is None:
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass

        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass


def _select_final_text(
    *,
    plan_body_completed: str | None,
    plan_body_started: str | None,
    assistant_text: str,
    result_text: str,
) -> str:
    """Pick the most plan-like text out of all the channels Cursor may use.

    Priority:
      1. Non-trivial CreatePlan body from a `tool_call.completed` event.
      2. Non-trivial CreatePlan body from a `tool_call.started` event
         (fallback when the session ended before a completed event was emitted).
      3. Last `assistant` text message.
      4. The `result` event's `result` field (joined assistant fragments).
    """
    for body in (plan_body_completed, plan_body_started):
        if not _is_trivial_plan_body(body):
            assert body is not None  # narrow for the type checker
            return body
    if assistant_text:
        return assistant_text
    return result_text


def parse_log(log_path: Path) -> dict[str, Any]:
    """Re-parse a stream-json log file (used by report.py to compute total cost).

    Returns {'usage': {...}, 'duration_ms': int, 'tool_call_count': int,
             'final_text': str | None}.

    `final_text` is selected from the same channel priority used by `run()`
    (they share `_consume_event` / `_StreamState`): a non-trivial CreatePlan
    body wins over assistant text, which wins over the `result` event's
    `result` field. An empty/missing final text is normalized to `None`.
    """
    out: dict[str, Any] = {
        "usage": {},
        "duration_ms": 0,
        "tool_call_count": 0,
        "final_text": None,
    }
    if not log_path.is_file():
        return out
    state = _StreamState()
    try:
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            _consume_event(state, evt)
    except OSError:
        pass

    out["usage"] = state.usage
    out["duration_ms"] = state.duration_ms
    out["tool_call_count"] = state.tool_call_count
    out["final_text"] = state.final_text() or None
    return out


def _resolve_cursor_bin() -> str | None:
    """Find cursor-agent. Tries PATH then the standard ~/.local/bin install dir."""
    found = shutil.which("cursor-agent")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / "cursor-agent",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _materialize_cursor_home(model: str, parent: Path) -> Path:
    """Build a managed `HOME` that isolates writes to `cli-config.json`.

    The cursor-agent foreground session uses the `--model` flag we pass
    at spawn time. Cursor's *background* task subsystem (the dispatcher
    behind `task_notification` events) instead reads the user's
    account-level preferred model from server state — set via the
    Cursor IDE's model picker, not via any local file.

    Given that, the managed HOME has exactly one job: never let an
    isolated cursor-agent invocation write to the user's real
    `~/.cursor/cli-config.json`. A stray local mutation could be
    interpreted by Cursor as a "user changed their preferred model"
    signal and synced to the server, silently overriding the model the
    user selected in their IDE. That's the leak we now have to prevent.

    The shim therefore:

      * symlinks every `~/.cursor/` entry (so `chats/`, `plans/`,
        `managed/`, statsig caches, etc. still resolve and any
        legitimate writes that ARE meant to share with the IDE go
        through unchanged), EXCEPT `cli-config.json`;
      * writes a fresh `cli-config.json` that copies the user's host
        config but STRIPS `model`, `selectedModel`, and
        `hasChangedDefaultModel`. Without those fields cursor-agent has
        nothing local to assert about the user's preferred model and
        cannot push a stale value to Cursor's backend. The `--model`
        flag passed at spawn time is authoritative for the foreground
        session;
      * also symlinks `~/.config/cursor/` since cursor-agent stores its
        login token at `~/.config/cursor/auth.json` (separately from
        `~/.cursor/`). Without this bridge every isolated invocation
        fails with "Authentication required".

    The `model` arg is intentionally still required for caller clarity
    even though we no longer write it into the managed config — future
    additions (e.g. enforcing that `--model` isn't accidentally elided
    in subprocesses we spawn elsewhere) may need it.
    """
    real_home = Path.home()
    real_cursor = real_home / ".cursor"
    managed_cursor = parent / ".cursor"
    managed_cursor.mkdir(parents=True, exist_ok=True)

    if real_cursor.is_dir():
        for entry in real_cursor.iterdir():
            if entry.name == "cli-config.json":
                continue
            target = managed_cursor / entry.name
            if target.exists() or target.is_symlink():
                continue
            try:
                target.symlink_to(entry)
            except OSError:
                # Fall back to a copy on filesystems that disallow symlinks
                # (rare, but cheap to handle: cursor-agent only reads here).
                if entry.is_dir():
                    shutil.copytree(entry, target, symlinks=True)
                else:
                    shutil.copy2(entry, target)

    # Bridge ~/.config/cursor/ (auth.json lives here, not in ~/.cursor/).
    real_config_cursor = real_home / ".config" / "cursor"
    if real_config_cursor.is_dir():
        managed_config = parent / ".config"
        managed_config.mkdir(parents=True, exist_ok=True)
        managed_config_cursor = managed_config / "cursor"
        if not (managed_config_cursor.exists() or managed_config_cursor.is_symlink()):
            try:
                managed_config_cursor.symlink_to(real_config_cursor)
            except OSError:
                shutil.copytree(real_config_cursor, managed_config_cursor, symlinks=True)

    cfg: dict[str, Any] = {}
    real_cfg_path = real_cursor / "cli-config.json"
    if real_cfg_path.is_file():
        try:
            parsed = json.loads(real_cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            cfg = dict(parsed)

    # The single safety-critical step: strip every field that asserts a
    # model preference. Anything cursor-agent might interpret as
    # "current user-selected model" must not appear here, or a
    # background sync to Cursor's backend could clobber the model the
    # user chose in their IDE.
    for key in ("model", "selectedModel", "hasChangedDefaultModel"):
        cfg.pop(key, None)
    # `model` is unused below but we keep it in the signature so callers
    # remain explicit about which slug self-evolve will pass via --model.
    _ = model
    cfg.setdefault("version", 1)
    cfg.setdefault("permissions", {"allow": [], "deny": []})
    cfg.setdefault("approvalMode", "allowlist")
    cfg.setdefault(
        "sandbox",
        {"mode": "disabled", "networkAccess": "user_config_with_defaults"},
    )

    (managed_cursor / "cli-config.json").write_text(
        json.dumps(cfg, indent=2)
    )
    return parent


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "CursorResult",
    "render_prompt",
    "run",
    "probe_init_model",
    "parse_log",
    "_is_trivial_plan_body",
    "_extract_plan_body",
    "_select_final_text",
    "_StreamState",
    "_consume_event",
    "_materialize_cursor_home",
]
