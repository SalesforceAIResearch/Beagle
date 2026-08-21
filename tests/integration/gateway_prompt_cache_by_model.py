#!/usr/bin/env python3
"""Smoke-test the correct prompt-cache request shape for each gateway backend.

This probe answers one practical question: when a conversation grows and a volatile
per-turn budget reminder changes, does the reusable history continue to be read from
cache?

The three backends require different request shapes:

* gpt-5.6 explicit
  - stable ``prompt_cache_key``
  - ``prompt_cache_options: {"mode": "explicit"}``
  - retained ``prompt_cache_breakpoint`` markers on tool-result content blocks
    (the exact growing-conversation shape previously validated on this gateway)
  - volatile reminder as a content part immediately after the marker in the
    newest tool result (the same ``[prefix marker][changing suffix]`` shape as
    the successful fixed-prefix probe)

* gpt-5.6 implicit (select with ``--gpt56-implicit``)
  - stable ``prompt_cache_key``
  - ``prompt_cache_options: {"mode": "implicit"}``
  - no explicit breakpoint markers
  - volatile reminder appended to the newest tool message, which directly tests
    whether the implicit latest-message checkpoint tolerates a changing suffix

* older GPT automatic (gpt-5.5)
  - no cache key, options, or markers
  - volatile reminder appended as the final part of the newest user/tool message
  - automatic longest-prefix matching reuses the content before that part

* Claude through the OpenAI-shaped gateway
  - Anthropic ``cache_control`` on the stable system block and last two user/tool
    messages
  - no OpenAI cache key or options
  - volatile reminder in a separate trailing message

At the defaults this sends 18 requests (3 models x 6 turns) at the gateway's natural
speed. The probe does not enforce a client-side request rate. HTTP 429 responses are
retried with backoff. ``--delay`` optionally waits after each successful response; it
defaults to zero and exists only to distinguish request-shape failures from cache
propagation/routing timing.

Run:
    .venv/bin/python tests/integration/gateway_prompt_cache_by_model.py

This is a manual live probe, not a pytest test. It reads the existing gateway URL and
credential list from ``.env`` and spends model tokens.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
DEFAULT_TURNS = 6
MAX_OUTPUT_TOKENS = 256
REUSE_MIN = 0.75
ANTHROPIC_WINDOW = 2

MODEL_STRATEGIES = {
    "gpt-5.6-sol": "explicit",
    "gpt-5.5": "automatic",
    "claude-opus-4-8": "anthropic",
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": "read a file",
        "parameters": {
            "type": "object",
            "properties": {"p": {"type": "string"}},
            "required": ["p"],
        },
    },
}]

STATIC_SYSTEM = (
    "STATIC BASE SYSTEM PROMPT that remains stable for the whole session. " * 30
)


def load_gateway() -> tuple[str, str]:
    values: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value

    base_url = values["LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL"].strip().rstrip("/")
    raw_keys = values["LLM_GATEWAY_EXPRESS_API_KEY_LIST"].strip()
    if raw_keys.startswith('"'):
        raw_keys = json.loads(raw_keys)
    keys = [
        key.strip()
        for key in raw_keys.replace(";", ",").replace(" ", ",").split(",")
        if key.strip()
    ]
    if not keys:
        raise SystemExit("no gateway API key found in .env")
    return base_url, keys[0]


@dataclass
class Turn:
    prompt: int | None = None
    cached: int | None = None
    write: int | None = None
    served: str | None = None
    error: str | None = None
    usage: dict | None = None


class Gateway:
    def __init__(self, base_url: str, api_key: str) -> None:
        parsed = urlparse(base_url)
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        self._connection_type = connection_type
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self._path = f"{parsed.path.rstrip('/')}/chat/completions"
        self._key = api_key
        self._conn = self._new_connection()

    def _new_connection(self):
        return self._connection_type(self._host, self._port, timeout=180)

    def send(
        self,
        model: str,
        messages: list,
        extra: dict | None,
        *,
        include_tools: bool = True,
    ) -> Turn:
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if include_tools:
            payload["tools"] = TOOLS
        if model.startswith("gpt-5"):
            payload["reasoning_effort"] = "low"
        if extra:
            payload.update(extra)

        body = json.dumps(payload)
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

        status: int | None = None
        data = ""
        error: str | None = None
        for attempt in range(6):
            try:
                self._conn.request("POST", self._path, body, headers)
                response = self._conn.getresponse()
                status = response.status
                retry_after = response.getheader("Retry-After")
                data = response.read().decode(errors="replace")
                if status == 429:
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
                    time.sleep(delay)
                    continue
                break
            except Exception as exc:
                error = str(exc)
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = self._new_connection()
                time.sleep(min(2 ** attempt, 8))

        if status != 200:
            return Turn(error=f"HTTP {status}: {(data or error or '')[:160]}")

        usage = None
        served = None
        for line in data.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            event_data = line[5:].strip()
            if not event_data or event_data == "[DONE]":
                continue
            try:
                event = json.loads(event_data)
            except json.JSONDecodeError:
                continue
            if served is None and event.get("model"):
                served = event["model"]
            if event.get("usage"):
                usage = event["usage"]

        if not usage:
            return Turn(served=served, error="HTTP 200 but no usage chunk")

        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = {}
        return Turn(
            prompt=usage.get("prompt_tokens"),
            cached=details.get("cached_tokens"),
            write=details.get("cache_write_tokens"),
            served=served,
            usage=usage,
        )


def _tool_text(nonce: str, turn: int, *, lines: int = 110) -> str:
    return f"[{nonce}] file {turn:02d}\n" + "\n".join(
        f"f{turn:02d} row {line:04d}: lorem ipsum dolor sit amet consectetur adipiscing {line}"
        for line in range(lines)
    )


def build_messages(
    nonce: str,
    completed_turns: int,
    total_turns: int,
    strategy: str,
    *,
    volatile: bool = True,
) -> tuple[list, dict | None]:
    """Build one request using the backend's correct cache configuration."""
    static = f"{STATIC_SYSTEM} session {nonce}"
    dynamic = "Platform: linux; cwd: /tmp/repo"

    if strategy == "anthropic":
        system = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": static,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic},
            ],
        }
    else:
        system = {"role": "system", "content": f"{static}\n\n{dynamic}"}

    messages = [
        system,
        {"role": "user", "content": f"[{nonce}] read the files in order"},
    ]
    user_tool_indexes = [1]
    tool_indexes = []

    for turn in range(1, completed_turns + 1):
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call-{turn}",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"p": f"f{turn}"}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call-{turn}",
            "content": _tool_text(nonce, turn),
        })
        user_tool_indexes.append(len(messages) - 1)
        tool_indexes.append(len(messages) - 1)

    if strategy == "explicit":
        # This is intentionally the minimal shape proven by the earlier live probe:
        # markers only on retained tool-result endpoints. Extra system/user markers
        # are not required because each tool endpoint already includes all preceding
        # system and user content in its reusable prefix.
        for index in tool_indexes:
            content = messages[index]["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
                messages[index]["content"] = content
            content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    elif strategy == "anthropic":
        # system(1) + rolling messages(2) stays within Anthropic's four-marker limit.
        for index in user_tool_indexes[-ANTHROPIC_WINDOW:]:
            content = messages[index]["content"]
            if isinstance(content, str):
                messages[index]["content"] = [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }]

    if volatile:
        reminder = (
            f"<token_budget turn={completed_turns} "
            f"remaining={total_turns - completed_turns}>"
        )
        if strategy in ("explicit", "implicit56", "automatic"):
            index = user_tool_indexes[-1]
            content = messages[index]["content"]
            parts = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else list(content)
            )
            messages[index]["content"] = parts + [{
                "type": "text",
                "text": reminder,
            }]
        else:
            messages.append({"role": "user", "content": reminder})

    extra = None
    if strategy == "explicit":
        extra = {
            "prompt_cache_key": f"cache-probe-{nonce}",
            "prompt_cache_options": {"mode": "explicit"},
        }
    elif strategy == "implicit56":
        extra = {
            "prompt_cache_key": f"cache-probe-{nonce}",
            "prompt_cache_options": {"mode": "implicit"},
        }
    return messages, extra


def validate_turn(turn: Turn, requested_model: str) -> str | None:
    if turn.error:
        return turn.error
    if turn.served is None:
        return "served model was not reported"
    if turn.served != requested_model:
        return f"served {turn.served!r}, requested {requested_model!r}"
    if turn.prompt is None or turn.cached is None:
        return (
            "prompt_tokens or cached_tokens was not reported; "
            f"usage={json.dumps(turn.usage, sort_keys=True)}"
        )
    if turn.prompt < 0 or turn.cached < 0 or turn.cached > turn.prompt:
        return "invalid prompt/cache token accounting"
    if turn.write is not None:
        if turn.write < 0 or turn.cached + turn.write > turn.prompt:
            return "invalid disjoint cache read/write accounting"
    return None


def run_model(
    gateway: Gateway,
    model: str,
    strategy: str,
    *,
    turns: int,
    delay: float,
    volatile: bool,
    include_tools: bool,
) -> bool:
    nonce = os.urandom(8).hex()
    previous_prompt = None
    reuse_ratios: list[float] = []

    print(f"\n{model}  strategy={strategy}  volatile={volatile}")
    print("turn   prompt   cached    write   reuse(previous prompt)")

    for completed_turns in range(1, turns + 1):
        messages, extra = build_messages(
            nonce,
            completed_turns,
            turns,
            strategy,
            volatile=volatile,
        )
        result = gateway.send(
            model,
            messages,
            extra,
            include_tools=include_tools,
        )
        problem = validate_turn(result, model)
        if problem:
            print(f"{completed_turns:>4}   INCONCLUSIVE: {problem}")
            return False

        ratio = (
            result.cached / previous_prompt
            if previous_prompt
            else None
        )
        if ratio is not None:
            reuse_ratios.append(ratio)
        write = "n/a" if result.write is None else str(result.write)
        ratio_text = "cold" if ratio is None else f"{ratio:.0%}"
        print(
            f"{completed_turns:>4} {result.prompt:>8} {result.cached:>8} "
            f"{write:>8} {ratio_text:>10}"
        )
        previous_prompt = result.prompt
        if delay:
            time.sleep(delay)
    # Automatic caching may write only an eligible token-aligned portion of the
    # first prompt, so its first reuse can legitimately be partial. Judge the
    # stable trajectory from the following turn onward.
    checked_ratios = reuse_ratios[1:]
    passed = bool(checked_ratios) and all(
        ratio >= REUSE_MIN for ratio in checked_ratios
    )
    if passed:
        if volatile:
            print(
                "PASS: the changing reminder stayed outside the reusable prefix; "
                "history cache reads grew with the conversation."
            )
        else:
            print(
                "PASS: growing history caches correctly without a volatile reminder "
                "(control only; volatile placement is not validated)."
            )
    else:
        shown = ", ".join(f"{ratio:.0%}" for ratio in reuse_ratios)
        print(
            f"FAIL: after the first warm-up reuse, expected each ratio >= "
            f"{REUSE_MIN:.0%}; observed [{shown}]."
        )
    return passed


def test_one_shot_volatile(
    gateway: Gateway,
    *,
    turns: int,
    delay: float,
    include_tools: bool,
    volatile_turns: set[int],
) -> bool:
    """Probe whether cache recovers after a rare send-time reminder.

    The normal volatile smoke test adds the reminder on every turn, so it can only prove the
    always-volatile shape fails. This diagnostic injects the same reminder shape on selected turns
    only, then removes it on following turns to answer the practical question: is a rare hint just a
    one-turn dip, or does it poison reuse after it disappears?
    """
    model = "gpt-5.6-sol"
    strategy = "explicit"
    nonce = os.urandom(8).hex()
    previous_prompt = None
    reuse_ratios: list[float] = []
    rows: list[tuple[int, bool, int, int, int | None, float | None]] = []

    print("\n" + "=" * 80)
    print(
        "ONE-SHOT VOLATILE — gpt-5.6 explicit; "
        f"volatile only on turns {sorted(volatile_turns)}"
    )
    print("=" * 80)
    print("turn hint   prompt   cached    write   reuse(previous prompt)")

    for completed_turns in range(1, turns + 1):
        hint_on = completed_turns in volatile_turns
        messages, extra = build_messages(
            nonce,
            completed_turns,
            turns,
            strategy,
            volatile=hint_on,
        )
        result = gateway.send(
            model,
            messages,
            extra,
            include_tools=include_tools,
        )
        problem = validate_turn(result, model)
        if problem:
            print(f"{completed_turns:>4} {'yes' if hint_on else ' no'}   INCONCLUSIVE: {problem}")
            return False

        ratio = (
            result.cached / previous_prompt
            if previous_prompt
            else None
        )
        if ratio is not None:
            reuse_ratios.append(ratio)
        write = result.write
        write_text = "n/a" if write is None else str(write)
        ratio_text = "cold" if ratio is None else f"{ratio:.0%}"
        print(
            f"{completed_turns:>4} {'yes' if hint_on else ' no':>4} "
            f"{result.prompt:>8} {result.cached:>8} {write_text:>8} {ratio_text:>10}"
        )
        rows.append((completed_turns, hint_on, result.prompt or 0, result.cached or 0, write, ratio))
        previous_prompt = result.prompt
        if delay:
            time.sleep(delay)

    # A recovery turn is a non-hint turn immediately after a hint turn. If rare hints are acceptable,
    # these turns should resume high reuse instead of staying at zero.
    recovery_ratios = [
        ratio
        for turn, hint_on, _prompt, _cached, _write, ratio in rows
        if not hint_on and (turn - 1) in volatile_turns and ratio is not None
    ]
    passed = bool(recovery_ratios) and all(ratio >= REUSE_MIN for ratio in recovery_ratios)
    shown = ", ".join(f"{ratio:.0%}" for ratio in recovery_ratios) or "none"
    if passed:
        print(f"PASS: cache recovered on post-hint turns ({shown}).")
    else:
        print(f"FAIL: cache did not recover on post-hint turns; observed [{shown}].")
    return passed


def build_marker_role_messages(
    nonce: str,
    role: str,
    suffix: str,
) -> list:
    """Build a fixed-prefix explicit-cache request, varying only marker role."""
    prefix_part = {
        "type": "text",
        # Match the ~9k-token prefix size of the earlier successful fixed-prefix
        # probe; the normal growing-conversation tool blocks remain smaller.
        "text": _tool_text(nonce, 99, lines=320),
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    suffix_part = {"type": "text", "text": suffix}
    system = {"role": "system", "content": f"Stable marker-role test {nonce}"}

    if role == "user":
        return [
            system,
            {"role": "user", "content": [prefix_part, suffix_part]},
        ]
    if role == "tool":
        return [
            system,
            {"role": "user", "content": "Read the supplied file."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "role-test-call",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"p": "fixed"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "role-test-call",
                "content": [prefix_part, suffix_part],
            },
        ]
    raise ValueError(f"unsupported marker role: {role}")


def test_marker_roles(
    gateway: Gateway,
    *,
    delay: float,
    include_tools: bool,
) -> bool:
    """Compare explicit breakpoints on user versus tool content blocks."""
    outcomes = []
    print(
        "\nGPT-5.6 explicit marker role A/B "
        f"(tools={include_tools}, ~9k fixed prefix, changing suffix)"
    )
    for role in ("user", "tool"):
        nonce = os.urandom(8).hex()
        extra = {
            "prompt_cache_key": f"marker-role-{role}-{nonce}",
            "prompt_cache_options": {"mode": "explicit"},
        }
        results: list[Turn | None] = []
        print(f"\nmarker role={role}")
        for label, suffix in (
            ("warm A", "SUFFIX A " * 12),
            ("repeat A", "SUFFIX A " * 12),
            ("change B", "COMPLETELY DIFFERENT SUFFIX B " * 12),
        ):
            result = gateway.send(
                "gpt-5.6-sol",
                build_marker_role_messages(nonce, role, suffix),
                extra,
                include_tools=include_tools,
            )
            problem = validate_turn(result, "gpt-5.6-sol")
            if problem:
                print(f"  {label:>8}: INCONCLUSIVE: {problem}")
                results.append(None)
            else:
                results.append(result)
                write = "n/a" if result.write is None else str(result.write)
                print(
                    f"  {label:>8}: prompt={result.prompt} cached={result.cached} "
                    f"write={write}"
                )
            if delay:
                time.sleep(delay)

        # The changed-suffix third request is decisive. Do not abort merely because
        # the repeated-A middle request omitted usage details.
        if len(results) != 3 or results[0] is None or results[2] is None:
            outcomes.append(False)
            continue
        first, changed = results[0], results[2]
        written_prefix = first.write or first.prompt
        passed = changed.cached >= REUSE_MIN * written_prefix
        outcomes.append(passed)
        print(
            "  PASS: changed suffix reused the marked prefix."
            if passed
            else "  FAIL: changed suffix did not reuse the marked prefix."
        )

    if outcomes == [True, False]:
        print("\nRESULT: user markers work but tool markers do not on this gateway path.")
    elif outcomes == [True, True]:
        print("\nRESULT: both marker roles work; the growing volatile failure is elsewhere.")
    elif outcomes == [False, False]:
        print("\nRESULT: neither role passed; explicit support/timing is inconclusive.")
    else:
        print("\nRESULT: unexpected role asymmetry; inspect the raw numbers.")
    return all(outcomes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_TURNS,
        help=f"requests per model (default: {DEFAULT_TURNS})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="diagnostic seconds after each successful response (default: 0)",
    )
    parser.add_argument(
        "--test-marker-roles",
        action="store_true",
        help="run a six-request user-vs-tool explicit breakpoint diagnostic",
    )
    parser.add_argument(
        "--test-one-shot-volatile",
        action="store_true",
        help="run a gpt-5.6 diagnostic with volatile reminders on selected turns only",
    )
    parser.add_argument(
        "--volatile-turn",
        action="append",
        type=int,
        help="turn to inject a volatile reminder for --test-one-shot-volatile (default: 3)",
    )
    parser.add_argument(
        "--gpt56-implicit",
        action="store_true",
        help="test GPT-5.6 implicit caching instead of explicit breakpoints",
    )
    parser.add_argument(
        "--no-volatile",
        action="store_true",
        help="omit the changing budget suffix to isolate suffix-related misses",
    )
    parser.add_argument(
        "--omit-tools",
        action="store_true",
        help="omit top-level tool definitions to reproduce the earlier successful probe",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_STRATEGIES),
        help="model to probe; repeat for multiple models (default: all)",
    )
    args = parser.parse_args()
    if args.turns < 2:
        parser.error("--turns must be at least 2")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.volatile_turn and any(turn < 1 for turn in args.volatile_turn):
        parser.error("--volatile-turn values must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    selected_models = args.model or list(MODEL_STRATEGIES)
    request_count = (
        args.turns
        if args.test_one_shot_volatile
        else 6 if args.test_marker_roles else len(selected_models) * args.turns
    )
    print(
        f"Prompt-cache configuration smoke probe: {request_count} requests, "
        f"delay={args.delay}s, volatile={not args.no_volatile}, "
        f"tools={not args.omit_tools}"
    )

    base_url, api_key = load_gateway()
    gateway = Gateway(base_url, api_key)
    if args.test_one_shot_volatile:
        volatile_turns = set(args.volatile_turn or [3])
        if any(turn > args.turns for turn in volatile_turns):
            raise SystemExit("--volatile-turn cannot exceed --turns")
        if not test_one_shot_volatile(
            gateway,
            turns=args.turns,
            delay=args.delay,
            include_tools=not args.omit_tools,
            volatile_turns=volatile_turns,
        ):
            raise SystemExit(1)
        return
    if args.test_marker_roles:
        if not test_marker_roles(
            gateway,
            delay=args.delay,
            include_tools=not args.omit_tools,
        ):
            raise SystemExit(1)
        return

    outcomes = []
    for model in selected_models:
        strategy = MODEL_STRATEGIES[model]
        if model == "gpt-5.6-sol" and args.gpt56_implicit:
            strategy = "implicit56"
        outcomes.append(run_model(
            gateway,
            model,
            strategy,
            turns=args.turns,
            delay=args.delay,
            volatile=not args.no_volatile,
            include_tools=not args.omit_tools,
        ))

    print("\nALL PASS" if all(outcomes) else "\nNOT ALL PASS")
    if not all(outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
