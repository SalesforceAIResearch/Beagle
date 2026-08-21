#!/usr/bin/env python3
"""Probe how the LLM Gateway Express (local proxy) caches prompts for gpt-5.6-sol.

Goal: decide, with raw HTTP and NO monet, whether this gateway supports
  (a) automatic longest-prefix caching (implicit breakpoint at the latest message), and
  (b) gpt-5.6's NATIVE explicit caching:
        top-level  prompt_cache_key           (stable routing/cache key)
        top-level  prompt_cache_options {mode:"explicit"}
        per-part   prompt_cache_breakpoint {mode:"explicit"}   (marks end of a reusable prefix)

If explicit caching works, a per-turn "budget" reminder can ride AFTER a breakpoint and stay cached
(the [history][breakpoint][budget] pattern), and monet can keep it as-is. If it does not, the reminder
must be made byte-stable another way (append-only history).

METHOD NOTES — this version fixes an earlier flawed probe (see the audit):
  * RETAINED breakpoints. Explicit mode does NOT add an implicit breakpoint, and prior-turn
    breakpoints are read-only hit points that must be RETAINED, not moved to the newest message each
    turn. So the explicit variants keep a breakpoint on a FIXED position (or on every prior tool
    block), never a marker that jumps forward.
  * STABLE prompt_cache_key on EVERY variant (consistent routing), not just the explicit ones.
  * Print cached_tokens AND cache_write_tokens — cached=0 with write>0 means "written, not yet
    matched" (honored), not "dropped".
  * Check HTTP status and fail loudly; never fold errors into (0,0).
  * One persistent connection (relay backend affinity), fresh nonce per variant (cold).

Run:
  cd /fsx/home/yutong/Github/beagle
  .venv/bin/python tests/integration/gateway_prompt_cache_probe.py
Reads LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL + LLM_GATEWAY_EXPRESS_API_KEY_LIST from beagle/.env.
Hits the live gateway (spends a little). Not a pytest test (no test_ prefix) — a manual live probe.
"""

from __future__ import annotations

import http.client
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
MODEL = "gpt-5.6-sol"


def load_gateway() -> "tuple[str, str]":
    env: "dict[str, str]" = {}
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v
    base = env["LLM_GATEWAY_EXPRESS_LOCAL_PROXY_URL"].strip().rstrip("/")
    raw = env["LLM_GATEWAY_EXPRESS_API_KEY_LIST"].strip()
    if raw[:1] == '"':
        raw = json.loads(raw)
    keys = [k.strip() for k in raw.replace(";", ",").replace(" ", ",").split(",") if k.strip()]
    if not keys:
        raise SystemExit("no LLM_GATEWAY_EXPRESS_API_KEY_LIST keys in .env")
    return base, keys[0]


class Result:
    __slots__ = ("status", "prompt", "cached", "write", "error")

    def __init__(self, status, prompt, cached, write, error=None):
        self.status, self.prompt, self.cached, self.write, self.error = status, prompt, cached, write, error

    def __str__(self):
        if self.error:
            return f"[HTTP {self.status} ERROR: {self.error}]"
        return f"prompt={self.prompt:>6} cached={self.cached:>6} write={self.write:>6}"


class Gateway:
    """One keep-alive connection reused by every send()."""

    def __init__(self, base_url: str, api_key: str) -> None:
        u = urlparse(base_url)
        self._host, self._port, self._key = u.hostname or "", u.port, api_key
        self._conn = http.client.HTTPConnection(self._host, self._port, timeout=180)

    def send(self, messages: list, extra: "dict | None" = None) -> Result:
        payload = {
            "model": MODEL,
            "messages": messages,
            "max_completion_tokens": 64,
            "reasoning_effort": "low",
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if extra:
            payload.update(extra)
        body = json.dumps(payload)
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                   "Connection": "keep-alive"}
        status, data, err = None, "", None
        for _ in range(3):
            try:
                self._conn.request("POST", "/chat/completions", body, headers)
                resp = self._conn.getresponse()
                status = resp.status
                data = resp.read().decode(errors="replace")
                break
            except Exception as e:  # connection dropped — reconnect and retry
                err = f"conn: {e}"
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = http.client.HTTPConnection(self._host, self._port, timeout=180)
                time.sleep(1)
        if status is None:
            return Result(None, 0, 0, 0, err or "no response")
        if status != 200:
            return Result(status, 0, 0, 0, f"non-200 body: {data[:200]}")
        usage = None
        for line in data.splitlines():
            line = line.strip()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    obj = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
        if not usage:
            return Result(status, 0, 0, 0, "200 but no usage chunk")
        d = usage.get("prompt_tokens_details") or {}
        return Result(status, usage.get("prompt_tokens", 0),
                      d.get("cached_tokens", 0), d.get("cache_write_tokens", 0))


def big_text(nonce: str, n_lines: int) -> str:
    return f"SESSION {nonce}\n" + "\n".join(
        f"{nonce} line {i:04d}: the quick brown fox jumps over the lazy dog by the river {i}" for i in range(n_lines))


# ---------------------------------------------------------------- EXPERIMENT 1 (the decisive one)
def experiment_1_explicit_fixed_prefix(gw: Gateway) -> None:
    """Fixed stable prefix P with a RETAINED explicit breakpoint at its end; only the suffix changes.
    This is the canonical explicit-cache pattern done correctly. If the gateway honors explicit
    breakpoints, turn 2 (suffix changed, P + its breakpoint unchanged) reads cached ~= |P|."""
    print("=" * 80)
    print("EXPERIMENT 1 — explicit breakpoint on a FIXED prefix, changing suffix (retained marker)")
    print("=" * 80)
    nonce = os.urandom(8).hex()
    P = big_text(nonce, 320)  # ~9k tokens, cold
    key = f"probe-explicit-{nonce}"

    def msgs(suffix: str, explicit: bool):
        prefix_part = {"type": "text", "text": P}
        if explicit:
            prefix_part["prompt_cache_breakpoint"] = {"mode": "explicit"}   # RETAINED, fixed position
        return [{"role": "user", "content": [prefix_part, {"type": "text", "text": suffix}]}]

    extra = {"prompt_cache_key": key, "prompt_cache_options": {"mode": "explicit"}}
    print("  warm  (P + suffixA, breakpoint on P):", gw.send(msgs("SUFFIX AAAA " * 10, True), extra)); time.sleep(2)
    print("  warm2 (P + suffixA, same):           ", gw.send(msgs("SUFFIX AAAA " * 10, True), extra)); time.sleep(2)
    r = gw.send(msgs("SUFFIX BBBB totally different " * 10, True), extra); time.sleep(2)
    print("  read  (P + suffixB, P+breakpoint retained):", r)
    print("  --> if honored: cached ~= |P| (~9k). If ignored/dropped: cached ~= 0.")
    if r.error:
        print(f"  INCONCLUSIVE — request errored: {r.error}")
    elif r.cached > 5000:
        print("  => explicit breakpoint IS honored (fixed prefix stayed cached across a changed suffix).")
    else:
        print("  => explicit breakpoint did NOT cache the fixed prefix here (cached≈0).")


# ---------------------------------------------------------------- growing-conversation variants
def _conversation(nonce: str, n_turns: int, mark_all_tools: bool) -> list:
    """system + user + N (assistant tool_call, tool result). If mark_all_tools, put a RETAINED
    explicit breakpoint on EVERY tool block (never moved) so prior-turn markers persist."""
    msgs: list = [
        {"role": "system", "content": f"You are a helpful agent. session {nonce}"},
        {"role": "user", "content": f"[{nonce}] read files in order"},
    ]
    for t in range(1, n_turns + 1):
        msgs.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{t}", "type": "function",
                                     "function": {"name": "read", "arguments": json.dumps({"p": f"f{t}"})}}]})
        body = f"[{nonce}] file {t:02d}\n" + "\n".join(
            f"f{t:02d} row {i:04d}: lorem ipsum dolor sit amet consectetur {i}" for i in range(110))
        part = {"type": "text", "text": body}
        if mark_all_tools:
            part["prompt_cache_breakpoint"] = {"mode": "explicit"}
        msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": [part]})
    return msgs


def experiment_2_growing(gw: Gateway) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 — growing conversation (stable prompt_cache_key on BOTH variants)")
    print("=" * 80)

    def run(label, explicit, turns=8):
        nonce = os.urandom(8).hex()
        key = f"probe-grow-{nonce}"
        extra = {"prompt_cache_key": key}
        if explicit:
            extra["prompt_cache_options"] = {"mode": "explicit"}
        print(f"  --- {label} ---")
        tot_p = tot_c = 0
        for n in range(1, turns + 1):
            r = gw.send(_conversation(nonce, n, mark_all_tools=explicit), extra)
            if r.error:
                print(f"    turn {n:>2}: {r}")
                continue
            tot_p += r.prompt; tot_c += r.cached
            print(f"    turn {n:>2}: {r}  ({100*r.cached/r.prompt if r.prompt else 0:3.0f}% hit)")
            time.sleep(3)
        print(f"    AGGREGATE hit = {100*tot_c/tot_p if tot_p else 0:.1f}%\n")

    run("A: implicit (no explicit hint), byte-stable latest", explicit=False)
    run("B: explicit mode, breakpoint RETAINED on every tool block", explicit=True)


def experiment_3_growing_with_changing_budget(gw: Gateway) -> None:
    """THE monet scenario: growing history with RETAINED explicit breakpoints on every tool block,
    plus a CHANGING per-turn 'budget' as a trailing message AFTER the last breakpoint. If the history
    keeps caching (cached grows) despite the changing budget, monet can keep the budget send-time
    after a breakpoint — no append-only needed."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 — growing history + retained explicit breakpoints + CHANGING budget after")
    print("=" * 80)
    nonce = os.urandom(8).hex()
    extra = {"prompt_cache_key": f"probe-budget-{nonce}", "prompt_cache_options": {"mode": "explicit"}}
    tot_p = tot_c = 0
    for n in range(1, 9):
        msgs = _conversation(nonce, n, mark_all_tools=True)
        msgs.append({"role": "user", "content": f"<budget turn {n}: {8 - n} turns left, used {n*137} tok>"})
        r = gw.send(msgs, extra)
        if r.error:
            print(f"    turn {n:>2}: {r}"); continue
        tot_p += r.prompt; tot_c += r.cached
        print(f"    turn {n:>2}: {r}  ({100*r.cached/r.prompt if r.prompt else 0:3.0f}% hit)")
        time.sleep(3)
    print(f"    AGGREGATE hit = {100*tot_c/tot_p if tot_p else 0:.1f}%")
    print("    --> if this GROWS like Exp 2, the changing budget rides after the breakpoint safely")
    print("        => keep the budget send-time via explicit caching; no append-only churn needed.")


def main() -> None:
    base, key = load_gateway()
    print(f"gateway: {base}  model: {MODEL}  (persistent connection, fresh nonce/key per variant)\n")
    gw = Gateway(base, key)
    experiment_1_explicit_fixed_prefix(gw)
    experiment_2_growing(gw)
    experiment_3_growing_with_changing_budget(gw)


if __name__ == "__main__":
    main()
