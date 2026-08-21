"""``trace_analyzer`` CLI.

    python -m trace_analyzer check     <trace> [--config default|monet|PATH] [--llm]
    python -m trace_analyzer summarize <trace> [--source auto] [--format text|json]
    python -m trace_analyzer normalize <trace> [-o out.messages.jsonl]
    python -m trace_analyzer ask       <trace> [<trace> ...] -q "..." [--model ...]
    python -m trace_analyzer profiles  [--config default]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import normalizers  # noqa: F401  (registers built-in normalizers)
from .config import builtin_configs, load_config
from .model import Severity
from .normalizer import NormalizeError, available, load
from .pipeline import run_qc
from .summarize import summarize


def _load_trace(source: str, path: str):
    try:
        return load(path, source=source)
    except NormalizeError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2)


def _maybe_client(args):
    if not getattr(args, "llm", False):
        return None
    from .llm import LLMError, OpenAIClient

    try:
        return OpenAIClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    except LLMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2)


# ── check (QC) ──────────────────────────────────────────────────────────────
def _cmd_check(args) -> int:
    traj = _load_trace(args.source, args.trace)
    client = _maybe_client(args)
    result = run_qc(
        traj, args.config, llm=client, min_severity=Severity.parse(args.min_severity)
    )
    if args.format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"{result.trace_id}: {len(result.issues)} issue(s)  [config={result.config}]")
        for i in result.issues:
            loc = f"msg {i.message_index}" if i.message_index is not None else "run"
            print(f"  [{str(i.severity).upper():6}] {i.category.value:24} ({loc}) {i.summary}")
            if i.evidence:
                print(f"           ↳ {i.evidence}")
            print(f"           · proposer={i.proposer}")
        if result.skipped:
            print(f"  proposers not run: {', '.join(result.skipped)}")
    if args.exit_code and any(i.severity >= Severity.parse(args.exit_code) for i in result.issues):
        return 1
    return 0


# ── summarize ───────────────────────────────────────────────────────────────
def _cmd_summarize(args) -> int:
    s = summarize(_load_trace(args.source, args.trace))
    if args.format == "json":
        print(json.dumps(s.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"trace_id : {s.trace_id}   (source={s.source})")
    print(f"turns    : {s.num_turns}   tool calls: {s.num_tool_calls}   errors: {s.num_tool_errors}")
    print(f"edits    : {s.num_edits}   test runs : {s.num_test_runs}   plan calls: {s.num_plan_calls}")
    print(f"terminal : {s.terminal}")
    if s.peak_prompt_tokens is not None:
        print(f"peak prompt tokens: {s.peak_prompt_tokens:,}")
    if s.tool_histogram:
        print("tools    : " + "  ".join(f"{k}={v}" for k, v in s.tool_histogram.items()))
    print(f"ends with success language: {s.ends_with_success_language}")
    if s.final_text_excerpt:
        print(f"final text: {s.final_text_excerpt}")
    return 0


# ── normalize ───────────────────────────────────────────────────────────────
def _cmd_normalize(args) -> int:
    traj = _load_trace(args.source, args.trace)
    rows = traj.messages()
    lines = [json.dumps(m, ensure_ascii=False) for m in rows]
    if args.output == "-":
        print("\n".join(lines))
        return 0
    out = Path(args.output) if args.output else Path(args.trace).with_name(
        traj.trace_id + ".messages.jsonl"
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stderr.write(f"wrote {len(rows)} messages -> {out}\n")
    return 0


# ── ask (QA) ────────────────────────────────────────────────────────────────
def _cmd_ask(args) -> int:
    from .llm import LLMError, ask

    trajs = [_load_trace(args.source, p) for p in args.trace]
    try:
        answer = ask(trajs, args.question, model=args.model, base_url=args.base_url, api_key=args.api_key)
    except LLMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    if args.format == "json":
        print(json.dumps({"trace_ids": [t.trace_id for t in trajs], "question": args.question,
                          "answer": answer}, ensure_ascii=False, indent=2))
    else:
        print(answer)
    return 0


# ── profiles ────────────────────────────────────────────────────────────────
def _cmd_profiles(args) -> int:
    cfg = load_config(args.config)
    print(f"config: {cfg.name}")
    print("proposers:")
    for p in cfg.proposers:
        print(f"  - {p.name}{'  (needs LLM)' if p.requires_llm else ''}")
    print("filters:  " + ", ".join(f.name for f in cfg.filters))
    print("mergers:  " + ", ".join(m.name for m in cfg.mergers))
    print(f"\nbuilt-in configs: {', '.join(builtin_configs())}")
    return 0


def _add_llm_args(p):
    p.add_argument("--llm", action="store_true", help="enable LLM proposers (needs creds)")
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trace_analyzer",
        description="Agent-trajectory analyzer: normalize → QC (proposers+filters) / ask.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    src = f"auto or one of {{{', '.join(available())}}}"

    pc = sub.add_parser("check", help="run the QC pipeline (proposers → filters → dedup)")
    pc.add_argument("trace")
    pc.add_argument("--source", default="auto", help=src)
    pc.add_argument("--config", default="default", help=f"profile: {', '.join(builtin_configs())} or a path")
    pc.add_argument("--format", choices=("text", "json"), default="text")
    pc.add_argument("--min-severity", default="info", help="info|low|medium|high")
    pc.add_argument("--exit-code", nargs="?", const="high", default=None,
                    help="exit 1 if any issue >= this severity")
    _add_llm_args(pc)
    pc.set_defaults(func=_cmd_check)

    ps = sub.add_parser("summarize", help="print per-run statistics")
    ps.add_argument("trace")
    ps.add_argument("--source", default="auto", help=src)
    ps.add_argument("--format", choices=("text", "json"), default="text")
    ps.set_defaults(func=_cmd_summarize)

    pn = sub.add_parser("normalize", help="export canonical numbered messages (JSONL)")
    pn.add_argument("trace")
    pn.add_argument("--source", default="auto", help=src)
    pn.add_argument("-o", "--output", default=None, help="output path, or '-' for stdout")
    pn.set_defaults(func=_cmd_normalize)

    pa = sub.add_parser("ask", help="ask an LLM about one or more traces")
    pa.add_argument("trace", nargs="+")
    pa.add_argument("-q", "--question", required=True)
    pa.add_argument("--source", default="auto")
    pa.add_argument("--format", choices=("text", "json"), default="text")
    pa.add_argument("--model", default=None)
    pa.add_argument("--base-url", default=None)
    pa.add_argument("--api-key", default=None)
    pa.set_defaults(func=_cmd_ask)

    pp = sub.add_parser("profiles", help="show a config's proposers/filters/mergers")
    pp.add_argument("--config", default="default")
    pp.set_defaults(func=_cmd_profiles)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
