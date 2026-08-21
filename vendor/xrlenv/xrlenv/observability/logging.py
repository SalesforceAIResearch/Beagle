"""Logging — JSON envelopes for shipping, ANSI-colorized lines for terminals.

Phase 0 supports two output shapes that share the same envelope semantics:

- :class:`JsonFormatter` — one JSON object per line on stdout, captured by
  ``systemd-journal`` on cloud nodes and by ``docker logs`` for the
  local-mode containers. Spec 08 §"Structured logs". This is what log
  shippers, ``jq``, and the admin panel's audit feed expect.
- :class:`PrettyFormatter` — short colorized line for an operator
  watching a terminal (``xrlenv up`` running interactively, examples,
  one-shot debug runs). Drops the floating-point ``ts`` for a
  ``HH:MM:SS`` clock and tags the level with ANSI color
  (red=ERROR, yellow=WARNING, green=INFO, dim=DEBUG).

The shared envelope fields are::

    ts       float    UNIX seconds with millisecond resolution
    level    str      DEBUG / INFO / WARNING / ERROR / CRITICAL
    event    str      logger name (e.g. ``xrlenv.control.coordinator``)
    message  str      formatted log message
    rollout_id?, sandbox_id?, node_id?  (when present in extras)

Both formatters preserve any field passed via
``LoggerAdapter.log(..., extra={...})``. The JSON formatter splices it
into the envelope; the pretty formatter renders it as ``key=value`` at
the end of the line.

The :func:`configure_logging` entry point picks a formatter from
``log_format``: ``"json"``, ``"pretty"``, or ``"auto"`` (default —
``pretty`` when the stream is a TTY, ``json`` otherwise so that
``journalctl`` and ``docker logs`` keep getting structured records).
The legacy :func:`configure_json_logging` is kept as a thin wrapper
for callers that want to lock in JSON output unconditionally.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

# Rotating-file defaults — a 50 MiB x 10-file ceiling caps the firehose at
# ~500 MiB on disk by construction, so a long-running control plane never
# fills the volume the way an unbounded stdout capture would.
_DEFAULT_LOG_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_LOG_BACKUP_COUNT = 10

# Standard LogRecord attributes we don't want to repeat as extras
# (Python sets these automatically; copying them into the JSON envelope
# adds noise and breaks the record/extras contract).
_RESERVED_LOGRECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        # PrettyFormatter sets `color` itself; never reflect it back as an extra.
        "color",
    }
)


# ANSI SGR escape codes. Kept as module-level constants so tests can patch
# them out, and so a `colorize=False` formatter can route the same code path
# through empty strings (no branching in the hot loop).
_RESET = "\x1b[0m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RED = "\x1b[31m"
_BRIGHT_RED = "\x1b[91m"
_YELLOW = "\x1b[33m"
_GREEN = "\x1b[32m"
_GREY = "\x1b[90m"


_LEVEL_COLOR: dict[int, str] = {
    logging.DEBUG: _DIM,
    logging.INFO: _GREEN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _BOLD + _BRIGHT_RED,
}


# Short fixed-width labels so columns line up in the operator's terminal.
_LEVEL_LABEL: dict[int, str] = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO ",
    logging.WARNING: "WARN ",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRIT ",
}


LogFormat = Literal["json", "pretty", "auto"]


class JsonFormatter(logging.Formatter):
    """Render :class:`logging.LogRecord` as a single JSON object.

    Drop-in replacement for ``logging.Formatter`` — install on any
    handler. Spec-08 envelope fields go first; arbitrary ``extra={}``
    fields follow.
    """

    def format(self, record: logging.LogRecord) -> str:
        envelope: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "event": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_ATTRS or key in envelope:
                continue
            envelope[key] = _coerce(value)
        if record.exc_info:
            envelope["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(envelope, default=_coerce, separators=(",", ":"))


class PrettyFormatter(logging.Formatter):
    """Render a record as one human-friendly colorized line.

    Layout::

        HH:MM:SS LEVEL  module.event: message  key=value key=value

    Where the ``LEVEL`` token is wrapped in ANSI SGR codes by default
    (red/yellow/green/dim per :data:`_LEVEL_COLOR`). Pass
    ``colorize=False`` (or set ``NO_COLOR=1`` in the environment) to
    emit plain ASCII — the layout is identical so log lines remain
    grep-friendly.

    Extras land at the end of the line as ``key=value`` pairs in the
    same order Python iterated ``LogRecord.__dict__``. ``exc_info``
    is appended on a new indented line, matching stdlib's traceback
    rendering.
    """

    def __init__(self, *, colorize: bool = True) -> None:
        super().__init__()
        self._colorize = colorize and not _no_color_env()

    def format(self, record: logging.LogRecord) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(record.created))
        label = _LEVEL_LABEL.get(record.levelno, record.levelname.ljust(5))
        if self._colorize:
            color = _LEVEL_COLOR.get(record.levelno, "")
            level_token = f"{color}{label}{_RESET}" if color else label
            event_token = f"{_GREY}{record.name}{_RESET}"
        else:
            level_token = label
            event_token = record.name
        message = record.getMessage()
        line = f"{clock} {level_token} {event_token}: {message}"

        extras: list[str] = []
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_ATTRS:
                continue
            extras.append(f"{key}={_pretty_extra(value)}")
        if extras:
            line = f"{line}  {' '.join(extras)}"

        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _pretty_extra(value: Any) -> str:
    """Render an extras value as a compact ``key=value`` token.

    Strings without whitespace stay bare; everything else round-trips
    through ``json.dumps`` so the output is parseable by eye.
    """
    if isinstance(value, str) and value and not any(c.isspace() for c in value):
        return value
    try:
        return json.dumps(value, default=_coerce, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _no_color_env() -> bool:
    """Honor the de-facto NO_COLOR convention (https://no-color.org).

    Any non-empty value disables color output regardless of TTY status.
    """
    return bool(os.environ.get("NO_COLOR", ""))


def _coerce(value: Any) -> Any:
    """Make ``value`` JSON-safe; fall back to ``repr`` for opaque types."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_coerce(v) for v in value]
    return repr(value)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_format: LogFormat = "auto",
    stream: Any = None,
    replace_handlers: bool = True,
    log_file: str | os.PathLike[str] | None = None,
    log_max_bytes: int = _DEFAULT_LOG_MAX_BYTES,
    log_backup_count: int = _DEFAULT_LOG_BACKUP_COUNT,
    stdout_level: int | str | None = None,
) -> logging.Handler:
    """Install handlers on the root logger.

    ``log_format`` selects the **console** (stdout) formatter:

    - ``"json"`` — :class:`JsonFormatter` (production / cloud).
    - ``"pretty"`` — :class:`PrettyFormatter` (interactive terminals).
    - ``"auto"`` (default) — pretty when ``stream`` is a TTY, json otherwise.
      That keeps ``xrlenv up`` colorized when the operator runs it by hand,
      while ``journalctl``/``docker logs`` capture continues to receive
      structured records on the cloud nodes.

    ``log_file`` adds a second, durable sink: a size-rotating
    :class:`~logging.handlers.RotatingFileHandler` (``log_max_bytes`` per
    file, ``log_backup_count`` rotations retained) that always emits JSON
    envelopes regardless of the console format. This is for long-running
    daemons whose stdout is captured to a file that would otherwise grow
    without bound (e.g. a Slurm ``--output`` capture). When ``log_file`` is
    set, the **firehose moves to the file at** ``level`` and the console is
    floored at ``WARNING`` (override via ``stdout_level``) so the stdout
    capture stays small while still surfacing crashes. Rotation keeps disk
    bounded by ``log_max_bytes * (log_backup_count + 1)``.

    Idempotent when called from the same entry point — re-invoking it
    with ``replace_handlers=True`` (the default) clears existing handlers
    so re-running ``xrlenv-node`` in a debugger doesn't cascade records.
    Returns the primary handler (the file handler when ``log_file`` is set,
    else the console handler) so callers can attach filters or remove it on
    shutdown.
    """
    level = _coerce_level(level, default=logging.INFO)
    target = stream if stream is not None else sys.stdout

    console = logging.StreamHandler(target)
    chosen = log_format
    if chosen == "auto":
        chosen = "pretty" if _stream_is_tty(target) else "json"
    if chosen == "pretty":
        console.setFormatter(PrettyFormatter(colorize=_stream_is_tty(target)))
    else:
        console.setFormatter(JsonFormatter())

    handlers: list[logging.Handler] = [console]
    primary: logging.Handler = console

    if log_file is not None:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
            delay=True,
        )
        # The rotating file is the durable, shippable firehose: always JSON
        # (spec-08 envelopes), regardless of the TTY-driven console format.
        file_handler.setFormatter(JsonFormatter())
        file_handler.setLevel(level)
        handlers.append(file_handler)
        primary = file_handler
        # Keep stdout (the unbounded-growth risk) to WARNING+ so a Slurm
        # --output capture holds only the boot banner + crashes.
        console.setLevel(_coerce_level(stdout_level, default=logging.WARNING))
    else:
        # Enforce the intended verbosity AT THE CONSOLE HANDLER, not only the root
        # logger. A library that force-sets its OWN loggers to DEBUG (pier's
        # ``setup_logger`` does exactly this, per-trial) otherwise floods the
        # console: those DEBUG records propagate straight to a NOTSET handler and
        # bypass ``level`` entirely. A handler-level floor drops them regardless of
        # the emitting logger's level. ``stdout_level`` still overrides when given.
        console.setLevel(_coerce_level(stdout_level, default=level))

    root = logging.getLogger()
    if replace_handlers:
        for existing in list(root.handlers):
            root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)

    # The root level gates records before any handler sees them, so it must
    # admit the most verbose handler; per-handler levels then route.
    set_levels = [h.level for h in handlers if h.level != logging.NOTSET]
    root.setLevel(min([level, *set_levels]))
    return primary


def _coerce_level(level: int | str | None, *, default: int) -> int:
    """Resolve a level name/number to an int, falling back to ``default``."""
    if level is None:
        return default
    if isinstance(level, str):
        return logging.getLevelNamesMapping().get(level.upper(), default)
    return level


def configure_json_logging(
    level: int | str = logging.INFO,
    *,
    stream: Any = None,
    replace_handlers: bool = True,
) -> logging.Handler:
    """Install :class:`JsonFormatter` on the root logger.

    Thin wrapper around :func:`configure_logging` that locks the format
    to ``"json"`` regardless of TTY status. Existing call sites keep
    their behavior; new call sites should prefer :func:`configure_logging`.
    """
    return configure_logging(
        level,
        log_format="json",
        stream=stream,
        replace_handlers=replace_handlers,
    )


def _stream_is_tty(stream: Any) -> bool:
    """Defensive ``stream.isatty()`` — ``StringIO`` and similar lack the method."""
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    try:
        return bool(isatty())
    except (ValueError, OSError):
        return False


__all__ = [
    "JsonFormatter",
    "LogFormat",
    "PrettyFormatter",
    "configure_json_logging",
    "configure_logging",
]
