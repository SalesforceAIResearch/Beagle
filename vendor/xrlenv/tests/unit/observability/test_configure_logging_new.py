"""Unit tests for the new configure_logging params and _coerce_level.

Covers (all regression-guard encoding of manual-verification listed in the
feature request):

1. No log_file — single StreamHandler; firehose at ``level`` on stdout.
2. With log_file — RotatingFileHandler added; file is JSON firehose at
   ``level``; console floored at WARNING.
3. stdout_level override — (a) with log_file + stdout_level="INFO" stdout
   mirrors firehose; (b) no log_file + stdout_level="WARNING" floors stdout.
4. Rotation bounds disk — small max_bytes + backup_count limits files on disk.
5. Parent dir auto-created — passing log_file under non-existent subdir creates
   the parent.
6. replace_handlers=True — re-invoking clears prior handlers.
7. Return value — RotatingFileHandler when log_file set; else StreamHandler.
8. _coerce_level — name→int, None→default, int passthrough, unknown→default.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import logging.handlers
from collections.abc import Generator
from pathlib import Path

import pytest
from xrlenv.observability.logging import _coerce_level, configure_logging

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: isolate the root logger state so tests don't bleed into each other
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Generator[None, None, None]:
    """Save root logger state and restore it after every test.

    The test instructions warn that these tests mutate the root logger.
    We capture the prior handlers + level and put them back in teardown,
    closing any file-based handlers installed by the SUT so there are no
    leaked file descriptors.
    """
    root = logging.getLogger()
    prior_level = root.level
    prior_handlers = list(root.handlers)

    yield

    # Close any handlers the test installed (file, rotating, etc.).
    for h in list(root.handlers):
        if h not in prior_handlers:
            h.flush()
            h.close()
            root.removeHandler(h)

    # Restore pre-test handlers and level.
    root.handlers = prior_handlers
    root.setLevel(prior_level)


# ──────────────────────────────────────────────────────────────────────────────
# 8. _coerce_level
# ──────────────────────────────────────────────────────────────────────────────


class TestCoerceLevel:
    def test_string_info_case_insensitive(self) -> None:
        assert _coerce_level("info", default=logging.WARNING) == logging.INFO

    def test_string_debug_uppercase(self) -> None:
        assert _coerce_level("DEBUG", default=logging.WARNING) == logging.DEBUG

    def test_string_warning_mixed_case(self) -> None:
        assert _coerce_level("Warning", default=logging.DEBUG) == logging.WARNING

    def test_string_error(self) -> None:
        assert _coerce_level("ERROR", default=logging.INFO) == logging.ERROR

    def test_string_critical(self) -> None:
        assert _coerce_level("CRITICAL", default=logging.INFO) == logging.CRITICAL

    def test_none_returns_default(self) -> None:
        assert _coerce_level(None, default=logging.WARNING) == logging.WARNING

    def test_none_returns_debug_default(self) -> None:
        assert _coerce_level(None, default=logging.DEBUG) == logging.DEBUG

    def test_int_passthrough(self) -> None:
        assert _coerce_level(logging.INFO, default=logging.DEBUG) == logging.INFO

    def test_int_passthrough_zero(self) -> None:
        assert _coerce_level(logging.NOTSET, default=logging.WARNING) == logging.NOTSET

    def test_unknown_name_returns_default(self) -> None:
        assert _coerce_level("NOTAREALEVEL", default=logging.WARNING) == logging.WARNING

    def test_empty_string_returns_default(self) -> None:
        # Empty string is not in the level names mapping.
        assert _coerce_level("", default=logging.INFO) == logging.INFO


# ──────────────────────────────────────────────────────────────────────────────
# 1. No log_file — single StreamHandler; full firehose at level on stdout
# ──────────────────────────────────────────────────────────────────────────────


class TestNoLogFile:
    def test_returns_stream_handler(self) -> None:
        stream = io.StringIO()
        h = configure_logging(level=logging.INFO, stream=stream)
        assert isinstance(h, logging.StreamHandler)
        # Must NOT be a RotatingFileHandler subclass.
        assert not isinstance(h, logging.handlers.RotatingFileHandler)

    def test_only_one_handler_installed(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream)
        root = logging.getLogger()
        assert len(root.handlers) == 1

    def test_debug_level_debug_message_reaches_stream(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        logging.getLogger("test.debug_firehose").debug("dbg-msg")
        output = stream.getvalue()
        assert "dbg-msg" in output

    def test_debug_level_info_message_reaches_stream(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        logging.getLogger("test.info_firehose").info("info-msg")
        assert "info-msg" in stream.getvalue()

    def test_debug_level_warning_message_reaches_stream(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        logging.getLogger("test.warn_firehose").warning("warn-msg")
        assert "warn-msg" in stream.getvalue()

    def test_debug_level_error_message_reaches_stream(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        logging.getLogger("test.error_firehose").error("err-msg")
        assert "err-msg" in stream.getvalue()

    def test_info_level_debug_messages_suppressed(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream)
        logging.getLogger("test.debug_suppressed").debug("hidden")
        assert "hidden" not in stream.getvalue()

    def test_info_level_info_message_reaches_stream(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream)
        logging.getLogger("test.info_present").info("shown")
        assert "shown" in stream.getvalue()

    def test_explicit_debug_logger_suppressed_at_info_console(self) -> None:
        # A library that force-sets its OWN logger to DEBUG (pier's setup_logger
        # does exactly this, per trial) must NOT flood an INFO console: the console
        # handler's level floor drops the DEBUG record even though the emitting
        # logger's explicit level would admit it. INFO+ from it still passes.
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream)
        noisy = logging.getLogger("thirdparty.forced_debug")
        noisy.setLevel(logging.DEBUG)   # library forces DEBUG on its own logger
        try:
            noisy.debug("flood")
            assert "flood" not in stream.getvalue()
            noisy.info("kept")
            assert "kept" in stream.getvalue()
        finally:
            noisy.setLevel(logging.NOTSET)


# ──────────────────────────────────────────────────────────────────────────────
# 2. With log_file — file is JSON firehose; console floored at WARNING
# ──────────────────────────────────────────────────────────────────────────────


class TestWithLogFile:
    def test_returns_rotating_file_handler(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        h = configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        assert isinstance(h, logging.handlers.RotatingFileHandler)

    def test_two_handlers_installed(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_info_message_lands_in_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        logging.getLogger("test.file_info").info("file-info-msg")
        # Close handlers so file is flushed.
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "file-info-msg" in content

    def test_warning_message_lands_in_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        logging.getLogger("test.file_warn").warning("file-warn-msg")
        for h in logging.getLogger().handlers:
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "file-warn-msg" in content

    def test_file_content_is_json_envelope(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, log_file=log_file,
            log_format="pretty",  # even with pretty on console, file must be JSON
            stream=stream,
        )
        logging.getLogger("test.file_json").info("json-check")
        for h in logging.getLogger().handlers:
            h.flush()
        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        assert lines, "expected at least one log line in file"
        record = json.loads(lines[0])  # must not raise
        assert record["message"] == "json-check"
        assert "ts" in record
        assert "level" in record

    def test_console_floored_at_warning_info_suppressed(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        logging.getLogger("test.stdout_floor_info").info("stdout-info-hidden")
        assert "stdout-info-hidden" not in stream.getvalue()

    def test_console_floored_at_warning_warning_visible(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        logging.getLogger("test.stdout_floor_warn").warning("stdout-warn-visible")
        assert "stdout-warn-visible" in stream.getvalue()

    def test_console_floored_at_warning_error_visible(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        logging.getLogger("test.stdout_floor_error").error("stdout-error-visible")
        assert "stdout-error-visible" in stream.getvalue()

    def test_debug_level_debug_reaches_file_not_stdout(self, tmp_path: Path) -> None:
        """Firehose is at the configured ``level`` (DEBUG) in the file,
        while stdout is still floored at WARNING."""
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, log_file=log_file, stream=stream)
        logging.getLogger("test.debug_file").debug("debug-only-file")
        for h in logging.getLogger().handlers:
            h.flush()
        # In file.
        content = log_file.read_text(encoding="utf-8")
        assert "debug-only-file" in content
        # Not on stdout.
        assert "debug-only-file" not in stream.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# 3. stdout_level override
# ──────────────────────────────────────────────────────────────────────────────


class TestStdoutLevel:
    def test_log_file_plus_stdout_info_mirrors_firehose(self, tmp_path: Path) -> None:
        """With log_file set + stdout_level="INFO", INFO reaches stdout too."""
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, log_file=log_file,
            stdout_level="INFO", stream=stream,
        )
        logging.getLogger("test.stdout_info_override").info("info-on-stdout")
        assert "info-on-stdout" in stream.getvalue()

    def test_log_file_plus_stdout_info_file_still_gets_info(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, log_file=log_file,
            stdout_level="INFO", stream=stream,
        )
        logging.getLogger("test.file_still_gets_info").info("both-sinks")
        for h in logging.getLogger().handlers:
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "both-sinks" in content

    def test_no_log_file_stdout_level_warning_floors_console(self) -> None:
        """Without log_file, stdout_level="WARNING" floors the console handler
        and no rotating file is created."""
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, stream=stream,
            stdout_level="WARNING",
        )
        root = logging.getLogger()
        # No RotatingFileHandler must be present.
        assert not any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in root.handlers
        )
        # INFO must not appear.
        logging.getLogger("test.no_file_stdout_floor_info").info("info-floored")
        assert "info-floored" not in stream.getvalue()

    def test_no_log_file_stdout_level_warning_warning_visible(self) -> None:
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, stream=stream,
            stdout_level="WARNING",
        )
        logging.getLogger("test.no_file_stdout_floor_warn").warning("warn-visible")
        assert "warn-visible" in stream.getvalue()

    def test_stdout_level_integer_accepted(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, log_file=log_file,
            stdout_level=logging.DEBUG, stream=stream,
        )
        logging.getLogger("test.stdout_int_level").debug("debug-via-int-stdout")
        # stdout_level=DEBUG means DEBUG reaches stdout even with log_file set.
        assert "debug-via-int-stdout" in stream.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Rotation bounds disk
# ──────────────────────────────────────────────────────────────────────────────


class TestRotation:
    def test_rotation_creates_backup_files(self, tmp_path: Path) -> None:
        """Filling the file past maxBytes causes rollover; backups are bounded."""
        log_file = tmp_path / "rotate.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.DEBUG,
            log_file=log_file,
            log_max_bytes=2000,
            log_backup_count=3,
            stream=stream,
        )
        logger = logging.getLogger("test.rotation")
        # Write enough data to force multiple rollovers.
        for i in range(200):
            logger.info("rotation test record %04d padding padding padding", i)

        # Flush and close handlers before inspecting files.
        for h in logging.getLogger().handlers:
            h.flush()
            if isinstance(h, logging.handlers.RotatingFileHandler):
                h.close()
                logging.getLogger().removeHandler(h)

        # Base file + up to 3 backups = at most 4 files total.
        log_files = list(tmp_path.glob("rotate.log*"))
        assert len(log_files) >= 2, "expected rollover to have produced backup files"
        assert len(log_files) <= 4, (
            f"expected at most 4 files (base+.1+.2+.3) but found {len(log_files)}: "
            f"{[f.name for f in log_files]}"
        )

    def test_rotation_max_bytes_respected(self, tmp_path: Path) -> None:
        """No single log file grows past log_max_bytes + one-record headroom."""
        log_file = tmp_path / "size.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.DEBUG,
            log_file=log_file,
            log_max_bytes=3000,
            log_backup_count=2,
            stream=stream,
        )
        logger = logging.getLogger("test.size_check")
        for i in range(300):
            logger.info("size test line %04d " + "x" * 50, i)

        for h in logging.getLogger().handlers:
            h.flush()
            if isinstance(h, logging.handlers.RotatingFileHandler):
                h.close()
                logging.getLogger().removeHandler(h)

        for f in tmp_path.glob("size.log*"):
            size = f.stat().st_size
            # Allow one record of headroom past the byte cap (stdlib rolls
            # *before* writing the next record, not mid-record).
            assert size < 3000 + 1000, (
                f"{f.name} grew to {size} bytes; expected < 4000"
            )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Parent dir auto-created
# ──────────────────────────────────────────────────────────────────────────────


class TestParentDirAutoCreated:
    def test_non_existent_subdir_is_created(self, tmp_path: Path) -> None:
        log_file = tmp_path / "subdir" / "nested" / "xrlenv.log"
        assert not log_file.parent.exists()
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        # The parent directory must now exist.
        assert log_file.parent.exists()

    def test_log_writes_succeed_after_mkdir(self, tmp_path: Path) -> None:
        log_file = tmp_path / "new_dir" / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        logging.getLogger("test.mkdir_write").info("after-mkdir-write")
        for h in logging.getLogger().handlers:
            h.flush()
        # The file must exist and contain the record.
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "after-mkdir-write" in content


# ──────────────────────────────────────────────────────────────────────────────
# 6. replace_handlers=True — re-invoking clears prior handlers
# ──────────────────────────────────────────────────────────────────────────────


class TestReplaceHandlers:
    def test_replace_true_clears_prior_handlers(self, tmp_path: Path) -> None:
        stream1 = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream1)
        stream2 = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream2, replace_handlers=True)
        root = logging.getLogger()
        # Only the second handler must remain.
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        # The handler must be on stream2, not stream1.
        assert handler.stream is stream2

    def test_replace_true_no_duplicate_file_handler(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        configure_logging(level=logging.INFO, log_file=log_file, stream=stream,
                          replace_handlers=True)
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1, (
            f"expected exactly 1 RotatingFileHandler after re-invoke; "
            f"got {len(file_handlers)}"
        )

    def test_replace_false_accumulates_handlers(self) -> None:
        stream1 = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream1, replace_handlers=True)
        stream2 = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream2, replace_handlers=False)
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_replace_true_message_not_duplicated(self, tmp_path: Path) -> None:
        """A record must appear exactly once on the stream after replace_handlers=True."""
        stream = io.StringIO()
        configure_logging(level=logging.INFO, stream=stream, replace_handlers=True)
        configure_logging(level=logging.INFO, stream=stream, replace_handlers=True)
        logging.getLogger("test.no_dup").info("unique-msg")
        count = stream.getvalue().count("unique-msg")
        assert count == 1, f"expected 1 occurrence of 'unique-msg'; got {count}"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Return value
# ──────────────────────────────────────────────────────────────────────────────


class TestReturnValue:
    def test_returns_stream_handler_when_no_log_file(self) -> None:
        stream = io.StringIO()
        primary = configure_logging(level=logging.INFO, stream=stream)
        assert isinstance(primary, logging.StreamHandler)
        assert not isinstance(primary, logging.handlers.RotatingFileHandler)

    def test_returns_rotating_file_handler_when_log_file_set(
        self, tmp_path: Path
    ) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        primary = configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        assert isinstance(primary, logging.handlers.RotatingFileHandler)

    def test_primary_handler_is_installed_on_root(self) -> None:
        stream = io.StringIO()
        primary = configure_logging(level=logging.INFO, stream=stream)
        root = logging.getLogger()
        assert primary in root.handlers

    def test_primary_file_handler_is_installed_on_root(self, tmp_path: Path) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        primary = configure_logging(level=logging.INFO, log_file=log_file, stream=stream)
        root = logging.getLogger()
        assert primary in root.handlers


# ──────────────────────────────────────────────────────────────────────────────
# Root-level gating: root.level must admit the most verbose handler
# ──────────────────────────────────────────────────────────────────────────────


class TestRootLevelGating:
    def test_root_level_set_to_configured_level_no_file(self) -> None:
        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)
        assert logging.getLogger().level == logging.DEBUG

    def test_root_level_set_to_min_of_file_and_console_levels(
        self, tmp_path: Path
    ) -> None:
        """With log_file=INFO and stdout_level=WARNING, the root must be INFO
        (the most verbose of the two) so INFO records reach the file handler."""
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO, log_file=log_file,
            stdout_level="WARNING", stream=stream,
        )
        # Root level must be INFO (not WARNING) so INFO reaches the file handler.
        assert logging.getLogger().level <= logging.INFO

    def test_root_level_debug_with_stdout_warning_admits_debug_to_file(
        self, tmp_path: Path
    ) -> None:
        log_file = tmp_path / "xrlenv.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.DEBUG, log_file=log_file,
            stdout_level="WARNING", stream=stream,
        )
        logging.getLogger("test.root_gates_debug").debug("debug-to-file")
        for h in logging.getLogger().handlers:
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "debug-to-file" in content
        # Still not on stdout.
        assert "debug-to-file" not in stream.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# CLI: the logging flags live on the `up` subparser (parsed AFTER `up`)
# ──────────────────────────────────────────────────────────────────────────────


class TestCliParser:
    def _parser(self) -> argparse.ArgumentParser:
        # Import inside to avoid paying CLI import cost for every test.
        from xrlenv.cli.__main__ import _build_parser
        return _build_parser()

    def test_log_file_flag_default_is_none(self) -> None:
        args = self._parser().parse_args(["up"])
        assert args.log_file is None

    def test_log_file_flag_accepts_value(self) -> None:
        args = self._parser().parse_args(["up", "--log-file", "/tmp/x.log"])
        assert args.log_file == "/tmp/x.log"

    def test_log_max_bytes_default(self) -> None:
        args = self._parser().parse_args(["up"])
        assert args.log_max_bytes == 50 * 1024 * 1024

    def test_log_max_bytes_accepts_value(self) -> None:
        args = self._parser().parse_args(["up", "--log-max-bytes", "1000000"])
        assert args.log_max_bytes == 1_000_000

    def test_log_backup_count_default(self) -> None:
        args = self._parser().parse_args(["up"])
        assert args.log_backup_count == 10

    def test_log_backup_count_accepts_value(self) -> None:
        args = self._parser().parse_args(["up", "--log-backup-count", "5"])
        assert args.log_backup_count == 5

    def test_stdout_log_level_default_is_none(self) -> None:
        args = self._parser().parse_args(["up"])
        assert args.stdout_log_level is None

    def test_stdout_log_level_accepts_value(self) -> None:
        args = self._parser().parse_args(
            ["up", "--stdout-log-level", "WARNING"]
        )
        assert args.stdout_log_level == "WARNING"

    def test_log_level_default_is_info(self) -> None:
        args = self._parser().parse_args(["up"])
        assert args.log_level == "INFO"

    def test_log_format_default_is_auto(self) -> None:
        args = self._parser().parse_args(["up"])
        assert args.log_format == "auto"

    def test_log_format_accepts_json(self) -> None:
        args = self._parser().parse_args(["up", "--log-format", "json"])
        assert args.log_format == "json"

    def test_log_format_accepts_pretty(self) -> None:
        args = self._parser().parse_args(["up", "--log-format", "pretty"])
        assert args.log_format == "pretty"

    def test_logging_flags_rejected_before_subcommand(self) -> None:
        # They are `up`-only now, not global — the old pre-subcommand
        # position must fail so the slurm scripts/docs stay honest.
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--log-file", "/tmp/x.log", "up"])
        with pytest.raises(SystemExit):
            self._parser().parse_args(["--log-level", "DEBUG", "up"])

    def test_non_up_subcommand_has_no_logging_attrs(self) -> None:
        # main() reads these via getattr(...) defaults for non-up commands.
        args = self._parser().parse_args(["nodes"])
        for attr in (
            "log_level", "log_format", "log_file",
            "log_max_bytes", "log_backup_count", "stdout_log_level",
        ):
            assert not hasattr(args, attr)

    def test_state_db_and_runs_root_remain_global(self) -> None:
        # Non-logging flags that many subcommands need stay on the top parser.
        args = self._parser().parse_args(
            ["--state-db", "/tmp/s.db", "--runs-root", "/tmp/r", "nodes"]
        )
        assert args.state_db == "/tmp/s.db"
        assert args.runs_root == "/tmp/r"
