"""Unit tests for configure_logging(queue_file_writes=True) — 2026-08-21.

The new queue_file_writes flag routes file + console handlers behind a
background QueueListener thread so the asyncio event loop never blocks on
a filesystem write (Lustre hiccup scenario from the 2026-08-21 outage).
"""

from __future__ import annotations

import io
import logging
import logging.handlers
from collections.abc import Generator
from pathlib import Path

import pytest
import xrlenv.observability.logging as _log_module
from xrlenv.observability.logging import _stop_listener, configure_logging

# ──────────────────────────────────────────────────────────────────────────────
# Fixture: restore root logger + stop any listener after every test
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_root_logger_and_listener() -> Generator[None, None, None]:
    """Save root logger state + listener state; restore both after each test.

    This prevents listener threads from one test leaking into the next and
    ensures the global _LISTENER + _ATEXIT_REGISTERED flags are cleaned up.
    """
    root = logging.getLogger()
    prior_level = root.level
    prior_handlers = list(root.handlers)

    yield

    # Stop any listener the SUT installed.
    _stop_listener()

    # Close and remove any handlers the test added.
    for h in list(root.handlers):
        if h not in prior_handlers:
            h.flush()
            h.close()
            root.removeHandler(h)

    root.handlers = prior_handlers
    root.setLevel(prior_level)

    # Reset module-level atexit flag so tests are independent.
    _log_module._ATEXIT_REGISTERED = False


# ──────────────────────────────────────────────────────────────────────────────
# 1. Records reach the file after listener flush
# ──────────────────────────────────────────────────────────────────────────────


class TestQueueFileWritesRecordsReachFile:
    def test_info_record_reaches_file_after_stop_listener(
        self, tmp_path: Path,
    ) -> None:
        """A record logged after configure_logging(queue_file_writes=True)
        must appear in the log file once the listener is stopped/flushed."""
        log_file = tmp_path / "queued.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        logging.getLogger("test.queue.info").info("queue-info-message")
        # Flush the listener; records buffered in the queue must drain.
        _stop_listener()
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "queue-info-message" in content

    def test_warning_record_reaches_file_after_stop_listener(
        self, tmp_path: Path,
    ) -> None:
        log_file = tmp_path / "queued_warn.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        logging.getLogger("test.queue.warn").warning("queue-warn-message")
        _stop_listener()
        content = log_file.read_text(encoding="utf-8")
        assert "queue-warn-message" in content

    def test_multiple_records_all_reach_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "multi.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.DEBUG,
            log_file=log_file,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        logger = logging.getLogger("test.queue.multi")
        for i in range(5):
            logger.info("record-%d", i)
        _stop_listener()
        content = log_file.read_text(encoding="utf-8")
        for i in range(5):
            assert f"record-{i}" in content


# ──────────────────────────────────────────────────────────────────────────────
# 2. Root carries a QueueHandler (not RotatingFileHandler directly)
# ──────────────────────────────────────────────────────────────────────────────


class TestRootHasQueueHandler:
    def test_root_carries_queue_handler_in_queue_mode(
        self, tmp_path: Path,
    ) -> None:
        log_file = tmp_path / "q.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        root = logging.getLogger()
        assert any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers), (
            f"Expected QueueHandler on root logger in queue_file_writes mode; "
            f"got handlers: {root.handlers}"
        )

    def test_root_does_not_carry_rotating_file_handler_directly(
        self, tmp_path: Path,
    ) -> None:
        """The RotatingFileHandler must be behind the QueueListener, not
        directly on the root logger."""
        log_file = tmp_path / "q2.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        root = logging.getLogger()
        assert not any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in root.handlers
        ), (
            "RotatingFileHandler must not be directly on root in queue mode; "
            f"got: {root.handlers}"
        )

    def test_root_carries_rotating_file_handler_when_queue_mode_off(
        self, tmp_path: Path,
    ) -> None:
        """Regression guard: without queue_file_writes, the old behaviour
        (direct RotatingFileHandler on root) must be preserved."""
        log_file = tmp_path / "direct.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=False,
            replace_handlers=True,
        )
        root = logging.getLogger()
        assert any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in root.handlers
        ), "RotatingFileHandler expected directly on root when queue_file_writes=False"

    def test_no_queue_handler_in_default_mode_without_log_file(self) -> None:
        """Default (no log_file, no queue_file_writes) — no QueueHandler."""
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            stream=stream,
            replace_handlers=True,
        )
        root = logging.getLogger()
        assert not any(
            isinstance(h, logging.handlers.QueueHandler) for h in root.handlers
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Re-invocation stops prior listener (no thread leak / no double-write)
# ──────────────────────────────────────────────────────────────────────────────


class TestReInvokeStopsPriorListener:
    def test_reinvoke_stops_prior_listener(self, tmp_path: Path) -> None:
        """Re-invoking configure_logging(queue_file_writes=True) must stop the
        old listener before installing a new one. After re-invocation, exactly
        one QueueHandler is on root."""
        log_file1 = tmp_path / "first.log"
        log_file2 = tmp_path / "second.log"
        stream = io.StringIO()

        configure_logging(
            level=logging.INFO,
            log_file=log_file1,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        first_listener = _log_module._LISTENER

        configure_logging(
            level=logging.INFO,
            log_file=log_file2,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        second_listener = _log_module._LISTENER

        # The first listener should have been stopped; it is a different object.
        assert second_listener is not first_listener

        root = logging.getLogger()
        queue_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.QueueHandler)
        ]
        assert len(queue_handlers) == 1, (
            f"Expected exactly 1 QueueHandler after re-invoke; "
            f"got {len(queue_handlers)}: {queue_handlers}"
        )

    def test_reinvoke_no_double_write_to_file(self, tmp_path: Path) -> None:
        """A record emitted after re-invocation must appear exactly once in
        the second log file (not duplicated from a leaked prior listener)."""
        log_file1 = tmp_path / "sink1.log"
        log_file2 = tmp_path / "sink2.log"
        stream = io.StringIO()

        configure_logging(
            level=logging.INFO,
            log_file=log_file1,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )

        configure_logging(
            level=logging.INFO,
            log_file=log_file2,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )

        logging.getLogger("test.no_double").info("unique-record")
        _stop_listener()

        content2 = log_file2.read_text(encoding="utf-8") if log_file2.exists() else ""
        count = content2.count("unique-record")
        assert count == 1, (
            f"Expected exactly 1 occurrence of 'unique-record' in second file; "
            f"got {count}"
        )

    def test_queue_mode_to_direct_mode_stops_listener(
        self, tmp_path: Path,
    ) -> None:
        """Switching from queue mode to direct mode (queue_file_writes=False)
        must stop the prior listener so the old thread doesn't linger."""
        log_file = tmp_path / "switch.log"
        stream = io.StringIO()

        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        assert _log_module._LISTENER is not None

        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=False,
            replace_handlers=True,
        )
        # After switching to direct mode, _LISTENER must be None.
        assert _log_module._LISTENER is None

    def test_direct_mode_root_has_rotating_handler_directly(
        self, tmp_path: Path,
    ) -> None:
        """Regression guard: queue=False preserves existing behaviour (direct
        RotatingFileHandler on root)."""
        log_file = tmp_path / "direct2.log"
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            log_file=log_file,
            stream=stream,
            queue_file_writes=False,
            replace_handlers=True,
        )
        root = logging.getLogger()
        assert any(
            isinstance(h, logging.handlers.RotatingFileHandler)
            for h in root.handlers
        )
        assert not any(
            isinstance(h, logging.handlers.QueueHandler)
            for h in root.handlers
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. queue_file_writes=True without log_file is a silent no-op (direct path)
# ──────────────────────────────────────────────────────────────────────────────


class TestQueueModeWithoutLogFile:
    def test_no_queue_handler_when_log_file_is_none(self) -> None:
        """queue_file_writes=True without a log_file falls back to the direct
        path — there is nothing to queue."""
        stream = io.StringIO()
        configure_logging(
            level=logging.INFO,
            stream=stream,
            queue_file_writes=True,
            replace_handlers=True,
        )
        root = logging.getLogger()
        assert not any(
            isinstance(h, logging.handlers.QueueHandler)
            for h in root.handlers
        )
        assert _log_module._LISTENER is None
