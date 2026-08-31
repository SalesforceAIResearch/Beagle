"""Tests for the pure helper functions in xrlenv/backends/docker.py.

These do NOT require a live Docker daemon — they exercise only the
module-level helpers that have no I/O dependency.
"""

from __future__ import annotations

import io
import tarfile

import pytest
from xrlenv.backends.docker import (
    _default_stub_transport,
    _extract_single_file,
    _make_single_file_tar,
    _parse_stats,
)

# ── _make_single_file_tar / _extract_single_file round-trip ──────────────────

def test_make_and_extract_round_trip() -> None:
    payload = b"hello docker tar"
    archive = _make_single_file_tar("file.txt", payload)
    result = _extract_single_file(iter([archive]), "file.txt")
    assert result == payload


def test_make_single_file_tar_produces_valid_tar() -> None:
    data = b"content"
    archive = _make_single_file_tar("data.bin", data)
    buf = io.BytesIO(archive)
    with tarfile.open(fileobj=buf, mode="r") as tf:
        members = tf.getmembers()
    assert len(members) == 1
    assert members[0].name == "data.bin"
    assert members[0].size == len(data)


def test_make_single_file_tar_empty_payload() -> None:
    archive = _make_single_file_tar("empty.txt", b"")
    result = _extract_single_file(iter([archive]), "empty.txt")
    assert result == b""


def test_extract_nonexistent_member_raises() -> None:
    archive = _make_single_file_tar("real.txt", b"x")
    with pytest.raises(KeyError):
        _extract_single_file(iter([archive]), "missing.txt")


def test_extract_accepts_chunked_iterator() -> None:
    """_extract_single_file must accept any iterator of bytes, not just a list."""
    payload = b"chunked data"
    archive = _make_single_file_tar("c.txt", payload)
    # Simulate multi-chunk delivery the same way docker-py does.
    chunk1 = archive[:len(archive) // 2]
    chunk2 = archive[len(archive) // 2:]
    result = _extract_single_file(iter([chunk1, chunk2]), "c.txt")
    assert result == payload


# ── _parse_stats ──────────────────────────────────────────────────────────────

_REALISTIC_STATS: dict = {
    "cpu_stats": {
        "cpu_usage": {"total_usage": 2_000_000_000},  # 2 seconds
    },
    "memory_stats": {"usage": 134_217_728},  # 128 MiB
    "networks": {
        "eth0": {"rx_bytes": 1024, "tx_bytes": 512},
        "eth1": {"rx_bytes": 256, "tx_bytes": 128},
    },
}


def test_parse_stats_realistic() -> None:
    usage = _parse_stats(_REALISTIC_STATS)
    assert usage.cpu_seconds == pytest.approx(2.0)
    assert usage.rss_bytes == 134_217_728
    assert usage.rx_bytes == 1280
    assert usage.tx_bytes == 640
    assert usage.disk_bytes == 0


def test_parse_stats_missing_all_fields() -> None:
    usage = _parse_stats({})
    assert usage.cpu_seconds == 0.0
    assert usage.rss_bytes == 0
    assert usage.rx_bytes == 0
    assert usage.tx_bytes == 0


def test_parse_stats_no_networks_key() -> None:
    raw = {
        "cpu_stats": {"cpu_usage": {"total_usage": 500_000_000}},
        "memory_stats": {"usage": 1024},
        # no "networks" key
    }
    usage = _parse_stats(raw)
    assert usage.rx_bytes == 0
    assert usage.tx_bytes == 0
    assert usage.cpu_seconds == pytest.approx(0.5)


def test_parse_stats_networks_is_none() -> None:
    raw = {
        "cpu_stats": {"cpu_usage": {"total_usage": 0}},
        "memory_stats": {"usage": 0},
        "networks": None,
    }
    usage = _parse_stats(raw)
    assert usage.rx_bytes == 0
    assert usage.tx_bytes == 0


# ── _default_stub_transport ───────────────────────────────────────────────────

def test_default_stub_transport_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert _default_stub_transport() == "uds"


def test_default_stub_transport_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert _default_stub_transport() == "tcp"


def test_default_stub_transport_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert _default_stub_transport() == "tcp"
