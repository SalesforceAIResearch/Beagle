"""Tests for the I/O-aware pull throttle env-var parsing helpers in cli.py.

Covers the glue introduced in ff354a1 that was not exercised by existing
test_node_cli.py tests:

- ``_pct_env`` (inner helper in ``_image_cache_config_from_env``) —
  validated by testing ``XRLENV_IO_UTIL_HIGH_PCT`` / ``XRLENV_IO_UTIL_LOW_PCT``
  through the public ``_image_cache_config_from_env``.
- ``_nonneg_int_env`` (module-level helper) — covers 0 as a valid value
  (disables the cap), negative, non-integer, unset.
- ``XRLENV_IO_THROTTLE`` bool parsing — all recognised falsey/truthy tokens
  plus an unrecognised token.
- ``io_util_low > io_util_high`` pre-check — the lone-low-override path and
  the both-override path.
- AIMD busy-signal: ``in_use`` busy AND ``io_saturated`` simultaneously;
  no sampler wired with throttle enabled.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest
from xrlenv.node.cli import _image_cache_config_from_env, _nonneg_int_env
from xrlenv.node.image_cache import ImageCacheConfig, ImageCacheManager

# ── helpers for env isolation ─────────────────────────────────────────────────

_ALL_OVERRIDE_VARS = (
    "XRLENV_PULL_CONCURRENCY",
    "XRLENV_PULL_CONCURRENCY_CEILING",
    "XRLENV_PULL_CONCURRENCY_INITIAL",
    "XRLENV_EVICT_THRESHOLD_CAP_GB",
    "XRLENV_EVICT_TARGET_CAP_GB",
    "XRLENV_IO_UTIL_HIGH_PCT",
    "XRLENV_IO_UTIL_LOW_PCT",
    "XRLENV_IO_THROTTLE",
)


def _clear_all(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in _ALL_OVERRIDE_VARS:
        monkeypatch.delenv(v, raising=False)


# ── _pct_env (tested via XRLENV_IO_UTIL_HIGH_PCT) ────────────────────────────


def test_pct_env_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_all(monkeypatch)
    cfg = _image_cache_config_from_env()
    # No overrides → None
    assert cfg is None


def test_pct_env_blank_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "   ")
    assert _image_cache_config_from_env() is None


def test_pct_env_non_numeric_warns_and_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "ninety")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        cfg = _image_cache_config_from_env()
    assert cfg is None  # only override was invalid → nothing to override
    assert any("XRLENV_IO_UTIL_HIGH_PCT" in r.message for r in caplog.records)


def test_pct_env_zero_out_of_range_warns_and_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Exactly 0 is out of range: constraint is 0 < pct <= 100."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "0")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        cfg = _image_cache_config_from_env()
    assert cfg is None
    assert any("XRLENV_IO_UTIL_HIGH_PCT" in r.message for r in caplog.records)


def test_pct_env_over_100_warns_and_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "101")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        cfg = _image_cache_config_from_env()
    assert cfg is None
    assert any("XRLENV_IO_UTIL_HIGH_PCT" in r.message for r in caplog.records)


def test_pct_env_100_is_valid_and_converts_to_1_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 % is the upper bound (allowed), converts to fraction 1.0."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "100")
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_util_high == pytest.approx(1.0)


def test_pct_env_valid_midrange_converts_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """85 % → 0.85 fraction stored in io_util_high."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "85")
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_util_high == pytest.approx(0.85)


def test_pct_env_low_valid_converts_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """60 % → 0.60 fraction stored in io_util_low."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_LOW_PCT", "60")
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_util_low == pytest.approx(0.60)


# ── XRLENV_IO_THROTTLE bool parsing ──────────────────────────────────────────


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "False", "OFF", "NO"])
def test_io_throttle_falsey_values_disable_throttle(
    monkeypatch: pytest.MonkeyPatch, val: str,
) -> None:
    """All recognised falsey tokens set io_throttle_enabled=False."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_THROTTLE", val)
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_throttle_enabled is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "True", "ON", "YES"])
def test_io_throttle_truthy_values_keep_throttle_enabled(
    monkeypatch: pytest.MonkeyPatch, val: str,
) -> None:
    """All recognised truthy tokens keep io_throttle_enabled=True (the default).

    Because the only override here is io_throttle_enabled=True (same as
    the library default), the helper either returns None or a config with
    io_throttle_enabled=True. Both are acceptable; what must NOT happen is
    that io_throttle_enabled ends up False."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_THROTTLE", val)
    cfg = _image_cache_config_from_env()
    if cfg is not None:
        assert cfg.io_throttle_enabled is True


def test_io_throttle_unrecognised_warns_and_leaves_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unrecognised value warns and defaults to enabled (fail-safe)."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_THROTTLE", "maybe")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        cfg = _image_cache_config_from_env()
    assert any("XRLENV_IO_THROTTLE" in r.message for r in caplog.records)
    # The unrecognised token must not disable throttling.
    if cfg is not None:
        assert cfg.io_throttle_enabled is True


def test_io_throttle_blank_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank value is silently ignored (treated as unset)."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_THROTTLE", "  ")
    cfg = _image_cache_config_from_env()
    # Blank → no override applied; no other vars set → None.
    assert cfg is None


# ── io_util_low > io_util_high pre-check ─────────────────────────────────────


def test_low_only_override_exceeding_default_high_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If only XRLENV_IO_UTIL_LOW_PCT is set and its value exceeds the
    *default* io_util_high (0.90), the low override is dropped with a
    warning so the ImageCacheConfig constructor does not raise.

    Edge case: a low-only override of e.g. 95% would produce low=0.95 >
    high=0.90 (default) — an invalid combination. The pre-check must catch
    this and drop the low override."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_LOW_PCT", "95")  # 0.95 > default high 0.90
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        cfg = _image_cache_config_from_env()
    # The invalid low is dropped; nothing else overridden → None returned.
    assert cfg is None
    assert any("LOW" in r.message.upper() for r in caplog.records)


def test_both_overrides_where_low_exceeds_high_drops_low(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When both HIGH and LOW are set and LOW > HIGH, LOW is dropped with
    a warning; HIGH alone is applied."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "70")  # 0.70
    monkeypatch.setenv("XRLENV_IO_UTIL_LOW_PCT", "80")   # 0.80 > 0.70 → invalid
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_util_high == pytest.approx(0.70)
    # low was dropped — must still equal the library default (0.70 default),
    # but more importantly must not exceed io_util_high.
    assert cfg.io_util_low <= cfg.io_util_high
    assert any("LOW" in r.message.upper() for r in caplog.records)


def test_valid_high_and_low_pair_both_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LOW < HIGH both overrides survive and reach ImageCacheConfig."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "85")  # 0.85
    monkeypatch.setenv("XRLENV_IO_UTIL_LOW_PCT", "65")   # 0.65 < 0.85 → valid
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_util_high == pytest.approx(0.85)
    assert cfg.io_util_low == pytest.approx(0.65)


def test_low_equal_to_high_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """low == high is the edge of the valid range (DiskIoSampler allows
    0 < low <= high <= 1). The pre-check must not drop it."""
    _clear_all(monkeypatch)
    monkeypatch.setenv("XRLENV_IO_UTIL_HIGH_PCT", "80")
    monkeypatch.setenv("XRLENV_IO_UTIL_LOW_PCT", "80")
    cfg = _image_cache_config_from_env()
    assert cfg is not None
    assert cfg.io_util_high == pytest.approx(0.80)
    assert cfg.io_util_low == pytest.approx(0.80)


# ── _nonneg_int_env ───────────────────────────────────────────────────────────


def test_nonneg_int_env_unset_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XRLENV_TEST_NN", raising=False)
    assert _nonneg_int_env("XRLENV_TEST_NN") is None


def test_nonneg_int_env_blank_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XRLENV_TEST_NN", "  ")
    assert _nonneg_int_env("XRLENV_TEST_NN") is None


def test_nonneg_int_env_non_integer_warns_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("XRLENV_TEST_NN", "four")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        result = _nonneg_int_env("XRLENV_TEST_NN")
    assert result is None
    assert any("XRLENV_TEST_NN" in r.message for r in caplog.records)


def test_nonneg_int_env_negative_warns_and_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("XRLENV_TEST_NN", "-1")
    with caplog.at_level("WARNING", logger="xrlenv.node"):
        result = _nonneg_int_env("XRLENV_TEST_NN")
    assert result is None
    assert any("XRLENV_TEST_NN" in r.message for r in caplog.records)


def test_nonneg_int_env_zero_is_valid_and_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 is a meaningful value that disables the concurrency cap — it must
    not be rejected the way _positive_int_env (used for PULL_CONCURRENCY)
    rejects 0."""
    monkeypatch.setenv("XRLENV_TEST_NN", "0")
    assert _nonneg_int_env("XRLENV_TEST_NN") == 0


def test_nonneg_int_env_positive_value_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XRLENV_TEST_NN", "4")
    assert _nonneg_int_env("XRLENV_TEST_NN") == 4


def test_nonneg_int_env_destroy_concurrency_zero_disables_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XRLENV_RAW_DESTROY_CONCURRENCY=0 must return 0 (cap disabled), not
    None (use default). This is the distinguishing contract vs the positive-
    only _positive_int_env used for pull concurrency."""
    monkeypatch.setenv("XRLENV_RAW_DESTROY_CONCURRENCY", "0")
    assert _nonneg_int_env("XRLENV_RAW_DESTROY_CONCURRENCY") == 0
    monkeypatch.delenv("XRLENV_RAW_DESTROY_CONCURRENCY")


# ── raw concurrency env → NodeAgentConfig field mapping ───────────────────────


def test_sysbox_create_concurrency_env_maps_to_config_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY reaches NodeAgentConfig via the
    override mapping, and the target field exists + accepts the value."""
    from xrlenv.node.agent import NodeAgentConfig
    from xrlenv.node.cli import _raw_concurrency_overrides

    monkeypatch.setenv("XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY", "2")
    overrides = _raw_concurrency_overrides()
    assert overrides["raw_sysbox_create_concurrency"] == 2
    # The field name is real — NodeAgentConfig accepts the kwarg.
    cfg = NodeAgentConfig(node_id="n1", backends={}, **overrides)
    assert cfg.raw_sysbox_create_concurrency == 2


def test_sysbox_create_concurrency_zero_disables_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 is meaningful (fall back to the general create cap), so it must map
    through as 0, not be dropped as 'unset'."""
    from xrlenv.node.cli import _raw_concurrency_overrides

    monkeypatch.setenv("XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY", "0")
    assert _raw_concurrency_overrides()["raw_sysbox_create_concurrency"] == 0


def test_sysbox_destroy_concurrency_env_maps_to_config_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XRLENV_RAW_SYSBOX_DESTROY_CONCURRENCY reaches NodeAgentConfig via the
    override mapping (symmetric with the create cap)."""
    from xrlenv.node.agent import NodeAgentConfig
    from xrlenv.node.cli import _raw_concurrency_overrides

    monkeypatch.setenv("XRLENV_RAW_SYSBOX_DESTROY_CONCURRENCY", "2")
    overrides = _raw_concurrency_overrides()
    assert overrides["raw_sysbox_destroy_concurrency"] == 2
    cfg = NodeAgentConfig(node_id="n1", backends={}, **overrides)
    assert cfg.raw_sysbox_destroy_concurrency == 2


def test_sysbox_create_concurrency_unset_leaves_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset → not in the overrides dict, so NodeAgentConfig's default (1) wins."""
    from xrlenv.node.agent import NodeAgentConfig
    from xrlenv.node.cli import _raw_concurrency_overrides

    monkeypatch.delenv("XRLENV_RAW_SYSBOX_CREATE_CONCURRENCY", raising=False)
    assert "raw_sysbox_create_concurrency" not in _raw_concurrency_overrides()
    assert NodeAgentConfig(node_id="n1", backends={}).raw_sysbox_create_concurrency == 1


def test_all_raw_concurrency_fields_exist_on_config() -> None:
    """Every env→field mapping targets a real NodeAgentConfig field — guards
    against a typo silently dropping an operator override."""
    from xrlenv.node.agent import NodeAgentConfig
    from xrlenv.node.cli import _RAW_CONCURRENCY_ENV_FIELDS

    fields = set(NodeAgentConfig.model_fields)  # pydantic BaseModel
    for env_name, field in _RAW_CONCURRENCY_ENV_FIELDS:
        assert field in fields, f"{env_name} maps to unknown field {field!r}"


# ── AIMD busy-signal interaction: in_use busy AND io_saturated simultaneously ─


class _FakeImageBackend:
    """Minimal SandboxBackend stand-in for AIMD-loop tests."""

    name = "fake-img"

    def __init__(self) -> None:
        from xrlenv.backends.base import SandboxCapabilities
        self.capabilities = SandboxCapabilities(
            supports_snapshot=False, supports_chainable_snapshot=False,
            live_state_captured=False, supports_port_forward=False,
            supports_gpu=False, isolation_class="container",
            fast_create_p50_ms=10,
        )

    async def list_images(self, *, include_shared_size: bool = False) -> list:
        return []

    async def free_disk_bytes(self) -> int:
        return 100 * 1024 ** 3

    async def total_disk_bytes(self) -> int:
        return 200 * 1024 ** 3


class _FixedSampler:
    """Duck-typed DiskIoSampler: always returns a fixed saturation value."""

    def __init__(self, *, saturated: bool) -> None:
        self._sat = saturated
        self.last_utilization = 0.99 if saturated else 0.10

    def saturated(self) -> bool:
        return self._sat


@pytest.mark.asyncio
async def test_aimd_both_in_use_busy_and_io_saturated_backs_off() -> None:
    """When both in_use > busy_threshold AND io_saturated=True, the OR
    combination must produce busy=True and the limit decays. Regression
    guard for the ``busy = in_use_total > threshold OR io_saturated``
    expression in run_pull_aimd_loop."""
    backend = _FakeImageBackend()  # type: ignore[arg-type]

    cfg = ImageCacheConfig(
        pull_concurrency=2,
        pull_concurrency_ceiling=64,
        pull_concurrency_initial=16,
        pull_aimd_interval_s=0.01,
        pull_busy_threshold=0,   # any in_use > 0 counts as busy
    )
    cache = ImageCacheManager(
        backend=backend,  # type: ignore[arg-type]
        config=cfg,
        disk_io_sampler=_FixedSampler(saturated=True),
    )
    # Simulate in_use > 0 by bumping the in_use counter directly.
    # The _in_use dict is keyed by image name; inject a fake entry.
    cache._in_use["some-image:v1"] = 3  # type: ignore[index]

    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.12)  # ~12 ticks: 16→8→4→2
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    # Must have reached the floor — both busy signals triggered backoff.
    assert cache.pull_concurrency_limit == 2


@pytest.mark.asyncio
async def test_aimd_no_sampler_wired_throttle_enabled_uses_in_use_only() -> None:
    """When disk_io_sampler=None (no sampler wired) but io_throttle_enabled=True,
    the loop must behave as if io_saturated=False — falls back to free-disk-only
    throttling. An idle node (in_use=0, ample disk) should ramp up."""
    cache = ImageCacheManager(
        backend=_FakeImageBackend(),  # type: ignore[arg-type]
        config=ImageCacheConfig(
            pull_concurrency=2,
            pull_concurrency_ceiling=64,
            pull_concurrency_initial=16,
            pull_aimd_interval_s=0.01,
            pull_busy_threshold=0,
            io_throttle_enabled=True,   # enabled, but no sampler
        ),
        disk_io_sampler=None,  # no sampler
    )
    task = asyncio.create_task(cache.run_pull_aimd_loop())
    try:
        await asyncio.sleep(0.12)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    # Idle + no IO signal → should have grown above initial 16.
    assert cache.pull_concurrency_limit > 16
