"""Multi-user (Slice C) — `xrlenv fairshare show` / `set` CLI commands.

``cmd_fairshare_show`` / ``cmd_fairshare_set`` open the state.db at ``state_db``
directly (no control plane, no port). Tests seed a real ``SqliteStateStore``
under ``tmp_path``, close it, then drive the command with an ``io.StringIO``
``out`` and re-open the store to assert the persisted policy.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from xrlenv.cli.commands import cmd_fairshare_set, cmd_fairshare_show
from xrlenv.control.state import SqliteStateStore


def _seed_empty_db(tmp_path: Path) -> Path:
    """Create (and close) an empty state.db so the cmd can open it."""
    db = tmp_path / "state.db"
    SqliteStateStore(db).close()
    return db


def test_show_on_disabled_policy_prints_disabled(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_show(state_db=db, out=out)
    assert rc == 0
    assert "DISABLED" in out.getvalue()


def test_set_default_cap_enables_and_persists(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(default_cap=4, state_db=db, out=out)
    assert rc == 0
    body = out.getvalue()
    assert "ENABLED" in body
    assert "default_cap=4" in body
    # Re-open the store: the change is durable.
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.capacity_basis == 4


def test_show_reflects_persisted_capacity(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(default_cap=4, state_db=db, out=io.StringIO())
    out = io.StringIO()
    cmd_fairshare_show(state_db=db, out=out)
    body = out.getvalue()
    assert "ENABLED" in body
    assert "default_cap=4" in body


def test_show_cap_override_replaces_default_capacity(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(default_cap=4, state_db=db, out=io.StringIO())
    cmd_fairshare_set(owner="alice", cap=32, state_db=db, out=io.StringIO())
    out = io.StringIO()
    cmd_fairshare_show(state_db=db, out=out)
    body = out.getvalue()
    assert "alice" in body
    assert "effective_cap=32" in body
    assert "owner_cap=32" in body


def test_set_owner_uncap_persists(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    rc = cmd_fairshare_set(owner="bob", uncap=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.overrides["bob"].uncapped is True


def test_set_owner_block_persists(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    rc = cmd_fairshare_set(
        owner="alice", block=True, state_db=db, out=io.StringIO(),
    )
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.overrides["alice"].blocked is True


def test_block_and_unblock_together_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(
        owner="bob", block=True, unblock=True, state_db=db, out=out,
    )
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_default_cap_and_disable_together_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(default_cap=4, disable=True, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_disable_after_enable_disables(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(default_cap=4, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(disable=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.enabled is False


# ── main() dispatch regression (prod TypeError on args.state_db=None) ─────────
# The `fairshare show`/`set` subparsers redefine `--state-db` (so it can follow
# the subcommand) with a None default; argparse copies that None over the
# top-level default. `main()` resolved `Path(args.state_db)` unconditionally for
# every subcommand *before* dispatch, so `xrlenv fairshare show` (no --state-db)
# crashed with `TypeError: argument should be a str ... not 'NoneType'`. main()
# must now fall back to DEFAULT_STATE_DB instead.
def test_main_fairshare_show_without_state_db_flag_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.cli.__main__ as cli_module

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(cli_module, "DEFAULT_STATE_DB", db)
    rc = cli_module.main(["fairshare", "show"])
    assert rc == 0


def test_main_fairshare_set_without_state_db_flag_persists_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.cli.__main__ as cli_module

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(cli_module, "DEFAULT_STATE_DB", db)
    rc = cli_module.main(["fairshare", "set", "--owner", "zhiyuan", "--cap", "2"])
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.overrides["zhiyuan"].hard_cap == 2


def test_main_fairshare_set_uses_default_cap_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.cli.__main__ as cli_module

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(cli_module, "DEFAULT_STATE_DB", db)
    rc = cli_module.main(["fairshare", "set", "--default-cap", "8"])
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.capacity_basis == 8


def test_main_fairshare_set_rejects_removed_capacity_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xrlenv.cli.__main__ as cli_module

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(cli_module, "DEFAULT_STATE_DB", db)
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["fairshare", "set", "--capacity", "8"])
    assert exc.value.code == 2


def test_main_fairshare_set_rejects_removed_weight_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--weight was removed from CLI; argparse must reject it with exit code 2."""
    import xrlenv.cli.__main__ as cli_module

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(cli_module, "DEFAULT_STATE_DB", db)
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["fairshare", "set", "--owner", "alice", "--weight", "2.0"])
    assert exc.value.code == 2


def test_main_fairshare_set_rejects_removed_floor_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--floor was removed from public CLI; argparse must reject it with exit code 2."""
    import xrlenv.cli.__main__ as cli_module

    db = _seed_empty_db(tmp_path)
    monkeypatch.setattr(cli_module, "DEFAULT_STATE_DB", db)
    with pytest.raises(SystemExit) as exc:
        cli_module.main(["fairshare", "set", "--floor", "2"])
    assert exc.value.code == 2


# ── Range validation (audit M6) ───────────────────────────────────────────────


def test_set_default_cap_zero_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(default_cap=0, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_set_default_cap_negative_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(default_cap=-1, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_set_owner_cap_zero_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(owner="alice", cap=0, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_set_owner_cap_negative_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(owner="alice", cap=-5, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


# ── --recap clears cap + uncap + blocked ──────────────────────────────────────


def test_recap_after_cap_clears_owner_cap(tmp_path: Path) -> None:
    """--recap must clear hard_cap → owner returns to default cap."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", cap=32, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", recap=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides.get("alice")
    # Either the row is gone entirely or hard_cap is cleared.
    assert ov is None or ov.hard_cap is None


def test_recap_after_uncap_clears_uncapped_flag(tmp_path: Path) -> None:
    """--recap must clear uncapped so the owner falls back under default cap."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", uncap=True, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", recap=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides.get("alice")
    assert ov is None or ov.uncapped is False


def test_recap_after_block_clears_blocked_flag(tmp_path: Path) -> None:
    """--recap must clear blocked so the owner re-enters normal admission."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", block=True, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", recap=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides.get("alice")
    assert ov is None or ov.blocked is False


def test_recap_and_uncap_together_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(owner="alice", recap=True, uncap=True, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


# ── --unblock preserves cap/uncap state ───────────────────────────────────────


def test_unblock_preserves_owner_cap(tmp_path: Path) -> None:
    """--unblock only flips blocked; it must not touch hard_cap."""
    db = _seed_empty_db(tmp_path)
    # Set cap=16 and block at the same time (two separate calls).
    cmd_fairshare_set(owner="alice", cap=16, state_db=db, out=io.StringIO())
    cmd_fairshare_set(owner="alice", block=True, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", unblock=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides["alice"]
    assert ov.blocked is False
    assert ov.hard_cap == 16  # cap preserved after unblock


def test_unblock_preserves_uncapped_flag(tmp_path: Path) -> None:
    """--unblock only flips blocked; it must not clear the uncapped flag."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", uncap=True, state_db=db, out=io.StringIO())
    cmd_fairshare_set(owner="alice", block=True, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", unblock=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides["alice"]
    assert ov.blocked is False
    assert ov.uncapped is True  # uncapped preserved after unblock


# ── --cap clears uncapped; --uncap clears hard_cap and blocked ────────────────


def test_cap_after_uncap_clears_uncapped_flag(tmp_path: Path) -> None:
    """--owner alice --cap N must clear the uncapped flag (they are mutually exclusive)."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", uncap=True, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", cap=8, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides["alice"]
    assert ov.uncapped is False
    assert ov.hard_cap == 8


def test_uncap_clears_hard_cap(tmp_path: Path) -> None:
    """--owner alice --uncap must remove any existing owner cap."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", cap=8, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", uncap=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides["alice"]
    assert ov.hard_cap is None
    assert ov.uncapped is True


def test_uncap_clears_blocked_flag(tmp_path: Path) -> None:
    """--owner alice --uncap must clear blocked (can't be both bypassed and blocked)."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", block=True, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", uncap=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides["alice"]
    assert ov.blocked is False
    assert ov.uncapped is True


def test_block_does_not_clear_owner_cap(tmp_path: Path) -> None:
    """--block only sets blocked; it must not erase a previously-set owner cap."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(owner="alice", cap=32, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(owner="alice", block=True, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    ov = pol.overrides["alice"]
    assert ov.blocked is True
    assert ov.hard_cap == 32  # cap still present; block only stops new admissions


# ── default-cap persistence ───────────────────────────────────────────────────


def test_default_cap_update_is_idempotent(tmp_path: Path) -> None:
    """Calling --default-cap twice with the same value must leave exactly that value."""
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(default_cap=4, state_db=db, out=io.StringIO())
    rc = cmd_fairshare_set(default_cap=4, state_db=db, out=io.StringIO())
    assert rc == 0
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.capacity_basis == 4


def test_default_cap_can_be_updated_to_new_value(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    cmd_fairshare_set(default_cap=4, state_db=db, out=io.StringIO())
    cmd_fairshare_set(default_cap=16, state_db=db, out=io.StringIO())
    store = SqliteStateStore(db)
    try:
        pol = store.get_fairness_policy()
    finally:
        store.close()
    assert pol.capacity_basis == 16


# ── owner_flags without --owner validation ────────────────────────────────────


def test_cap_without_owner_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(cap=4, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_block_without_owner_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(block=True, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()


def test_uncap_without_owner_is_error(tmp_path: Path) -> None:
    db = _seed_empty_db(tmp_path)
    out = io.StringIO()
    rc = cmd_fairshare_set(uncap=True, state_db=db, out=out)
    assert rc == 2
    assert "error" in out.getvalue().lower()
