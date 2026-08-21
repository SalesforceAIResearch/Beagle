"""Two-user multi-tenancy smoke — REAL traffic against a live ``xrlenv up``.

No faked records. This drives the actual owner-stamping path end to end: it
mints per-user **consumer** tokens for alice and bob, then — as each user, over
a separate ``Client.grpc(...)`` connection — submits real work on your running
cluster. The control plane stamps every rollout / session with the owner read
off that user's verified token (never client-supplied), exactly as in
production.

It exercises **both** consumer use-cases:

1. **Raw containers** (the primary, heavily-developed path) — each user
   acquires N long-lived containers (``acquire_container`` → exec → kept alive),
   so you can see live owner-scoped sessions in the admin panel. Destroyed when
   you stop the smoke.
2. **Gym/step rollouts** (optional, ``--rollout-template``) — each user runs M
   short rollouts to completion, leaving owner-scoped sealed trajectories. Uses
   hello-shell-style ``{"cmd": ...}`` step actions; pass a template registered
   on your cluster.

One token per user: the same consumer token submits jobs AND opens the admin
panel (read-only, scoped to that user's own jobs) — no separate viewer token.
It also ensures an **operator** token exists (for the see-all view), written
into the cluster's secrets root so your running ``xrlenv up`` hot-reloads them
— no restart.

Prerequisites
-------------
- A running ``xrlenv up`` with >=1 node attached, reachable at
  ``--connect-host:--connect-port`` (gRPC, default 50051), and its admin panel
  bound **non-loopback** (so per-user auth engages — a loopback bind bypasses
  auth and every page renders as the admin view).
- Auth must be on (any token issued) for owner-stamping to engage. This smoke
  issues tokens into ``--secrets-root`` (default ``~/.xrlenv/secrets`` — point
  it at whatever ``xrlenv up`` loads).
- The ``--image`` (default ``busybox:latest``) present on / pullable by nodes.
- For the rollout path: ``--rollout-template`` registered on the cluster.

Run it
------
    .venv/bin/python tests/smoke/multi_tenancy/two_user_cluster_smoke.py \\
        --connect-host 127.0.0.1 --connect-port 50051 \\
        --admin-url http://127.0.0.1:8080 \\
        --jobs-per-user 2
    # add real rollout traffic too:
    #   --rollout-template hello-shell --rollouts-per-user 2

Then follow the printed checklist (log in to the admin panel as operator /
alice / bob and confirm each sees only their own work). Ctrl-C tears the live
sessions down.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

from xrlenv import Client
from xrlenv.control.security import (
    DEFAULT_SECRETS_ROOT,
    TokenStore,
    generate_token,
    token_full_id,
    token_sha256,
    write_secret_file,
    write_user_record,
)

_USERS = ("alice", "bob")
# Smoke-only plaintext cache (0600) so re-runs REUSE the same per-user tokens.
# users.json stores only hashes, so the plaintext can't be recovered from it.
_TOKEN_CACHE_NAME = ".smoke_multi_tenancy_tokens.json"


def _load_cache(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (json.dumps(cache, indent=2) + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def _prune_stale_consumer_rows(users_path: Path, keep_shas: set[str]) -> int:
    """Drop alice/bob consumer rows from users.json whose sha isn't in
    ``keep_shas`` — so re-running the smoke can't pile up duplicate per-user
    tokens. Leaves every other token (other owners, viewer/operator) untouched.
    Returns how many rows were removed."""
    if not users_path.exists():
        return 0
    try:
        rows = json.loads(users_path.read_text(encoding="utf-8")) or []
    except (OSError, json.JSONDecodeError):
        return 0
    kept = [
        r for r in rows
        if not (
            r.get("role") == "consumer"
            and r.get("owner_id") in _USERS
            and r.get("token_sha") not in keep_shas
        )
    ]
    removed = len(rows) - len(kept)
    if removed:
        fd = os.open(str(users_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, (json.dumps(kept, indent=2) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    return removed


def _issue_tokens(secrets_root: Path) -> dict[str, str]:
    """Idempotently provide one per-user consumer token per user + ensure an
    operator token exists.

    Re-running the smoke **reuses** each user's existing consumer token (stable
    ``token_id``; your browser login keeps working) instead of minting a fresh
    one every time, and **prunes** any stale duplicates from earlier runs.
    Reuse needs the plaintext — which the hashed ``users.json`` can't give back —
    so minted plaintexts are cached 0600 at
    ``<secrets_root>/.smoke_multi_tenancy_tokens.json`` (a smoke-only file). A
    cached token is reused only if it still verifies as that owner's consumer
    token (so a revoked / rotated / deleted token self-heals to a fresh mint).
    Written into ``secrets_root`` so a running ``xrlenv up`` hot-reloads them.

    One token per user: the consumer token submits jobs AND opens the admin
    panel (read-only, scoped to that user's own jobs)."""
    out: dict[str, str] = {}
    users = secrets_root / "users.json"
    cache_path = secrets_root / _TOKEN_CACHE_NAME
    cache = _load_cache(cache_path)
    store = TokenStore.load(secrets_root=secrets_root, env={})
    keep_shas: set[str] = set()
    for owner in _USERS:
        key = f"{owner}_consumer"
        cached = cache.get(key)
        ident = store.verify(cached) if cached else None
        if ident is not None and ident.role == "consumer" and ident.owner_id == owner:
            out[key] = cached  # reuse — same token_id across runs
        else:
            tok = generate_token("consumer")
            write_user_record(users, token=tok, role="consumer", owner_id=owner,
                               display_name=owner.title())
            out[key] = tok
            cache[key] = tok
        keep_shas.add(token_sha256(out[key]))
    removed = _prune_stale_consumer_rows(users, keep_shas)
    if removed:
        print(f"  (pruned {removed} stale per-user consumer row(s) from {users})",
              file=sys.stderr)
    _save_cache(cache_path, cache)
    op_file = secrets_root / "operator.token"
    if not op_file.exists():
        op = generate_token("operator")
        write_secret_file(op_file, op)
        out["operator"] = op
    else:
        out["operator"] = "(existing operator.token — reuse your own)"
    return out


async def _run_gym_rollouts(
    *, who: str, client: Client, template: str, n: int, steps: int,
) -> list[str]:
    """Real gym/step traffic: run ``n`` short rollouts to completion as ``who``.

    Uses hello-shell-style ``{"cmd": ...}`` actions. Returns rollout_ids. On a
    template/contract mismatch, logs and returns what it managed to start so the
    smoke still demonstrates the raw path.
    """
    ids: list[str] = []
    for r in range(n):
        try:
            session = await client.rollout(
                template=template,
                init={"max_steps": steps},
                task_key=f"{who}-rollout-{r}",
            )
            async with session:
                ids.append(session.rollout_id)
                await session.step({"cmd": "echo hello"})
                guard = 0
                while not session.done and guard < steps + 2:
                    await session.step({"cmd": f"echo step-{session.steps_taken}"})
                    guard += 1
            print(f"  [{who}] rollout {session.rollout_id} finished "
                  f"({session.steps_taken} steps)", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"  [{who}] rollout {r} skipped: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            break
    return ids


async def _acquire_raw_jobs(
    stack: contextlib.AsyncExitStack,
    *,
    who: str,
    client: Client,
    image: str,
    n: int,
) -> list[str]:
    """Real raw-container traffic: acquire ``n`` long-lived containers as ``who``
    (kept alive via ``stack`` for admin inspection). Returns rollout_ids."""
    ids: list[str] = []
    for i in range(n):
        session = await stack.enter_async_context(
            await client.acquire_container(
                image=image,
                command=["sleep", "infinity"],
                # task_key is a first-class admin-visible column + the reliable
                # per-job marker. Anti-affinity also spreads same-key acquires.
                task_key=f"{who}-task-{i}",
            )
        )
        # Prove the container is real + alive.
        result = await session.exec(["sh", "-c", "echo owner-check"], timeout_s=15.0)
        ok = result.exit_code == 0 and not result.timed_out
        ids.append(session.rollout_id)
        print(f"  [{who}] acquired rollout={session.rollout_id} "
              f"node={session.node_id} exec_ok={ok}", file=sys.stderr, flush=True)
    return ids


def _print_checklist(
    *, admin_url: str, tokens: dict[str, str],
    raw_ids: dict[str, list[str]], gym_ids: dict[str, list[str]],
) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print("  Two users submitted REAL jobs. Verify scoping in the admin panel.")
    print(bar)
    print(f"\n  Admin: {admin_url}/    (non-loopback bind required for auth; "
          "use 127.0.0.1 in the URL, not localhost)\n")
    print("  Basic-auth login — the USERNAME box is COSMETIC (type anything); "
          "the token is the PASSWORD. Each user opens the panel with the SAME "
          "consumer token they submit jobs with:\n")
    print(f"    operator (sees all) : password = {tokens['operator']}")
    print(f"    alice (own jobs)    : password = {tokens['alice_consumer']}")
    print(f"    bob (own jobs)      : password = {tokens['bob_consumer']}")
    print("\n  Raw containers (LIVE — destroyed on Ctrl-C); marked by task_key "
          "alice-task-* / bob-task-*:\n")
    print("   [ ] operator → /rollouts/raw lists alice's AND bob's sessions.")
    print("   [ ] alice    → /rollouts/raw lists ONLY alice-task-* .")
    print("   [ ] bob      → /rollouts/raw lists ONLY bob-task-* .")
    print("   [ ] /sandboxes scopes the same way (alice sees only hers).")
    if raw_ids.get("bob"):
        print(f"   [ ] cross-owner 404: as ALICE open "
              f"{admin_url}/raw-rollouts/{raw_ids['bob'][0]} → 404; your own "
              f"{admin_url}/raw-rollouts/{raw_ids['alice'][0]} → 200.")
    if any(gym_ids.values()):
        print("\n  Gym/step rollouts (completed — sealed trajectories):\n")
        print("   [ ] /rollouts/template — operator sees both users'; alice "
              "sees only alice-rollout-* .")
        if gym_ids.get("bob"):
            print(f"   [ ] cross-owner 404: as ALICE open "
                  f"{admin_url}/rollouts/{gym_ids['bob'][0]} → 404.")
    print("\n  Fair-share demo (separate terminal, against the SAME state.db "
          "your cluster uses):")
    print("    .venv/bin/python -m xrlenv.cli fairshare show")
    print("    .venv/bin/python -m xrlenv.cli fairshare set --default-cap 2   "
          "# then re-run with more --jobs-per-user and watch one user park")
    print("    .venv/bin/python -m xrlenv.cli fairshare set --owner bob --block\n")
    print(f"{bar}\n  Ctrl-C to destroy the live raw sessions and exit.\n{bar}\n",
          flush=True)


async def _amain(args: argparse.Namespace) -> int:
    secrets_root = (
        Path(args.secrets_root).expanduser()
        if args.secrets_root else DEFAULT_SECRETS_ROOT
    )
    tokens = _issue_tokens(secrets_root)
    print(f"issued per-user tokens into {secrets_root}/users.json "
          "(your running `xrlenv up` hot-reloads them):")
    for who in ("alice_consumer", "bob_consumer"):
        print(f"  {who:<16} token_id={token_full_id(tokens[who])}")

    # Let the control plane hot-reload the new tokens before we dial with them
    # (its mtime watch fires on the next RPC / poll).
    await asyncio.sleep(1.5)

    clients: dict[str, Client] = {
        who: Client.grpc(
            host=args.connect_host, port=args.connect_port,
            token=tokens[f"{who}_consumer"],
        )
        for who in _USERS
    }
    raw_ids: dict[str, list[str]] = {u: [] for u in _USERS}
    gym_ids: dict[str, list[str]] = {u: [] for u in _USERS}
    try:
        # 1. Gym/step rollouts (optional) — run to completion first.
        if args.rollout_template and args.rollouts_per_user > 0:
            print(f"\n[rollouts] template={args.rollout_template!r}",
                  file=sys.stderr, flush=True)
            for who in _USERS:
                gym_ids[who] = await _run_gym_rollouts(
                    who=who, client=clients[who], template=args.rollout_template,
                    n=args.rollouts_per_user, steps=args.rollout_steps,
                )
        # 2. Raw containers — acquire + keep alive for inspection.
        async with contextlib.AsyncExitStack() as stack:
            print("\n[raw] acquiring long-lived containers…",
                  file=sys.stderr, flush=True)
            for who in _USERS:
                raw_ids[who] = await _acquire_raw_jobs(
                    stack, who=who, client=clients[who],
                    image=args.image, n=args.jobs_per_user,
                )
            _print_checklist(admin_url=args.admin_url, tokens=tokens,
                             raw_ids=raw_ids, gym_ids=gym_ids)
            try:
                while True:
                    await asyncio.sleep(3600)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\n[smoke] tearing down live raw sessions…",
                      file=sys.stderr, flush=True)
    finally:
        for c in clients.values():
            with contextlib.suppress(Exception):
                await c.close()
    print("[smoke] done.", file=sys.stderr, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="two_user_cluster_smoke")
    p.add_argument("--connect-host", default="127.0.0.1",
                   help="Control-plane gRPC host (default 127.0.0.1).")
    p.add_argument("--connect-port", type=int, default=50051,
                   help="Control-plane gRPC port (default 50051).")
    p.add_argument("--admin-url", default="http://127.0.0.1:8080",
                   help="Admin panel base URL for the printed checklist.")
    p.add_argument("--secrets-root", default=None,
                   help="Where to write per-user tokens (default "
                        "~/.xrlenv/secrets — must match what `xrlenv up` loads).")
    p.add_argument("--jobs-per-user", type=int, default=2,
                   help="Long-lived raw containers to acquire per user "
                        "(default 2).")
    p.add_argument("--image", default="busybox:latest",
                   help="Image to acquire for raw jobs (present/pullable on "
                        "nodes).")
    p.add_argument("--rollout-template", default=None,
                   help="If set, also run real gym/step rollouts per user with "
                        "this template (hello-shell-style {cmd:...} actions).")
    p.add_argument("--rollouts-per-user", type=int, default=2,
                   help="Gym rollouts per user when --rollout-template is set.")
    p.add_argument("--rollout-steps", type=int, default=3,
                   help="max_steps per gym rollout (default 3).")
    args = p.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
