"""Phase-0 security primitives (spec 19).

Three concerns live here:

1. **Identity**: who is this caller? Phase 0 has three roles —
   ``node``, ``consumer``, ``operator`` — each with a single shared
   bearer token loaded from ``~/.xrlenv/secrets/<role>.token`` (per
   spec 19 §"Token lifecycle"). Phase 1 splits these into per-identity
   tokens with rotation + revocation; the slot is here so the gRPC
   interceptor's call site doesn't change.

2. **Scopes**: what may that identity do? Spec 19 §"API authz scopes"
   defines three coarse scopes — ``node.report``, ``consumer.rollout``,
   ``operator.admin`` — plus phase-1 sub-scopes
   (``port_forward.allow``, ``audit.read``, ``template.publish``)
   not yet enforced.

3. **Audit hints**: never log the bearer token itself; log the first
   6 chars of its SHA-256 instead so an operator can correlate
   ``auth.denied`` events with token rotations / leaks.

This module is intentionally small + sync: it's loaded at startup +
hit on every gRPC method call, so the hot path stays free of asyncio
overhead.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from xrlenv import paths

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Roles + scopes
# ──────────────────────────────────────────────────────────────────────────────


Role = Literal["node", "consumer", "operator", "viewer"]
Scope = Literal[
    "node.report", "consumer.rollout", "operator.admin", "admin.read",
]


# Spec 19 §"API authz scopes": each role's default scope. Phase 0 keeps the
# mapping flat — each role holds exactly one coarse scope. Phase 1's per-
# identity tokens carry an explicit scope set instead, and the role-to-scope
# fallback below becomes a backwards-compat default.
#
# B7.3 (P1.x slice 3) adds the ``viewer`` role + ``admin.read`` scope: a
# read-only operator-facing identity. The admin HTTP middleware gates write
# routes on ``operator`` and read routes on ``consumer | viewer | operator``
# (multi-user: a per-user ``consumer`` token opens the admin read-only, scoped
# to its own ``owner_id``). The gRPC scope dispatcher doesn't consult admin
# routes, so viewer tokens are silently inert outside the admin HTTP surface.
ROLE_DEFAULT_SCOPE: dict[Role, Scope] = {
    "node": "node.report",
    "consumer": "consumer.rollout",
    "operator": "operator.admin",
    "viewer": "admin.read",
}

# B7.3 (P1.x slice 3): generator-side prefix that travels with new admin
# tokens so an operator pasting one into chat / a runbook can tell at a glance
# whether the share grants reads or writes. The prefix is decorative — the
# server gates by ``identity.role``, not by the prefix — but it gives the
# share UX teeth without leaking any bearer material.
ROLE_TOKEN_PREFIX: dict[Role, str] = {
    "node": "",
    "consumer": "",
    "operator": "write_",
    "viewer": "read_",
}

# Spec 19 §"API authz scopes" + spec 05 / spec 21: gRPC method → required
# scope. The interceptor refuses any call to a listed method whose token
# doesn't carry the matching scope.
#   - NodeControl methods require ``node.report`` (spec 21 bidi stream).
#   - RolloutControl methods require ``consumer.rollout`` (spec 05 — added
#     when Client.grpc / RolloutControlServicer landed).
# Operator-facing gRPC (admin RPCs) lands later and registers under
# ``operator.admin`` the same way.
METHOD_REQUIRED_SCOPE: dict[str, Scope] = {
    "/xrlenv.node_control.v1.NodeControl/NodeControlStream":      "node.report",
    "/xrlenv.rollout_control.v1.RolloutControl/StartRollout":     "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/Step":             "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/Finish":           "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/Cancel":           "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/CancelGroup":      "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/Replay":           "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/Heartbeat":        "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/SetFinalReward":   "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/ListNodes":        "consumer.rollout",
    # Raw container session RPCs (case-2/3 docker-py drop-in). Same
    # ``consumer.rollout`` scope as the gym/step lifecycle — without these
    # entries ``required_scope_for_method`` returns ``None`` and *any* valid
    # token role (viewer / operator / node) would pass the scope check and be
    # able to acquire + drive raw sessions (audit M4).
    "/xrlenv.rollout_control.v1.RolloutControl/AcquireContainer":     "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/QueueStatus":          "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/ContainerExec":        "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/ContainerExecStream":  "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/DestroyContainer":     "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/ContainerPutArchive":  "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/ContainerGetArchive":  "consumer.rollout",
    "/xrlenv.rollout_control.v1.RolloutControl/ApplyEgress":          "consumer.rollout",
    # Operator-only: ``xrlenv images plan`` already authenticates with an
    # operator token (XRLENV_OPERATOR_TOKEN). It is a cluster-planning op, not
    # a consumer surface, so it requires ``operator.admin`` (audit M4 open Q).
    "/xrlenv.rollout_control.v1.RolloutControl/PlanImageDistribution": "operator.admin",
}


# ``$XRLENV_HOME/secrets`` (default ``~/.xrlenv/secrets``). Relocating
# ``XRLENV_HOME`` per checkout keeps a dev cluster's token store from sharing
# prod's on a common FSx home — see :mod:`xrlenv.paths`.
DEFAULT_SECRETS_ROOT = paths.secrets_root()


# ──────────────────────────────────────────────────────────────────────────────
# Identity
# ──────────────────────────────────────────────────────────────────────────────


class TokenIdentity(BaseModel):
    """One verified caller. The gRPC interceptor stashes this in the call
    context so handlers can read who's calling without re-verifying.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    scope: Scope
    owner_id: str = "default"
    """Tenant the caller acts as (multi-user, ``notes/multi-user-fairshare-plan.md``).
    Legacy shared role-tokens (one ``<role>.token`` file per role) map to
    ``"default"`` — back-compat for single-tenant deployments. Per-user
    tokens minted via ``xrlenv tokens issue --owner <id>`` carry a distinct
    ``owner_id`` that the control plane server-stamps onto every rollout the
    caller starts (Slice B), and the admin panel scopes reads by (Slice B)."""

    display_name: str | None = None
    """Optional human label for the owner, shown in ``xrlenv tokens list``.
    Cosmetic — never used for authz decisions, and not surfaced in the admin
    panel today."""

    digest_hint: str
    """First 6 chars of the bearer token's SHA-256 digest. Audit / structured
    logs use this in place of the raw token bytes (spec 19 §"Token lifecycle":
    *INFO logs include the first 6 chars of the token's SHA-256 digest as an
    identity hint, never the token bytes*)."""

    token_id: str
    """First 12 chars of the bearer token's SHA-256 digest. Stable identifier
    used by ``xrlenv tokens revoke <token-id>`` — long enough to be unique in
    operator practice, short enough for an operator to copy from the audit log
    or paste from a chat thread. The first 6 chars equal ``digest_hint`` so a
    revoke-by-prefix-of-6 still works for operators only quoting log lines."""


def token_digest_hint(token: str) -> str:
    """Stable opaque identifier safe to log: first 6 hex chars of SHA-256."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:6]


def generate_token(role: Role) -> str:
    """Emit a fresh bearer for ``role`` with the role-appropriate prefix.

    Admin-tier roles (``viewer`` / ``operator``) get a ``read_`` /
    ``write_`` prefix respectively so an operator pasting the token into
    chat or a runbook can see the privilege at a glance. ``node`` /
    ``consumer`` tokens stay unprefixed — they're systemd-installed or
    workflow-script-installed and rarely hand-shared.

    The prefix is generator-side only; ``TokenStore.verify`` matches the
    raw string. Existing unprefixed tokens issued before B7.3 keep
    working.
    """
    import secrets as _secrets
    return ROLE_TOKEN_PREFIX[role] + _secrets.token_urlsafe(32)


def token_full_id(token: str) -> str:
    """12-hex-char canonical token identifier used by ``tokens revoke``.

    The first 6 characters match :func:`token_digest_hint`, so an operator
    quoting a log-line ``digest_hint`` and passing it to
    ``xrlenv tokens revoke`` still resolves to a unique match in the small
    phase-0 role set.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def token_sha256(token: str) -> str:
    """Full 64-hex-char SHA-256 of a bearer token.

    Per-user tokens (``xrlenv tokens issue --owner``) are persisted **hashed**
    in ``users.json`` keyed by this full digest, so the plaintext bearer is
    never written to disk after the one-time issue print. :meth:`TokenStore.verify`
    re-hashes the presented bearer and looks it up here. The full digest (not
    the 12-char :func:`token_full_id`) is the at-rest key so two distinct
    bearers can't collide into one identity.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# TokenStore
# ──────────────────────────────────────────────────────────────────────────────


class TokenStore:
    """In-memory map from raw bearer token to :class:`TokenIdentity`.

    Built at startup from ``~/.xrlenv/secrets/<role>.token`` files
    (mode 0600) plus the ``XRLENV_{NODE,CONSUMER,OPERATOR,VIEWER}_TOKEN`` env
    vars per spec 09. Verifying a bearer is a hash-table lookup.

    Phase 0 deliberately keeps the lookup **constant-time-not-required**:
    the bearer-vs-stored comparison uses a plain dict get, *not*
    :py:func:`hmac.compare_digest`, because the threat model documents the
    primary risk as token *leakage* (visible in logs / fs / env), not
    timing oracle. Phase 1's per-identity tokens move to constant-time
    + rotation-aware checks.
    """

    def __init__(self) -> None:
        self._by_token: dict[str, TokenIdentity] = {}
        self._by_role: dict[Role, str] = {}
        # Multi-user (notes/multi-user-fairshare-plan.md, Slice A): per-user
        # tokens carrying a distinct ``owner_id``. Keyed by the bearer's full
        # SHA-256 (``token_sha256``) so the plaintext is never held in memory
        # or on disk after issue. ``verify`` re-hashes the presented bearer
        # and consults this map after the legacy raw-token miss. Loaded from
        # ``<secrets_root>/users.json``.
        self._by_token_sha: dict[str, TokenIdentity] = {}
        # B5.2 rotate-with-grace: a token whose role has just been
        # rotated stays accepted until its grace expiry (unix ts).
        # The verify path rejects entries whose expiry has elapsed
        # even before the next hot-reload picks up the on-disk state.
        self._grace_expires: dict[str, float] = {}
        # B5.2 revoke: token_ids the operator has explicitly killed.
        # ``verify`` returns ``None`` for any matching identity.
        # Persisted in ``<secrets_root>/revoked.json``.
        self._revoked_token_ids: set[str] = set()
        # Hot-reload bookkeeping. ``load(...)`` records the secrets dir
        # + env snapshot it loaded from, plus the per-secret-file mtime
        # observed at load time. ``maybe_reload()`` re-stats those files
        # on each call and rebuilds the store when any mtime advanced
        # (or a new role file appears), so ``xrlenv tokens issue ...``
        # while the control plane is already running takes effect on
        # the next RPC instead of requiring a restart. The grace + revoked
        # sidecars below get the same mtime watch so rotate / revoke take
        # effect just as quickly.
        self._secrets_root: Path | None = None
        self._env_snapshot: dict[str, str] | None = None
        self._file_mtimes: dict[Role, float | None] = {}
        self._grace_file_mtimes: dict[Role, float | None] = {}
        self._revoked_file_mtime: float | None = None
        self._users_file_mtime: float | None = None
        # Serialises the in-memory maps across the two threads that touch a
        # shared store: the gRPC interceptor (control-plane loop) and the
        # admin server (its own daemon thread — see ``AdminServer.start``).
        # Both call ``maybe_reload`` + ``verify``; ``maybe_reload`` rebuilds
        # the maps non-atomically (clear-then-repopulate), so without this a
        # ``verify`` on one thread could observe a half-cleared store mid
        # reload on the other and spuriously reject a valid token.
        self._reload_lock = threading.Lock()

    @property
    def is_empty(self) -> bool:
        return not self._by_token and not self._by_token_sha

    @property
    def known_roles(self) -> list[Role]:
        return list(self._by_role)

    def users(self) -> list[TokenIdentity]:
        """Per-user identities (those minted with ``--owner``), newest-load
        order. Excludes the legacy shared role-tokens (which live in
        :attr:`_by_role` and all carry ``owner_id="default"``)."""
        return list(self._by_token_sha.values())

    def add(self, role: Role, token: str) -> TokenIdentity:
        """Register ``token`` for ``role``. Replaces any existing token for
        that role; returns the resulting :class:`TokenIdentity`.

        Empty-string tokens are rejected — an empty bearer matches every
        request whose ``Authorization`` metadata is missing, which is a
        silent-bypass bug we never want.
        """
        if not token or not token.strip():
            raise ValueError(
                f"refusing to register empty token for role {role!r}"
            )
        prior_token = self._by_role.get(role)
        if prior_token is not None and prior_token != token:
            # Plain replace evicts the prior token unconditionally. Use
            # ``rotate`` if you want a grace window. Clear any stale
            # grace entry on the prior token so a later ``rotate`` of
            # the same role can re-add it cleanly.
            self._by_token.pop(prior_token, None)
            self._grace_expires.pop(prior_token, None)
        identity = TokenIdentity(
            role=role,
            scope=ROLE_DEFAULT_SCOPE[role],
            digest_hint=token_digest_hint(token),
            token_id=token_full_id(token),
        )
        self._by_token[token] = identity
        self._by_role[role] = token
        return identity

    def register_user(
        self,
        *,
        token_sha: str,
        role: Role,
        owner_id: str,
        display_name: str | None = None,
    ) -> TokenIdentity:
        """Register a per-user identity keyed by the bearer's full SHA-256.

        Used by the ``users.json`` load path (and directly by tests). The
        plaintext bearer never reaches this method — only its
        :func:`token_sha256` digest. ``token_id`` is the first 12 chars of
        that digest, so revocation by ``token_id`` works identically to the
        role-token path. An empty ``owner_id`` is rejected: an unset tenant
        would silently collapse into the ``"default"`` scope and see every
        other default-tenant job.
        """
        if not owner_id or not owner_id.strip():
            raise ValueError("refusing to register a user token with empty owner_id")
        if len(token_sha) != 64:
            raise ValueError(
                f"token_sha must be a 64-hex SHA-256 digest; got {len(token_sha)} chars",
            )
        identity = TokenIdentity(
            role=role,
            scope=ROLE_DEFAULT_SCOPE[role],
            owner_id=owner_id,
            display_name=display_name,
            digest_hint=token_sha[:6],
            token_id=token_sha[:12],
        )
        self._by_token_sha[token_sha] = identity
        return identity

    def rotate(
        self,
        role: Role,
        new_token: str,
        *,
        grace_s: float = 0.0,
    ) -> TokenIdentity:
        """Register ``new_token`` as the active token for ``role``.

        ``grace_s == 0`` (the default) is an immediate cutover — the prior
        token is dropped from ``_by_token`` synchronously, so any call
        carrying it 401s starting with the next RPC.

        ``grace_s > 0`` keeps the prior token valid until ``time.time() +
        grace_s``. Both tokens accept calls during the window; ``verify``
        rejects the old one once the wall-clock crosses the expiry. Use
        this only for deployment-rollover cases where the new token isn't
        yet propagated everywhere; the security-default is no grace.
        """
        if not new_token or not new_token.strip():
            raise ValueError(
                f"refusing to rotate role {role!r} to an empty token",
            )
        prior_token = self._by_role.get(role)
        identity = TokenIdentity(
            role=role,
            scope=ROLE_DEFAULT_SCOPE[role],
            digest_hint=token_digest_hint(new_token),
            token_id=token_full_id(new_token),
        )
        self._by_token[new_token] = identity
        self._by_role[role] = new_token
        if prior_token is not None and prior_token != new_token:
            if grace_s > 0:
                self._grace_expires[prior_token] = time.time() + grace_s
            else:
                self._by_token.pop(prior_token, None)
                self._grace_expires.pop(prior_token, None)
        return identity

    def revoke(self, token_id_or_prefix: str) -> str:
        """Mark a token revoked by its 12-char ``token_id`` (or a unique
        prefix ≥ 6 chars). Returns the full token_id that matched.

        Resolution is in-memory only — operators run the on-disk update
        via ``xrlenv tokens revoke``, which persists to ``revoked.json``
        and lets a running control plane pick the change up on the next
        ``maybe_reload``. Direct callers (tests, the CLI's own check
        path) get the same fail-loud behaviour: no match raises
        ``LookupError``; an ambiguous prefix raises ``ValueError``.
        """
        prefix = token_id_or_prefix.strip().lower()
        if len(prefix) < 6:
            raise ValueError(
                f"refusing to revoke {token_id_or_prefix!r}: prefix must be "
                "at least 6 hex chars to disambiguate",
            )
        matches = sorted({
            identity.token_id
            for identity in (*self._by_token.values(), *self._by_token_sha.values())
            if identity.token_id.startswith(prefix)
        })
        if not matches:
            raise LookupError(
                f"no known token matches token_id prefix {prefix!r}",
            )
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous token_id prefix {prefix!r} — matches "
                f"{matches!r}; pass more characters",
            )
        full_id = matches[0]
        self._revoked_token_ids.add(full_id)
        return full_id

    def verify(self, token: str | None) -> TokenIdentity | None:
        """Return the :class:`TokenIdentity` for ``token`` or ``None``.

        Refuses tokens whose ``token_id`` is in the revocation set, and
        tokens whose grace window has elapsed (the latter is also evicted
        from the in-memory map so subsequent calls are O(1) misses).

        Holds ``_reload_lock`` over the map reads so a concurrent
        ``maybe_reload`` on the other thread can't expose its half-cleared
        store to this lookup. The lock is uncontended on the hot path (a
        rebuild only happens when a secret file's mtime advanced).
        """
        if not token:
            return None
        with self._reload_lock:
            identity = self._by_token.get(token)
            if identity is None:
                # Multi-user fallback: per-user tokens are held hashed, so re-hash
                # the presented bearer and consult the SHA map. One extra SHA-256
                # per otherwise-miss; the common role-token path stays a plain get.
                sha_identity = self._by_token_sha.get(token_sha256(token))
                if sha_identity is None:
                    return None
                if sha_identity.token_id in self._revoked_token_ids:
                    return None
                return sha_identity
            if identity.token_id in self._revoked_token_ids:
                return None
            expires_at = self._grace_expires.get(token)
            if expires_at is not None and time.time() >= expires_at:
                # Grace elapsed — drop the entry so the next call short-circuits.
                self._by_token.pop(token, None)
                self._grace_expires.pop(token, None)
                return None
            return identity

    def maybe_reload(self) -> bool:
        """Re-read the secrets directory if any token file's mtime
        advanced (or a new role file appeared) since the last load.

        Idempotent + cheap: the hot path is one ``stat()`` per known
        role file plus an existence check for any role we haven't yet
        loaded. Returns ``True`` when the store was actually rebuilt
        — the auth interceptor uses that to log a single line per
        rotation rather than spamming on every RPC.

        Stores constructed without ``load(...)`` (test fakes calling
        ``add()`` directly) skip the reload silently — there's no
        secrets dir to watch.

        Runs under ``_reload_lock`` so the clear-then-repopulate rebuild is
        atomic with respect to a ``verify`` (or a second ``maybe_reload``) on
        the other thread; the inner ``_load_from`` does not re-take the lock.
        """
        with self._reload_lock:
            if self._secrets_root is None:
                return False
            changed = False
            roles: tuple[Role, ...] = ("node", "consumer", "operator", "viewer")
            for role in roles:
                path = self._secrets_root / f"{role}.token"
                current = _safe_mtime(path)
                previous = self._file_mtimes.get(role)
                if current != previous:
                    changed = True
                    break
                prev_grace_path = self._secrets_root / f"{role}.token.previous.json"
                current_grace = _safe_mtime(prev_grace_path)
                previous_grace = self._grace_file_mtimes.get(role)
                if current_grace != previous_grace:
                    changed = True
                    break
            if not changed:
                revoked_current = _safe_mtime(self._secrets_root / "revoked.json")
                if revoked_current != self._revoked_file_mtime:
                    changed = True
            if not changed:
                # Multi-user: pick up ``xrlenv tokens issue --owner`` writes to
                # users.json while the control plane runs (no restart needed).
                users_current = _safe_mtime(self._secrets_root / "users.json")
                if users_current != self._users_file_mtime:
                    changed = True
            if not changed:
                return False
            # Rebuild from scratch — simpler than incremental updates and
            # the volume is tiny (3 roles + ≤ a handful of revocations).
            self._by_token.clear()
            self._by_token_sha.clear()
            self._by_role.clear()
            self._grace_expires.clear()
            self._revoked_token_ids.clear()
            self._file_mtimes.clear()
            self._grace_file_mtimes.clear()
            env_map = self._env_snapshot if self._env_snapshot is not None else dict(os.environ)
            self._load_from(self._secrets_root, env_map)
            LOGGER.info(
                "TokenStore: hot-reloaded from %s; roles loaded=%s users=%d "
                "grace_tokens=%d revoked=%d",
                self._secrets_root, sorted(self._by_role), len(self._by_token_sha),
                len(self._grace_expires), len(self._revoked_token_ids),
            )
            return True

    @classmethod
    def load(
        cls,
        *,
        secrets_root: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> TokenStore:
        """Construct a :class:`TokenStore` from disk + environment.

        Roles populated when:

        - ``<secrets_root>/<role>.token`` exists and is mode 0600
          (warning logged + skipped if mode is more permissive); OR
        - ``XRLENV_{NODE,CONSUMER,OPERATOR,VIEWER}_TOKEN`` is set in ``env``.

        Env-var values take precedence so an operator can override the
        on-disk secret in tests / one-off runs without rewriting files.
        Returns an empty store if no source matched — the gRPC
        interceptor treats an empty store as "auth disabled" (the
        slice-1+ behaviour) so phase-0-only smoke tests keep working.

        Two sidecars are also picked up when present:

        - ``<role>.token.previous.json`` — JSON ``{"token": "...",
          "grace_until": "<iso8601>"}`` written by ``xrlenv tokens
          rotate --grace``. Entries past their ``grace_until`` are
          ignored (the file may linger on disk for operator visibility,
          but ``verify`` won't accept the token).
        - ``revoked.json`` — JSON list of ``{"token_id": "...",
          "revoked_at": "..."}`` appended by ``xrlenv tokens revoke``.
          Any matching identity is refused.
        """
        store = cls()
        secrets_root = secrets_root or DEFAULT_SECRETS_ROOT
        env_map = env if env is not None else dict(os.environ)
        store._load_from(secrets_root, env_map)
        # Snapshot the watch state so ``maybe_reload()`` can pick up
        # ``xrlenv tokens issue`` / ``rotate`` / ``revoke`` writes
        # without requiring a restart. Only files (not env vars) get
        # hot-reloaded — env-var-only tokens are usually one-off test
        # scaffolding where rotation isn't expected.
        store._secrets_root = secrets_root
        store._env_snapshot = dict(env_map) if env is not None else None
        return store

    def _load_from(self, secrets_root: Path, env_map: dict[str, str]) -> None:
        roles: tuple[Role, ...] = ("node", "consumer", "operator", "viewer")
        # Step 1: load revocations first so add() / rotate() see them.
        revoked_path = secrets_root / "revoked.json"
        self._revoked_token_ids = _load_revocations(revoked_path)
        self._revoked_file_mtime = _safe_mtime(revoked_path)
        # Step 2: active tokens.
        for role in roles:
            token = _load_role_token(role, secrets_root, env_map)
            if token is not None:
                self.add(role, token)
            self._file_mtimes[role] = _safe_mtime(
                secrets_root / f"{role}.token",
            )
        # Step 3: grace (previous) tokens. Skip if expired or if the
        # role has no active token to grace from (defensive — an
        # orphaned previous file shouldn't accidentally authorize).
        for role in roles:
            grace_path = secrets_root / f"{role}.token.previous.json"
            self._grace_file_mtimes[role] = _safe_mtime(grace_path)
            grace_record = _load_grace_record(grace_path)
            if grace_record is None:
                continue
            grace_token, grace_until = grace_record
            if time.time() >= grace_until:
                continue
            if role not in self._by_role:
                continue
            if self._by_role[role] == grace_token:
                continue
            identity = TokenIdentity(
                role=role,
                scope=ROLE_DEFAULT_SCOPE[role],
                digest_hint=token_digest_hint(grace_token),
                token_id=token_full_id(grace_token),
            )
            if identity.token_id in self._revoked_token_ids:
                continue
            self._by_token[grace_token] = identity
            self._grace_expires[grace_token] = grace_until
        # Step 4: per-user tokens (multi-user). Held hashed in users.json.
        # Load EVERY record regardless of revocation state — exactly like
        # role tokens, which stay in _by_token and are refused by verify()
        # rather than dropped at load. Keeping a revoked identity resolvable
        # is what makes `tokens revoke` idempotent: a second revoke of the
        # same token_id still finds the identity and returns success instead
        # of "no known token matches". verify() refuses any identity whose
        # token_id is in the revocation set (loaded in Step 1), so loading a
        # revoked record grants no access.
        users_path = secrets_root / "users.json"
        self._users_file_mtime = _safe_mtime(users_path)
        for record in _load_user_records(users_path):
            self.register_user(
                token_sha=record["token_sha"],
                role=record["role"],
                owner_id=record["owner_id"],
                display_name=record.get("display_name"),
            )

        # Step 5: collision reconciliation — per-user identity wins. A token
        # registered both as a shared legacy role-token (Step 2/3) AND as a
        # per-user token (Step 4) is almost always a footgun: a user's per-user
        # bearer leaked into the control plane as ``XRLENV_<ROLE>_TOKEN`` (or
        # ``<role>.token``) — e.g. an operator who is also a user launching
        # ``xrlenv up`` with their client ``.env`` loaded. The shared role-token
        # carries ``owner_id="default"`` and ``verify`` matches the legacy map
        # first, so that user would *silently* authenticate as the shared
        # "default" tenant, collapsing their isolation. Resolve it the intuitive
        # way: the more-specific per-user identity wins. Dropping the colliding
        # entry from the legacy map makes ``verify`` fall through to the per-user
        # identity (no change to its hot path), and we WARN so the operator can
        # clean up the control plane's environment.
        for raw_token in list(self._by_token):
            user_identity = self._by_token_sha.get(token_sha256(raw_token))
            if user_identity is None:
                continue
            legacy_role = self._by_token[raw_token].role
            self._by_token.pop(raw_token, None)
            self._grace_expires.pop(raw_token, None)
            LOGGER.warning(
                "security: a shared %s role-token (token_id=%s) has the same "
                "value as a per-user token (owner=%r); the per-user identity "
                "now wins, so this token authenticates as owner=%r — NOT the "
                "shared 'default' tenant. This usually means a client bearer "
                "(XRLENV_%s_TOKEN / %s.token) leaked into the control plane's "
                "environment; remove it there to silence this warning.",
                legacy_role, user_identity.token_id, user_identity.owner_id,
                user_identity.owner_id, legacy_role.upper(), legacy_role,
            )


def _safe_mtime(path: Path) -> float | None:
    """``path.stat().st_mtime`` if the path exists, else ``None``.

    Tolerant of races between ``exists()`` and ``stat()`` (file removed
    between checks) and of OS-level errors that shouldn't crash the
    interceptor's hot path.
    """
    try:
        return path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return None


def _load_role_token(
    role: Role,
    secrets_root: Path,
    env: dict[str, str],
) -> str | None:
    env_var = f"XRLENV_{role.upper()}_TOKEN"
    if env.get(env_var):
        return env[env_var]
    path = secrets_root / f"{role}.token"
    if not path.exists():
        return None
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return None
    if mode & 0o077:
        # Spec 19 §"Token lifecycle": *Mode 0600. Never logged at INFO*.
        # Refuse loose-permissions secret files to avoid silently using a
        # token an unrelated user could read. Operators get a clear
        # message from the warning + the role being unauthorized.
        LOGGER.warning(
            "security: refusing to load %s — file mode is %o, expected 0600",
            path, mode,
        )
        return None
    return path.read_text(encoding="utf-8").strip() or None


# ──────────────────────────────────────────────────────────────────────────────
# Method scope helpers
# ──────────────────────────────────────────────────────────────────────────────


def required_scope_for_method(method: str) -> Scope | None:
    """Return the spec-19 scope required to invoke ``method``.

    ``None`` means the method is unauthenticated (spec-19 phase-0 ships
    no such methods on the bidi gRPC, but the value is reserved for
    future health-probe RPCs).
    """
    return METHOD_REQUIRED_SCOPE.get(method)


def scope_satisfies(have: Scope, need: Scope) -> bool:
    """Whether a token holding ``have`` may invoke a method needing ``need``.

    Phase 0 is intentionally flat: each role holds one scope and
    ``operator.admin`` does **not** imply the others. A node token cannot
    call consumer methods even though node ops "feel" lower-trust — the
    threat model in spec 19 lists consumer-token reuse against the control
    plane as a real risk we explicitly defend against.
    """
    return have == need


def write_secret_file(path: Path, token: str) -> None:
    """Atomically write ``token`` to ``path`` with mode 0600.

    Parent directory is created with mode 0700 so a stale ``~/.xrlenv``
    with looser perms doesn't leak the secret. Raises if a non-token
    sibling file at ``path`` cannot be replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, path)


def write_grace_record(path: Path, token: str, grace_until: _dt.datetime) -> None:
    """Write a ``<role>.token.previous.json`` sidecar with mode 0600.

    The file is bytewise sensitive (it contains the prior bearer token)
    so it inherits the same 0600 perms + atomic-replace as the active
    token file. ``grace_until`` is serialized as an ISO-8601 string in
    UTC for operator readability.
    """
    payload = json.dumps({
        "token": token,
        "grace_until": grace_until.astimezone(_dt.UTC).isoformat(),
    })
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, path)


def append_revocation(path: Path, token_id: str) -> None:
    """Append ``token_id`` to ``revoked.json`` (a list of records).

    Creates the file if missing; otherwise reads the existing list and
    appends. ``mode 0644`` (world-readable) is fine here: revocation IDs
    are not secrets — they're SHA-256 prefixes — and operators sometimes
    grep this file from a non-owning login. The control plane only needs
    read access.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing: list[dict[str, str]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            LOGGER.warning(
                "revoked.json at %s is malformed; rewriting from scratch",
                path,
            )
            existing = []
    if any(r.get("token_id") == token_id for r in existing):
        return  # idempotent — re-revoking the same id is a no-op.
    existing.append({
        "token_id": token_id,
        "revoked_at": _dt.datetime.now(_dt.UTC).isoformat(),
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_grace_record(path: Path) -> tuple[str, float] | None:
    """Load ``(token, grace_until_unix_ts)`` from a previous-token sidecar.

    Returns ``None`` on missing file, malformed JSON, missing fields,
    bad timestamp, or loose file mode (the sidecar holds a still-valid
    bearer token, so we treat permissions the same way as the active
    token file).
    """
    if not path.exists():
        return None
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return None
    if mode & 0o077:
        LOGGER.warning(
            "security: refusing to load %s — file mode is %o, expected 0600",
            path, mode,
        )
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("security: failed to parse %s; ignoring", path)
        return None
    token = body.get("token") if isinstance(body, dict) else None
    grace_until_raw = body.get("grace_until") if isinstance(body, dict) else None
    if not isinstance(token, str) or not isinstance(grace_until_raw, str):
        return None
    try:
        grace_until_dt = _dt.datetime.fromisoformat(grace_until_raw)
    except ValueError:
        LOGGER.warning(
            "security: bad grace_until %r in %s; ignoring",
            grace_until_raw, path,
        )
        return None
    if grace_until_dt.tzinfo is None:
        grace_until_dt = grace_until_dt.replace(tzinfo=_dt.UTC)
    return token, grace_until_dt.timestamp()


def write_user_record(
    path: Path,
    *,
    token: str,
    role: Role,
    owner_id: str,
    display_name: str | None = None,
) -> str:
    """Append a per-user token record to ``users.json``; return its token_id.

    Stores the bearer's **SHA-256 digest**, never the plaintext — the issue
    CLI prints the raw token once and it is unrecoverable afterwards. The
    record carries ``owner_id`` (the tenant the bearer acts as), ``role``,
    an optional ``display_name``, and ``created_at``. Idempotent on the
    digest: re-writing the same bearer is a no-op.

    Mode 0600 + atomic replace — the file holds only hashes + metadata (no
    secret material), but it lives under ``~/.xrlenv/secrets`` so it inherits
    the tight perms by convention.
    """
    token_sha = token_sha256(token)
    token_id = token_sha[:12]
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            LOGGER.warning(
                "users.json at %s is malformed; rewriting from scratch", path,
            )
            existing = []
    if any(r.get("token_sha") == token_sha for r in existing):
        return token_id  # idempotent — same bearer already recorded.
    existing.append({
        "token_sha": token_sha,
        "token_id": token_id,
        "role": role,
        "owner_id": owner_id,
        "display_name": display_name,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (json.dumps(existing, indent=2) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return token_id


def _load_user_records(path: Path) -> list[dict[str, Any]]:
    """Read validated per-user token records from ``users.json``.

    Drops malformed entries (missing ``token_sha`` / ``owner_id``, unknown
    ``role``, wrong digest length) with a warning rather than failing the
    whole load — one bad row shouldn't lock every user out.
    """
    if not path.exists():
        return []
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("security: failed to parse %s; treating as empty", path)
        return []
    if not isinstance(body, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in body:
        if not isinstance(entry, dict):
            continue
        token_sha = entry.get("token_sha")
        owner_id = entry.get("owner_id")
        role = entry.get("role")
        if not isinstance(token_sha, str) or len(token_sha) != 64:
            continue
        if not isinstance(owner_id, str) or not owner_id:
            continue
        if role not in ROLE_DEFAULT_SCOPE:
            LOGGER.warning(
                "security: user record %s has unknown role %r; skipping",
                token_sha[:12], role,
            )
            continue
        out.append(entry)
    return out


def _load_revocations(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("security: failed to parse %s; treating as empty", path)
        return set()
    if not isinstance(body, list):
        return set()
    out: set[str] = set()
    for entry in body:
        if isinstance(entry, dict):
            token_id = entry.get("token_id")
            if isinstance(token_id, str) and token_id:
                out.add(token_id)
    return out


__all__ = [
    "DEFAULT_SECRETS_ROOT",
    "METHOD_REQUIRED_SCOPE",
    "ROLE_DEFAULT_SCOPE",
    "ROLE_TOKEN_PREFIX",
    "Role",
    "Scope",
    "TokenIdentity",
    "TokenStore",
    "append_revocation",
    "generate_token",
    "required_scope_for_method",
    "scope_satisfies",
    "token_digest_hint",
    "token_full_id",
    "token_sha256",
    "write_grace_record",
    "write_secret_file",
    "write_user_record",
]
