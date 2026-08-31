"""Wire-level constants shared across the spec-21 bidi protocol surface.

Centralised here so the control-plane gRPC server, the node-side
outbound link, and the consumer-facing Client transport all agree on
message-size limits without each defaulting to the gRPC stack's
4 MB ceiling.

Audit M1 (2026-04-29) called out that ``PutArchiveCommand`` ships
verifier-asset tarballs over the bidi stream; relying on Python
gRPC's default ``grpc.max_*_message_length`` (4 MB) would silently
fail at remote rollout time when a benchmark's ``tests/`` directory
exceeds that bound. We pin the ceiling here so the cap is one place
to change and one place to test.
"""

from __future__ import annotations

# Maximum bidi message size, in bytes. 128 MB is the sub-slice 1.b
# tarball-build commitment (default tarball cap is 100 MB; 28 MB
# headroom covers the proto envelope + any future field growth on
# BuildImageCommand). It's also comfortably above every
# terminal-bench-2 verifier-asset tarball measured to date
# (typical: 1-50 KB; tail outliers under 1 MB) and SWE-bench-Lite
# / OSWorld test suites; trajectory replies fit too (spec-17
# PlatformJsonlSink writes ~1-5 MB bodies for typical rollouts).
DEFAULT_MAX_MESSAGE_BYTES: int = 128 * 1024 * 1024

# Wire-chunk size for streamed-reply payloads (currently
# ``container_get_archive``). A single ``ContainerGetArchiveReply``
# carrying the whole tarball is the prod failure that took nodes
# "lost": a >128 MiB archive trips gRPC's send ceiling
# (``RESOURCE_EXHAUSTED``) and a >2 GiB one trips protobuf's hard
# serialize limit (``EncodeError``); either tears down the bidi
# stream the heartbeat shares, so the control plane marks the node
# lost and seals every in-flight rollout there as ``node_lost``.
# Chunking keeps every NodeMsg far below both ceilings. 4 MiB is the
# classic gRPC-friendly chunk — two orders of magnitude under the
# 128 MiB cap (so envelope overhead is irrelevant) while keeping the
# per-archive chunk count modest for typical 1-500 MB verifier dirs.
ARCHIVE_CHUNK_BYTES: int = 4 * 1024 * 1024

# Transport safety net: a NodeMsg at/above this serialized size must
# never be put on the wire — doing so raises at the gRPC layer and
# severs the stream (taking the heartbeat with it). The node-side
# outbound pump catches a reply this large and substitutes a clean
# ``FAILED`` CommandReply for the same command_id, degrading an
# un-chunked oversized reply (e.g. a huge batched-exec ExecReply)
# into a single failed command instead of a whole-node outage. Set
# 1 MiB below the hard cap to leave room for HTTP/2 framing overhead
# the ``ByteSize()`` estimate doesn't include.
MAX_OUTBOUND_MESSAGE_GUARD_BYTES: int = DEFAULT_MAX_MESSAGE_BYTES - (1 * 1024 * 1024)

# Control-plane relay cap for ``container_get_archive`` (node-lost
# guardrail / plane-split enforcement). The control plane is the
# metadata + orchestration channel, NOT a bulk-data pipe — spec 00
# invariant 6 keeps blobs on disk / object store, out of the control
# path. A caller pulling a whole container filesystem through it (e.g.
# EvoClaw's ``docker cp {c}:/testbed .`` — hundreds of MB of many small
# files per task) both (a) makes the node stream large tar volumes and
# (b) forces the single-process control plane to buffer the whole
# reassembled tarball in RAM. The node refuses a get_archive whose
# streamed size exceeds this cap, failing THAT transfer cleanly
# (``ArchiveTooLarge``) without touching the rollout. 128 MiB is
# generous vs every legitimate small read (verifier/reward files 1-50
# KB, patches/logs/trajectory bodies 1-5 MB — two-plus orders of
# magnitude under the cap) while blocking whole-repo copies. Operators
# tune via ``XRLENV_MAX_GET_ARCHIVE_RELAY_BYTES``; ``0`` disables the cap
# (unbounded — legacy / tests). Large-artifact capture is the job of the
# artifact-export primitive (notes/artifact-export-primitive-proposal.md),
# which keeps the bytes off the control plane entirely.
DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES: int = 128 * 1024 * 1024

# Default cap on operator-supplied tarball-source build contexts
# shipped over ``BuildImageCommand``. Per the locked F1 decision in
# notes/source-build-dispatch.md (sub-slice 1.b): 100 MB default,
# operator-tunable via ``xrlenv up --build-tarball-max-bytes``.
# Build-context tarballs over this cap get rejected at apply time
# on the operator's side (clear ``ManifestInvalid``), before any
# wire traffic — operators iterate locally rather than failing
# mid-cluster on an oversized payload.
DEFAULT_BUILD_TARBALL_MAX_BYTES: int = 100 * 1024 * 1024

# gRPC channel/server option pairs consumed by ``grpc.aio.server`` and
# ``grpc.aio.insecure_channel`` / ``grpc.aio.secure_channel``. Apply
# these unconditionally on any channel/server that carries bidi
# traffic so a single-VM smoke and the multi-VM acceptance gate see
# the same cap.
# Per-beat deadline for the raw-session keepalive RPC. Must stay well under the
# SDK's beat cadence (XRLENV_RAW_HEARTBEAT_INTERVAL_S, 30 s) so a wedged beat
# fails and is retried by the next one rather than stalling the loop: the
# keepalive is single-threaded, so one never-returning RPC silences a whole
# consumer process and every quiet session it holds dies at the quarantine
# horizon.
HEARTBEAT_RPC_TIMEOUT_S: float = 10.0


# Sanity ceiling for the keepalive cadence. Not a tuning limit — a typo filter.
# The keepalive exists to beat several times inside the control plane's liveness
# TTL (120 s by default), so any cadence measured in hours is indistinguishable
# from having no keepalive at all: it beats once at registration and never again,
# which is precisely the "looks healthy while the control plane hears nothing"
# failure the validation exists to prevent. A day is far above any legitimate
# setting and far below the values that cause it (1e300 parses and is finite).
MAX_HEARTBEAT_INTERVAL_S: float = 86_400.0


# How often a client channel sends an HTTP/2 keepalive ping while otherwise idle.
# Without this, a half-open TCP connection (a silently dropped flow, a NAT that
# forgot us) is invisible: an RPC issued on it hangs instead of failing, which is
# how a keepalive loop gets wedged. Kept equal to the beat cadence so an idle
# consumer still exercises the path.
GRPC_KEEPALIVE_TIME_MS: int = 30_000
GRPC_KEEPALIVE_TIMEOUT_MS: int = 10_000
# The server must TOLERATE pings at least as often as clients send them. gRPC
# servers default to allowing one ping per 5 minutes without data and answer a
# GOAWAY ``too_many_pings`` beyond that — so a client-only change would break the
# very connections it is meant to protect. This bound must stay <=
# GRPC_KEEPALIVE_TIME_MS; a test pins that relationship.
GRPC_SERVER_MIN_PING_INTERVAL_MS: int = 10_000

_GRPC_MESSAGE_SIZE_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
]

# CLIENT-side channels (consumer SDK + the node's outbound link).
GRPC_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    *_GRPC_MESSAGE_SIZE_OPTIONS,
    ("grpc.keepalive_time_ms", GRPC_KEEPALIVE_TIME_MS),
    ("grpc.keepalive_timeout_ms", GRPC_KEEPALIVE_TIMEOUT_MS),
    # Ping even with no RPC in flight — the idle case is the whole point.
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
]

# SERVER-side. Deliberately NOT the client list: a server given
# ``keepalive_time_ms`` starts pinging its own clients, which is a different
# decision. What it needs is permission to RECEIVE frequent pings.
GRPC_SERVER_OPTIONS: list[tuple[str, int]] = [
    *_GRPC_MESSAGE_SIZE_OPTIONS,
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms",
     GRPC_SERVER_MIN_PING_INTERVAL_MS),
    # Never GOAWAY a client purely for ping cadence.
    ("grpc.http2.max_ping_strikes", 0),
]


__all__ = [
    "ARCHIVE_CHUNK_BYTES",
    "DEFAULT_BUILD_TARBALL_MAX_BYTES",
    "DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "GRPC_CHANNEL_OPTIONS",
    "GRPC_KEEPALIVE_TIME_MS",
    "GRPC_SERVER_MIN_PING_INTERVAL_MS",
    "GRPC_SERVER_OPTIONS",
    "HEARTBEAT_RPC_TIMEOUT_S",
    "MAX_HEARTBEAT_INTERVAL_S",
    "MAX_OUTBOUND_MESSAGE_GUARD_BYTES",
]
