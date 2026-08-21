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
GRPC_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_send_message_length", DEFAULT_MAX_MESSAGE_BYTES),
    ("grpc.max_receive_message_length", DEFAULT_MAX_MESSAGE_BYTES),
]


__all__ = [
    "ARCHIVE_CHUNK_BYTES",
    "DEFAULT_BUILD_TARBALL_MAX_BYTES",
    "DEFAULT_MAX_GET_ARCHIVE_RELAY_BYTES",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "GRPC_CHANNEL_OPTIONS",
    "MAX_OUTBOUND_MESSAGE_GUARD_BYTES",
]
