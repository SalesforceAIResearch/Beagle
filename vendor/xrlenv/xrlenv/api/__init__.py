"""xrlenv API surface — protobuf definitions and generated gRPC stubs.

The wire formats live under ``proto/``:

- ``node_control.proto`` (spec 21) — control-plane ↔ node-agent bidi
  stream.
- ``rollout_control.proto`` (spec 05) — consumer-facing unary RPCs the
  trainer / smoke driver dials via :py:meth:`xrlenv.client.Client.grpc`.

Generated stubs live in :mod:`xrlenv.api._pb2` and are re-exported here.

Regenerate stubs after editing the .proto with::

    scripts/gen_protos.sh
"""

from xrlenv.api._pb2 import (
    node_control_pb2,
    node_control_pb2_grpc,
    rollout_control_pb2,
    rollout_control_pb2_grpc,
)

__all__ = [
    "node_control_pb2",
    "node_control_pb2_grpc",
    "rollout_control_pb2",
    "rollout_control_pb2_grpc",
]
