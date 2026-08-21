"""Per-host node-agent daemon (spec 04).

The node agent owns the local backend driver, the in-flight sandbox table, and
the hardware probe. In phase 0 it speaks to the control plane over a single
outbound bidi gRPC stream (spec 21); for Slice 1 the same surface is reachable
via direct method calls so the control plane and node agent can run in the
same Python process while the wire format is being shaken down.
"""

from xrlenv.node.agent import NodeAgent, NodeAgentConfig
from xrlenv.node.hw_probe import HardwareInfo, probe_hardware

__all__ = [
    "HardwareInfo",
    "NodeAgent",
    "NodeAgentConfig",
    "probe_hardware",
]
