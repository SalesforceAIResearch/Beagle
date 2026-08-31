"""Consumer-facing SDK (spec 05).

The phase-0 surface is deliberately small: a :class:`Client` with a
``rollout`` async context manager and a ``batch_rollout`` workhorse. Framework
adapters (Slime, verl) layer on top in phase 1.

For Slice 1 only ``Client.rollout`` is implemented end-to-end; ``batch_rollout``
and ``replay`` land in subsequent slices once the StateStore has trajectory
durability and the scheduler can pack concurrent placements.
"""

from xrlenv.client.client import Client, Template
from xrlenv.client.dotenv import load_dotenv, parse_dotenv, upload_dotenv
from xrlenv.client.session import RolloutSession

__all__ = [
    "Client",
    "RolloutSession",
    "Template",
    "load_dotenv",
    "parse_dotenv",
    "upload_dotenv",
]
