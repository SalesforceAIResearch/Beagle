"""Built-in normalizers — imported for their registration side effects.

Importing this subpackage registers every shipped normalizer with the registry
in :mod:`trace_analyzer.normalizer`. Keep new built-ins listed here so a plain
``import trace_analyzer`` makes them discoverable via ``--source`` / auto-sniff.
"""

from __future__ import annotations

from . import monet, openai_messages  # noqa: F401

__all__ = ["monet", "openai_messages"]
