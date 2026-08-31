#!/usr/bin/env python3
"""Thin shim for ``python -m beagle.tools.onboard`` (see that module's docstring).

    python scripts/onboard/onboard_agent.py --upstream <url> --ref <ref> --repo <org>/<name> --dir <path>
"""

from __future__ import annotations

import sys

from beagle.tools.onboard import main

if __name__ == "__main__":
    sys.exit(main())
