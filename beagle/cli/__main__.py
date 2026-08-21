"""Enable ``python -m beagle.cli`` (the package needs a __main__ entry)."""

from __future__ import annotations

from beagle.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
