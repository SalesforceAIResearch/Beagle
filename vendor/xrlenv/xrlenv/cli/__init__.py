"""CLI entry points (spec 09).

The full CLI surface (``xrlenv up``, ``xrlenv nodes``, ``xrlenv rollouts``,
``xrlenv capacity``, ``xrlenv warmup``) lands in Slice 4. Slice 1 ships only
the ``xrlenv`` console script as a stub so the package's entry-point wiring
can be exercised by ``uv run xrlenv``.
"""
