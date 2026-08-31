"""Integration shim package: a top-level ``runner`` that the vendored DarwinX eval shells as
``python -m runner.run``. Resolved by putting this ``_shims/`` dir on ``PYTHONPATH`` at launch
(``DarwinX.evolve``) — NOT a real beagle subpackage (that's why it's under ``_shims/``)."""
