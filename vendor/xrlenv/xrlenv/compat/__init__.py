"""Drop-in compatibility shims for upstream client APIs.

Currently houses :mod:`xrlenv.compat.docker_client`, which lets
benchmark code that already speaks ``docker.from_env()`` / docker-py
(SWE-bench, terminal-bench, OSWorld, ...) run unmodified against an
xrlenv-orchestrated cluster.
"""
