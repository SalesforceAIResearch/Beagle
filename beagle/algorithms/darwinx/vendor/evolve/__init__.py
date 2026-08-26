"""Self-evolve pipeline for monet_code.

This package implements an automated pipeline that uses cursor-agent (GPT-5.5
Extra High by default) to iteratively improve `monet_code` against terminal-bench:

    parent node -> claim <=2 failing tasks -> analyze -> implement -> review
        -> mini-eval -> (loop up to 10) -> final eval -> PR if score improved

Multiple pipelines run in parallel against a shared SQLite-backed campaign
tree at ``<reports-root>/self_evolve/<campaign>/``. ``reports-root`` defaults
to ``<repo-parent>/reports`` and can be overridden via the
``DARWINX_EVAL_REPORTS_ROOT`` env var or ``--reports-root`` on every CLI.
"""

from __future__ import annotations
