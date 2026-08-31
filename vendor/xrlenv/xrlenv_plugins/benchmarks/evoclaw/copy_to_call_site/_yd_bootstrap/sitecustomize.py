"""Startup hook that re-applies the YD fixes in spawn / forkserver eval children.

EvoClaw runs evals in a ``ProcessPoolExecutor`` (harness ``orchestrator.py``). Under the
**fork** start method a child inherits the parent's in-process monkey-patches for free;
under **spawn** / **forkserver** the child boots a *fresh* interpreter that would NOT — so
the eval-level fixes (git-clean GT-test preservation, go-benchmark reassembly, n2p eval-retry)
would silently not apply.

Python imports ``sitecustomize`` at interpreter startup (``site`` init) in **every** process,
including spawned pool workers. This module lives in a dedicated dir that ``run_e2e_xrlenv.py``
prepends to ``PYTHONPATH`` (only) when ``--apply-yd-fixes`` is active, and it re-runs
``apply_yd_fixes`` when ``EVOCLAW_APPLY_YD_FIXES=1`` (also set by the wrapper, inherited by the
children). ``apply_yd_fixes`` is idempotent, so a fork child that already inherited the patch
is unaffected. Defensive: does nothing without the env var, and never raises (a broken hook must
not take down an unrelated interpreter).
"""
import os

if os.environ.get("EVOCLAW_APPLY_YD_FIXES") == "1":
    try:
        import yd_fixes  # on PYTHONPATH alongside this dir (set by run_e2e_xrlenv)

        yd_fixes.apply_yd_fixes()
    except Exception:
        # Never break a subprocess because the hook couldn't import/patch — the fix is
        # opt-in robustness, not a hard requirement.
        pass
