# beagle integration — CI/CD orchestration

The back-to-back runner for the `tests/smoke/` suite: runs each beagle smoke in
sequence and exits non-zero if *any* regressed, so a release gate is one command.
Mirrors `vendor/xrlenv/tests/integration/run_all_smoke.sh` in spirit.

Carries no test logic of its own — it orchestrates the smokes and, like xrlenv's
runner, **re-reads each smoke's on-disk artifacts** rather than trusting exit codes
(a smoke can exit 0 while every trial silently scored 0).
