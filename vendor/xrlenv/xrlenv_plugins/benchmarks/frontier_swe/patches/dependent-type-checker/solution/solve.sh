#!/usr/bin/env bash
set -euo pipefail

echo "=== Oracle Solution: Dependent Type Checker ==="

# ── xrlenv content overlay (frontier-swe) ─────────────────────────────────────
# Upstream's solve.sh read the reference implementation from /tests/reference_impl:
#     TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../tests" && pwd)"   # = /tests
# but harbor mounts /tests ONLY during the verify phase, NOT during solve — a
# platform contract upstream itself documents in its sibling oracles
# (cranelift/solve.sh: "Cannot access /tests/ — bundle any needed resources under
# solution/"; libexpat/solve.sh: "/tests/ is only mounted during verification").
# So `cd /solution/../tests` failed with "No such file or directory", solve.sh
# aborted (set -e), no checker was built, and the verifier's empty checker rejected
# all 174 valid programs (accept 0/174 → reward 0).
#
# The FIX follows upstream's own documented pattern: the SAME reference_impl (Cargo.toml
# + src/main.rs, byte-identical to the copy inside tests/tests-bundle.tar.gz that the
# verifier itself builds and compares against) is bundled under solution/reference_impl/
# by build_cache.py's patch stage, and we copy it from there — a path present during
# the solve phase. This changes only WHERE the oracle reads the reference, never the
# reference itself; the verifier's anti-cheat hash check is oracle-exempt
# (HARBOR_ORACLE_MODE=1), so the byte-identical copy is legitimately accepted.
REF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/reference_impl" && pwd)"

mkdir -p /app/type-checker/src
cp "$REF_DIR/Cargo.toml" /app/type-checker/Cargo.toml
cp "$REF_DIR/src/main.rs" /app/type-checker/src/main.rs

# Fix the binary name to match what the verifier expects
sed -i 's/name = "type-checker-reference"/name = "type-checker"/' /app/type-checker/Cargo.toml

cd /app/type-checker
cargo build --release 2>&1

echo "Oracle solution built at /app/type-checker/"
