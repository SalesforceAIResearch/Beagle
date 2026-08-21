#!/usr/bin/env bash
# Pre-pull SWE-bench Verified prebuilt images on the local Docker
# daemon. Run on EACH cluster node before consumers drive the
# smoke through xrlenv (the docker-py drop-in does NOT pull
# implicitly — operator-pre-pulled contract).
#
# Pulls upstream's ``swebench/sweb.eval.x86_64.<instance>:latest``
# images. The 8 default smoke instances total ~25 GB on disk; the
# full Verified set is ~500 instances × few GB each.
#
# Usage:
#   ./pre-pull-images.sh                    # default: 8 smoke instances
#   ./pre-pull-images.sh --all              # full 500-instance Verified set
#   ./pre-pull-images.sh inst-a inst-b ...  # custom instance list
#
# Mirrors ``SMOKE_INSTANCES`` in smoke.py — keep these two lists
# in lock-step. Out-of-sync = smoke asks for an instance you didn't
# pull, fails fast at acquire_container with ImageNotFound.

set -euo pipefail

SMOKE_INSTANCES=(
    "astropy__astropy-7166"
    "django__django-11099"
    "sympy__sympy-18189"
    "astropy__astropy-12907"
    "astropy__astropy-14182"
    "sympy__sympy-13615"
    "django__django-11138"
    "sympy__sympy-12489"
)

NAMESPACE="swebench"
TAG="latest"

if [[ "${1:-}" == "--all" ]]; then
    echo "[pre-pull] fetching full Verified instance list..."
    if ! command -v python >/dev/null 2>&1; then
        echo "[pre-pull] python required for --all (loads dataset list)" >&2
        exit 2
    fi
    # Use the same dataset loader the smoke uses so the lists
    # can't drift.
    mapfile -t INSTANCES < <(python - <<'PY'
from swebench.harness.run_evaluation import load_swebench_dataset
for inst in load_swebench_dataset(
    "SWE-bench/SWE-bench_Verified", "test",
):
    print(inst["instance_id"])
PY
    )
elif (( $# > 0 )); then
    INSTANCES=("$@")
else
    INSTANCES=("${SMOKE_INSTANCES[@]}")
fi

echo "[pre-pull] pulling ${#INSTANCES[@]} instance image(s)..."
fail=0
for inst in "${INSTANCES[@]}"; do
    # The "__" → "_1776_" mangling matches swebench's
    # TestSpec.instance_image_key when namespace is set; see
    # swebench.harness.test_spec.test_spec.
    mangled="${inst//__/_1776_}"
    image="${NAMESPACE}/sweb.eval.x86_64.${mangled}:${TAG}"
    image="${image,,}"  # docker requires lowercase
    echo "[pre-pull] -> ${image}"
    if ! docker pull "${image}"; then
        echo "[pre-pull] FAILED: ${image}" >&2
        fail=$((fail + 1))
    fi
done

if (( fail > 0 )); then
    echo "[pre-pull] ${fail} image(s) failed to pull." >&2
    exit 1
fi
echo "[pre-pull] all ${#INSTANCES[@]} image(s) present."
