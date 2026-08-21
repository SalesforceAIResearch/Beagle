#!/usr/bin/env bash
# Regenerate gRPC stubs from xrlenv/api/proto/*.proto into xrlenv/api/_pb2/.
# Run from the repo root after editing any .proto file:
#
#     bash scripts/gen_protos.sh
#
# CI verifies that the committed _pb2 files match what this script produces.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${REPO_ROOT}/xrlenv/api/proto"
OUT_DIR="${REPO_ROOT}/xrlenv/api/_pb2"

mkdir -p "${OUT_DIR}"

# Pick the python that has grpcio-tools installed.
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python)"
fi

cd "${REPO_ROOT}"
# Make protoc-gen-mypy / protoc-gen-mypy_grpc reachable via PATH so protoc
# can shell out to them when --mypy_out / --mypy_grpc_out are passed.
PY_BIN_DIR="$(dirname "${PY}")"
PATH="${PY_BIN_DIR}:${PATH}" "${PY}" -m grpc_tools.protoc \
    --proto_path="${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    --mypy_out="${OUT_DIR}" \
    --mypy_grpc_out="${OUT_DIR}" \
    "${PROTO_DIR}"/*.proto

# protoc's python_out emits absolute imports (`import <name>_pb2`); the
# generated stubs live inside the xrlenv.api._pb2 package, so we rewrite the
# imports to be package-qualified. This lets `python -c "from xrlenv.api
# import <name>_pb2"` work without a sys.path hack. Pattern is generic so
# new .proto files don't require touching this script.
for f in "${OUT_DIR}"/*_pb2_grpc.py; do
    sed -i.bak -E 's/^import ([a-z_]+_pb2) as/from xrlenv.api._pb2 import \1 as/' "$f"
    rm -f "${f}.bak"
done

echo "regenerated stubs in ${OUT_DIR}:"
ls -la "${OUT_DIR}" | grep -v __pycache__
