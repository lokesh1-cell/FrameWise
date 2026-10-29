#!/usr/bin/env bash
# Compile proto/*.proto into Python modules under src/physical_ai_pipeline/proto_gen.
#
# Uses the system `protoc` when available, otherwise grpcio-tools' bundled
# protoc (`python -m grpc_tools.protoc`), which also ships the well-known
# google/protobuf/*.proto includes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="${ROOT}/proto"
OUT_DIR="${ROOT}/src/physical_ai_pipeline/proto_gen"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

mkdir -p "${OUT_DIR}/foxglove"

if command -v protoc >/dev/null 2>&1 && [[ "${USE_GRPC_TOOLS:-0}" != "1" ]]; then
  PROTOC=(protoc)
  INCLUDES=(-I "${PROTO_DIR}")
else
  PROTOC=("${PYTHON}" -m grpc_tools.protoc)
  WKT_DIR="$("${PYTHON}" -c 'import grpc_tools, os; print(os.path.join(os.path.dirname(grpc_tools.__file__), "_proto"))')"
  INCLUDES=(-I "${PROTO_DIR}" -I "${WKT_DIR}")
fi

echo "[compile_proto] using: ${PROTOC[*]}"
"${PROTOC[@]}" "${INCLUDES[@]}" \
  --python_out="${OUT_DIR}" \
  --pyi_out="${OUT_DIR}" \
  "${PROTO_DIR}/physical_ai.proto" \
  "${PROTO_DIR}/foxglove/foxglove.proto"

touch "${OUT_DIR}/__init__.py" "${OUT_DIR}/foxglove/__init__.py"

# Also emit a binary FileDescriptorSet for tooling (mcap CLI, schema registry).
"${PROTOC[@]}" "${INCLUDES[@]}" --include_imports \
  --descriptor_set_out="${OUT_DIR}/physical_ai.fds" \
  "${PROTO_DIR}/physical_ai.proto" "${PROTO_DIR}/foxglove/foxglove.proto"

echo "[compile_proto] wrote ${OUT_DIR}"
