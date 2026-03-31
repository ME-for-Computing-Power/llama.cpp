#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build_arm_llama"
TOOLCHAIN_FILE="$ROOT_DIR/aarch64.toolchain.cmake"

GGML_FMSH=${GGML_FMSH:-ON}
GGML_FMSH_ICRAFT=${GGML_FMSH_ICRAFT:-ON}
GGML_FMSH_ICRAFT_ROOT=${GGML_FMSH_ICRAFT_ROOT:-}

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DGGML_CUDA=OFF \
  -DGGML_FMSH="$GGML_FMSH" \
  -DGGML_FMSH_ICRAFT="$GGML_FMSH_ICRAFT" \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_BACKEND_DL=ON \
  ${GGML_FMSH_ICRAFT_ROOT:+-DGGML_FMSH_ICRAFT_ROOT="$GGML_FMSH_ICRAFT_ROOT"} \
  "$ROOT_DIR"

cmake --build . -j$(nproc)


# # store password in script (insecure)
# SSH_PASSWORD='fmsh'
# TARGET_USER=root
# TARGET_HOST=192.168.110.88
# TARGET_PATH="/tmp/bin_$(date +%Y%m%d_%H%M%S).tar.gz"


# # requires sshpass to be installed
# sshpass -p "$SSH_PASSWORD" scp ./bin.tar.gz "${TARGET_USER}@${TARGET_HOST}:${TARGET_PATH}"
# echo "File transferred to ${TARGET_HOST}:${TARGET_PATH}"


# # r