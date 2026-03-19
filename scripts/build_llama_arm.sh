#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build_arm_llama"
TOOLCHAIN_FILE="$ROOT_DIR/aarch64.toolchain.cmake"

# 清理旧的缓存（非常重要！旧的错误路径会缓存在 CMakeCache.txt 中）
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_BACKEND_DL=ON \
  "$ROOT_DIR"

cmake --build . -j$(nproc)