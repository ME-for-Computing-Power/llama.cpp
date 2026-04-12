#!/usr/bin/env bash
set -euo pipefail

source_dir=${FMSH_ZG330_SOURCE_DIR:?}
build_dir=${FMSH_ZG330_BUILD_DIR:?}
install_prefix=${FMSH_ZG330_INSTALL_PREFIX:?}
generator=${FMSH_ZG330_GENERATOR:?}
build_type=${FMSH_ZG330_BUILD_TYPE:?}
jobs=${FMSH_ZG330_JOBS:?}
target_arch=${FMSH_ZG330_ARCH:-x86_64}
debug_symbols=${FMSH_ZG330_DEBUG_SYMBOLS:-on}
configure_only=${FMSH_ZG330_CONFIGURE_ONLY:-0}
no_install=${FMSH_ZG330_NO_INSTALL:-1}
toolchain_file=${FMSH_ZG330_TOOLCHAIN_FILE:-}
modelzoo_root=${FMSH_ZG330_MODELZOO_ROOT:-/ModelzooDeps}

cmake_arch=x64
if [[ "$target_arch" == "aarch64" || "$target_arch" == "arm64" ]]; then
  cmake_arch=arm64
fi

cmake_args=(
  -S "$source_dir"
  -B "$build_dir"
  -G "$generator"
  -DCMAKE_BUILD_TYPE="$build_type"
  -DCMAKE_INSTALL_PREFIX="$install_prefix"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  -DGGML_FMSH_ZG330=ON
  -DGGML_NATIVE=OFF
  -DFMSH_ZG330_ARCH="$cmake_arch"
  -DFMSH_ZG330_MODELZOO_ROOT="$modelzoo_root"
  "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS:-}"
  "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS:-}"
)

if [[ "$cmake_arch" == "arm64" && -z "$toolchain_file" ]]; then
  if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 && command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    auto_toolchain_file="$build_dir/auto-aarch64-toolchain.cmake"
    mkdir -p "$build_dir"
    cat > "$auto_toolchain_file" <<'TOOLCHAIN_EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
TOOLCHAIN_EOF
    toolchain_file="$auto_toolchain_file"
  else
    echo "fatal: aarch64 cross compiler not found in image. Set FMSH_ZG330_TOOLCHAIN_FILE=..." >&2
    exit 2
  fi
fi

if [[ -n "$toolchain_file" ]]; then
  cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=$toolchain_file")
fi

if [[ "$debug_symbols" == "on" || "$debug_symbols" == "1" || "$debug_symbols" == "true" ]]; then
  if [[ "$build_type" == "Release" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g")
  fi
fi

if [[ -n "${FMSH_ZG330_EXTRA_CMAKE_ARGS:-}" ]]; then
  eval "set -- ${FMSH_ZG330_EXTRA_CMAKE_ARGS}"
  cmake_args+=("$@")
fi

if [[ "$cmake_arch" == "arm64" ]]; then
  rm -f "$build_dir/CMakeCache.txt"
  rm -rf "$build_dir/CMakeFiles"
fi

echo "+ cmake ${cmake_args[*]}"
cmake "${cmake_args[@]}"

if (( configure_only )); then
  exit 0
fi

build_args=(--parallel "$jobs" --config "$build_type")
if [[ -n "${FMSH_ZG330_BUILD_TARGETS:-}" ]]; then
  eval "set -- ${FMSH_ZG330_BUILD_TARGETS}"
  build_args+=(--target "$@")
fi

echo "+ cmake --build $build_dir ${build_args[*]}"
cmake --build "$build_dir" "${build_args[@]}"

if (( ! no_install )); then
  echo "+ cmake --install $build_dir --prefix $install_prefix"
  cmake --install "$build_dir" --prefix "$install_prefix"
fi
