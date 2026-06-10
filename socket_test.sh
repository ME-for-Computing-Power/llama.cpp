#!/usr/bin/env bash
set -euo pipefail

LOCK_DIR="$(pwd)/.socket_test.lock"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: socket_test.sh is already running (lock ${LOCK_DIR} exists)." >&2
    echo "If you are sure no instance is running, remove it with: rm -rf ${LOCK_DIR}" >&2
    exit 1
fi
trap 'rm -rf "${LOCK_DIR}"' EXIT

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
WORKSPACE_DIR="$(pwd)"
LLAMA_LOG="${WORKSPACE_DIR}/llama.log"
BACKEND_LOG="${WORKSPACE_DIR}/.cache/deploy/backend.log"
LIVE_LOG="${WORKSPACE_DIR}/socket_test.live.log"

rm -f "${LLAMA_LOG}"
rm -f "${LIVE_LOG}"

mkdir -p "${WORKSPACE_DIR}/.cache/deploy"

set +e
timeout --signal=INT "${TIMEOUT_SECONDS}" \
    docker run --network host --rm \
        -v "${WORKSPACE_DIR}:/workspace" \
        fpai-icraft:latest \
        bash -lc '
            set -o pipefail
            cd /workspace/build-fmsh-zg330-x64/bin
            export LD_LIBRARY_PATH=/workspace/build-fmsh-zg330-x64/bin:/ModelzooDeps/x64/Dynamic/lib:${LD_LIBRARY_PATH:-}
            export GGML_FMSH_ZG330_LOG=1
            export GGML_FMSH_ZG330_CACHE_DIR=/workspace/.cache/deploy
            stdbuf -oL -eL ./llama-cli \
                -m /workspace/Qwen3.5-0.8B-Q4_K_M.gguf \
                --device FMSH_ZG330 \
                --reasoning-budget 0 \
                -p "Hello there" \
                -n 10 \
                -c 1024 \
                --no-warmup \
                --single-turn \
                --seed 1024 \
                --verbose \
                --log-file /workspace/llama.log \
                --simple-io \
                2>&1 | tee /workspace/socket_test.live.log
        '
status=$?
set -e

if [[ ${status} -eq 124 ]]; then
    echo "socket inference timed out after ${TIMEOUT_SECONDS}s"
elif [[ ${status} -ne 0 ]]; then
    echo "socket inference exited with status ${status}"
fi

echo
echo "===== llama.log ====="
if [[ -f "${LLAMA_LOG}" ]]; then
    echo "llama.log is at ${LLAMA_LOG}"
else
    echo "missing ${LLAMA_LOG}"
fi

echo
echo "===== backend.log ====="
if [[ -f "${BACKEND_LOG}" ]]; then
    echo "backend.log is at ${BACKEND_LOG}"
else
    echo "missing ${BACKEND_LOG}"
fi

# echo
# echo "===== grep: error|warn|fail ====="
# grep -Ein 'error|warn|fail' "${LLAMA_LOG}" "${BACKEND_LOG}" || true

exit "${status}"
