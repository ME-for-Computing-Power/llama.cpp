#!/usr/bin/env bash
set -euo pipefail

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
REMOTE_HOST="${REMOTE_HOST:-root@192.168.110.114}"
REMOTE_DIR="${REMOTE_DIR:-/root/llama}"
WORKSPACE_DIR="$(pwd)"
LOCAL_CACHE_DIR="${WORKSPACE_DIR}/.cache"
LOCAL_BIN_DIR="${WORKSPACE_DIR}/build-fmsh-zg330-arm64/bin"
LLAMA_LOG="${WORKSPACE_DIR}/llama_arm.log"
BACKEND_LOG="${WORKSPACE_DIR}/backend_arm.log"
LIVE_LOG="${WORKSPACE_DIR}/socket_test_arm.live.log"

rm -f "${LLAMA_LOG}"
rm -f "${BACKEND_LOG}"
rm -f "${LIVE_LOG}"

mkdir -p "${LOCAL_CACHE_DIR}/deploy"

echo "===== upload cache ====="
rsync -avh --info=progress2 --stats "${LOCAL_CACHE_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"

echo
echo "===== upload bin ====="
rsync -avh --info=progress2 --stats "${LOCAL_BIN_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"

echo
echo "===== remote inference ====="
set +e
timeout --signal=INT "${TIMEOUT_SECONDS}" \
    ssh "${REMOTE_HOST}" "
        set -euo pipefail
        cd '${REMOTE_DIR}/bin'
        export LD_LIBRARY_PATH='${REMOTE_DIR}/bin:/ModelzooDeps/aarch64/Dynamic/lib:\${LD_LIBRARY_PATH:-}'
        export GGML_FMSH_ZG330_LOG=1
        export GGML_FMSH_ZG330_CACHE_DIR='${REMOTE_DIR}/.cache/deploy'
        stdbuf -oL -eL ./llama-cli \
            -m '${REMOTE_DIR}/Qwen3.5-0.8B-Q4_K_M.gguf' \
            --device FMSH_ZG330 \
            --reasoning-budget 0 \
            -p 'Hello there' \
            -n 10 \
            -c 1100 \
            --no-warmup \
            --single-turn \
            --seed 1024 \
            --verbose \
            -fa on \
            --simple-io \
            --log-file '${REMOTE_DIR}/llama.log'
    " 2>&1 | tee "${LIVE_LOG}"
status=$?
set -e

if [[ ${status} -eq 124 ]]; then
    echo "arm inference timed out after ${TIMEOUT_SECONDS}s"
elif [[ ${status} -ne 0 ]]; then
    echo "arm inference exited with status ${status}"
fi

echo
echo "===== download logs ====="
scp "${REMOTE_HOST}:${REMOTE_DIR}/llama.log" "${LLAMA_LOG}"
scp "${REMOTE_HOST}:${REMOTE_DIR}/.cache/deploy/backend.log" "${BACKEND_LOG}"

echo
echo "===== llama.log ====="
if [[ -f "${LLAMA_LOG}" ]]; then
    echo "llama log is at ${LLAMA_LOG}"
else
    echo "missing ${LLAMA_LOG}"
fi

echo
echo "===== backend.log ====="
if [[ -f "${BACKEND_LOG}" ]]; then
    echo "backend log is at ${BACKEND_LOG}"
else
    echo "missing ${BACKEND_LOG}"
fi

# echo
# echo "===== grep: error|warn|fail ====="
# grep -Ein 'error|warn|fail' "${LLAMA_LOG}" "${BACKEND_LOG}" || true

exit "${status}"
