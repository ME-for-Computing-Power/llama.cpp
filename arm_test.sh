#!/usr/bin/env bash
set -euo pipefail

# Verification for the LM-head/unembedding const-matmul offload
# (zg330-unembedding-velvety-bonbon.md, section "3. Verification"). Two gates,
# both run on the real 30tai ARM board over ssh:
#
#   A. Memory-safety gate (requirements A/A2): run the FMSH-enabled feature
#      build twice back to back — GGML_FMSH_ZG330_OFFLOAD_UNEMBED=0, then =1 —
#      polling /proc/$PID/status VmRSS during each run. Expected direction:
#      peak_VmRSS(on) < peak_VmRSS(off) (not just "not much higher" — see the
#      plan doc for the full mechanism explanation).
#
#   B. Performance gate (requirement C): compare against the separate pure-CPU
#      baseline build (no fmsh code linked at all) on the same board/prompt.
#      Named gate: feature Prompt t/s > 3.8 and Generation t/s > 2.2, AND
#      unembed_stage_summary calls>0 with fallback_not_baked=0 /
#      fallback_m_mismatch=0 (i.e. the op actually ran on the NPU every time).
#
# Old env knobs from prior experiments (GGML_FMSH_ZG330_DISABLE_DEVICE_INPUT_CHAIN,
# GGML_FMSH_ZG330_RMS_NORM_CUSTOM_OP, ...) are dropped — no backward-compat
# requirement for this rewrite.
#
# If a gate isn't met, this script still prints the measured numbers/breakdown
# instead of hiding the failure — that's the point of the gate.

# NOTE on runtime: this backend does per-op host<->NPU round trips for every
# offloaded elementwise/matmul op (see project memory: known-slow before the
# cache is fully warm / with some ops missing from .cache). A handful of
# generated tokens with --device FMSH_ZG330 can legitimately take many minutes,
# not seconds. TIMEOUT_SECONDS must comfortably exceed that or a pass looks
# "hung" when it is only slow — and worse, a local timeout that kills the ssh
# session does NOT stop the backgrounded remote llama-cli process, which then
# keeps running and contends with the next pass. run_remote_pass writes a
# remote pidfile and always issues a follow-up kill after the timeout wrapper
# returns (success OR timeout) specifically to prevent that.
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
REMOTE_HOST="${REMOTE_HOST:-root@192.168.110.114}"
REMOTE_DIR="${REMOTE_DIR:-/root/llama}"

WORKSPACE_DIR="$(pwd)"
LOCAL_CACHE_DIR="${WORKSPACE_DIR}/.cache"
LOCAL_BIN_DIR="${WORKSPACE_DIR}/build-fmsh-zg330-arm64/bin"
BASELINE_BIN_DIR="${BASELINE_BIN_DIR:-/home/ray/git/llama.cpp-f36838f/build-arm64-nofmsh/bin}"

MODEL_NAME="${MODEL_NAME:-Qwen3.5-0.8B-Q4_K_M.gguf}"
PROMPT_TEXT="${PROMPT_TEXT:-介绍你自己}"
N_PREDICT="${N_PREDICT:-8}"
N_CTX="${N_CTX:-1100}"

PERF_PROMPT_GATE_TPS="${PERF_PROMPT_GATE_TPS:-3.8}"
# See ggml-fmsh-zg330.cpp: this branch has no pre-baked elementwise (ADD/MUL)
# net cache yet, and ARM cannot icraft-compile, so every miss is retried on
# every occurrence at real subprocess-spawn cost — prohibitively slow. Default
GGML_FMSH_ZG330_DISABLE_ELEMENTWISE="${GGML_FMSH_ZG330_DISABLE_ELEMENTWISE:-1}"
PERF_GEN_GATE_TPS="${PERF_GEN_GATE_TPS:-2.2}"

LOG_DIR="${WORKSPACE_DIR}/arm_test_logs"
mkdir -p "${LOG_DIR}"

overall_status=0
CURRENT_REMOTE_PIDFILE=""

cleanup_current_remote_pass() {
    if [[ -n "${CURRENT_REMOTE_PIDFILE}" ]]; then
        timeout 15 ssh "${REMOTE_HOST}" \
            "if [ -f '${CURRENT_REMOTE_PIDFILE}' ]; then kill -9 \$(cat '${CURRENT_REMOTE_PIDFILE}') 2>/dev/null; rm -f '${CURRENT_REMOTE_PIDFILE}'; fi" \
            >/dev/null 2>&1 || true
    fi
}
trap cleanup_current_remote_pass EXIT

echo "===== pre-flight: killing any leftover llama-cli from prior runs on the board ====="
timeout 15 ssh "${REMOTE_HOST}" "pkill -9 -f '${REMOTE_DIR}/.*/llama-cli' 2>/dev/null; true" || true

echo "===== upload cache ====="
mkdir -p "${LOCAL_CACHE_DIR}/deploy"
rsync -avh --info=progress2 --stats "${LOCAL_CACHE_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"

echo
echo "===== upload feature bin ====="
rsync -avh --info=progress2 --stats "${LOCAL_BIN_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"

echo
echo "===== upload baseline (pure-CPU, no fmsh linked) bin ====="
if [[ -d "${BASELINE_BIN_DIR}" ]]; then
    rsync -avh --info=progress2 --stats "${BASELINE_BIN_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/baseline_bin/"
else
    echo "WARNING: baseline bin dir not found at ${BASELINE_BIN_DIR}; performance gate will be skipped" >&2
fi

# run_remote_pass LABEL BIN_SUBDIR CLI_EXTRA_ARGS EXTRA_ENV_LINES
# Sets globals: PASS_PEAK_RSS_KB PASS_EXIT_CODE PASS_STDOUT_LOCAL PASS_BACKEND_LOG_LOCAL
run_remote_pass() {
    local label="$1" bin_subdir="$2" cli_extra="$3" extra_env="$4"
    local remote_out="${REMOTE_DIR}/run_${label}.out"
    local remote_pidfile="${REMOTE_DIR}/run_${label}.pid"
    local remote_script="
set -uo pipefail
cd '${REMOTE_DIR}/${bin_subdir}'
export LD_LIBRARY_PATH='${REMOTE_DIR}/${bin_subdir}:/ModelzooDeps/aarch64/Dynamic/lib:\${LD_LIBRARY_PATH:-}'
${extra_env}
: > '${remote_out}'
stdbuf -oL -eL ./llama-cli \
    -m '${REMOTE_DIR}/${MODEL_NAME}' \
    --reasoning-budget 0 \
    -p '${PROMPT_TEXT}' \
    -n ${N_PREDICT} \
    -c ${N_CTX} \
    --no-warmup \
    --single-turn \
    --seed 1024 \
    --verbose \
    --simple-io \
    ${cli_extra} \
    --log-file '${REMOTE_DIR}/llama_${label}.log' \
    > '${remote_out}' 2>&1 &
PID=\$!
echo \$PID > '${remote_pidfile}'
PEAK=0
while kill -0 \"\$PID\" 2>/dev/null; do
    RSS=\$(awk '/VmRSS/{print \$2}' /proc/\$PID/status 2>/dev/null)
    if [ -n \"\$RSS\" ]; then
        if [ \"\$RSS\" -gt \"\$PEAK\" ] 2>/dev/null; then PEAK=\$RSS; fi
    fi
    sleep 0.05
done
wait \"\$PID\"
EXIT_CODE=\$?
rm -f '${remote_pidfile}'
echo \"__PEAK_VMRSS_KB__=\${PEAK}\"
echo \"__EXIT_CODE__=\${EXIT_CODE}\"
"
    echo "--- running pass: ${label} (timeout ${TIMEOUT_SECONDS}s) ---"
    CURRENT_REMOTE_PIDFILE="${remote_pidfile}"
    set +e
    local result
    result="$(timeout --signal=INT "${TIMEOUT_SECONDS}" ssh "${REMOTE_HOST}" bash -s <<< "${remote_script}")"
    local ssh_status=$?
    set -e

    # Always clean up: if the local timeout killed our ssh session, the
    # backgrounded remote llama-cli survives unless we explicitly kill it here.
    timeout 15 ssh "${REMOTE_HOST}" \
        "if [ -f '${remote_pidfile}' ]; then kill -9 \$(cat '${remote_pidfile}') 2>/dev/null; rm -f '${remote_pidfile}'; fi" \
        >/dev/null 2>&1 || true
    CURRENT_REMOTE_PIDFILE=""

    echo "${result}"

    PASS_PEAK_RSS_KB="$(echo "${result}" | sed -n 's/^__PEAK_VMRSS_KB__=//p' | tail -n1)"
    PASS_EXIT_CODE="$(echo "${result}" | sed -n 's/^__EXIT_CODE__=//p' | tail -n1)"
    [[ -z "${PASS_PEAK_RSS_KB}" ]] && PASS_PEAK_RSS_KB=0
    [[ -z "${PASS_EXIT_CODE}" ]] && PASS_EXIT_CODE=${ssh_status}
    if [[ "${ssh_status}" -eq 124 ]]; then
        echo "WARNING: pass=${label} timed out after ${TIMEOUT_SECONDS}s (remote process killed); consider raising TIMEOUT_SECONDS"
    fi

    PASS_STDOUT_LOCAL="${LOG_DIR}/run_${label}.out"
    PASS_BACKEND_LOG_LOCAL="${LOG_DIR}/backend_${label}.log"
    scp -q "${REMOTE_HOST}:${remote_out}" "${PASS_STDOUT_LOCAL}" 2>/dev/null || true
    scp -q "${REMOTE_HOST}:${REMOTE_DIR}/.cache/deploy/backend.log" "${PASS_BACKEND_LOG_LOCAL}" 2>/dev/null || true

    echo "pass=${label} peak_vmrss_kb=${PASS_PEAK_RSS_KB} exit_code=${PASS_EXIT_CODE}"
}

parse_tps() {
    # prints "PROMPT_TPS GEN_TPS" parsed from a llama-cli stdout log, from the
    # "[ Prompt: X t/s | Generation: Y t/s ]" line (tools/cli/cli.cpp:632).
    local file="$1"
    awk '
        match($0, /Prompt: ([0-9.]+) t\/s \| Generation: ([0-9.]+) t\/s/, m) { prompt_tps = m[1]; gen_tps = m[2] }
        END { printf "%s %s", (prompt_tps==""?"0":prompt_tps), (gen_tps==""?"0":gen_tps) }
    ' "${file}" 2>/dev/null || echo "0 0"
}

echo
echo "===== gate A: memory-safety (RSS on < RSS off) ====="
run_remote_pass "unembed_off" "bin" "--device FMSH_ZG330" \
    "export GGML_FMSH_ZG330_LOG=1
export GGML_FMSH_ZG330_CACHE_DIR='${REMOTE_DIR}/.cache/deploy'
export GGML_FMSH_ZG330_DISABLE_ELEMENTWISE='${GGML_FMSH_ZG330_DISABLE_ELEMENTWISE}'
export GGML_FMSH_ZG330_OFFLOAD_UNEMBED=0"
RSS_OFF_KB="${PASS_PEAK_RSS_KB}"

run_remote_pass "unembed_on" "bin" "--device FMSH_ZG330" \
    "export GGML_FMSH_ZG330_LOG=1
export GGML_FMSH_ZG330_CACHE_DIR='${REMOTE_DIR}/.cache/deploy'
export GGML_FMSH_ZG330_DISABLE_ELEMENTWISE='${GGML_FMSH_ZG330_DISABLE_ELEMENTWISE}'
export GGML_FMSH_ZG330_OFFLOAD_UNEMBED=1"
RSS_ON_KB="${PASS_PEAK_RSS_KB}"
BACKEND_LOG_ON="${PASS_BACKEND_LOG_LOCAL}"

echo
echo "peak_VmRSS(off) = ${RSS_OFF_KB} kB"
echo "peak_VmRSS(on)  = ${RSS_ON_KB} kB"
if awk -v on="${RSS_ON_KB}" -v off="${RSS_OFF_KB}" 'BEGIN{exit !(on < off)}'; then
    echo "PASS: gate A (peak_VmRSS(on) < peak_VmRSS(off))"
else
    echo "FAIL: gate A (peak_VmRSS(on) >= peak_VmRSS(off)) — see plan doc mechanism section for the expected explanation"
    overall_status=1
fi

echo
echo "===== unembed_stage_summary / resident_summary (unembed_on pass) ====="
if [[ -f "${BACKEND_LOG_ON}" ]]; then
    grep -E 'unembed_stage_summary|resident_summary' "${BACKEND_LOG_ON}" || echo "(none found)"
else
    echo "missing ${BACKEND_LOG_ON}"
fi

echo
echo "===== gate B: performance (feature > pure-CPU baseline, named thresholds) ====="
if [[ -d "${BASELINE_BIN_DIR}" ]]; then
    run_remote_pass "baseline_cpu" "baseline_bin" "" ""
    read -r BASELINE_PROMPT_TPS BASELINE_GEN_TPS <<< "$(parse_tps "${PASS_STDOUT_LOCAL}")"
else
    BASELINE_PROMPT_TPS="0"; BASELINE_GEN_TPS="0"
    echo "skipped (no baseline bin uploaded)"
fi

run_remote_pass "feature_perf" "bin" "--device FMSH_ZG330" \
    "export GGML_FMSH_ZG330_LOG=1
export GGML_FMSH_ZG330_CACHE_DIR='${REMOTE_DIR}/.cache/deploy'
export GGML_FMSH_ZG330_DISABLE_ELEMENTWISE='${GGML_FMSH_ZG330_DISABLE_ELEMENTWISE}'
export GGML_FMSH_ZG330_OFFLOAD_UNEMBED=1"
read -r FEATURE_PROMPT_TPS FEATURE_GEN_TPS <<< "$(parse_tps "${PASS_STDOUT_LOCAL}")"
FEATURE_BACKEND_LOG="${PASS_BACKEND_LOG_LOCAL}"

echo
echo "baseline (pure CPU): Prompt=${BASELINE_PROMPT_TPS} t/s | Generation=${BASELINE_GEN_TPS} t/s"
echo "feature  (FMSH+unembed): Prompt=${FEATURE_PROMPT_TPS} t/s | Generation=${FEATURE_GEN_TPS} t/s"
echo "named gate thresholds: Prompt > ${PERF_PROMPT_GATE_TPS} t/s, Generation > ${PERF_GEN_GATE_TPS} t/s"

CALLS=0; FALLBACK_NOT_BAKED=0; FALLBACK_M_MISMATCH=0
if [[ -f "${FEATURE_BACKEND_LOG}" ]]; then
    LINE="$(grep -m1 'unembed_stage_summary' "${FEATURE_BACKEND_LOG}" || true)"
    if [[ -n "${LINE}" ]]; then
        CALLS="$(echo "${LINE}" | sed -n 's/.*calls=\([0-9]*\).*/\1/p')"
        FALLBACK_NOT_BAKED="$(echo "${LINE}" | sed -n 's/.*fallback_not_baked=\([0-9]*\).*/\1/p')"
        FALLBACK_M_MISMATCH="$(echo "${LINE}" | sed -n 's/.*fallback_m_mismatch=\([0-9]*\).*/\1/p')"
    fi
fi
echo "unembed_stage_summary: calls=${CALLS} fallback_not_baked=${FALLBACK_NOT_BAKED} fallback_m_mismatch=${FALLBACK_M_MISMATCH}"

gate_b_pass=1
if ! awk -v v="${FEATURE_PROMPT_TPS}" -v g="${PERF_PROMPT_GATE_TPS}" 'BEGIN{exit !(v > g)}'; then gate_b_pass=0; fi
if ! awk -v v="${FEATURE_GEN_TPS}" -v g="${PERF_GEN_GATE_TPS}" 'BEGIN{exit !(v > g)}'; then gate_b_pass=0; fi
if [[ -z "${CALLS}" || "${CALLS}" -eq 0 ]]; then gate_b_pass=0; fi
if [[ -n "${FALLBACK_NOT_BAKED}" && "${FALLBACK_NOT_BAKED}" -ne 0 ]]; then gate_b_pass=0; fi
if [[ -n "${FALLBACK_M_MISMATCH}" && "${FALLBACK_M_MISMATCH}" -ne 0 ]]; then gate_b_pass=0; fi

if [[ "${gate_b_pass}" -eq 1 ]]; then
    echo "PASS: gate B (perf thresholds met and unembed ran on NPU every call)"
else
    echo "FAIL: gate B — report the measured numbers and stage breakdown (gather vs device vs unpack) honestly"
    if [[ -f "${FEATURE_BACKEND_LOG}" ]]; then
        grep -E 'unembed_stage_summary|op_summary op=' "${FEATURE_BACKEND_LOG}" || true
    fi
    overall_status=1
fi

echo
echo "===== logs ====="
echo "local logs saved under: ${LOG_DIR}"

exit "${overall_status}"
