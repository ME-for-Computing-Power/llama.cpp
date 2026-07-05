#include "ggml-fmsh-zg330.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-fmsh-zg330-netmake.h"

#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>

#include <chrono>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using namespace icraft::xrt;
namespace zg330 = icraft::xrt::zg330;

struct ggml_fmsh_zg330_op_signature {
    uint32_t op = 0;
    uint32_t dtype = 0;
    int64_t shape[GGML_MAX_DIMS] = {0, 0, 0, 0};

    bool operator==(const ggml_fmsh_zg330_op_signature & other) const {
        return op == other.op && dtype == other.dtype && std::memcmp(shape, other.shape, sizeof(shape)) == 0;
    }
};

struct ggml_fmsh_zg330_op_signature_hash {
    size_t operator()(const ggml_fmsh_zg330_op_signature & sig) const noexcept {
        size_t h = static_cast<size_t>(sig.op) * 1315423911ULL ^ static_cast<size_t>(sig.dtype);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            h ^= static_cast<size_t>(sig.shape[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        }
        return h;
    }
};

struct ggml_fmsh_zg330_session_entry {
    struct cached_weight_tensor {
        uintptr_t key = 0;
        Tensor tensor;
    };

    ggml_fmsh_zg330_op_signature signature;
    uint64_t hit_count = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    ggml::fmsh::netmake::MatmulZgNetworkBundle bundle;
    Session session;
    TensorType input_type_a;
    TensorType input_type_b;
    Tensor input_tensor_a;
    bool input_tensor_a_ready = false;
    std::vector<cached_weight_tensor> weight_tensors;
};

struct ggml_fmsh_zg330_elementwise_session_entry {
    ggml_fmsh_zg330_op_signature signature;
    uint64_t hit_count = 0;
    ggml::fmsh::netmake::ElementwiseZgOp op;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t compiled_rows = 0;
    ggml::fmsh::netmake::ElementwiseZgNetworkBundle bundle;
    Session session;
    std::vector<TensorType> input_types;
    std::vector<Tensor> input_tensors;
};

// Single, dedicated, never-evicted session for the LM-head/unembedding constant
// matmul. Deliberately not a cache/map entry: there is exactly one such session
// for the process lifetime (see zg330-unembedding-velvety-bonbon.md "Session
// storage" decision) — a bucket/multi-shape design was explicitly rejected.
struct ggml_fmsh_zg330_unembed_session_entry {
    bool usable = false; // true once bundle found + Session created + apply() succeeded
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    ggml::fmsh::netmake::ConstMatmulZgNetworkBundle bundle;
    Session session;
    TensorType input_type_a;
    Tensor input_tensor_a;
    bool input_tensor_a_ready = false;
    uint64_t raw_file_bytes = 0;
};

struct ggml_backend_fmsh_zg330_context {
    ggml_backend_t cpu_backend = nullptr;

    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_session_entry>> session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_elementwise_session_entry>, ggml_fmsh_zg330_op_signature_hash> elementwise_session_cache;

    bool strict_mode = false;
    bool enable_log = true;
    int log_level = 1; // 0 debug, 1 info, 2 warn, 3 error
    bool offload_cpy_dup = false;
    bool offload_soft_max = false;
    bool offload_rms_norm = false;
    bool disable_elementwise = false;

    bool offload_unembed = false;
    int64_t unembed_m = 1;
    std::filesystem::path unembed_cache_dir;
    std::string unembed_tensor_name_override;
    std::unique_ptr<ggml_fmsh_zg330_unembed_session_entry> unembed_session;

    struct unembed_stage_perf {
        uint64_t calls = 0;
        uint64_t exact_m_hits = 0;
        uint64_t fallback_not_baked = 0;
        uint64_t fallback_m_mismatch = 0;
        double gather_ms = 0.0;
        double device_memcpy_ms = 0.0;
        double device_hard_ms = 0.0;
        double unpack_ms = 0.0;
        double total_ms = 0.0;
    } unembed_perf;

    std::filesystem::path cache_dir;
    std::filesystem::path log_file;

    Device zg_device;
    bool device_opened = false;
    std::string device_url;

    struct op_perf {
        uint64_t calls = 0;
        double total_ms = 0.0;
        double memcpy_ms = 0.0;
        double hard_ms = 0.0;
    };
    std::unordered_map<uint32_t, op_perf> perf;

    struct zg_reuse_pool {
        uint64_t instr_bytes = 0;
        uint64_t weight_bytes = 0;
        uint64_t ftmp_bytes = 0;
        uint64_t io_in_bytes = 0;
        uint64_t io_out_bytes = 0;
        int64_t io_in_v_id = -1;
        int64_t io_out_v_id = -1;
        MemChunk instr_chunk;
        MemChunk weight_chunk;
        MemChunk ftmp_chunk;
        MemChunk io_in_chunk;
        MemChunk io_out_chunk;
    };
    std::unordered_map<std::string, zg_reuse_pool> reuse_pools;

    uint64_t mul_mat_total = 0;
    uint64_t mul_mat_offloaded = 0;
    uint64_t mul_mat_fallback = 0;
    uint64_t batched_mul_mat_total = 0;
    uint64_t batched_mul_mat_offloaded = 0;
    uint64_t mul_mat_device_input_chain = 0;

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
    bool debug_compare = false;
    double debug_compare_atol = 1e-1;
    double debug_compare_rtol = 1e-1;
    uint64_t debug_compare_total = 0;
    uint64_t debug_compare_mismatch = 0;
    uint64_t debug_compare_log_limit = 200;
    uint64_t debug_compare_logged = 0;
#endif

    std::mutex mu;
};

static ggml_fmsh_zg330_op_signature ggml_fmsh_make_signature(const ggml_tensor * node) {
    ggml_fmsh_zg330_op_signature sig = {};
    sig.op = static_cast<uint32_t>(node->op);
    sig.dtype = static_cast<uint32_t>(node->type);
    std::memcpy(sig.shape, node->ne, sizeof(sig.shape));
    return sig;
}

static int64_t ggml_fmsh_next_pow2_i64(int64_t v) {
    if (v <= 1) {
        return 1;
    }
    int64_t p = 1;
    while (p < v && p < (1LL << 30)) {
        p <<= 1;
    }
    return p < v ? v : p;
}

static int64_t ggml_fmsh_elementwise_compile_rows(ggml::fmsh::netmake::ElementwiseZgOp kind, int64_t rows) {
    if (rows <= 0) {
        return rows;
    }
    // icraft/ZG330 RMS_NORM has known failure on 1-row compile shapes.
    if (kind == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM && rows == 1) {
        return 2;
    }
    // Bucket dynamic row counts to reduce per-token recompilation.
    // Keep exact shape for tiny rows to avoid unnecessary padding overhead.
    if (rows <= 2) {
        return rows;
    }
    return ggml_fmsh_next_pow2_i64(rows);
}

static ggml_fmsh_zg330_op_signature ggml_fmsh_make_elementwise_signature(
    const ggml_tensor * node,
    ggml::fmsh::netmake::ElementwiseZgOp kind,
    int64_t rows,
    int64_t cols) {
    ggml_fmsh_zg330_op_signature sig = {};
    sig.op = static_cast<uint32_t>(node->op);
    sig.dtype = static_cast<uint32_t>(node->type);
    sig.shape[0] = cols;
    sig.shape[1] = ggml_fmsh_elementwise_compile_rows(kind, rows);
    sig.shape[2] = 1;
    sig.shape[3] = 1;
    return sig;
}

static bool ggml_fmsh_get_env_bool(const char * key, bool def) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return def;
    }
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0) {
        return true;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0 || std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0) {
        return false;
    }
    return def;
}

static double ggml_fmsh_get_env_double(const char * key, double def) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const double parsed = std::strtod(v, &end);
    if (end == v) {
        return def;
    }
    return parsed;
}

static uint64_t ggml_fmsh_get_env_u64(const char * key, uint64_t def) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return def;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 10);
    if (end == v) {
        return def;
    }
    return static_cast<uint64_t>(parsed);
}

static int ggml_fmsh_get_log_level(void) {
    const char * v = std::getenv("GGML_FMSH_ZG330_LOG_LEVEL");
    if (!v || v[0] == '\0') {
        return 1;
    }
    if (std::isdigit(static_cast<unsigned char>(v[0])) != 0) {
        const int level = std::atoi(v);
        return level < 0 ? 0 : (level > 3 ? 3 : level);
    }
    if (std::strcmp(v, "DEBUG") == 0 || std::strcmp(v, "debug") == 0) return 0;
    if (std::strcmp(v, "INFO") == 0 || std::strcmp(v, "info") == 0) return 1;
    if (std::strcmp(v, "WARN") == 0 || std::strcmp(v, "warn") == 0) return 2;
    if (std::strcmp(v, "ERROR") == 0 || std::strcmp(v, "error") == 0) return 3;
    return 1;
}

static const char * ggml_fmsh_log_tag(int level) {
    switch (level) {
        case 0: return "DEBUG";
        case 1: return "INFO";
        case 2: return "WARN";
        default: return "ERROR";
    }
}

static void ggml_fmsh_log_locked(ggml_backend_fmsh_zg330_context * ctx, int level, const std::string & msg) {
    if (!ctx->enable_log || level < ctx->log_level) {
        return;
    }
    std::ofstream ofs(ctx->log_file, std::ios::app);
    if (ofs) {
        ofs << "[" << ggml_fmsh_log_tag(level) << "] " << msg << "\n";
    }
}

static bool ggml_fmsh_is_meta_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool ggml_fmsh_is_host_dispatch_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return true;
        default:
            return false;
    }
}

static bool ggml_fmsh_is_elementwise_zg_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
            return true;
        default:
            return false;
    }
}

static bool ggml_fmsh_is_cpy_dup_op(const ggml_tensor * op) {
    return op->op == GGML_OP_CPY || op->op == GGML_OP_DUP;
}

static bool ggml_fmsh_map_elementwise_op(
    ggml_op op,
    ggml::fmsh::netmake::ElementwiseZgOp * out) {
    switch (op) {
        case GGML_OP_CPY:      *out = ggml::fmsh::netmake::ElementwiseZgOp::CPY; return true;
        case GGML_OP_DUP:      *out = ggml::fmsh::netmake::ElementwiseZgOp::DUP; return true;
        case GGML_OP_ADD:      *out = ggml::fmsh::netmake::ElementwiseZgOp::ADD; return true;
        case GGML_OP_MUL:      *out = ggml::fmsh::netmake::ElementwiseZgOp::MUL; return true;
        case GGML_OP_SCALE:    *out = ggml::fmsh::netmake::ElementwiseZgOp::SCALE; return true;
        case GGML_OP_RMS_NORM: *out = ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM; return true;
        case GGML_OP_SOFT_MAX: *out = ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX; return true;
        default: return false;
    }
}

// Identifies the LM-head/unembedding constant matmul (e.g. `token_embd.weight`
// tied embedding). Reads its own env vars directly (not via ctx) because the
// device-level `supports_op` callback below has no ggml_backend_fmsh_zg330_context
// available; the same predicate is reused, unchanged, from graph_compute where a
// ctx does exist, so both call sites stay consistent by construction.
static bool ggml_fmsh_is_lm_head_mul_mat(const ggml_tensor * node);

static bool ggml_fmsh_is_supported_op(const ggml_tensor * op) {
    if (ggml_fmsh_is_meta_op(op)) {
        return true;
    }
    if (ggml_fmsh_is_elementwise_zg_op(op)) {
        return true;
    }
    if (ggml_fmsh_is_host_dispatch_op(op)) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    if (ggml_fmsh_is_lm_head_mul_mat(op)) {
        // Bypass the F32-only rejection below for this one identified, quantized-weight op.
        return op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
    }
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

static bool ggml_fmsh_should_run_elementwise_zg(
    const ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * op) {
    // Diagnostic-only escape hatch (default off, no effect on normal behavior):
    // this branch has no pre-baked ADD/MUL/SCALE net cache yet on some hosts,
    // and compile-on-miss is retried on every single occurrence with no
    // negative-caching, which is prohibitively slow on ARM (which cannot
    // icraft compile at all). Set to force all elementwise ops to CPU.
    if (ctx->disable_elementwise) {
        return false;
    }
    if (op->op == GGML_OP_SOFT_MAX) {
        return ctx->offload_soft_max;
    }
    if (op->op == GGML_OP_RMS_NORM) {
        return ctx->offload_rms_norm;
    }
    if (ggml_fmsh_is_elementwise_zg_op(op)) {
        return true;
    }
    return ctx->offload_cpy_dup && ggml_fmsh_is_cpy_dup_op(op);
}

static bool ggml_fmsh_can_run_mul_mat_zg(const ggml_tensor * node) {
    if (node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || node->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (node->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src0->ne[2] <= 0 || src0->ne[3] <= 0 ||
        src1->ne[0] <= 0 || src1->ne[1] <= 0 || src1->ne[2] <= 0 || src1->ne[3] <= 0) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    if (node->ne[0] != src0->ne[1] || node->ne[1] != src1->ne[1] || node->ne[2] != src1->ne[2] || node->ne[3] != src1->ne[3]) {
        return false;
    }
    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || node->nb[0] != sizeof(float)) {
        return false;
    }
    return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(node);
}

static bool ggml_fmsh_is_lm_head_mul_mat(const ggml_tensor * node) {
    if (node == nullptr || node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (!ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_UNEMBED", false)) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    // Leaf weight: not a view, not the output of another op (matches a GGUF-loaded tensor).
    if (src0->op != GGML_OP_NONE || src0->view_src != nullptr) {
        return false;
    }
    if (!ggml_is_quantized(src0->type)) {
        return false;
    }
    const char * name = ggml_get_name(src0);
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * override_name = std::getenv("GGML_FMSH_ZG330_UNEMBED_TENSOR_NAME");
    if (override_name != nullptr && override_name[0] != '\0') {
        return std::strcmp(name, override_name) == 0;
    }
    return std::strcmp(name, "token_embd.weight") == 0 || std::strcmp(name, "output.weight") == 0;
}

static std::string ggml_fmsh_get_device_url() {
#if defined(__aarch64__) || defined(_M_ARM64)
    const char * axi = std::getenv("GGML_FMSH_ZG330_AXI_URL");
    if (axi && axi[0]) {
        return axi;
    }
    return "axi://zg330aiu?npu=0x40000000&dma=0x80000000";
#else
    const char * ip = std::getenv("GGML_FMSH_ZG330_IP");
    const char * port = std::getenv("GGML_FMSH_ZG330_PORT");
    std::string ip_str = (ip && ip[0]) ? ip : "192.168.110.114";
    std::string port_str = (port && port[0]) ? port : "9981";
    return "socket://zg330aiu@" + ip_str + ":" + port_str;
#endif
}

static bool ggml_fmsh_validate_mul_mat(const ggml_tensor * node, int64_t * m, int64_t * k, int64_t * n) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1) {
        return false;
    }

    // ggml mul_mat conventions:
    // src0: [k, n, ne02, ne03]
    // src1: [k, m, ne12, ne13]
    // dst : [n, m, ne12, ne13], with broadcast: ne12 % ne02 == 0, ne13 % ne03 == 0
    const int64_t k0 = src0->ne[0];
    const int64_t n0 = src0->ne[1];
    const int64_t k1 = src1->ne[0];
    const int64_t m1 = src1->ne[1];
    const int64_t n_dst = node->ne[0];
    const int64_t m_dst = node->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    if (k0 <= 0 || n0 <= 0 || k1 <= 0 || m1 <= 0 || ne02 <= 0 || ne03 <= 0 || ne12 <= 0 || ne13 <= 0) {
        return false;
    }
    if (k0 != k1 || n0 != n_dst || m1 != m_dst) {
        return false;
    }
    if (node->ne[2] != ne12 || node->ne[3] != ne13) {
        return false;
    }
    if (ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        return false;
    }

    *m = m1;
    *k = k0;
    *n = n0;
    return true;
}

static std::string ggml_fmsh_make_mul_mat_cache_key(const ggml_tensor * node, int64_t m, int64_t k, int64_t n) {
    return std::to_string(static_cast<uint32_t>(node->op)) + "|" +
           std::to_string(static_cast<uint32_t>(node->type)) + "|" +
           std::to_string(node->ne[0]) + "," +
           std::to_string(node->ne[1]) + "," +
           std::to_string(node->ne[2]) + "," +
           std::to_string(node->ne[3]) + "|" +
           std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
}

static bool ggml_fmsh_validate_elementwise(
    const ggml_tensor * node,
    ggml::fmsh::netmake::ElementwiseZgOp * op_kind,
    int64_t * rows,
    int64_t * cols) {
    if (!node || !op_kind || !rows || !cols) {
        return false;
    }
    if (!ggml_fmsh_map_elementwise_op(node->op, op_kind)) {
        return false;
    }
    if (node->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t c = node->ne[0];
    const int64_t r = node->ne[1] * node->ne[2] * node->ne[3];
    if (r <= 0 || c <= 0) {
        return false;
    }
    if (node->nb[0] != sizeof(float)) {
        return false;
    }
    *rows = r;
    *cols = c;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return src0 &&
                   src0->type == GGML_TYPE_F32 &&
                   ggml_nelements(src0) == ggml_nelements(node);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            if (!src0 || !src1 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (src0->ne[d] != node->ne[d] && src0->ne[d] != 1) return false;
                if (src1->ne[d] != node->ne[d] && src1->ne[d] != 1) return false;
            }
            return true;
        case GGML_OP_SCALE:
            if (!src0 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (src1) {
                if (src1->type != GGML_TYPE_F32 || src1->data == nullptr || ggml_nelements(src1) < 1) {
                    return false;
                }
            }
            return true;
        case GGML_OP_RMS_NORM:
            return src0 && src0->type == GGML_TYPE_F32;
        case GGML_OP_SOFT_MAX:
            if (!src0 || src0->type != GGML_TYPE_F32) {
                return false;
            }
            if (node->src[2] != nullptr) {
                return false;
            }
            if (src1 != nullptr && src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
                return false;
            }
            return true;
        default:
            return false;
    }
}

static bool ggml_fmsh_open_device_if_needed(ggml_backend_fmsh_zg330_context * ctx, std::string * err) {
    if (ctx->device_opened) {
        return true;
    }
    try {
        ctx->device_url = ggml_fmsh_get_device_url();
        ctx->zg_device = Device::Open(ctx->device_url);
        ctx->device_opened = true;
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        GGML_LOG_ERROR("%s: failed to open zg330 device: %s\n", __func__, e.what());
        std::exit(EXIT_FAILURE);
    }
}

static ggml_fmsh_zg330_session_entry * ggml_fmsh_get_or_create_mul_mat_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    int64_t m = 0, k = 0, n = 0;
    if (!ggml_fmsh_validate_mul_mat(node, &m, &k, &n)) {
        if (err) *err = "invalid MUL_MAT shape";
        return nullptr;
    }
    const ggml_fmsh_zg330_op_signature sig = ggml_fmsh_make_signature(node);
    const std::string cache_key = ggml_fmsh_make_mul_mat_cache_key(node, m, k, n);
    auto it = ctx->session_cache.find(cache_key);
    if (it != ctx->session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_matmul_zg_network(ctx->cache_dir, m, k, n);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);

        ggml_fmsh_log_locked(
            ctx, 0,
            "session_opt_config net=" + bundle.net_name + " reuse_ok=0 connect_ok=0 instr_bytes=0 weight_bytes=0 ftmp_bytes=0 in_v_id=-1 out_v_id=-1");
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_session_entry>();
        entry->signature = sig;
        entry->hit_count = 1;
        entry->m = m;
        entry->k = k;
        entry->n = n;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        entry->input_type_a = entry->bundle.network.inputs()[0].tensorType().clone();
        entry->input_type_b = entry->bundle.network.inputs()[1].tensorType().clone();

        ggml_fmsh_zg330_session_entry * ptr = entry.get();
        ctx->session_cache.emplace(cache_key, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
}

static ggml_fmsh_zg330_elementwise_session_entry * ggml_fmsh_get_or_create_elementwise_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    ggml::fmsh::netmake::ElementwiseZgOp kind;
    int64_t rows = 0;
    int64_t cols = 0;
    if (!ggml_fmsh_validate_elementwise(node, &kind, &rows, &cols)) {
        if (err) *err = "invalid elementwise shape/layout";
        return nullptr;
    }
    const ggml_fmsh_zg330_op_signature sig = ggml_fmsh_make_elementwise_signature(node, kind, rows, cols);
    auto it = ctx->elementwise_session_cache.find(sig);
    if (it != ctx->elementwise_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    int64_t compile_rows = ggml_fmsh_elementwise_compile_rows(kind, rows);

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_elementwise_zg_network(ctx->cache_dir, kind, compile_rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_elementwise_session_entry>();
        entry->signature = sig;
        entry->hit_count = 1;
        entry->op = kind;
        entry->rows = rows;
        entry->cols = cols;
        entry->compiled_rows = compile_rows;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        for (const auto & in : entry->bundle.network.inputs()) {
            entry->input_types.push_back(in.tensorType().clone());
        }
        entry->input_tensors.resize(entry->input_types.size());

        ggml_fmsh_zg330_elementwise_session_entry * ptr = entry.get();
        ctx->elementwise_session_cache.emplace(sig, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
}

// Load-only: creates the single, process-lifetime unembed session on first call.
// Never compiles. Fails soft (returns nullptr, sets *err) if the pre-baked artifact
// for (m,k,n) is missing or the device/session setup throws — callers must CPU-fallback.
// Subsequent calls (regardless of outcome) just return the cached result, so a missing
// artifact is only ever looked up once per process.
static ggml_fmsh_zg330_unembed_session_entry * ggml_fmsh_get_or_create_unembed_session(
    ggml_backend_fmsh_zg330_context * ctx,
    int64_t m,
    int64_t k,
    int64_t n,
    std::string * err) {
    if (ctx->unembed_session) {
        if (!ctx->unembed_session->usable && err) {
            *err = "unembed prebaked network unavailable (checked once, cached miss)";
        }
        return ctx->unembed_session->usable ? ctx->unembed_session.get() : nullptr;
    }

    auto entry = std::make_unique<ggml_fmsh_zg330_unembed_session_entry>();
    entry->m = m;
    entry->k = k;
    entry->n = n;

    try {
        entry->bundle = ggml::fmsh::netmake::load_prebaked_const_matmul_zg_network(
            ctx->unembed_cache_dir, m, k, n, "unembed");
        if (!entry->bundle.found) {
            if (err) {
                *err = "prebaked unembed network not found for shape " +
                       std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n) +
                       " under " + ctx->unembed_cache_dir.string();
            }
            ctx->unembed_session = std::move(entry);
            return nullptr;
        }

        if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
            ctx->unembed_session = std::move(entry);
            return nullptr;
        }

        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            entry->bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);
        session.apply();

        entry->session = std::move(session);
        entry->input_type_a = entry->bundle.network.inputs()[0].tensorType().clone();

        std::error_code ec;
        const auto raw_bytes = std::filesystem::file_size(entry->bundle.raw_path, ec);
        entry->raw_file_bytes = ec ? 0 : static_cast<uint64_t>(raw_bytes);
        entry->usable = true;

        ggml_fmsh_zg330_unembed_session_entry * ptr = entry.get();
        ctx->unembed_session = std::move(entry);
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        ctx->unembed_session = std::move(entry);
        return nullptr;
    }
}

// Executes the LM-head constant matmul on the already-created, exact-shape-matched
// unembed session. Packs src1's M rows (gather_ms), issues exactly one
// session.forward({input_a}) call, reads back M*n floats (unpack_ms). Device
// memcpy/hard time comes from session.timeProfileResults() — the only confirmed
// timing surface icraft exposes.
static bool ggml_fmsh_execute_unembed(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_unembed_session_entry * entry,
    std::string * err) {
    const ggml_tensor * src1 = node->src[1];
    if (!src1 || src1->data == nullptr || node->data == nullptr) {
        if (err) *err = "unembed tensor data is null";
        return false;
    }
    try {
        if (!entry->input_tensor_a_ready) {
            entry->input_tensor_a = Tensor(entry->input_type_a.clone());
            entry->input_tensor_a.mallocOn(HostDevice::MemRegion());
            entry->input_tensor_a_ready = true;
        }

        const auto tg0 = std::chrono::high_resolution_clock::now();
        float * a_dst = reinterpret_cast<float *>(entry->input_tensor_a.data().cptr());
        for (int64_t row = 0; row < entry->m; ++row) {
            const char * a_src_row = static_cast<const char *>(src1->data) + row * src1->nb[1];
            float * a_row_dst = a_dst + row * entry->k;
            if (src1->nb[0] == static_cast<size_t>(sizeof(float))) {
                std::memcpy(a_row_dst, a_src_row, static_cast<size_t>(entry->k * sizeof(float)));
            } else {
                for (int64_t col = 0; col < entry->k; ++col) {
                    float v = 0.0f;
                    std::memcpy(&v, a_src_row + col * src1->nb[0], sizeof(float));
                    a_row_dst[col] = v;
                }
            }
        }
        const auto tg1 = std::chrono::high_resolution_clock::now();

        auto outputs = entry->session.forward({entry->input_tensor_a});
        if (outputs.empty()) {
            if (err) *err = "unembed session.forward returned empty output";
            return false;
        }

        const auto tu0 = std::chrono::high_resolution_clock::now();
        const size_t out_elems = static_cast<size_t>(entry->m * entry->n);
        std::vector<float> out_tmp(out_elems);
        outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, out_elems * sizeof(float));
        for (int64_t row = 0; row < entry->m; ++row) {
            char * dst_row = static_cast<char *>(node->data) + row * node->nb[1];
            const float * src_row = out_tmp.data() + row * entry->n;
            if (node->nb[0] == static_cast<size_t>(sizeof(float))) {
                std::memcpy(dst_row, src_row, static_cast<size_t>(entry->n * sizeof(float)));
            } else {
                for (int64_t col = 0; col < entry->n; ++col) {
                    std::memcpy(dst_row + col * node->nb[0], &src_row[col], sizeof(float));
                }
            }
        }
        const auto tu1 = std::chrono::high_resolution_clock::now();

        const double gather_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
        const double unpack_ms = std::chrono::duration<double, std::milli>(tu1 - tu0).count();

        double device_memcpy_ms = 0.0;
        double device_hard_ms = 0.0;
        for (const auto & kv : entry->session.timeProfileResults()) {
            device_memcpy_ms += std::get<1>(kv.second);
            device_hard_ms += std::get<2>(kv.second);
        }

        auto & p = ctx->unembed_perf;
        p.exact_m_hits++;
        p.gather_ms += gather_ms;
        p.unpack_ms += unpack_ms;
        p.device_memcpy_ms += device_memcpy_ms;
        p.device_hard_ms += device_hard_ms;

        ggml_fmsh_log_locked(
            ctx, 0,
            "unembed_dispatch m_actual=" + std::to_string(entry->m) + " matched=true" +
            " gather_ms=" + std::to_string(gather_ms) +
            " device_memcpy_ms=" + std::to_string(device_memcpy_ms) +
            " device_hard_ms=" + std::to_string(device_hard_ms) +
            " unpack_ms=" + std::to_string(unpack_ms));
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

static inline float ggml_fmsh_read_f32_broadcast(
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3);

static inline void ggml_fmsh_write_f32_indexed(
    ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3,
    float v);

static inline float ggml_fmsh_read_f32_linear(const ggml_tensor * t, size_t idx);

static bool ggml_fmsh_execute_elementwise(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_elementwise_session_entry * entry,
    const Tensor * input_override,
    bool keep_device_output_only,
    Tensor * chained_output,
    std::string * err) {
    if (!node || !entry || node->data == nullptr || node->src[0] == nullptr || node->src[0]->data == nullptr) {
        if (err) *err = "elementwise tensor data is null";
        return false;
    }
    try {
        ggml_fmsh_log_locked(ctx, 1, "elementwise_begin op=" + std::string(ggml_op_name(node->op)));
        const int64_t exec_rows = entry->compiled_rows > 0 ? entry->compiled_rows : entry->rows;
        const size_t total_elems = static_cast<size_t>(exec_rows * entry->cols);
        const size_t total_bytes = total_elems * sizeof(float);
        for (size_t i = 0; i < entry->input_types.size(); ++i) {
            if (!entry->input_tensors[i].defined()) {
                entry->input_tensors[i] = Tensor(entry->input_types[i].clone());
                entry->input_tensors[i].mallocOn(HostDevice::MemRegion());
            }
        }

        if (chained_output != nullptr) {
            *chained_output = Tensor();
        }
        float * in0 = reinterpret_cast<float *>(entry->input_tensors[0].data().cptr());
        const ggml_tensor * src0 = node->src[0];
        const bool can_chain_input0 =
            input_override != nullptr &&
            entry->op != ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX &&
            entry->op != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM;
        Tensor input0 = can_chain_input0 ? *input_override : entry->input_tensors[0];

        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX) {
            float scale = 1.0f;
            float max_bias = 0.0f;
            std::memcpy(&scale, (float *) node->op_params + 0, sizeof(float));
            std::memcpy(&max_bias, (float *) node->op_params + 1, sizeof(float));

            const ggml_tensor * src1 = node->src[1];
            const ggml_tensor * src2 = node->src[2];
            if (src2 != nullptr) {
                if (err) *err = "SOFT_MAX with sinks(src2) is not supported";
                return false;
            }

            const int64_t ne12 = src1 ? src1->ne[2] : 1;
            const int64_t ne13 = src1 ? src1->ne[3] : 1;
            const uint32_t n_head = static_cast<uint32_t>(node->ne[2]);
            const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(n_head))));
            const float m0 = std::pow(2.0f, -(max_bias) / n_head_log2);
            const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
            const bool use_f16 = src1 && src1->type == GGML_TYPE_F16;

            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const uint32_t h = static_cast<uint32_t>(i2);
                        const float slope = (max_bias > 0.0f)
                            ? (h < n_head_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_head_log2) + 1))
                            : 1.0f;

                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                        const float * sp = reinterpret_cast<const float *>(
                            static_cast<const char *>(src0->data) + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);

                        const int64_t i11 = i1;
                        const int64_t i12 = i2 % ne12;
                        const int64_t i13 = i3 % ne13;
                        const char * mp_base = src1
                            ? static_cast<const char *>(src1->data) + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]
                            : nullptr;

                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            float v = sp[i0] * scale;
                            if (mp_base) {
                                const float mv = use_f16
                                    ? GGML_FP16_TO_FP32(reinterpret_cast<const ggml_fp16_t *>(mp_base)[i0])
                                    : reinterpret_cast<const float *>(mp_base)[i0];
                                v += slope * mv;
                            }
                            in0[row_off + static_cast<size_t>(i0)] = v;
                        }
                    }
                }
            }
        } else if (!can_chain_input0) {
            if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::CPY ||
                entry->op == ggml::fmsh::netmake::ElementwiseZgOp::DUP) {
                const size_t n = ggml_nelements(node);
                for (size_t idx = 0; idx < n; ++idx) {
                    in0[idx] = ggml_fmsh_read_f32_linear(src0, idx);
                }
            } else {
                for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                            const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                            const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                            for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                                in0[row_off + static_cast<size_t>(i0)] = ggml_fmsh_read_f32_broadcast(src0, i0, i1, i2, i3);
                            }
                        }
                    }
                }
            }
        }
        if (!can_chain_input0 && exec_rows > entry->rows) {
            float * in0_pad = reinterpret_cast<float *>(entry->input_tensors[0].data().cptr());
            const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(float);
            for (int64_t r = entry->rows; r < exec_rows; ++r) {
                std::memcpy(
                    in0_pad + static_cast<size_t>(r * entry->cols),
                    in0_pad + static_cast<size_t>((entry->rows - 1) * entry->cols),
                    row_bytes);
            }
        }

        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::ADD || entry->op == ggml::fmsh::netmake::ElementwiseZgOp::MUL) {
            if (!node->src[1] || !node->src[1]->data) {
                if (err) *err = "elementwise binary src1 is null";
                return false;
            }
            float * in1 = reinterpret_cast<float *>(entry->input_tensors[1].data().cptr());
            const ggml_tensor * src1 = node->src[1];
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            in1[row_off + static_cast<size_t>(i0)] = ggml_fmsh_read_f32_broadcast(src1, i0, i1, i2, i3);
                        }
                    }
                }
            }
            if (exec_rows > entry->rows) {
                float * in1_pad = reinterpret_cast<float *>(entry->input_tensors[1].data().cptr());
                const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(float);
                for (int64_t r = entry->rows; r < exec_rows; ++r) {
                    std::memcpy(
                        in1_pad + static_cast<size_t>(r * entry->cols),
                        in1_pad + static_cast<size_t>((entry->rows - 1) * entry->cols),
                        row_bytes);
                }
            }
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SCALE) {
            float scale = 1.0f;
            if (node->src[1] && node->src[1]->data && ggml_nelements(node->src[1]) >= 1) {
                // SCALE's factor tensor can be a view/strided scalar; always read via ggml strides.
                scale = ggml_fmsh_read_f32_broadcast(node->src[1], 0, 0, 0, 0);
            } else {
                std::memcpy(&scale, node->op_params, sizeof(float));
            }
            std::memcpy(entry->input_tensors[1].data().cptr(), &scale, sizeof(float));
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::CPY ||
                   entry->op == ggml::fmsh::netmake::ElementwiseZgOp::DUP) {
            const float one = 1.0f;
            std::memcpy(entry->input_tensors[1].data().cptr(), &one, sizeof(float));
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params, sizeof(float));
            std::memcpy(entry->input_tensors[1].data().cptr(), &eps, sizeof(float));
        }
        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX &&
            node->src[1] != nullptr &&
            node->src[1]->data != nullptr &&
            entry->input_tensors.size() > 1 &&
            entry->input_tensors[1].defined() &&
            exec_rows > entry->rows) {
            float * in1_pad = reinterpret_cast<float *>(entry->input_tensors[1].data().cptr());
            const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(float);
            for (int64_t r = entry->rows; r < exec_rows; ++r) {
                std::memcpy(
                    in1_pad + static_cast<size_t>(r * entry->cols),
                    in1_pad + static_cast<size_t>((entry->rows - 1) * entry->cols),
                    row_bytes);
            }
        }

        std::vector<Tensor> session_inputs = entry->input_tensors;
        session_inputs[0] = input0;
        auto outputs = entry->session.forward(session_inputs);
        if (outputs.empty()) {
            if (err) *err = "elementwise session.forward returned empty output";
            return false;
        }
        if (chained_output != nullptr) {
            *chained_output = outputs[0];
        }
        ggml_fmsh_log_locked(ctx, 1, "elementwise_forward_done op=" + std::string(ggml_op_name(node->op)));

        if (!keep_device_output_only) {
            std::vector<float> out_tmp(total_elems);
            outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);
            ggml_fmsh_log_locked(ctx, 1, "elementwise_read_done op=" + std::string(ggml_op_name(node->op)));
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            ggml_fmsh_write_f32_indexed(
                                node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                        }
                    }
                }
            }
        } else {
            ggml_fmsh_log_locked(ctx, 0, "elementwise_read_skip op=" + std::string(ggml_op_name(node->op)));
        }
        ggml_fmsh_log_locked(ctx, 1, "elementwise_end op=" + std::string(ggml_op_name(node->op)));
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

static inline float ggml_fmsh_read_f32_broadcast(
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const int64_t x0 = t->ne[0] == 1 ? 0 : i0;
    const int64_t x1 = t->ne[1] == 1 ? 0 : i1;
    const int64_t x2 = t->ne[2] == 1 ? 0 : i2;
    const int64_t x3 = t->ne[3] == 1 ? 0 : i3;
    const char * p = static_cast<const char *>(t->data) + x0 * t->nb[0] + x1 * t->nb[1] + x2 * t->nb[2] + x3 * t->nb[3];
    return *reinterpret_cast<const float *>(p);
}

static inline void ggml_fmsh_write_f32_indexed(
    ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3,
    float v) {
    char * p = static_cast<char *>(t->data) + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    *reinterpret_cast<float *>(p) = v;
}

static inline float ggml_fmsh_read_f32_indexed(
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const char * p = static_cast<const char *>(t->data) + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    return *reinterpret_cast<const float *>(p);
}

static inline float ggml_fmsh_read_f32_linear(const ggml_tensor * t, size_t idx) {
    const size_t ne0 = static_cast<size_t>(t->ne[0]);
    const size_t ne1 = static_cast<size_t>(t->ne[1]);
    const size_t ne2 = static_cast<size_t>(t->ne[2]);
    const size_t i0 = idx % ne0;
    idx /= ne0;
    const size_t i1 = idx % ne1;
    idx /= ne1;
    const size_t i2 = idx % ne2;
    idx /= ne2;
    const size_t i3 = idx;
    return ggml_fmsh_read_f32_indexed(
        t,
        static_cast<int64_t>(i0),
        static_cast<int64_t>(i1),
        static_cast<int64_t>(i2),
        static_cast<int64_t>(i3));
}

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
static bool ggml_fmsh_tensor_snapshot_f32(const ggml_tensor * t, std::vector<float> & out, std::string * err) {
    if (!t || !t->data) {
        if (err) *err = "tensor data null";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) *err = "tensor type is not F32";
        return false;
    }
    const size_t n = ggml_nelements(t);
    out.resize(n);
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    out[idx++] = ggml_fmsh_read_f32_indexed(t, i0, i1, i2, i3);
                }
            }
        }
    }
    return true;
}

static bool ggml_fmsh_tensor_restore_f32(ggml_tensor * t, const std::vector<float> & in, std::string * err) {
    if (!t || !t->data) {
        if (err) *err = "tensor data null";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) *err = "tensor type is not F32";
        return false;
    }
    const size_t n = ggml_nelements(t);
    if (in.size() != n) {
        if (err) *err = "tensor restore size mismatch";
        return false;
    }
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    ggml_fmsh_write_f32_indexed(t, i0, i1, i2, i3, in[idx++]);
                }
            }
        }
    }
    return true;
}

struct ggml_fmsh_debug_saved_tensor {
    ggml_tensor * t = nullptr;
    std::vector<float> data;
};

static bool ggml_fmsh_debug_backup_tensor(
    ggml_tensor * t,
    std::vector<ggml_fmsh_debug_saved_tensor> & saved,
    std::unordered_set<ggml_tensor *> & seen,
    std::string * err) {
    if (!t || !t->data || t->type != GGML_TYPE_F32) {
        return true;
    }
    if (!seen.insert(t).second) {
        return true;
    }
    ggml_fmsh_debug_saved_tensor s;
    s.t = t;
    if (!ggml_fmsh_tensor_snapshot_f32(t, s.data, err)) {
        return false;
    }
    saved.push_back(std::move(s));
    return true;
}

static bool ggml_fmsh_debug_restore_tensors(
    std::vector<ggml_fmsh_debug_saved_tensor> & saved,
    std::string * err) {
    for (auto & s : saved) {
        if (!ggml_fmsh_tensor_restore_f32(s.t, s.data, err)) {
            return false;
        }
    }
    return true;
}

static void ggml_fmsh_debug_compare_and_log(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    const std::vector<float> & ref,
    int node_index) {
    if (!ctx->debug_compare || !node || node->type != GGML_TYPE_F32 || !node->data) {
        return;
    }
    const size_t n = ggml_nelements(node);
    if (ref.size() != n || n == 0) {
        return;
    }

    size_t mismatches = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t first_bad = static_cast<size_t>(-1);
    float first_ref = 0.0f;
    float first_npu = 0.0f;

    size_t idx = 0;
    for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                    const float got = ggml_fmsh_read_f32_indexed(node, i0, i1, i2, i3);
                    const float exp = ref[idx];
                    const double abs_err = std::fabs(static_cast<double>(got) - static_cast<double>(exp));
                    const double rel_err = abs_err / (std::fabs(static_cast<double>(exp)) + 1e-12);
                    if (abs_err > max_abs) max_abs = abs_err;
                    if (rel_err > max_rel) max_rel = rel_err;
                    if (abs_err > ctx->debug_compare_atol + ctx->debug_compare_rtol * std::fabs(static_cast<double>(exp))) {
                        if (first_bad == static_cast<size_t>(-1)) {
                            first_bad = idx;
                            first_ref = exp;
                            first_npu = got;
                        }
                        ++mismatches;
                    }
                    ++idx;
                }
            }
        }
    }

    ctx->debug_compare_total++;
    if (mismatches > 0) {
        ctx->debug_compare_mismatch++;
        if (ctx->debug_compare_logged < ctx->debug_compare_log_limit) {
            ctx->debug_compare_logged++;
            ggml_fmsh_log_locked(
                ctx, 2,
                "debug_compare_mismatch op=" + std::string(ggml_op_name(node->op)) +
                " node_idx=" + std::to_string(node_index) +
                " shape=[" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," +
                std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + "]" +
                " n=" + std::to_string(n) +
                " mismatches=" + std::to_string(mismatches) +
                " max_abs=" + std::to_string(max_abs) +
                " max_rel=" + std::to_string(max_rel) +
                " first_bad_idx=" + std::to_string(first_bad) +
                " first_ref=" + std::to_string(first_ref) +
                " first_npu=" + std::to_string(first_npu));
        }
    }
}
#endif

static bool ggml_fmsh_execute_mul_mat(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_session_entry * entry,
    const Tensor * input_override,
    bool keep_device_output_only,
    Tensor * chained_output,
    std::string * err) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1 || src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        if (err) *err = "MUL_MAT tensor data is null";
        return false;
    }

    try {
        struct ggml_fmsh_f32_stats {
            float min_v = 0.0f;
            float max_v = 0.0f;
            size_t inf_cnt = 0;
            size_t nan_cnt = 0;
        };
        const auto calc_f32_stats = [](const float * data, size_t n) -> ggml_fmsh_f32_stats {
            ggml_fmsh_f32_stats s;
            if (!data || n == 0) {
                return s;
            }
            s.min_v = data[0];
            s.max_v = data[0];
            for (size_t i = 0; i < n; ++i) {
                const float v = data[i];
                if (std::isnan(v)) {
                    s.nan_cnt++;
                    continue;
                }
                if (std::isinf(v)) {
                    s.inf_cnt++;
                    continue;
                }
                if (v < s.min_v) s.min_v = v;
                if (v > s.max_v) s.max_v = v;
            }
            return s;
        };
        static std::atomic<int> s_inf_trace_budget{20};

        if (entry->bundle.network.inputs().size() < 2) {
            if (err) *err = "compiled network input count < 2";
            return false;
        }

        if (!entry->input_tensor_a_ready) {
            entry->input_tensor_a = Tensor(entry->input_type_a.clone());
            entry->input_tensor_a.mallocOn(HostDevice::MemRegion());
            entry->input_tensor_a_ready = true;
        }

        const int64_t ne02 = src0->ne[2];
        const int64_t ne03 = src0->ne[3];
        const int64_t ne12 = src1->ne[2];
        const int64_t ne13 = src1->ne[3];
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        const size_t out_bytes = static_cast<size_t>(entry->m * entry->n * static_cast<int64_t>(sizeof(float)));
        std::vector<float> out_tmp;
        if (!keep_device_output_only) {
            out_tmp.resize(static_cast<size_t>(entry->m * entry->n));
        }

        const bool allow_input_chain = input_override != nullptr && ne12 == 1 && ne13 == 1;
        if (chained_output != nullptr) {
            *chained_output = Tensor();
        }

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            const int64_t i03 = i13 / r3;
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                const int64_t i02 = i12 / r2;

                const char * src1_slice = static_cast<const char *>(src1->data) + i12 * src1->nb[2] + i13 * src1->nb[3];
                const char * src0_slice = static_cast<const char *>(src0->data) + i02 * src0->nb[2] + i03 * src0->nb[3];
                char * dst_slice = static_cast<char *>(node->data) + i12 * node->nb[2] + i13 * node->nb[3];

                Tensor input_a = entry->input_tensor_a;
                if (allow_input_chain && i12 == 0 && i13 == 0) {
                    input_a = *input_override;
                } else {
                    // A = src1 slice : [m, k]
                    float * a_dst = reinterpret_cast<float *>(entry->input_tensor_a.data().cptr());
                    for (int64_t row = 0; row < entry->m; ++row) {
                        const char * a_src_row = src1_slice + row * src1->nb[1];
                        float * a_row_dst = a_dst + row * entry->k;
                        if (src1->nb[0] == static_cast<size_t>(sizeof(float))) {
                            std::memcpy(a_row_dst, a_src_row, static_cast<size_t>(entry->k * sizeof(float)));
                        } else {
                            for (int64_t col = 0; col < entry->k; ++col) {
                                float v = 0.0f;
                                std::memcpy(&v, a_src_row + col * src1->nb[0], sizeof(float));
                                a_row_dst[col] = v;
                            }
                        }
                    }
                }

                // Only cache transpose for static src0 tensors.
                // Dynamic tensors (e.g. KV-related views) can keep same address but changed content across tokens.
                const bool src0_is_static =
                    (src0->view_src == nullptr) &&
                    (src0->op == GGML_OP_NONE) &&
                    ((src0->flags & GGML_TENSOR_FLAG_INPUT) == 0);

                Tensor dynamic_weight;
                const uintptr_t weight_key = reinterpret_cast<uintptr_t>(src0_slice);
                Tensor * weight_tensor = nullptr;
                if (src0_is_static) {
                    for (auto & cached : entry->weight_tensors) {
                        if (cached.key == weight_key) {
                            weight_tensor = &cached.tensor;
                            break;
                        }
                    }
                }

                if (!weight_tensor) {
                    Tensor new_weight(entry->input_type_b.clone());
                    new_weight.mallocOn(HostDevice::MemRegion());

                    // B = transpose(src0 slice): src0 [n, k] -> B [k, n]
                    float * w_dst = reinterpret_cast<float *>(new_weight.data().cptr());
                    for (int64_t i = 0; i < entry->n; ++i) {
                        const char * w_src_row = src0_slice + i * src0->nb[1];
                        for (int64_t j = 0; j < entry->k; ++j) {
                            float v = 0.0f;
                            std::memcpy(&v, w_src_row + j * src0->nb[0], sizeof(float));
                            w_dst[static_cast<size_t>(j * entry->n + i)] = v;
                        }
                    }

                    if (src0_is_static) {
                        entry->weight_tensors.push_back({weight_key, std::move(new_weight)});
                        weight_tensor = &entry->weight_tensors.back().tensor;
                    } else {
                        dynamic_weight = std::move(new_weight);
                        weight_tensor = &dynamic_weight;
                    }
                }

                auto outputs = entry->session.forward({input_a, *weight_tensor});
                if (outputs.empty()) {
                    if (err) *err = "session.forward returned empty output";
                    return false;
                }
                if (chained_output != nullptr && i12 == 0 && i13 == 0 && ne12 == 1 && ne13 == 1) {
                    *chained_output = outputs[0];
                }

                if (!keep_device_output_only) {
                    outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, out_bytes);

                    const ggml_fmsh_f32_stats out_stats =
                        calc_f32_stats(out_tmp.data(), static_cast<size_t>(entry->m * entry->n));
                    if ((out_stats.inf_cnt > 0 || out_stats.nan_cnt > 0) &&
                        s_inf_trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                        const float * a_ptr = reinterpret_cast<const float *>(input_a.data().cptr());
                        const float * w_ptr = reinterpret_cast<const float *>(weight_tensor->data().cptr());
                        const ggml_fmsh_f32_stats a_stats =
                            calc_f32_stats(a_ptr, static_cast<size_t>(entry->m * entry->k));
                        const ggml_fmsh_f32_stats w_stats =
                            calc_f32_stats(w_ptr, static_cast<size_t>(entry->k * entry->n));
                        ggml_fmsh_log_locked(
                            ctx, 2,
                            "mul_mat_inf_trace shape=[" + std::to_string(node->ne[0]) + "," +
                            std::to_string(node->ne[1]) + "," +
                            std::to_string(node->ne[2]) + "," +
                            std::to_string(node->ne[3]) + "]" +
                            " tile_i12=" + std::to_string(i12) +
                            " tile_i13=" + std::to_string(i13) +
                            " src0_nb0=" + std::to_string(src0->nb[0]) +
                            " src1_nb0=" + std::to_string(src1->nb[0]) +
                            " dst_nb0=" + std::to_string(node->nb[0]) +
                            " A[min,max,inf,nan]=[" + std::to_string(a_stats.min_v) + "," +
                            std::to_string(a_stats.max_v) + "," +
                            std::to_string(a_stats.inf_cnt) + "," +
                            std::to_string(a_stats.nan_cnt) + "]" +
                            " W[min,max,inf,nan]=[" + std::to_string(w_stats.min_v) + "," +
                            std::to_string(w_stats.max_v) + "," +
                            std::to_string(w_stats.inf_cnt) + "," +
                            std::to_string(w_stats.nan_cnt) + "]" +
                            " Y[min,max,inf,nan]=[" + std::to_string(out_stats.min_v) + "," +
                            std::to_string(out_stats.max_v) + "," +
                            std::to_string(out_stats.inf_cnt) + "," +
                            std::to_string(out_stats.nan_cnt) + "]");
                    }

                    for (int64_t row = 0; row < entry->m; ++row) {
                        char * dst_row = dst_slice + row * node->nb[1];
                        const float * src_row = out_tmp.data() + row * entry->n;
                        if (node->nb[0] == static_cast<size_t>(sizeof(float))) {
                            std::memcpy(dst_row, src_row, static_cast<size_t>(entry->n * sizeof(float)));
                        } else {
                            for (int64_t col = 0; col < entry->n; ++col) {
                                std::memcpy(dst_row + col * node->nb[0], &src_row[col], sizeof(float));
                            }
                        }
                    }
                }
            }
        }
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

static void ggml_fmsh_accumulate_profile(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    const Session & session,
    double total_ms) {
    auto & p = ctx->perf[static_cast<uint32_t>(node->op)];
    p.calls++;
    p.total_ms += total_ms;

    const auto profile = session.timeProfileResults();
    double memcpy_ms = 0.0;
    double hard_ms = 0.0;
    for (const auto & kv : profile) {
        const auto & t = kv.second;
        memcpy_ms += std::get<1>(t);
        hard_ms += std::get<2>(t);
    }
    p.memcpy_ms += memcpy_ms;
    p.hard_ms += hard_ms;
}

static bool ggml_fmsh_is_batched_mul_mat(const ggml_tensor * node) {
    return node != nullptr && node->op == GGML_OP_MUL_MAT && (node->ne[2] > 1 || node->ne[3] > 1);
}

static std::string ggml_fmsh_tensor_loc_str(ggml_backend_fmsh_zg330_context * ctx, const Tensor & t) {
    if (!t.defined() || !t.hasData()) {
        return "none";
    }
    if (t.isOn(HostDevice::MemRegion())) {
        return "Host";
    }
    if (ctx->device_opened && t.isOn(ctx->zg_device.defaultMemRegion())) {
        return "ZG330";
    }
    return "Other";
}

static void ggml_fmsh_materialize_if_device(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    std::unordered_map<const ggml_tensor *, Tensor> & device_tensor_map) {
    if (!t || t->data == nullptr) {
        return;
    }
    auto it = device_tensor_map.find(t);
    if (it == device_tensor_map.end()) {
        return;
    }
    const size_t bytes = ggml_nbytes(t);
    if (bytes > 0) {
        it->second.read(reinterpret_cast<char *>(t->data), 0, bytes);
    }
    ggml_fmsh_log_locked(
        ctx, 1,
        "materialize_to_host bytes=" + std::to_string(bytes) +
        " op_src=" + std::string(ggml_op_name(t->op)));
    device_tensor_map.erase(it);
}

static int ggml_fmsh_count_future_consumers(ggml_cgraph * cgraph, int from_idx, const ggml_tensor * producer) {
    int cnt = 0;
    for (int j = from_idx + 1; j < cgraph->n_nodes; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == producer) {
                ++cnt;
            }
        }
    }
    return cnt;
}

static enum ggml_status ggml_fmsh_compute_cpu_fallback(ggml_backend_fmsh_zg330_context * ctx, ggml_cgraph * cgraph) {
    if (!ctx->cpu_backend) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute(ctx->cpu_backend, cgraph);
}

static enum ggml_status ggml_fmsh_compute_cpu_node(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_cgraph * cgraph,
    int node_index) {
    if (!ctx->cpu_backend) {
        return GGML_STATUS_FAILED;
    }
    ggml_cgraph node_view = ggml_graph_view(cgraph, node_index, node_index + 1);
    return ggml_backend_graph_compute(ctx->cpu_backend, &node_view);
}

static const char * ggml_backend_fmsh_zg330_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FMSH_ZG330";
}

static void ggml_backend_fmsh_zg330_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_fmsh_zg330_context *>(backend->context);
    if (ctx) {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ggml_fmsh_log_locked(
            ctx, 1,
            "backend free: cache_size=" + std::to_string(ctx->session_cache.size()) +
            " elementwise_cache_size=" + std::to_string(ctx->elementwise_session_cache.size()));
        for (const auto & kv : ctx->perf) {
            const auto & p = kv.second;
            ggml_fmsh_log_locked(
                ctx, 1,
                "op_summary op=" + std::to_string(kv.first) +
                " calls=" + std::to_string(p.calls) +
                " total_ms=" + std::to_string(p.total_ms) +
                " memcpy_ms=" + std::to_string(p.memcpy_ms) +
                " hard_ms=" + std::to_string(p.hard_ms));
        }
        const double mm_hit = ctx->mul_mat_total == 0 ? 0.0 : (100.0 * static_cast<double>(ctx->mul_mat_offloaded) / static_cast<double>(ctx->mul_mat_total));
        const double bmm_hit = ctx->batched_mul_mat_total == 0 ? 0.0 : (100.0 * static_cast<double>(ctx->batched_mul_mat_offloaded) / static_cast<double>(ctx->batched_mul_mat_total));
        double memcpy_ratio = 0.0;
        auto it = ctx->perf.find(static_cast<uint32_t>(GGML_OP_MUL_MAT));
        if (it != ctx->perf.end() && it->second.total_ms > 0.0) {
            memcpy_ratio = 100.0 * it->second.memcpy_ms / it->second.total_ms;
        }
        ggml_fmsh_log_locked(
            ctx, 1,
            "offload_summary op=MUL_MAT total=" + std::to_string(ctx->mul_mat_total) +
            " offloaded=" + std::to_string(ctx->mul_mat_offloaded) +
            " fallback=" + std::to_string(ctx->mul_mat_fallback) +
            " hit_rate_pct=" + std::to_string(mm_hit) +
            " batched_total=" + std::to_string(ctx->batched_mul_mat_total) +
            " batched_offloaded=" + std::to_string(ctx->batched_mul_mat_offloaded) +
            " batched_hit_rate_pct=" + std::to_string(bmm_hit) +
            " chained_input_hits=" + std::to_string(ctx->mul_mat_device_input_chain) +
            " memcpy_ratio_pct=" + std::to_string(memcpy_ratio));
        if (ctx->offload_unembed) {
            const auto & up = ctx->unembed_perf;
            const auto avg = [](double sum, uint64_t n) { return n > 0 ? sum / static_cast<double>(n) : 0.0; };
            ggml_fmsh_log_locked(
                ctx, 1,
                "unembed_stage_summary calls=" + std::to_string(up.calls) +
                " exact_m_hits=" + std::to_string(up.exact_m_hits) +
                " avg_gather_ms=" + std::to_string(avg(up.gather_ms, up.exact_m_hits)) +
                " avg_device_memcpy_ms=" + std::to_string(avg(up.device_memcpy_ms, up.exact_m_hits)) +
                " avg_device_hard_ms=" + std::to_string(avg(up.device_hard_ms, up.exact_m_hits)) +
                " avg_unpack_ms=" + std::to_string(avg(up.unpack_ms, up.exact_m_hits)) +
                " avg_total_ms=" + std::to_string(avg(up.total_ms, up.exact_m_hits)) +
                " fallback_not_baked=" + std::to_string(up.fallback_not_baked) +
                " fallback_m_mismatch=" + std::to_string(up.fallback_m_mismatch));
            if (ctx->unembed_session && ctx->unembed_session->usable) {
                // Device-side artifact size, a documented proxy for on-device residency —
                // real host-RAM verification is external (see arm_test.sh VmRSS gate).
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "resident_summary raw_file_bytes=" + std::to_string(ctx->unembed_session->raw_file_bytes));
            }
        }
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
        ggml_fmsh_log_locked(
            ctx, 1,
            "debug_compare_summary total=" + std::to_string(ctx->debug_compare_total) +
            " mismatches=" + std::to_string(ctx->debug_compare_mismatch) +
            " logged=" + std::to_string(ctx->debug_compare_logged) +
            " atol=" + std::to_string(ctx->debug_compare_atol) +
            " rtol=" + std::to_string(ctx->debug_compare_rtol));
#endif
    }

    if (ctx && ctx->device_opened) {
        try {
            Device::Close(ctx->zg_device);
        } catch (...) {
            // ignore close errors
        }
        ctx->device_opened = false;
    }
    if (ctx && ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_fmsh_zg330_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_fmsh_zg330_context *>(backend->context);
    if (!ctx) {
        return GGML_STATUS_FAILED;
    }

    std::lock_guard<std::mutex> lock(ctx->mu);

    bool last_dispatched_valid = false;
    bool last_dispatched_zg = false;
    bool has_prev_device_output = false;
    Tensor prev_device_output;
    ggml_tensor * prev_device_output_node = nullptr;
    std::unordered_map<const ggml_tensor *, Tensor> device_tensor_map;

    auto find_single_future_consumer =
        [&](int from_idx, const ggml_tensor * producer, ggml_tensor ** out_node, int * out_src_idx) -> bool {
            ggml_tensor * found = nullptr;
            int found_src = -1;
            for (int j = from_idx + 1; j < cgraph->n_nodes; ++j) {
                ggml_tensor * cand = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (cand->src[s] != producer) {
                        continue;
                    }
                    if (found != nullptr) {
                        return false;
                    }
                    found = cand;
                    found_src = s;
                }
            }
            if (!found) {
                return false;
            }
            *out_node = found;
            *out_src_idx = found_src;
            return true;
        };

    auto can_consume_from_device =
        [&](ggml_tensor * consumer, int src_idx) -> bool {
            if (consumer->op == GGML_OP_MUL_MAT && src_idx == 1 && ggml_fmsh_can_run_mul_mat_zg(consumer) &&
                consumer->ne[2] == 1 && consumer->ne[3] == 1) {
                return true;
            }
            return false;
        };
    auto log_dispatch_boundary = [&](bool current_is_zg, const ggml_tensor * node, const char * reason) {
        if (!last_dispatched_valid || last_dispatched_zg == current_is_zg) {
            return;
        }
        const size_t src0_bytes = node->src[0] ? ggml_nbytes(node->src[0]) : 0;
        ggml_fmsh_log_locked(
            ctx, 1,
            std::string("fallback_boundary reason=") + reason +
            " direction=" + (current_is_zg ? "Host->ZG330" : "ZG330->Host") +
            " op=" + ggml_op_name(node->op) +
            " tensor_bytes=" + std::to_string(ggml_nbytes(node)) +
            " src0_bytes=" + std::to_string(src0_bytes));
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (ggml_fmsh_is_meta_op(node)) {
            continue;
        }
        ggml_fmsh_log_locked(
            ctx, 1,
            "node_enter op=" + std::string(ggml_op_name(node->op)) +
            " shape=[" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," +
            std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + "]" +
            " dtype=" + std::to_string(static_cast<int>(node->type)));
        if (ggml_nelements(node) == 0) {
            ggml_fmsh_log_locked(
                ctx, 0,
                "skip op=" + std::string(ggml_op_name(node->op)) +
                " reason=zero_elements shape=[" +
                std::to_string(node->ne[0]) + "," +
                std::to_string(node->ne[1]) + "," +
                std::to_string(node->ne[2]) + "," +
                std::to_string(node->ne[3]) + "]");
            has_prev_device_output = false;
            prev_device_output_node = nullptr;
            device_tensor_map.erase(node);
            last_dispatched_valid = true;
            last_dispatched_zg = false;
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT) {
            ctx->mul_mat_total++;
            if (ggml_fmsh_is_batched_mul_mat(node)) {
                ctx->batched_mul_mat_total++;
            }

            if (ggml_fmsh_is_lm_head_mul_mat(node)) {
                ctx->unembed_perf.calls++;
                const auto tu0 = std::chrono::high_resolution_clock::now();
                const ggml_tensor * src0u = node->src[0];
                const ggml_tensor * src1u = node->src[1];
                std::string uerr;
                bool uok = false;
                if (src0u == nullptr || src1u == nullptr || node->type != GGML_TYPE_F32 || src1u->type != GGML_TYPE_F32) {
                    uerr = "unembed: unsupported dtype";
                } else if (src0u->ne[2] != 1 || src0u->ne[3] != 1 || src1u->ne[2] != 1 || src1u->ne[3] != 1) {
                    uerr = "unembed: batched shapes unsupported (single fixed [m,k]->[m,n] session only)";
                } else if (!ggml_is_contiguous(src1u) || !ggml_is_contiguous(node)) {
                    uerr = "unembed: non-contiguous src1/dst unsupported";
                } else {
                    const int64_t k = src0u->ne[0];
                    const int64_t n = src0u->ne[1];
                    const int64_t m_actual = src1u->ne[1];
                    ggml_fmsh_zg330_unembed_session_entry * uentry =
                        ggml_fmsh_get_or_create_unembed_session(ctx, ctx->unembed_m, k, n, &uerr);
                    if (!uentry) {
                        ctx->unembed_perf.fallback_not_baked++;
                    } else if (m_actual != uentry->m) {
                        ctx->unembed_perf.fallback_m_mismatch++;
                        uerr = "unembed: m_actual(" + std::to_string(m_actual) + ") != unembed_m(" +
                               std::to_string(uentry->m) + ")";
                        ggml_fmsh_log_locked(
                            ctx, 1, "unembed_dispatch m_actual=" + std::to_string(m_actual) + " matched=false");
                    } else {
                        uok = ggml_fmsh_execute_unembed(ctx, node, uentry, &uerr);
                        if (!uok) {
                            ctx->unembed_perf.fallback_m_mismatch++;
                        }
                    }
                }

                if (uok) {
                    const auto tu1 = std::chrono::high_resolution_clock::now();
                    ctx->unembed_perf.total_ms += std::chrono::duration<double, std::milli>(tu1 - tu0).count();
                    ctx->mul_mat_offloaded++;
                    has_prev_device_output = false;
                    prev_device_output_node = nullptr;
                    device_tensor_map.erase(node);
                    last_dispatched_valid = true;
                    last_dispatched_zg = true;
                    continue;
                }

                // Per design: any miss here (not baked, m mismatch, forward() exception, ...) is a
                // soft CPU fallback for this single node — this op isn't routed to FMSH at all when
                // GGML_FMSH_ZG330_OFFLOAD_UNEMBED is off, so a graceful miss is strictly safer than a
                // new hard-failure surface for the whole graph.
                ggml_fmsh_log_locked(ctx, 1, "fallback op=MUL_MAT(unembed) reason=" + uerr);
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                const enum ggml_status ust = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
                if (ust != GGML_STATUS_SUCCESS) {
                    return ust;
                }
                ctx->mul_mat_fallback++;
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = false;
                continue;
            }

            if (!ggml_fmsh_can_run_mul_mat_zg(node)) {
                ggml_fmsh_log_locked(
                    ctx, 3,
                    "fatal op=MUL_MAT reason=unsupported_layout_or_dtype dst_type=" +
                    std::to_string(static_cast<int>(node->type)) +
                    " src0_type=" + std::to_string(static_cast<int>(node->src[0]->type)) +
                    " src1_type=" + std::to_string(static_cast<int>(node->src[1]->type)));
                return GGML_STATUS_FAILED;
            }

            // Temporarily skip boundary accounting for elementwise path to avoid
            // touching malformed intermediate views while bringing up operators.
            bool created = false;
            std::string err;
            ggml_fmsh_zg330_session_entry * entry = ggml_fmsh_get_or_create_mul_mat_session(ctx, node, &created, &err);
            if (!entry) {
                ggml_fmsh_log_locked(ctx, 3, "fatal op=MUL_MAT reason=" + err);
                return GGML_STATUS_FAILED;
            }

            ggml_fmsh_log_locked(
                ctx, 1,
                std::string(created ? "session_create" : "session_hit") +
                " op=MUL_MAT shape=[" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + "]" +
                " dtype=" + std::to_string(static_cast<int>(node->type)) +
                " hits=" + std::to_string(entry->hit_count) +
                " net=" + entry->bundle.net_name +
                " net_cache=" + std::string(entry->bundle.ram_cache_hit ? "HIT" : "MISS") +
                " compile_now=" + std::string((created && entry->bundle.compiled_now) ? "YES" : "NO"));

            const auto t0 = std::chrono::high_resolution_clock::now();
            const Tensor * chained_input_ptr = nullptr;
            const ggml_tensor * chained_input_src = nullptr;
            auto it_dev_in = device_tensor_map.find(node->src[1]);
            if (it_dev_in != device_tensor_map.end() && node->ne[2] == 1 && node->ne[3] == 1) {
                chained_input_ptr = &it_dev_in->second;
                chained_input_src = node->src[1];
            } else if (has_prev_device_output && prev_device_output_node == node->src[1] &&
                       node->ne[2] == 1 && node->ne[3] == 1) {
                chained_input_ptr = &prev_device_output;
                chained_input_src = node->src[1];
            }
            ggml_tensor * next_consumer = nullptr;
            int next_src_idx = -1;
            const bool keep_device_output_only =
                node->ne[2] == 1 && node->ne[3] == 1 &&
                find_single_future_consumer(i, node, &next_consumer, &next_src_idx) &&
                can_consume_from_device(next_consumer, next_src_idx);
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            std::vector<float> cpu_ref_out;
            if (ctx->debug_compare && ctx->device_url.rfind("socket://", 0) == 0) {
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                std::vector<ggml_fmsh_debug_saved_tensor> debug_saved;
                std::unordered_set<ggml_tensor *> debug_seen;
                std::string dbg_err;
                const auto backup_if_alias = [&](ggml_tensor * s) -> bool {
                    if (s == nullptr || s->data == nullptr || node->data == nullptr) {
                        return true;
                    }
                    if (s != node && s->data != node->data) {
                        return true;
                    }
                    return ggml_fmsh_debug_backup_tensor(s, debug_saved, debug_seen, &dbg_err);
                };
                if (!ggml_fmsh_debug_backup_tensor(node, debug_saved, debug_seen, &dbg_err) ||
                    !backup_if_alias(node->src[0]) ||
                    !backup_if_alias(node->src[1]) ||
                    !backup_if_alias(node->src[2])) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_backup_failed op=MUL_MAT reason=" + dbg_err);
                }
                const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
                if (cpu_st != GGML_STATUS_SUCCESS) {
                    return cpu_st;
                }
                std::string snap_err;
                if (!ggml_fmsh_tensor_snapshot_f32(node, cpu_ref_out, &snap_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=MUL_MAT reason=" + snap_err);
                }
                if (!debug_saved.empty()) {
                    std::string restore_err;
                    if (!ggml_fmsh_debug_restore_tensors(debug_saved, &restore_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "debug_compare_restore_failed op=MUL_MAT reason=" + restore_err);
                        return GGML_STATUS_FAILED;
                    }
                }
                chained_input_ptr = nullptr;
                chained_input_src = nullptr;
            }
            const bool keep_device_output_effective =
                (ctx->debug_compare && ctx->device_url.rfind("socket://", 0) == 0) ? false : keep_device_output_only;
#else
            const bool keep_device_output_effective = keep_device_output_only;
#endif
            Tensor chained_output;
            if (!ggml_fmsh_execute_mul_mat(ctx, node, entry, chained_input_ptr, keep_device_output_effective, &chained_output, &err)) {
                ggml_fmsh_log_locked(ctx, 3, "fatal op=MUL_MAT reason=" + err);
                return GGML_STATUS_FAILED;
            }
            if (chained_input_ptr != nullptr) {
                ctx->mul_mat_device_input_chain++;
                ggml_fmsh_log_locked(
                    ctx, 0,
                    "dispatch op=MUL_MAT chained_input=1 input_loc=" +
                    ggml_fmsh_tensor_loc_str(ctx, *chained_input_ptr));
                if (chained_input_src != nullptr) {
                    device_tensor_map.erase(chained_input_src);
                }
            }
            if (node->ne[2] == 1 && node->ne[3] == 1) {
                prev_device_output = chained_output;
                prev_device_output_node = node;
                has_prev_device_output = true;
                if (keep_device_output_effective) {
                    device_tensor_map[node] = chained_output;
                    ggml_fmsh_log_locked(ctx, 0, "dispatch op=MUL_MAT keep_device_output=1");
                } else {
                    device_tensor_map.erase(node);
                }
            } else {
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                device_tensor_map.erase(node);
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ggml_fmsh_accumulate_profile(ctx, node, entry->session, total_ms);
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            if (!cpu_ref_out.empty()) {
                ggml_fmsh_debug_compare_and_log(ctx, node, cpu_ref_out, i);
            }
#endif
            ctx->mul_mat_offloaded++;
            if (ggml_fmsh_is_batched_mul_mat(node)) {
                ctx->batched_mul_mat_offloaded++;
            }
            last_dispatched_valid = true;
            last_dispatched_zg = true;
            continue;
        }

        if (ggml_fmsh_should_run_elementwise_zg(ctx, node)) {
            ggml_fmsh_log_locked(ctx, 1, "elementwise_dispatch_enter op=" + std::string(ggml_op_name(node->op)));
            ggml::fmsh::netmake::ElementwiseZgOp ek;
            int64_t erows = 0;
            int64_t ecols = 0;
            if (!ggml_fmsh_validate_elementwise(node, &ek, &erows, &ecols)) {
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "fallback op=" + std::string(ggml_op_name(node->op)) +
                    " reason=unsupported_layout_or_dtype");
                const enum ggml_status st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = false;
                continue;
            }
            ggml_fmsh_log_locked(
                ctx, 1,
                "elementwise_dispatch_valid op=" + std::string(ggml_op_name(node->op)) +
                " rows=" + std::to_string(erows) +
                " cols=" + std::to_string(ecols));

            log_dispatch_boundary(true, node, "dispatch_switch");
            bool created = false;
            std::string err;
            ggml_fmsh_log_locked(ctx, 1, "elementwise_get_session op=" + std::string(ggml_op_name(node->op)));
            ggml_fmsh_zg330_elementwise_session_entry * entry =
                ggml_fmsh_get_or_create_elementwise_session(ctx, node, &created, &err);
            if (!entry) {
                ggml_fmsh_log_locked(ctx, 2, "fallback op=" + std::string(ggml_op_name(node->op)) + " reason=" + err);
                if (ctx->strict_mode) {
                    return GGML_STATUS_FAILED;
                }
                const enum ggml_status st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                last_dispatched_valid = true;
                last_dispatched_zg = false;
                continue;
            }

            ggml_fmsh_log_locked(
                ctx, 1,
                std::string(created ? "session_create" : "session_hit") +
                " op=" + std::string(ggml_op_name(node->op)) +
                " shape=[" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," +
                std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + "]" +
                " dtype=" + std::to_string(static_cast<int>(node->type)) +
                " hits=" + std::to_string(entry->hit_count) +
                " net=" + entry->bundle.net_name +
                " net_cache=" + std::string(entry->bundle.ram_cache_hit ? "HIT" : "MISS") +
                " compile_now=" + std::string((created && entry->bundle.compiled_now) ? "YES" : "NO"));

            const auto t0 = std::chrono::high_resolution_clock::now();
            const Tensor * chained_input_ptr = nullptr;
            const bool keep_device_output_only = false;
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            std::vector<float> cpu_ref_out;
            if (ctx->debug_compare && ctx->device_url.rfind("socket://", 0) == 0) {
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                std::vector<ggml_fmsh_debug_saved_tensor> debug_saved;
                std::unordered_set<ggml_tensor *> debug_seen;
                std::string dbg_err;
                const auto backup_if_alias = [&](ggml_tensor * s) -> bool {
                    if (s == nullptr || s->data == nullptr || node->data == nullptr) {
                        return true;
                    }
                    if (s != node && s->data != node->data) {
                        return true;
                    }
                    return ggml_fmsh_debug_backup_tensor(s, debug_saved, debug_seen, &dbg_err);
                };
                if (!ggml_fmsh_debug_backup_tensor(node, debug_saved, debug_seen, &dbg_err) ||
                    !backup_if_alias(node->src[0]) ||
                    !backup_if_alias(node->src[1]) ||
                    !backup_if_alias(node->src[2])) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_backup_failed op=" + std::string(ggml_op_name(node->op)) + " reason=" + dbg_err);
                }
                const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
                if (cpu_st != GGML_STATUS_SUCCESS) {
                    return cpu_st;
                }
                std::string snap_err;
                if (!ggml_fmsh_tensor_snapshot_f32(node, cpu_ref_out, &snap_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=" + std::string(ggml_op_name(node->op)) + " reason=" + snap_err);
                }
                if (!debug_saved.empty()) {
                    std::string restore_err;
                    if (!ggml_fmsh_debug_restore_tensors(debug_saved, &restore_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "debug_compare_restore_failed op=" + std::string(ggml_op_name(node->op)) + " reason=" + restore_err);
                        return GGML_STATUS_FAILED;
                    }
                }
            }
#endif
            Tensor chained_output;
            if (!ggml_fmsh_execute_elementwise(ctx, node, entry, chained_input_ptr, keep_device_output_only, &chained_output, &err)) {
                ggml_fmsh_log_locked(ctx, 2, "fallback op=" + std::string(ggml_op_name(node->op)) + " reason=" + err);
                if (ctx->strict_mode) {
                    return GGML_STATUS_FAILED;
                }
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = false;
                continue;
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ggml_fmsh_accumulate_profile(ctx, node, entry->session, total_ms);
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            if (!cpu_ref_out.empty()) {
                ggml_fmsh_debug_compare_and_log(ctx, node, cpu_ref_out, i);
            }
#endif

            has_prev_device_output = false;
            prev_device_output_node = nullptr;
            device_tensor_map.erase(node);
            last_dispatched_valid = true;
            last_dispatched_zg = true;
            continue;
        }

        if (ggml_fmsh_is_host_dispatch_op(node)) {
            log_dispatch_boundary(false, node, "host_dispatch_op");
            std::string reason = "host_dispatch_op";
            if (ggml_fmsh_is_cpy_dup_op(node) && !ctx->offload_cpy_dup) {
                reason = "cpy_dup_host_default";
            }
            ggml_fmsh_log_locked(
                ctx, 0,
                "dispatch op=" + std::string(ggml_op_name(node->op)) +
                " backend=HostBackend shape=[" +
                std::to_string(node->ne[0]) + "," +
                std::to_string(node->ne[1]) + "," +
                std::to_string(node->ne[2]) + "," +
                std::to_string(node->ne[3]) + "]" +
                " dtype=" + std::to_string(static_cast<int>(node->type)) +
                " reason=" + reason);
            ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
            const enum ggml_status st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
            if (st != GGML_STATUS_SUCCESS) {
                ggml_fmsh_log_locked(ctx, 2, "host dispatch failed op=" + std::string(ggml_op_name(node->op)));
                return st;
            }
            has_prev_device_output = false;
            prev_device_output_node = nullptr;
            device_tensor_map.erase(node);
            last_dispatched_valid = true;
            last_dispatched_zg = false;
            continue;
        }

        ggml_fmsh_log_locked(
            ctx, ctx->strict_mode ? 3 : 2,
            "fallback op=" + std::string(ggml_op_name(node->op)) +
            " bytes=" + std::to_string(ggml_nbytes(node)) +
            " reason=unsupported_op");
        if (ctx->strict_mode) {
            return GGML_STATUS_FAILED;
        }
        log_dispatch_boundary(false, node, "unsupported_op");
        ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
        ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
        ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
        const enum ggml_status st = ggml_fmsh_compute_cpu_node(ctx, cgraph, i);
        if (st != GGML_STATUS_SUCCESS) {
            return st;
        }
        has_prev_device_output = false;
        prev_device_output_node = nullptr;
        device_tensor_map.erase(node);
        last_dispatched_valid = true;
        last_dispatched_zg = false;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_fmsh_zg330_i = {
    /* .get_name                = */ ggml_backend_fmsh_zg330_get_name,
    /* .free                    = */ ggml_backend_fmsh_zg330_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_fmsh_zg330_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_fmsh_zg330_guid(void) {
    static ggml_guid guid = { 0x46, 0xac, 0x9f, 0x39, 0x21, 0x44, 0x59, 0x63, 0x91, 0x85, 0xe2, 0x66, 0x17, 0x8c, 0x04, 0x30 };
    return &guid;
}

static ggml_backend_t ggml_backend_fmsh_zg330_init_impl(void) {
    auto * ctx = new ggml_backend_fmsh_zg330_context;
    ctx->cpu_backend = ggml_backend_cpu_init();
    if (!ctx->cpu_backend) {
        delete ctx;
        return nullptr;
    }

    ctx->strict_mode = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_STRICT", false);
    ctx->enable_log = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_LOG", true);
    ctx->log_level = ggml_fmsh_get_log_level();
    ctx->offload_cpy_dup = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_CPY_DUP", false);
    ctx->offload_soft_max = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_SOFT_MAX", false);
    ctx->offload_rms_norm = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_RMS_NORM", false);
    ctx->disable_elementwise = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_DISABLE_ELEMENTWISE", false);
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
    ctx->debug_compare = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_DEBUG_COMPARE", true);
    ctx->debug_compare_atol = ggml_fmsh_get_env_double("GGML_FMSH_ZG330_DEBUG_COMPARE_ATOL", 1e-4);
    ctx->debug_compare_rtol = ggml_fmsh_get_env_double("GGML_FMSH_ZG330_DEBUG_COMPARE_RTOL", 1e-3);
    ctx->debug_compare_log_limit = ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_DEBUG_COMPARE_LOG_LIMIT", 200);
#endif

    
    const char * cache_dir = std::getenv("GGML_FMSH_ZG330_CACHE_DIR");
    ctx->cache_dir = cache_dir && cache_dir[0] ? std::filesystem::path(cache_dir) : std::filesystem::path(".cache/ggml-fmsh_zg330");
    std::error_code ec;
    std::filesystem::create_directories(ctx->cache_dir, ec);
    ctx->log_file = ctx->cache_dir / "backend.log";

    ctx->offload_unembed = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_UNEMBED", false);
    ctx->unembed_m = static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_UNEMBED_M", 1));
    {
        const char * unembed_cache_dir = std::getenv("GGML_FMSH_ZG330_UNEMBED_CACHE_DIR");
        ctx->unembed_cache_dir = unembed_cache_dir && unembed_cache_dir[0]
            ? std::filesystem::path(unembed_cache_dir)
            : ctx->cache_dir;
    }
    {
        const char * unembed_tensor_name = std::getenv("GGML_FMSH_ZG330_UNEMBED_TENSOR_NAME");
        ctx->unembed_tensor_name_override = unembed_tensor_name && unembed_tensor_name[0] ? unembed_tensor_name : "";
    }



    // delete prev log file if exists, and create a new empty log
    if (std::filesystem::exists(ctx->log_file)) {
        std::filesystem::remove(ctx->log_file, ec);
        std::ofstream ofs(ctx->log_file, std::ios::out);

        // write current timestamp to log
        const auto now = std::chrono::system_clock::now();
        const std::time_t now_c = std::chrono::system_clock::to_time_t
            (now - std::chrono::hours(24)); // backdate 24h to ensure log is always newer than any cached session
        ofs << "Log start time: " << std::ctime(&now_c) << std::flush;

        // release ofs
        ofs.close();
    }

    
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_fmsh_zg330_guid(),
        /* .iface   = */ ggml_backend_fmsh_zg330_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_fmsh_zg330_reg(), 0),
        /* .context = */ ctx,
    };

    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ggml_fmsh_log_locked(
            ctx, 1,
            "backend init strict=" + std::to_string(ctx->strict_mode ? 1 : 0) +
            " log_level=" + std::to_string(ctx->log_level) +
            " offload_cpy_dup=" + std::to_string(ctx->offload_cpy_dup ? 1 : 0) +
            " offload_soft_max=" + std::to_string(ctx->offload_soft_max ? 1 : 0) +
            " offload_rms_norm=" + std::to_string(ctx->offload_rms_norm ? 1 : 0) +
            " offload_unembed=" + std::to_string(ctx->offload_unembed ? 1 : 0) +
            " unembed_m=" + std::to_string(ctx->unembed_m) +
            " unembed_cache_dir=" + ctx->unembed_cache_dir.string() +
            " unembed_tensor_name_override=" + (ctx->unembed_tensor_name_override.empty() ? "(default)" : ctx->unembed_tensor_name_override) +
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            " debug_compare=" + std::to_string(ctx->debug_compare ? 1 : 0) +
            " debug_compare_atol=" + std::to_string(ctx->debug_compare_atol) +
            " debug_compare_rtol=" + std::to_string(ctx->debug_compare_rtol) +
#endif
            " cache_dir=" + ctx->cache_dir.string());
    }

    return backend;
}

static bool ggml_backend_is_fmsh_zg330_impl(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_fmsh_zg330_guid());
}

static void ggml_backend_fmsh_zg330_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (!ggml_backend_is_fmsh_zg330_impl(backend)) {
        return;
    }
    auto * ctx = static_cast<ggml_backend_fmsh_zg330_context *>(backend->context);
    ggml_backend_cpu_set_n_threads(ctx->cpu_backend, n_threads);
}

// device interface
static const char * ggml_backend_fmsh_zg330_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH_ZG330";
}

static const char * ggml_backend_fmsh_zg330_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH ZG330 Backend";
}

static void ggml_backend_fmsh_zg330_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_fmsh_zg330_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_fmsh_zg330_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_fmsh_zg330_device_get_name(dev);
    props->description = ggml_backend_fmsh_zg330_device_get_description(dev);
    props->type = ggml_backend_fmsh_zg330_device_get_type(dev);
    ggml_backend_fmsh_zg330_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_fmsh_zg330_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_fmsh_zg330_init_impl();
}

static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(void) {
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_fmsh_zg330_buffer_type_impl();
}

static ggml_backend_buffer_t ggml_backend_fmsh_zg330_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_fmsh_zg330_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_fmsh_is_supported_op(op);
}

static bool ggml_backend_fmsh_zg330_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_fmsh_zg330_device_i = {
    /* .get_name             = */ ggml_backend_fmsh_zg330_device_get_name,
    /* .get_description      = */ ggml_backend_fmsh_zg330_device_get_description,
    /* .get_memory           = */ ggml_backend_fmsh_zg330_device_get_memory,
    /* .get_type             = */ ggml_backend_fmsh_zg330_device_get_type,
    /* .get_props            = */ ggml_backend_fmsh_zg330_device_get_props,
    /* .init_backend         = */ ggml_backend_fmsh_zg330_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_fmsh_zg330_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_fmsh_zg330_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_fmsh_zg330_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fmsh_zg330_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// reg interface
static const char * ggml_backend_fmsh_zg330_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FMSH_ZG330";
}

static size_t ggml_backend_fmsh_zg330_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_fmsh_zg330_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_fmsh_zg330_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &dev;
}

static void * ggml_backend_fmsh_zg330_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return reinterpret_cast<void *>(ggml_backend_fmsh_zg330_set_n_threads);
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_fmsh_zg330_reg_i = {
    /* .get_name         = */ ggml_backend_fmsh_zg330_reg_get_name,
    /* .get_device_count = */ ggml_backend_fmsh_zg330_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fmsh_zg330_reg_get_device,
    /* .get_proc_address = */ ggml_backend_fmsh_zg330_reg_get_proc_address,
};

} // namespace

ggml_backend_t ggml_backend_fmsh_zg330_init(void) {
    return ggml_backend_fmsh_zg330_init_impl();
}

bool ggml_backend_is_fmsh_zg330(ggml_backend_t backend) {
    return ggml_backend_is_fmsh_zg330_impl(backend);
}

ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type(void) {
    return ggml_backend_fmsh_zg330_buffer_type_impl();
}

ggml_backend_reg_t ggml_backend_fmsh_zg330_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fmsh_zg330_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fmsh_zg330_reg)
