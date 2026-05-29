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
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
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
    TensorType output_type;
    MemChunk output_chunk;
    int64_t output_value_id = -1;
    size_t output_bytes = 0;
};

// Two-phase RMS_NORM session: avoids the ZG-unsupported Sqrt op.
// pre session: X[R,C], eps[1,1] → r_sq[R,1]  (Mul+ReduceSum+Mul+Add)
// post session: X[R,C], r_inv[R,1] → Y[R,C]   (Mul)
// Shared X: pre_input_tensors[0] == post_input_tensors[0] (same Tensor object),
// so X is only written once into the shared host buffer before both forwards.
struct ggml_fmsh_zg330_rmsnorm_split_session_entry {
    ggml_fmsh_zg330_op_signature signature;
    uint64_t hit_count = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t compiled_rows = 0;
    ggml::fmsh::netmake::RmsNormSplitNetworkBundle bundle;
    Session session_pre;
    Session session_post;
    // Shared input X tensor (pre[0] and post[0] point to the same allocation).
    TensorType x_input_type;
    Tensor     x_input_tensor;
    // Pre-only: eps[1,1].
    TensorType eps_input_type;
    Tensor     eps_input_tensor;
    // Post-only: r_inv[R,1].
    TensorType r_inv_input_type;
    Tensor     r_inv_input_tensor;
    // Post output: preallocated in ZG DDR via userConnectNetwork so output can stay on device.
    TensorType post_output_type;
    MemChunk   post_output_chunk;
    int64_t    post_output_value_id = -1;
    size_t     post_output_bytes    = 0;
};

// ROPE NeoX session: X[N,d], theta[N,d/2] → Y[N,d]
// theta is computed on host per call (position × dim-pair angles).
struct ggml_fmsh_zg330_rope_session_entry {
    ggml_fmsh_zg330_op_signature signature;
    uint64_t hit_count = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    ggml::fmsh::netmake::RopeZgNetworkBundle bundle;
    Session session;
    std::vector<TensorType> input_types;
    std::vector<Tensor>     input_tensors; // [0]=X, [1]=theta
};

// Runtime chain descriptor (not cached; rebuilt each dispatch iteration).
struct ggml_fmsh_zg330_fused_ew_chain {
    std::vector<ggml_tensor *> nodes;
    std::vector<ggml::fmsh::netmake::ElementwiseZgOp> ops;
    std::vector<int> cgraph_indices; // cgraph index of each chain node
    int64_t rows         = 0;
    int64_t cols         = 0;
    int64_t compiled_rows = 0;
};

// Cached fused session entry keyed by net_name.
struct ggml_fmsh_zg330_fused_ew_entry {
    std::string net_name;
    uint64_t hit_count = 0;
    std::vector<ggml::fmsh::netmake::ElementwiseZgOp> ops;
    int64_t rows = 0, cols = 0, compiled_rows = 0;
    ggml::fmsh::netmake::ElementwiseZgNetworkBundle bundle;
    Session session;
    std::vector<TensorType> input_types;
    std::vector<Tensor> input_tensors;
    MemChunk output_chunk;
    int64_t output_value_id = -1;
    size_t output_bytes = 0;
};

struct ggml_fmsh_zg330_bf16_bridge_session_entry {
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t hit_count = 0;
    ggml::fmsh::netmake::ElementwiseZgNetworkBundle bundle;
    Session session;
    TensorType input_type;
    Tensor input_tensor;
};

struct ggml_fmsh_flash_attn_validation {
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;
    int64_t value_dim = 0;
    int64_t q_len = 0;
    int64_t kv_len = 0;
    int64_t n_batch = 0;

    int64_t q_bucket = 0;
    int64_t kv_bucket = 0;
    int64_t softmax_cols = 0;

    bool has_mask = false;
    bool causal = false;
    bool use_logit_softcap = false;

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
};

struct ggml_fmsh_zg330_flash_attn_session_entry {
    ggml_fmsh_zg330_op_signature signature;
    ggml_fmsh_flash_attn_validation info;
    uint64_t hit_count = 0;
    ggml::fmsh::netmake::FlashAttnZgNetworkBundle bundle;
    Session session;
    std::vector<TensorType> input_types;
    std::vector<Tensor> input_tensors;

    // AXI 模式同步：Session 内部复用同一个 TensorNode，第一次 waitForReady 成功后
    // ready_=true 永久置位，后续调用立即返回导致读到上一帧数据（1帧延迟 bug）。
    // 修复：forward() 后立即 setReady(false) + 安装新 check_func_，再 waitForReady()。
    // 第一次 forward 用 layerCount 边沿触发学习增量，后续用固定增量同步。
    ZG330Device zg_device;        // AXI 模式下有效；socket 模式下为空
    uint32_t layer_increment = 0; // 每次 forward 的 layerCount 增量，第一次 forward 时学习
};

struct ggml_fmsh_zg330_buft_ctx; // forward declaration
struct ggml_fmsh_zg330_device_context; // forward declaration

struct ggml_backend_fmsh_zg330_context {
    ggml_backend_t cpu_backend = nullptr;

    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_session_entry>> session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_elementwise_session_entry>, ggml_fmsh_zg330_op_signature_hash> elementwise_session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_rmsnorm_split_session_entry>, ggml_fmsh_zg330_op_signature_hash> rmsnorm_split_session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_rope_session_entry>, ggml_fmsh_zg330_op_signature_hash> rope_session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_flash_attn_session_entry>> flash_attn_session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_bf16_bridge_session_entry>> bridge_session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_fused_ew_entry>> fused_ew_session_cache;

    bool strict_mode = false;
    bool enable_log = true;
    int log_level = 1; // 0 debug, 1 info, 2 warn, 3 error
    bool offload_cpy_dup = false;  // bf16 roundtrip destroys KV cache precision; must stay CPU
    bool offload_soft_max = true;
    bool offload_rms_norm = true;
    bool offload_flash_attn_ext = true;
    bool offload_rope = true;
    bool rms_norm_native_sqrt = false; // false=split (pre+host_sqrt+post), true=fused+Sqrt@hostt
    int64_t flash_softmax_cu = 8;
    int64_t flash_precompile_kv_depth = 1;
    int64_t flash_kv_bucket_max = 8192;

    std::filesystem::path cache_dir;
    std::filesystem::path log_file;

    Device zg_device;
    bool device_opened = false;
    std::string device_url;

    // Per-graph-compute staging buffers for ZG buffer tensors that need CPU access.
    // When a ZG buffer tensor (weight/KV) is materialized, D2H data goes here and
    // tensor->data is redirected to the staging buffer for the duration of graph_compute.
    std::unordered_map<const ggml_tensor *, std::vector<char>> zg_staging_bufs;

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
    uint64_t flash_attn_total = 0;
    uint64_t flash_attn_offloaded = 0;
    uint64_t flash_attn_fallback = 0;
    uint64_t elementwise_output_chunk_hits = 0;
    uint64_t elementwise_output_chunk_misses = 0;

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
    bool debug_compare = false;
    double debug_compare_atol = 0.1;
    double debug_compare_rtol = 0.05;
    uint64_t debug_compare_total = 0;
    uint64_t debug_compare_mismatch = 0;
    uint64_t debug_compare_log_limit = 200;
    uint64_t debug_compare_logged = 0;
#endif

    std::mutex mu;
};

// Buffer type context: holds a reference to the ZG device for allocation.
// Stored on ggml_fmsh_zg330_device_context so it outlives individual backend instances.
// backend_ctx is updated each time a new backend is initialized (may be null during probe).
// dev_ctx is always set (points back to the owning device context).
struct ggml_fmsh_zg330_buft_ctx {
    ggml_backend_fmsh_zg330_context * backend_ctx = nullptr;
    void * dev_ctx_ptr = nullptr; // opaque pointer to ggml_fmsh_zg330_device_context
};

// Per-allocation ZG DDR buffer context.
struct ggml_fmsh_zg330_buffer_ctx {
    MemChunk chunk;
    size_t   size;
};

// Forward declarations (defined after graph_compute helpers).
static bool ggml_fmsh_is_zg_buffer_tensor(
        const ggml_tensor * t, MemChunk * out_chunk, size_t * out_offset);
static ggml_backend_buffer_t ggml_fmsh_zg330_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size);

struct ggml_fmsh_zg330_device_context {
    std::mutex mu;
    Device memory_device;
    bool memory_device_opened = false;
    std::string memory_device_url;
    bool memory_cache_valid = false;
    size_t memory_free = 0;
    size_t memory_total = 0;
    // Owned ZG buffer type for this device; backend_ctx is updated on init/free.
    ggml_fmsh_zg330_buft_ctx * buft_ctx = nullptr;
    ggml_backend_buffer_type * cached_buft = nullptr; // owns the single buft object
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

static int64_t ggml_fmsh_align_up_i64(int64_t v, int64_t a) {
    if (a <= 1) {
        return v;
    }
    const int64_t r = v % a;
    return r == 0 ? v : (v + (a - r));
}

static inline uint16_t ggml_fmsh_f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static bool ggml_fmsh_wait_tensor_ready(
    const Tensor & t,
    std::string * err,
    const char * where,
    int64_t timeout_ms = 30000) {
    if (timeout_ms < 0) {
        timeout_ms = 0;
    }
    if (!t.waitForReady(std::chrono::milliseconds(timeout_ms))) {
        if (err) {
            *err = std::string(where) + " output waitForReady timeout";
        }
        return false;
    }
    return true;
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

static int64_t ggml_fmsh_flash_bucket_len(int64_t len) {
    if (len <= 1) {
        return 1;
    }
    return ggml_fmsh_next_pow2_i64(len);
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

static bool ggml_fmsh_is_supported_op(const ggml_tensor * op) {
    if (ggml_fmsh_is_meta_op(op)) {
        return true;
    }
    if (op->op == GGML_OP_FLASH_ATTN_EXT) {
        return true;
    }
    if (op->op == GGML_OP_ROPE) {
        if (op->type != GGML_TYPE_F32 || op->src[0] == nullptr || op->src[0]->type != GGML_TYPE_F32) {
            return false;
        }
        const int32_t mode = ((const int32_t *) op->op_params)[2];
        // Support NeoX (mode 2) and IMROPE (40 = NeoX|MROPE, text Qwen3.5) on PL.
        // Pure MROPE 8 (vision cross-modal) goes to CPU.
        return mode != GGML_ROPE_TYPE_MROPE;
    }
    if (ggml_fmsh_is_elementwise_zg_op(op)) {
        return true;
    }
    if (ggml_fmsh_is_host_dispatch_op(op)) {
        return true;
    }
    if (op->op == GGML_OP_SET_ROWS) {
        return true;
    }
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_BF16) {
        return false;
    }
    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

static bool ggml_fmsh_should_run_elementwise_zg(
    const ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * op) {
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
    const bool src1_is_bf16 = (src1->type == GGML_TYPE_BF16);
    if (node->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with non-F32 dst/src0 type is not supported for ZG330 offload");
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && !src1_is_bf16) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT src1 type must be F32 or BF16 for ZG330 offload");
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[1] <= 0 || src0->ne[2] <= 0 || src0->ne[3] <= 0 ||
        src1->ne[0] <= 0 || src1->ne[1] <= 0 || src1->ne[2] <= 0 || src1->ne[3] <= 0) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with non-positive dimension is not supported for ZG330 offload");
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with mismatched k dimension is not supported for ZG330 offload");
        return false;
    }
    if (node->ne[0] != src0->ne[1] || node->ne[1] != src1->ne[1] || node->ne[2] != src1->ne[2] || node->ne[3] != src1->ne[3]) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with mismatched output dimensions is not supported for ZG330 offload");
        return false;
    }
    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with incompatible broadcast dimensions is not supported for ZG330 offload");
        return false;
    }
    if (src0->nb[0] != sizeof(float) || node->nb[0] != sizeof(float)) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with non-contiguous tensors is not supported for ZG330 offload");
        return false;
    }
    if (!src1_is_bf16 && src1->nb[0] != sizeof(float)) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with non-contiguous src1 is not supported for ZG330 offload");
        return false;
    }
    if (src1_is_bf16 && src1->nb[0] != sizeof(uint16_t)) {
        ggml_fmsh_log_locked(nullptr, 2, "MUL_MAT with non-contiguous BF16 src1 is not supported for ZG330 offload");
        return false;
    }
    return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(node);
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
            // Only handle contiguous src and dst — non-contiguous dst (e.g. KV cache views)
            // would require element-by-element stride-aware writes which PL cannot do directly.
            return src0 &&
                   src0->type == GGML_TYPE_F32 &&
                   ggml_nelements(src0) == ggml_nelements(node) &&
                   ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(node);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            if (!src0 || !src1 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
                ggml_fmsh_log_locked(nullptr, 2, "ELEMENTWISE with non-F32 type is not supported for ZG330 offload");
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (src0->ne[d] != node->ne[d] && src0->ne[d] != 1) return false;
                if (src1->ne[d] != node->ne[d] && src1->ne[d] != 1) return false;
            }
            return true;
        case GGML_OP_SCALE:
            if (!src0 || src0->type != GGML_TYPE_F32) {
                ggml_fmsh_log_locked(nullptr, 2, "ELEMENTWISE with non-F32 type is not supported for ZG330 offload");
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

static float ggml_fmsh_read_mask_f32(
    const ggml_tensor * mask,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const char * p = static_cast<const char *>(mask->data) +
                     i0 * mask->nb[0] + i1 * mask->nb[1] +
                     i2 * mask->nb[2] + i3 * mask->nb[3];
    if (mask->type == GGML_TYPE_F16) {
        return GGML_FP16_TO_FP32(*reinterpret_cast<const ggml_fp16_t *>(p));
    }
    return *reinterpret_cast<const float *>(p);
}

static bool ggml_fmsh_is_probably_causal_mask(
    const ggml_tensor * mask,
    int64_t q_len,
    int64_t kv_len,
    int64_t mask_head,
    int64_t mask_batch) {
    if (!mask || !mask->data || q_len <= 0 || kv_len <= 0) {
        return false;
    }
    const int64_t rows_to_check = std::min<int64_t>(q_len, 4);
    const float k_mask_inf = -1e8f;
    for (int64_t r = 0; r < rows_to_check; ++r) {
        const int64_t q_row = r == rows_to_check - 1 ? (q_len - 1) : r;
        const int64_t causal_limit = kv_len - q_len + q_row;
        bool has_non_inf_before_limit = false;
        for (int64_t c = 0; c < kv_len; ++c) {
            const float mv = ggml_fmsh_read_mask_f32(mask, c, q_row, mask_head, mask_batch);
            if (c > causal_limit) {
                if (mv > k_mask_inf) {
                    return false;
                }
            } else if (mv > k_mask_inf) {
                has_non_inf_before_limit = true;
            }
        }
        if (!has_non_inf_before_limit) {
            return false;
        }
    }
    return true;
}

static bool ggml_fmsh_validate_flash_attn_ext(
    const ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    ggml_fmsh_flash_attn_validation * out,
    std::string * err) {
    if (!ctx || !node || !out || node->op != GGML_OP_FLASH_ATTN_EXT) {
        if (err) *err = "invalid FLASH_ATTN_EXT op";
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    const ggml_tensor * sinks = node->src[4];
    if (!q || !k || !v) {
        if (err) *err = "FLASH_ATTN_EXT missing q/k/v";
        return false;
    }
    if (sinks != nullptr) {
        if (err) *err = "FLASH_ATTN_EXT with sinks is not supported";
        return false;
    }
    if (node->type != GGML_TYPE_F32 || q->type != GGML_TYPE_F32) {
        if (err) *err = "FLASH_ATTN_EXT requires dst/q type F32";
        return false;
    }
    if ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32)) {
        if (err) *err = "FLASH_ATTN_EXT requires k/v type F16 or F32";
        return false;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32)) {
        if (err) *err = "FLASH_ATTN_EXT mask type must be F16/F32";
        return false;
    }
    if (q->nb[0] != sizeof(float) ||
        k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) ||
        node->nb[0] != sizeof(float) ||
        (mask != nullptr && mask->nb[0] != ggml_type_size(mask->type))) {
        if (err) *err = "FLASH_ATTN_EXT requires row-contiguous inputs";
        return false;
    }
    if (!ggml_is_contiguous(node) || (mask != nullptr && !ggml_is_contiguous(mask))) {
        if (err) *err = "FLASH_ATTN_EXT requires contiguous dst/mask";
        return false;
    }
    if (q->ne[0] <= 0 || q->ne[1] <= 0 || q->ne[2] <= 0 || q->ne[3] <= 0 ||
        k->ne[0] <= 0 || k->ne[1] <= 0 || k->ne[2] <= 0 || k->ne[3] <= 0 ||
        v->ne[0] <= 0 || v->ne[1] <= 0 || v->ne[2] <= 0 || v->ne[3] <= 0) {
        if (err) *err = "FLASH_ATTN_EXT has non-positive dim";
        return false;
    }

    const int64_t head_dim = q->ne[0];
    const int64_t q_len = q->ne[1];
    const int64_t n_head = q->ne[2];
    const int64_t n_batch = q->ne[3];
    const int64_t kv_len = k->ne[1];
    const int64_t n_head_kv = k->ne[2];
    const int64_t n_batch_kv = k->ne[3];
    const int64_t value_dim = v->ne[0];

    if (k->ne[0] != head_dim || v->ne[1] != kv_len || v->ne[2] != n_head_kv || v->ne[3] != n_batch_kv) {
        if (err) *err = "FLASH_ATTN_EXT q/k/v shape mismatch";
        return false;
    }
    if (node->ne[0] != value_dim || node->ne[1] != n_head || node->ne[2] != q_len || node->ne[3] != n_batch) {
        if (err) *err = "FLASH_ATTN_EXT dst shape mismatch";
        return false;
    }
    if (n_head % n_head_kv != 0 || n_batch % n_batch_kv != 0) {
        if (err) *err = "FLASH_ATTN_EXT q/k broadcast mismatch";
        return false;
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale, (const float *) node->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) node->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) node->op_params + 2, sizeof(float));
    if (logit_softcap < 0.0f) {
        if (err) *err = "FLASH_ATTN_EXT logit_softcap must be >= 0";
        return false;
    }
    if (max_bias > 0.0f && mask == nullptr) {
        if (err) *err = "FLASH_ATTN_EXT max_bias requires mask";
        return false;
    }

    bool causal = false;
    if (mask != nullptr) {
        if (mask->ne[0] < kv_len || mask->ne[1] < q_len || mask->ne[2] <= 0 || mask->ne[3] <= 0) {
            if (err) *err = "FLASH_ATTN_EXT mask shape too small";
            return false;
        }
        if (n_head % mask->ne[2] != 0 || n_batch % mask->ne[3] != 0) {
            if (err) *err = "FLASH_ATTN_EXT mask broadcast mismatch";
            return false;
        }
        causal = ggml_fmsh_is_probably_causal_mask(mask, q_len, kv_len, 0, 0);
    }

    out->n_head = n_head;
    out->n_head_kv = n_head_kv;
    out->head_dim = head_dim;
    out->value_dim = value_dim;
    out->q_len = q_len;
    out->kv_len = kv_len;
    out->n_batch = n_batch;
    out->q_bucket = ggml_fmsh_flash_bucket_len(q_len);
    out->kv_bucket = ggml_fmsh_flash_bucket_len(kv_len);
    // icraft AXI bug: q_bucket=2 produces incorrect NPU results regardless
    // of the compiled network shape. Fall back to CPU for q_len=2.
    if (out->q_bucket == 2) {
        if (err) *err = "FLASH_ATTN_EXT q_bucket=2 is unsupported on AXI (icraft bug)";
        return false;
    }
    out->softmax_cols = ggml_fmsh_align_up_i64(out->q_bucket, std::max<int64_t>(1, ctx->flash_softmax_cu));
    out->has_mask = mask != nullptr;
    out->causal = causal;
    out->scale = scale;
    out->max_bias = max_bias;
    out->logit_softcap = logit_softcap;
    out->use_logit_softcap = std::fabs(logit_softcap) > 0.0f;
    return true;
}

static std::string ggml_fmsh_make_flash_attn_cache_key(
    const ggml_tensor * node,
    const ggml_fmsh_flash_attn_validation & v) {
    return std::to_string(static_cast<uint32_t>(node->op)) + "|" +
           std::to_string(static_cast<uint32_t>(node->type)) + "|" +
           "nh=" + std::to_string(v.n_head) +
           "|nkvh=" + std::to_string(v.n_head_kv) +
           "|hd=" + std::to_string(v.head_dim) +
           "|dv=" + std::to_string(v.value_dim) +
           "|q=" + std::to_string(v.q_bucket) +
           "|kv=" + std::to_string(v.kv_bucket) +
           "|scol=" + std::to_string(v.softmax_cols) +
           "|mask=" + std::to_string(v.has_mask ? 1 : 0) +
           "|causal=" + std::to_string(v.causal ? 1 : 0) +
           "|softcap=" + std::to_string(v.use_logit_softcap ? 1 : 0);
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

static bool ggml_fmsh_query_pl_memory(Device & device, size_t * free_bytes, size_t * total_bytes, std::string * err) {
    try {
        MemRegion region;
        try {
            region = device.getMemRegion("plddr");
        } catch (...) {
            region = device.defaultMemRegion();
        }

        const MemManager manager = region.memManager();
        const auto & info = manager.getMemRegionInfo();
        auto total_it = info.find("byte_size");
        if (total_it == info.end()) {
            if (err) {
                *err = "PL memory region does not expose byte_size";
            }
            return false;
        }

        const auto * total_imm = total_it->second.as<icraft::xir::IntImm::NodeType>();
        if (!total_imm || total_imm->value <= 0) {
            if (err) {
                *err = "PL memory region byte_size is invalid";
            }
            return false;
        }

        uint64_t allocated = 0;
        for (const MemChunk & chunk : manager.getAllMemChunk()) {
            if (chunk.defined()) {
                allocated += chunk->byte_size;
            }
        }

        const uint64_t total_u64 = static_cast<uint64_t>(total_imm->value);
        const uint64_t free_u64 = allocated < total_u64 ? total_u64 - allocated : 0;
        const uint64_t max_size_t = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
        *total_bytes = static_cast<size_t>(std::min(total_u64, max_size_t));
        *free_bytes = static_cast<size_t>(std::min(free_u64, max_size_t));
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    } catch (...) {
        if (err) {
            *err = "unknown PL memory query error";
        }
        return false;
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

    const bool enable_user_connect =
        kind == ggml::fmsh::netmake::ElementwiseZgOp::MUL ||
        kind == ggml::fmsh::netmake::ElementwiseZgOp::ADD ||
        kind == ggml::fmsh::netmake::ElementwiseZgOp::SCALE ||
        kind == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM;

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_elementwise_zg_network(ctx->cache_dir, kind, compile_rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);

        MemChunk output_chunk;
        TensorType output_type;
        int64_t output_value_id = -1;
        size_t output_bytes = 0;
        if (enable_user_connect) {
            auto zg_backend = session->backends.at(0).cast<zg330::ZG330Backend>();
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork prepare op=" + std::string(ggml_op_name(node->op)));
            const auto & out_values = bundle.network.outputs();
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork outputs op=" + std::string(ggml_op_name(node->op)) +
                " count=" + std::to_string(out_values.size()));
            if (out_values.size() != 1) {
                if (err) *err = "fatal: elementwise outputs size != 1 for userConnectNetwork";
                return nullptr;
            }
            const auto & out_value = out_values[0];
            output_type = out_value.tensorType().clone();
            output_value_id = out_value->v_id;
            output_bytes = out_value.storageBytes();
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork alloc op=" + std::string(ggml_op_name(node->op)) +
                " v_id=" + std::to_string(output_value_id) +
                " bytes=" + std::to_string(output_bytes));
            output_chunk = ctx->zg_device.defaultMemRegion().malloc(output_bytes, true, 4096);
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork alloc_ok op=" + std::string(ggml_op_name(node->op)));

            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork try_backend op=" + std::string(ggml_op_name(node->op)));
            zg_backend.userConnectNetwork(output_chunk, output_value_id);
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork ok op=" + std::string(ggml_op_name(node->op)) +
                " v_id=" + std::to_string(output_value_id) +
                " bytes=" + std::to_string(output_bytes));
        }
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
        entry->output_type = output_type;
        entry->output_chunk = output_chunk;
        entry->output_value_id = output_value_id;
        entry->output_bytes = output_bytes;
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

static ggml_fmsh_zg330_rmsnorm_split_session_entry * ggml_fmsh_get_or_create_rmsnorm_split_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    ggml::fmsh::netmake::ElementwiseZgOp kind;
    int64_t rows = 0, cols = 0;
    if (!ggml_fmsh_validate_elementwise(node, &kind, &rows, &cols)) {
        if (err) *err = "invalid rmsnorm shape/layout";
        return nullptr;
    }
    if (kind != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
        if (err) *err = "not an RMS_NORM op";
        return nullptr;
    }
    const int64_t compile_rows = ggml_fmsh_elementwise_compile_rows(kind, rows);
    const ggml_fmsh_zg330_op_signature sig = ggml_fmsh_make_elementwise_signature(node, kind, compile_rows, cols);
    auto it = ctx->rmsnorm_split_session_cache.find(sig);
    if (it != ctx->rmsnorm_split_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }
    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }
    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_rmsnorm_split_zg_networks(
            ctx->cache_dir, compile_rows, cols);

        Session session_pre = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network_pre.view(0), {ctx->zg_device, HostDevice::Default()});
        session_pre.enableTimeProfile(true);
        session_pre.apply();

        Session session_post = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network_post.view(0), {ctx->zg_device, HostDevice::Default()});
        session_post.enableTimeProfile(true);

        // Preallocate post output in ZG DDR so the result can stay on device and
        // be consumed directly by the next elementwise op (e.g. MUL) without D2H.
        MemChunk   post_output_chunk;
        TensorType post_output_type;
        int64_t    post_output_value_id = -1;
        size_t     post_output_bytes    = 0;
        {
            auto zg_backend = session_post->backends.at(0).cast<zg330::ZG330Backend>();
            const auto & out_values = bundle.network_post.outputs();
            if (out_values.size() == 1) {
                post_output_type     = out_values[0].tensorType().clone();
                post_output_value_id = out_values[0]->v_id;
                post_output_bytes    = out_values[0].storageBytes();
                post_output_chunk    = ctx->zg_device.defaultMemRegion().malloc(post_output_bytes, true, 4096);
                zg_backend.userConnectNetwork(post_output_chunk, post_output_value_id);
                ggml_fmsh_log_locked(ctx, 1,
                    "rmsnorm_post userConnectNetwork v_id=" + std::to_string(post_output_value_id) +
                    " bytes=" + std::to_string(post_output_bytes));
            }
        }
        session_post.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_rmsnorm_split_session_entry>();
        entry->signature     = sig;
        entry->hit_count     = 1;
        entry->rows          = rows;
        entry->cols          = cols;
        entry->compiled_rows = compile_rows;
        entry->bundle        = std::move(bundle);
        entry->session_pre   = std::move(session_pre);
        entry->session_post  = std::move(session_post);
        entry->post_output_type     = post_output_type;
        entry->post_output_chunk    = post_output_chunk;
        entry->post_output_value_id = post_output_value_id;
        entry->post_output_bytes    = post_output_bytes;
        // Extract input types from both networks; X type from pre[0] (shared with post[0]).
        const auto & pre_inputs = entry->bundle.network_pre.inputs();
        const auto & post_inputs = entry->bundle.network_post.inputs();
        if (pre_inputs.size() < 2 || post_inputs.size() < 2) {
            if (err) *err = "rmsnorm_split: unexpected network input count";
            return nullptr;
        }
        entry->x_input_type    = pre_inputs[0].tensorType().clone();
        entry->eps_input_type  = pre_inputs[1].tensorType().clone();
        entry->r_inv_input_type = post_inputs[1].tensorType().clone();

        ggml_fmsh_zg330_rmsnorm_split_session_entry * ptr = entry.get();
        ctx->rmsnorm_split_session_cache.emplace(sig, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

static ggml_fmsh_zg330_rope_session_entry * ggml_fmsh_get_or_create_rope_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    if (!node || node->op != GGML_OP_ROPE) {
        if (err) *err = "not a ROPE op";
        return nullptr;
    }
    if (node->type != GGML_TYPE_F32 || !node->src[0] || node->src[0]->type != GGML_TYPE_F32) {
        if (err) *err = "ROPE: unsupported dtype";
        return nullptr;
    }
    const int32_t rope_mode = ((const int32_t *) node->op_params)[2];
    // Support NeoX (mode 2), IMROPE (mode 40=NEOX|MROPE|8) used by Qwen3.5 text.
    // For text-only IMROPE the 4 position groups (t/h/w/extra) are all equal,
    // so the rotation is identical to plain NeoX — we read pos[i2] (time positions).
    // Pure MROPE mode 8 (vision cross-modal) is not supported.
    const bool is_neox  = (rope_mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool is_imrope = rope_mode == GGML_ROPE_TYPE_IMROPE; // 40
    const bool is_mrope_only = (rope_mode == GGML_ROPE_TYPE_MROPE);   // 8, vision/cross-modal
    if (!is_neox && !is_imrope && rope_mode != 0) {
        if (err) *err = "ROPE: unsupported mode (not NeoX/IMROPE)";
        return nullptr;
    }
    if (is_mrope_only) {
        if (err) *err = "ROPE: pure MROPE (vision) not supported on PL";
        return nullptr;
    }
    const int64_t cols = node->ne[0]; // head_dim
    if (cols <= 0 || cols % 2 != 0) {
        if (err) *err = "ROPE: head_dim must be even";
        return nullptr;
    }
    const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3]; // n_head * seq * batch
    if (rows <= 0) {
        if (err) *err = "ROPE: zero rows";
        return nullptr;
    }

    ggml_fmsh_zg330_op_signature sig = {};
    sig.op = static_cast<uint32_t>(node->op);
    sig.dtype = static_cast<uint32_t>(node->type);
    sig.shape[0] = cols;
    sig.shape[1] = rows;
    sig.shape[2] = 0;
    sig.shape[3] = 0;

    auto it = ctx->rope_session_cache.find(sig);
    if (it != ctx->rope_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_rope_zg_network(ctx->cache_dir, rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_rope_session_entry>();
        entry->signature = sig;
        entry->hit_count = 1;
        entry->rows = rows;
        entry->cols = cols;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        for (const auto & in : entry->bundle.network.inputs()) {
            entry->input_types.push_back(in.tensorType().clone());
        }
        entry->input_tensors.resize(entry->input_types.size());

        ggml_fmsh_zg330_rope_session_entry * ptr = entry.get();
        ctx->rope_session_cache.emplace(sig, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

static ggml_fmsh_zg330_fused_ew_entry * ggml_fmsh_get_or_create_fused_ew_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_fmsh_zg330_fused_ew_chain & chain,
    bool * created,
    std::string * err) {
    // Build net_name: "fused_add_rmsnorm_mul_bf16_2x1024"
    auto op_name_str = [](ggml::fmsh::netmake::ElementwiseZgOp op) -> std::string {
        switch (op) {
            case ggml::fmsh::netmake::ElementwiseZgOp::ADD:      return "add";
            case ggml::fmsh::netmake::ElementwiseZgOp::MUL:      return "mul";
            case ggml::fmsh::netmake::ElementwiseZgOp::SCALE:    return "scale";
            case ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM: return "rmsnorm";
            default:                                              return "unknown";
        }
    };
    std::string net_name = "fused";
    for (auto op : chain.ops) {
        net_name += "_" + op_name_str(op);
    }
    net_name += "_bf16_" + std::to_string(chain.compiled_rows) + "x" + std::to_string(chain.cols);

    auto it = ctx->fused_ew_session_cache.find(net_name);
    if (it != ctx->fused_ew_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_fused_ew_zg_network(
            ctx->cache_dir, chain.ops, chain.compiled_rows, chain.cols);

        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);

        MemChunk output_chunk;
        int64_t output_value_id = -1;
        size_t output_bytes = 0;
        {
            auto zg_backend = session->backends.at(0).cast<zg330::ZG330Backend>();
            const auto & out_values = bundle.network.outputs();
            if (out_values.size() != 1) {
                if (err) *err = "fatal: fused_ew outputs size != 1";
                return nullptr;
            }
            const auto & out_value = out_values[0];
            output_value_id = out_value->v_id;
            output_bytes = out_value.storageBytes();
            output_chunk = ctx->zg_device.defaultMemRegion().malloc(output_bytes, true, 4096);
            zg_backend.userConnectNetwork(output_chunk, output_value_id);
            ggml_fmsh_log_locked(ctx, 1,
                "fused_ew_userConnectNetwork net=" + net_name +
                " v_id=" + std::to_string(output_value_id) +
                " bytes=" + std::to_string(output_bytes));
        }

        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_fused_ew_entry>();
        entry->net_name      = net_name;
        entry->hit_count     = 1;
        entry->ops           = chain.ops;
        entry->rows          = chain.rows;
        entry->cols          = chain.cols;
        entry->compiled_rows = chain.compiled_rows;
        entry->bundle        = std::move(bundle);
        entry->session       = std::move(session);
        entry->output_chunk  = output_chunk;
        entry->output_value_id = output_value_id;
        entry->output_bytes  = output_bytes;
        for (const auto & in : entry->bundle.network.inputs()) {
            entry->input_types.push_back(in.tensorType().clone());
        }
        entry->input_tensors.resize(entry->input_types.size());

        ggml_fmsh_zg330_fused_ew_entry * ptr = entry.get();
        ctx->fused_ew_session_cache.emplace(net_name, std::move(entry));
        if (created) *created = true;
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_session_create net=" + net_name);
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

static ggml_fmsh_zg330_flash_attn_session_entry * ggml_fmsh_get_or_create_flash_attn_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    ggml_fmsh_flash_attn_validation v;
    if (!ggml_fmsh_validate_flash_attn_ext(ctx, node, &v, err)) {
        return nullptr;
    }

    std::string cache_key = ggml_fmsh_make_flash_attn_cache_key(node, v);
    auto it = ctx->flash_attn_session_cache.find(cache_key);
    if (it != ctx->flash_attn_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    std::string last_compile_err = "unknown";
    const int max_softmax_col_retries = 4;
    for (int attempt = 0; attempt <= max_softmax_col_retries; ++attempt) {
        cache_key = ggml_fmsh_make_flash_attn_cache_key(node, v);
        it = ctx->flash_attn_session_cache.find(cache_key);
        if (it != ctx->flash_attn_session_cache.end()) {
            it->second->hit_count++;
            if (created) *created = false;
            return it->second.get();
        }

        try {
            auto bundle = ggml::fmsh::netmake::get_or_compile_flash_attn_zg_network(
                ctx->cache_dir,
                v.head_dim,
                v.value_dim,
                v.q_bucket,
                v.kv_bucket,
                v.softmax_cols,
                v.use_logit_softcap);
            Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
                bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
            session.enableTimeProfile(true);
            session.apply();

            auto entry = std::make_unique<ggml_fmsh_zg330_flash_attn_session_entry>();
            entry->signature = ggml_fmsh_make_signature(node);
            entry->info = v;
            entry->hit_count = 1;
            entry->bundle = std::move(bundle);
            entry->session = std::move(session);
            if (ctx->zg_device.is<ZG330Device>()) {
                entry->zg_device = ctx->zg_device.cast<ZG330Device>();
            }
            for (const auto & in : entry->bundle.network.inputs()) {
                entry->input_types.push_back(in.tensorType().clone());
            }
            entry->input_tensors.resize(entry->input_types.size());

            ggml_fmsh_zg330_flash_attn_session_entry * ptr = entry.get();
            ctx->flash_attn_session_cache.emplace(cache_key, std::move(entry));
            if (created) *created = true;

            int64_t pre_kv = v.kv_bucket;
            for (int64_t d = 0; d < ctx->flash_precompile_kv_depth; ++d) {
                pre_kv *= 2;
                if (pre_kv <= 0 || pre_kv > ctx->flash_kv_bucket_max) {
                    break;
                }
                try {
                    (void) ggml::fmsh::netmake::get_or_compile_flash_attn_zg_network(
                        ctx->cache_dir,
                        v.head_dim,
                        v.value_dim,
                        v.q_bucket,
                        pre_kv,
                        v.softmax_cols,
                        v.use_logit_softcap);
                } catch (...) {
                    break;
                }
            }
            return ptr;
        } catch (const std::exception & e) {
            last_compile_err = e.what();
            v.softmax_cols = ggml_fmsh_next_pow2_i64(v.softmax_cols + 1);
        }
    }

    if (err) {
        *err = "FLASH_ATTN_EXT compile failed: " + last_compile_err;
    }
    return nullptr;
}

static std::string ggml_fmsh_make_bf16_bridge_cache_key(int64_t rows, int64_t cols) {
    return "bf16_bridge|" + std::to_string(rows) + "x" + std::to_string(cols);
}

static ggml_fmsh_zg330_bf16_bridge_session_entry * ggml_fmsh_get_or_create_bf16_bridge_session(
    ggml_backend_fmsh_zg330_context * ctx,
    int64_t rows,
    int64_t cols,
    bool * created,
    std::string * err) {
    const std::string cache_key = ggml_fmsh_make_bf16_bridge_cache_key(rows, cols);
    auto it = ctx->bridge_session_cache.find(cache_key);
    if (it != ctx->bridge_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_bf16_bridge_zg_network(ctx->cache_dir, rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_bf16_bridge_session_entry>();
        entry->rows = rows;
        entry->cols = cols;
        entry->hit_count = 1;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        entry->input_type = entry->bundle.network.inputs()[0].tensorType().clone();

        ggml_fmsh_zg330_bf16_bridge_session_entry * ptr = entry.get();
        ctx->bridge_session_cache.emplace(cache_key, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
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
        uint16_t * in0 = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
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
                            in0[row_off + static_cast<size_t>(i0)] = ggml_fmsh_f32_to_bf16(v);
                        }
                    }
                }
            }
        } else if (!can_chain_input0) {
            if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::CPY ||
                entry->op == ggml::fmsh::netmake::ElementwiseZgOp::DUP) {
                const size_t n = ggml_nelements(node);
                for (size_t idx = 0; idx < n; ++idx) {
                    in0[idx] = ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_linear(src0, idx));
                }
            } else {
                for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                            const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                            const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                            for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                                in0[row_off + static_cast<size_t>(i0)] = ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(src0, i0, i1, i2, i3));
                            }
                        }
                    }
                }
            }
        }
        if (!can_chain_input0 && exec_rows > entry->rows) {
            uint16_t * in0_pad = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
            const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
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
            uint16_t * in1 = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
            const ggml_tensor * src1 = node->src[1];
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            in1[row_off + static_cast<size_t>(i0)] = ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(src1, i0, i1, i2, i3));
                        }
                    }
                }
            }
            if (exec_rows > entry->rows) {
                uint16_t * in1_pad = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
                const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
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
            const uint16_t scale_bf16 = ggml_fmsh_f32_to_bf16(scale);
            std::memcpy(entry->input_tensors[1].data().cptr(), &scale_bf16, sizeof(uint16_t));
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::CPY ||
                   entry->op == ggml::fmsh::netmake::ElementwiseZgOp::DUP) {
            const uint16_t one_bf16 = ggml_fmsh_f32_to_bf16(1.0f);
            std::memcpy(entry->input_tensors[1].data().cptr(), &one_bf16, sizeof(uint16_t));
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params, sizeof(float));
            const uint16_t eps_bf16 = ggml_fmsh_f32_to_bf16(eps);
            std::memcpy(entry->input_tensors[1].data().cptr(), &eps_bf16, sizeof(uint16_t));
        }
        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX &&
            node->src[1] != nullptr &&
            node->src[1]->data != nullptr &&
            entry->input_tensors.size() > 1 &&
            entry->input_tensors[1].defined() &&
            exec_rows > entry->rows) {
            uint16_t * in1_pad = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
            const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
            for (int64_t r = entry->rows; r < exec_rows; ++r) {
                std::memcpy(
                    in1_pad + static_cast<size_t>(r * entry->cols),
                    in1_pad + static_cast<size_t>((entry->rows - 1) * entry->cols),
                    row_bytes);
            }
        }

        if (can_chain_input0) {
            // D2H copy: PLDDR F32 → BF16 host buffer. Read actual_rows only; the PLDDR
            // tensor from the previous op may have fewer rows than exec_rows (e.g. ADD
            // compiled_rows=1 but RMS_NORM compiled_rows=2 for 1-row decode tensors).
            const int64_t actual_rows = entry->rows;
            const size_t act_elems =
                static_cast<size_t>(actual_rows) * static_cast<size_t>(entry->cols);
            std::vector<float> d2h_buf(act_elems);
            input0.read(reinterpret_cast<char *>(d2h_buf.data()), 0, act_elems * sizeof(float));
            for (size_t idx = 0; idx < act_elems; ++idx) {
                in0[idx] = ggml_fmsh_f32_to_bf16(d2h_buf[idx]);
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(in0 + r * entry->cols, in0 + (actual_rows - 1) * entry->cols, row_bytes);
                }
            }
        }
        std::vector<Tensor> session_inputs = entry->input_tensors;
        std::vector<Tensor> outputs = entry->session.forward(session_inputs);
        if (outputs.empty()) {
            if (err) *err = "elementwise session.forward returned empty output";
            return false;
        }
        if (entry->output_chunk.defined()) {
            if (outputs[0].chunk() == entry->output_chunk) {
                ctx->elementwise_output_chunk_hits++;
            } else {
                ctx->elementwise_output_chunk_misses++;
                ggml_fmsh_log_locked(
                    ctx, 2,
                    "userConnectNetwork chunk mismatch op=" + std::string(ggml_op_name(node->op)) +
                    " v_id=" + std::to_string(entry->output_value_id) +
                    " bytes=" + std::to_string(entry->output_bytes));
            }
        }
        if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "elementwise")) {
            return false;
        }
        Tensor effective_output = outputs[0];
        if (chained_output != nullptr) {
            *chained_output = outputs[0];
        }
        ggml_fmsh_log_locked(ctx, 1, "elementwise_forward_done op=" + std::string(ggml_op_name(node->op)));

        const bool should_write_host_output =
            !keep_device_output_only
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            || (ctx->debug_compare == 1)
#endif
            ;
        if (should_write_host_output) {
            std::vector<float> out_tmp(total_elems);
            effective_output.read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);
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

// Two-phase RMS_NORM execution: PL computes r² per row, host does scalar sqrt, PL applies X*r_inv.
// X is written once into a shared buffer used by both pre and post networks.
static bool ggml_fmsh_execute_rmsnorm_split(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_rmsnorm_split_session_entry * entry,
    const Tensor * input_override,    // device-resident X from upstream (or nullptr)
    bool keep_device_output,          // if true, leave post output in ZG DDR
    Tensor * chained_output,          // out: device tensor if keep_device_output
    std::string * err) {
    if (!node || !entry || node->data == nullptr || node->src[0] == nullptr || node->src[0]->data == nullptr) {
        if (err) *err = "rmsnorm_split: tensor data is null";
        return false;
    }
    if (chained_output) *chained_output = Tensor();
    try {
        const int64_t exec_rows   = entry->compiled_rows;
        const int64_t actual_rows = entry->rows;
        const int64_t cols        = entry->cols;

        // Lazily allocate shared X tensor (used by both pre and post).
        if (!entry->x_input_tensor.defined()) {
            entry->x_input_tensor = Tensor(entry->x_input_type.clone());
            entry->x_input_tensor.mallocOn(HostDevice::MemRegion());
        }
        if (!entry->eps_input_tensor.defined()) {
            entry->eps_input_tensor = Tensor(entry->eps_input_type.clone());
            entry->eps_input_tensor.mallocOn(HostDevice::MemRegion());
        }
        if (!entry->r_inv_input_tensor.defined()) {
            entry->r_inv_input_tensor = Tensor(entry->r_inv_input_type.clone());
            entry->r_inv_input_tensor.mallocOn(HostDevice::MemRegion());
        }

        // Fill X: if upstream produced a device-resident tensor, D2H from it;
        // otherwise read from src[0] host memory.
        uint16_t * x_buf = reinterpret_cast<uint16_t *>(entry->x_input_tensor.data().cptr());
        if (input_override != nullptr) {
            // Upstream (e.g. ADD) left output in ZG DDR — read bf16 directly.
            const size_t act_bytes = static_cast<size_t>(actual_rows) * static_cast<size_t>(cols) * sizeof(float);
            std::vector<float> d2h(static_cast<size_t>(actual_rows) * static_cast<size_t>(cols));
            input_override->read(reinterpret_cast<char *>(d2h.data()), 0, act_bytes);
            for (size_t idx = 0; idx < d2h.size(); ++idx) {
                x_buf[idx] = ggml_fmsh_f32_to_bf16(d2h[idx]);
            }
        } else {
            const ggml_tensor * src0 = node->src[0];
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row    = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < cols; ++i0) {
                            x_buf[row_off + static_cast<size_t>(i0)] =
                                ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(src0, i0, i1, i2, i3));
                        }
                    }
                }
            }
        }
        if (exec_rows > actual_rows) {
            const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
            for (int64_t r = actual_rows; r < exec_rows; ++r) {
                std::memcpy(x_buf + r * cols, x_buf + (actual_rows - 1) * cols, row_bytes);
            }
        }

        float eps = 0.0f;
        std::memcpy(&eps, node->op_params, sizeof(float));
        const uint16_t eps_bf16 = ggml_fmsh_f32_to_bf16(eps);
        std::memcpy(entry->eps_input_tensor.data().cptr(), &eps_bf16, sizeof(uint16_t));

        // Run pre: [X, eps] → r_sq[exec_rows, 1].
        std::vector<Tensor> pre_inputs = {entry->x_input_tensor, entry->eps_input_tensor};
        auto pre_outputs = entry->session_pre.forward(pre_inputs);
        if (pre_outputs.empty()) {
            if (err) *err = "rmsnorm_split pre session.forward returned empty";
            return false;
        }
        if (!ggml_fmsh_wait_tensor_ready(pre_outputs[0], err, "rmsnorm_split_pre")) {
            return false;
        }

        // Read r_sq (exec_rows scalars — tiny transfer regardless of cols).
        std::vector<float> r_sq(static_cast<size_t>(exec_rows));
        pre_outputs[0].read(reinterpret_cast<char *>(r_sq.data()), 0,
                             static_cast<size_t>(exec_rows) * sizeof(float));

        // Compute r_inv on host.
        uint16_t * r_inv_buf = reinterpret_cast<uint16_t *>(entry->r_inv_input_tensor.data().cptr());
        for (int64_t r = 0; r < exec_rows; ++r) {
            const float rsq_val = (r < actual_rows) ? r_sq[static_cast<size_t>(r)]
                                                    : r_sq[static_cast<size_t>(actual_rows - 1)];
            const float r_inv = (rsq_val > 0.0f) ? (1.0f / std::sqrt(rsq_val)) : 0.0f;
            r_inv_buf[static_cast<size_t>(r)] = ggml_fmsh_f32_to_bf16(r_inv);
        }

        // Evict any tensor already occupying post_output_chunk.
        // (mirrors the elementwise output_chunk eviction pattern)
        // Run post: [X (shared, no re-upload needed), r_inv] → Y.
        std::vector<Tensor> post_inputs = {entry->x_input_tensor, entry->r_inv_input_tensor};
        auto post_outputs = entry->session_post.forward(post_inputs);
        if (post_outputs.empty()) {
            if (err) *err = "rmsnorm_split post session.forward returned empty";
            return false;
        }
        if (!ggml_fmsh_wait_tensor_ready(post_outputs[0], err, "rmsnorm_split_post")) {
            return false;
        }

        // Check output chunk consistency.
        if (entry->post_output_chunk.defined()) {
            if (post_outputs[0].chunk() != entry->post_output_chunk) {
                ggml_fmsh_log_locked(ctx, 2, "rmsnorm_split post output chunk mismatch");
            }
        }

        const bool should_readback = !keep_device_output;
        if (should_readback) {
            const size_t total_elems = static_cast<size_t>(exec_rows) * static_cast<size_t>(cols);
            std::vector<float> out_tmp(total_elems);
            post_outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, total_elems * sizeof(float));
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row    = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < cols; ++i0) {
                            ggml_fmsh_write_f32_indexed(node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                        }
                    }
                }
            }
        } else if (chained_output != nullptr) {
            // Return device tensor for the next consumer to use directly.
            Tensor dev_out = post_outputs[0];
            if (entry->post_output_chunk.defined() && dev_out.chunk() == entry->post_output_chunk) {
                // Retype to actual rows if compiled_rows > actual_rows.
                if (exec_rows != actual_rows && entry->post_output_type.defined()) {
                    TensorType rtyped = entry->post_output_type.clone();
                    rtyped.setShape(icraft::xir::Array<int64_t>{actual_rows, cols});
                    dev_out = Tensor(rtyped, dev_out.chunk(), dev_out.offset());
                }
            }
            *chained_output = dev_out;
            ggml_fmsh_log_locked(ctx, 0, "rmsnorm_split keep_device_output=1");
        }
        ggml_fmsh_log_locked(ctx, 1, "rmsnorm_split_done rows=" + std::to_string(actual_rows) + " cols=" + std::to_string(cols));
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

// ROPE NeoX execution: compute theta table on host, upload, run PL network.
static bool ggml_fmsh_execute_rope(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_rope_session_entry * entry,
    std::string * err) {
    if (!node || !entry || node->data == nullptr || !node->src[0] || node->src[0]->data == nullptr) {
        if (err) *err = "rope_exec: null tensor";
        return false;
    }
    try {
        // Lazily allocate input tensors.
        for (size_t i = 0; i < entry->input_types.size(); ++i) {
            if (!entry->input_tensors[i].defined()) {
                entry->input_tensors[i] = Tensor(entry->input_types[i].clone());
                entry->input_tensors[i].mallocOn(HostDevice::MemRegion());
            }
        }

        const ggml_tensor * src0 = node->src[0]; // X[ne0=d, ne1=n_head, ne2=seq, ne3=batch]
        const ggml_tensor * src1 = node->src[1]; // positions [ne0=seq]

        const int32_t n_dims     = ((const int32_t *) node->op_params)[1];
        float freq_base  = 10000.0f;
        float freq_scale = 1.0f;
        std::memcpy(&freq_base,  (const float *) node->op_params + 5, sizeof(float));
        std::memcpy(&freq_scale, (const float *) node->op_params + 6, sizeof(float));

        const int64_t ne0 = node->ne[0]; // head_dim = cols
        const int64_t ne1 = node->ne[1]; // n_head
        const int64_t ne2 = node->ne[2]; // seq_len
        const int64_t ne3 = node->ne[3]; // batch
        const int64_t half = ne0 / 2;

        // Fill X input (bf16, row-major [rows, cols]).
        uint16_t * x_in = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const int64_t row = (i3 * ne2 + i2) * ne1 + i1;
                    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(ne0);
                    const float * src_row = reinterpret_cast<const float *>(
                        static_cast<const char *>(src0->data) +
                        i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        x_in[row_off + static_cast<size_t>(i0)] = ggml_fmsh_f32_to_bf16(src_row[i0]);
                    }
                }
            }
        }

        // Fill theta input (bf16, row-major [rows, half]).
        // theta[row_of_head, j] = position * freq_base^(-2j / n_dims) * freq_scale
        // NeoX layout: first half dims [0..half) use theta[j] for j=0..half-1.
        uint16_t * theta_in = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
        // Position tensor: src1 is int32 [seq_len] (shared across all batches and heads).
        const int32_t * pos_data = (src1 && src1->data)
            ? reinterpret_cast<const int32_t *>(src1->data)
            : nullptr;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                // Position for this sequence slot (shared across batches and heads).
                const int32_t pos = pos_data
                    ? pos_data[i2 < src1->ne[0] ? i2 : src1->ne[0] - 1]
                    : static_cast<int32_t>(i2);
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const int64_t row = (i3 * ne2 + i2) * ne1 + i1;
                    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(half);
                    for (int64_t j = 0; j < half; ++j) {
                        float theta;
                        if (j < n_dims / 2) {
                            // Active rotation dimension.
                            theta = static_cast<float>(pos) * freq_scale *
                                std::pow(freq_base, -2.0f * static_cast<float>(j) / static_cast<float>(n_dims));
                        } else {
                            // Passthrough dimension: theta=0 → cos=1, sin=0.
                            theta = 0.0f;
                        }
                        theta_in[row_off + static_cast<size_t>(j)] = ggml_fmsh_f32_to_bf16(theta);
                    }
                }
            }
        }

        // Run the PL ROPE network.
        auto outputs = entry->session.forward(entry->input_tensors);
        if (outputs.empty()) {
            if (err) *err = "rope_exec: session.forward returned empty";
            return false;
        }
        if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "rope_exec")) {
            return false;
        }

        // Read output Y and write to host.
        const size_t total_elems = static_cast<size_t>(entry->rows) * static_cast<size_t>(ne0);
        const size_t total_bytes = total_elems * sizeof(float);
        std::vector<float> out_tmp(total_elems);
        outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const int64_t row = (i3 * ne2 + i2) * ne1 + i1;
                    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(ne0);
                    float * dst_row = reinterpret_cast<float *>(
                        static_cast<char *>(node->data) +
                        i1 * node->nb[1] + i2 * node->nb[2] + i3 * node->nb[3]);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = out_tmp[row_off + static_cast<size_t>(i0)];
                    }
                }
            }
        }
        ggml_fmsh_log_locked(ctx, 1, "rope_exec_done rows=" + std::to_string(entry->rows) + " cols=" + std::to_string(ne0));
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

static bool ggml_fmsh_execute_fused_ew(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_fmsh_zg330_fused_ew_chain & chain,
    ggml_fmsh_zg330_fused_ew_entry * entry,
    const Tensor * input_override,
    bool keep_device_output_only,
    Tensor * chained_output,
    std::string * err) {
    ggml_tensor * out_node = chain.nodes.back();
    ggml_tensor * in_node  = chain.nodes[0];

    if (!out_node || !in_node || out_node->data == nullptr) {
        if (err) *err = "fused_ew: tensor data is null";
        return false;
    }
    if (input_override == nullptr && (in_node->src[0] == nullptr || in_node->src[0]->data == nullptr)) {
        if (err) *err = "fused_ew: primary src[0] is null";
        return false;
    }

    try {
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_begin net=" + entry->net_name);

        const int64_t exec_rows   = entry->compiled_rows;
        const int64_t actual_rows = chain.rows;
        const int64_t cols        = entry->cols;
        const size_t  total_elems = static_cast<size_t>(exec_rows) * static_cast<size_t>(cols);
        const size_t  total_bytes = total_elems * sizeof(float);

        // Lazily allocate host input tensors.
        for (size_t idx = 0; idx < entry->input_types.size(); ++idx) {
            if (!entry->input_tensors[idx].defined()) {
                entry->input_tensors[idx] = Tensor(entry->input_types[idx].clone());
                entry->input_tensors[idx].mallocOn(HostDevice::MemRegion());
            }
        }

        if (chained_output) *chained_output = Tensor();

        // ── Primary input (input_tensors[0]) ──────────────────────────────────
        uint16_t * in0 = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
        if (input_override != nullptr) {
            // D2H: PLDDR F32 → BF16 host buffer. Read actual_rows only; the PLDDR tensor
            // from the previous op may have fewer rows than exec_rows (RMS_NORM pads 1→2).
            const size_t act_elems = static_cast<size_t>(actual_rows) * static_cast<size_t>(cols);
            std::vector<float> d2h_buf(act_elems);
            input_override->read(reinterpret_cast<char *>(d2h_buf.data()), 0, act_elems * sizeof(float));
            for (size_t idx = 0; idx < act_elems; ++idx) {
                in0[idx] = ggml_fmsh_f32_to_bf16(d2h_buf[idx]);
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(in0 + r * cols, in0 + (actual_rows - 1) * cols, row_bytes);
                }
            }
        } else {
            const ggml_tensor * src0 = in_node->src[0];
            for (int64_t i3 = 0; i3 < in_node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < in_node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < in_node->ne[1]; ++i1) {
                        const int64_t row     = ((i3 * in_node->ne[2]) + i2) * in_node->ne[1] + i1;
                        const size_t  row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < in_node->ne[0]; ++i0) {
                            in0[row_off + static_cast<size_t>(i0)] =
                                ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(src0, i0, i1, i2, i3));
                        }
                    }
                }
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(
                        in0 + r * cols,
                        in0 + (actual_rows - 1) * cols,
                        row_bytes);
                }
            }
        }

        // ── Secondary inputs ──────────────────────────────────────────────────
        int secondary_idx = 1;
        for (size_t k = 0; k < chain.ops.size(); ++k) {
            const auto     op    = chain.ops[k];
            ggml_tensor * knode = chain.nodes[k];
            if (op == ggml::fmsh::netmake::ElementwiseZgOp::ADD ||
                op == ggml::fmsh::netmake::ElementwiseZgOp::MUL) {
                if (!knode->src[1] || !knode->src[1]->data) {
                    if (err) *err = "fused_ew: ADD/MUL src[1] null at op " + std::to_string(k);
                    return false;
                }
                uint16_t * sec = reinterpret_cast<uint16_t *>(
                    entry->input_tensors[secondary_idx].data().cptr());
                const ggml_tensor * src1  = knode->src[1];
                const int64_t       krows = knode->ne[1] * knode->ne[2] * knode->ne[3];
                for (int64_t i3 = 0; i3 < knode->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < knode->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < knode->ne[1]; ++i1) {
                            const int64_t row     = ((i3 * knode->ne[2]) + i2) * knode->ne[1] + i1;
                            const size_t  row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                            for (int64_t i0 = 0; i0 < knode->ne[0]; ++i0) {
                                sec[row_off + static_cast<size_t>(i0)] =
                                    ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(src1, i0, i1, i2, i3));
                            }
                        }
                    }
                }
                if (exec_rows > krows) {
                    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
                    for (int64_t r = krows; r < exec_rows; ++r) {
                        std::memcpy(
                            sec + r * cols,
                            sec + (krows - 1) * cols,
                            row_bytes);
                    }
                }
                secondary_idx++;
            } else if (op == ggml::fmsh::netmake::ElementwiseZgOp::SCALE) {
                float scale = 1.0f;
                if (knode->src[1] && knode->src[1]->data && ggml_nelements(knode->src[1]) >= 1) {
                    scale = ggml_fmsh_read_f32_broadcast(knode->src[1], 0, 0, 0, 0);
                } else {
                    std::memcpy(&scale, knode->op_params, sizeof(float));
                }
                const uint16_t sv = ggml_fmsh_f32_to_bf16(scale);
                std::memcpy(entry->input_tensors[secondary_idx].data().cptr(), &sv, sizeof(uint16_t));
                secondary_idx++;
            } else if (op == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
                float eps = 0.0f;
                std::memcpy(&eps, knode->op_params, sizeof(float));
                const uint16_t ev = ggml_fmsh_f32_to_bf16(eps);
                std::memcpy(entry->input_tensors[secondary_idx].data().cptr(), &ev, sizeof(uint16_t));
                secondary_idx++;
            }
        }

        // ── Forward ───────────────────────────────────────────────────────────
        std::vector<Tensor> session_inputs = entry->input_tensors;
        std::vector<Tensor> outputs = entry->session.forward(session_inputs);
        if (outputs.empty()) {
            if (err) *err = "fused_ew: session.forward returned empty output";
            return false;
        }
        if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "fused_ew")) {
            return false;
        }

        if (chained_output) *chained_output = outputs[0];
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_forward_done net=" + entry->net_name);

        const bool should_write_host =
            !keep_device_output_only
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            || (ctx->debug_compare == 1)
#endif
            ;
        if (should_write_host) {
            std::vector<float> out_tmp(total_elems);
            outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);
            for (int64_t i3 = 0; i3 < out_node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < out_node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < out_node->ne[1]; ++i1) {
                        const int64_t row     = ((i3 * out_node->ne[2]) + i2) * out_node->ne[1] + i1;
                        const size_t  row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < out_node->ne[0]; ++i0) {
                            ggml_fmsh_write_f32_indexed(
                                out_node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                        }
                    }
                }
            }
        } else {
            ggml_fmsh_log_locked(ctx, 0, "fused_ew_read_skip net=" + entry->net_name);
        }
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_end net=" + entry->net_name);
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

        const bool src1_is_bf16 = (src1->type == GGML_TYPE_BF16);
        const bool allow_input_chain = input_override != nullptr && ne12 == 1 && ne13 == 1;
        if (chained_output != nullptr) {
            *chained_output = Tensor();
        }

        ggml_fmsh_log_locked(
            ctx, 1,
            "mul_mat_exec_enter m=" + std::to_string(entry->m) +
            " k=" + std::to_string(entry->k) +
            " n=" + std::to_string(entry->n) +
            " src1_type=" + std::to_string(static_cast<int>(src1->type)) +
            " src1_is_bf16=" + std::to_string(src1_is_bf16 ? 1 : 0) +
            " allow_input_chain=" + std::to_string(allow_input_chain ? 1 : 0) +
            " ne12=" + std::to_string(ne12) + " ne13=" + std::to_string(ne13));

        ggml_fmsh_zg330_bf16_bridge_session_entry * bridge_entry = nullptr;
        if (src1_is_bf16 && !allow_input_chain) {
            ggml_fmsh_log_locked(ctx, 0, "bf16_bridge_enter rows=" + std::to_string(entry->m) + " cols=" + std::to_string(entry->k));
            std::string bridge_err;
            bool bridge_created = false;
            bridge_entry = ggml_fmsh_get_or_create_bf16_bridge_session(ctx, entry->m, entry->k, &bridge_created, &bridge_err);
            if (!bridge_entry) {
                ggml_fmsh_log_locked(ctx, 3, "bf16_bridge_session_create_failed: " + bridge_err);
                if (err) *err = "BF16 bridge session creation failed: " + bridge_err;
                return false;
            }
            if (!bridge_entry->input_tensor.defined()) {
                bridge_entry->input_tensor = Tensor(bridge_entry->input_type.clone());
                bridge_entry->input_tensor.mallocOn(HostDevice::MemRegion());
            }
            ggml_fmsh_log_locked(
                ctx, 1,
                std::string(bridge_created ? "bf16_bridge_session_create" : "bf16_bridge_session_hit") +
                " rows=" + std::to_string(entry->m) + " cols=" + std::to_string(entry->k) +
                " net=" + bridge_entry->bundle.net_name +
                " net_cache=" + std::string(bridge_entry->bundle.ram_cache_hit ? "HIT" : "MISS") +
                " compiled_now=" + std::string(bridge_entry->bundle.compiled_now ? "YES" : "NO"));
        } else if (src1_is_bf16 && allow_input_chain) {
            ggml_fmsh_log_locked(ctx, 1, "bf16_bridge_skip_reason=input_chain_already_provided");
        } else {
            ggml_fmsh_log_locked(ctx, 1, "bf16_bridge_skip_reason=src1_not_bf16 src1_type=" + std::to_string(static_cast<int>(src1->type)));
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
                } else if (src1_is_bf16) {
                    // BF16 bridge path: copy BF16 data directly (2 bytes per element), no CPU conversion
                    ggml_fmsh_log_locked(ctx, 0, "bf16_bridge_fwd_enter i12=" + std::to_string(i12) + " i13=" + std::to_string(i13));
                    uint16_t * bridge_dst = reinterpret_cast<uint16_t *>(bridge_entry->input_tensor.data().cptr());
                    for (int64_t row = 0; row < entry->m; ++row) {
                        const char * a_src_row = src1_slice + row * src1->nb[1];
                        uint16_t * bridge_row_dst = bridge_dst + row * entry->k;
                        if (src1->nb[0] == static_cast<size_t>(sizeof(uint16_t))) {
                            std::memcpy(bridge_row_dst, a_src_row, static_cast<size_t>(entry->k) * sizeof(uint16_t));
                        } else {
                            for (int64_t col = 0; col < entry->k; ++col) {
                                uint16_t v = 0;
                                std::memcpy(&v, a_src_row + col * src1->nb[0], sizeof(uint16_t));
                                bridge_row_dst[col] = v;
                            }
                        }
                    }
                    auto bridge_outputs = bridge_entry->session.forward({bridge_entry->input_tensor});
                    if (bridge_outputs.empty()) {
                        ggml_fmsh_log_locked(ctx, 3, "bf16_bridge_fwd_empty_output");
                        if (err) *err = "BF16 bridge forward returned empty output";
                        return false;
                    }
                    if (!ggml_fmsh_wait_tensor_ready(bridge_outputs[0], err, "bf16_bridge")) {
                        ggml_fmsh_log_locked(ctx, 3, "bf16_bridge_wait_ready_failed");
                        return false;
                    }
                    input_a = bridge_outputs[0];
                    ctx->mul_mat_device_input_chain++;
                    ggml_fmsh_log_locked(ctx, 0, "bf16_bridge_fwd_done chaining_to_matmul=1");
                } else {
                    // A = src1 slice : [m, k]
                    float * a_dst = reinterpret_cast<float *>(entry->input_tensor_a.data().cptr());
                    for (int64_t row = 0; row < entry->m; ++row) {
                        const char * a_src_row = src1_slice + row * src1->nb[1];
                        float * a_row_dst = a_dst + row * entry->k;
                        if (src1->nb[0] == static_cast<size_t>(sizeof(float))) {
                            const float * src = reinterpret_cast<const float *>(a_src_row);
                            for (int64_t col = 0; col < entry->k; ++col) {
                                a_row_dst[col] = src[col];
                            }
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
                    // Build transposed weight in a host buffer: src0 [n, k] -> B [k, n]
                    const size_t weight_bytes = static_cast<size_t>(entry->k * entry->n) * sizeof(float);
                    std::vector<float> host_w_buf(static_cast<size_t>(entry->k * entry->n));
                    for (int64_t i = 0; i < entry->n; ++i) {
                        const char * w_src_row = src0_slice + i * src0->nb[1];
                        for (int64_t j = 0; j < entry->k; ++j) {
                            float v = 0.0f;
                            std::memcpy(&v, w_src_row + j * src0->nb[0], sizeof(float));
                            host_w_buf[static_cast<size_t>(j * entry->n + i)] = v;
                        }
                    }

                    Tensor new_weight(entry->input_type_b.clone());
                    // Static weights live on NPU PL memory to avoid per-token DMA re-upload.
                    // Dynamic weights (KV views) stay on host since they change each call.
                    if (src0_is_static && ctx->device_opened) {
                        new_weight.mallocOn(ctx->zg_device.defaultMemRegion());
                        new_weight.write(0, reinterpret_cast<char *>(host_w_buf.data()), weight_bytes);
                    } else {
                        new_weight.mallocOn(HostDevice::MemRegion());
                        std::memcpy(reinterpret_cast<float *>(new_weight.data().cptr()), host_w_buf.data(), weight_bytes);
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
                if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "mul_mat")) {
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
                        const float * w_ptr = reinterpret_cast<const float *>(weight_tensor->data().cptr());
                        const ggml_fmsh_f32_stats w_stats =
                            calc_f32_stats(w_ptr, static_cast<size_t>(entry->k * entry->n));
                        std::string a_trace = "bridge_bf16";
                        if (!src1_is_bf16) {
                            const float * a_ptr = reinterpret_cast<const float *>(input_a.data().cptr());
                            const ggml_fmsh_f32_stats a_stats =
                                calc_f32_stats(a_ptr, static_cast<size_t>(entry->m * entry->k));
                            a_trace = "A[min,max,inf,nan]=[" + std::to_string(a_stats.min_v) + "," +
                                std::to_string(a_stats.max_v) + "," +
                                std::to_string(a_stats.inf_cnt) + "," +
                                std::to_string(a_stats.nan_cnt) + "]";
                        }
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
                            " " + a_trace +
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

static bool ggml_fmsh_execute_flash_attn_ext(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_flash_attn_session_entry * entry,
    std::string * err) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    if (!q || !k || !v || !node->data || !q->data) {
        if (err) *err = "FLASH_ATTN_EXT tensor data is null";
        return false;
    }

    // K and V may live in ZG DDR (fake pointers, not dereferenceable on host).
    // If so, D2H the entire root tensor into a staging buffer and temporarily
    // redirect k->data / v->data so that the existing per-head stride loops work correctly
    // even for permuted/view tensors whose strides differ from contiguous layout.
    std::vector<char> k_root_staging, v_root_staging;
    void * k_orig_data = k->data;
    void * v_orig_data = v->data;
    {
        auto stage_kv = [&](const ggml_tensor * t, std::vector<char> & staging) -> void * {
            MemChunk chunk; size_t off = 0;
            const ggml_tensor * root = t;
            while (root->view_src) root = root->view_src;
            if (!ggml_fmsh_is_zg_buffer_tensor(root, &chunk, &off)) return t->data;
            const size_t root_bytes = ggml_nbytes(root);
            staging.resize(root_bytes);
            chunk.read(staging.data(), off, root_bytes);
            // Redirect: t's host address = staging_base + (t_zg_addr - root_zg_addr).
            const uintptr_t root_zg = reinterpret_cast<uintptr_t>(root->data);
            const uintptr_t t_zg    = reinterpret_cast<uintptr_t>(t->data);
            return staging.data() + (t_zg - root_zg);
        };
        const_cast<ggml_tensor *>(k)->data = stage_kv(k, k_root_staging);
        const_cast<ggml_tensor *>(v)->data = stage_kv(v, v_root_staging);
    }
    const void * k_data = k->data;
    const void * v_data = v->data;


    const auto & cfg = entry->info;
    const int64_t head_dim = cfg.head_dim;
    const int64_t value_dim = cfg.value_dim;
    // Use actual tensor dimensions, not stale session-creation values.
    // Session is keyed by buckets, so actual <= bucket is always guaranteed.
    const int64_t q_len = q->ne[1];
    const int64_t kv_len = k->ne[1];
    const int64_t q_bucket = cfg.q_bucket;
    const int64_t kv_bucket = cfg.kv_bucket;
    const int64_t n_head = cfg.n_head;
    const int64_t n_head_kv = cfg.n_head_kv;
    const int64_t n_batch = cfg.n_batch;
    const int64_t n_batch_kv = k->ne[3];
    const int64_t rk2 = n_head / n_head_kv;
    const int64_t rk3 = n_batch / n_batch_kv;

    const size_t k_bytes = static_cast<size_t>(head_dim * kv_bucket) * sizeof(float);
    const size_t v_bytes = static_cast<size_t>(kv_bucket * value_dim) * sizeof(float);
    const size_t m_bytes = static_cast<size_t>(q_bucket * kv_bucket) * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(q_bucket * value_dim) * sizeof(float);
    const float k_mask_neg_inf = -1e9f;
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
    static std::atomic<int> s_flash_inf_trace_budget{32};

    struct ggml_fmsh_f32_stats_local {
        float min_v = std::numeric_limits<float>::infinity();
        float max_v = -std::numeric_limits<float>::infinity();
        size_t inf_cnt = 0;
        size_t nan_cnt = 0;
    };
    const auto calc_f32_stats_local = [](const float * data, size_t n) -> ggml_fmsh_f32_stats_local {
        ggml_fmsh_f32_stats_local s;
        if (!data || n == 0) {
            s.min_v = 0.0f;
            s.max_v = 0.0f;
            return s;
        }
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
            s.min_v = std::min(s.min_v, v);
            s.max_v = std::max(s.max_v, v);
        }
        if (!std::isfinite(s.min_v)) s.min_v = 0.0f;
        if (!std::isfinite(s.max_v)) s.max_v = 0.0f;
        return s;
    };
#endif

    for (size_t i = 0; i < entry->input_types.size(); ++i) {
        if (!entry->input_tensors[i].defined()) {
            entry->input_tensors[i] = Tensor(entry->input_types[i].clone());
            entry->input_tensors[i].mallocOn(HostDevice::MemRegion());
        }
    }

    float * in_q     = reinterpret_cast<float *>(entry->input_tensors[0].data().cptr());
    float * in_k     = reinterpret_cast<float *>(entry->input_tensors[1].data().cptr());
    float * in_v     = reinterpret_cast<float *>(entry->input_tensors[2].data().cptr());
    float * in_scale = reinterpret_cast<float *>(entry->input_tensors[3].data().cptr());
    float * in_mask  = reinterpret_cast<float *>(entry->input_tensors[4].data().cptr());
    float * in_softcap = nullptr;
    if (cfg.use_logit_softcap && entry->input_tensors.size() > 5) {
        in_softcap = reinterpret_cast<float *>(entry->input_tensors[5].data().cptr());
    }

    std::vector<float> out_tmp(static_cast<size_t>(q_bucket * value_dim), 0.0f);
    const uint32_t n_head_u32 = static_cast<uint32_t>(std::max<int64_t>(1, n_head));
    const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(n_head_u32))));
    const float m0 = std::pow(2.0f, -(cfg.max_bias) / n_head_log2);
    const float m1 = std::pow(2.0f, -(cfg.max_bias / 2.0f) / n_head_log2);

    for (int64_t ib = 0; ib < n_batch; ++ib) {
        const int64_t k_batch = ib / rk3;
        const int64_t mask_batch = mask ? (ib % mask->ne[3]) : 0;
        for (int64_t ih = 0; ih < n_head; ++ih) {
            const int64_t k_head = ih / rk2;
            const int64_t mask_head = mask ? (ih % mask->ne[2]) : 0;

            // Q: [q_bucket, head_dim]
            for (int64_t iq = 0; iq < q_bucket; ++iq) {
                float * q_row = in_q + static_cast<size_t>(iq * head_dim);
                if (iq < q_len) {
                    const float * src_q_row = reinterpret_cast<const float *>(
                        static_cast<const char *>(q->data) + iq * q->nb[1] + ih * q->nb[2] + ib * q->nb[3]);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        q_row[d] = src_q_row[d];
                    }
                } else {
                    std::memset(q_row, 0, static_cast<size_t>(head_dim) * sizeof(float));
                }
            }

            // K: [head_dim, kv_bucket]
            std::memset(in_k, 0, k_bytes);
            for (int64_t ikv = 0; ikv < kv_len; ++ikv) {
                const char * src_k_row = static_cast<const char *>(k_data) +
                                         ikv * k->nb[1] + k_head * k->nb[2] + k_batch * k->nb[3];
                if (k->type == GGML_TYPE_F32) {
                    const float * src = reinterpret_cast<const float *>(src_k_row);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        in_k[static_cast<size_t>(d * kv_bucket + ikv)] = src[d];
                    }
                } else {
                    const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(src_k_row);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        in_k[static_cast<size_t>(d * kv_bucket + ikv)] = GGML_FP16_TO_FP32(src[d]);
                    }
                }
            }

            // V: [kv_bucket, value_dim]
            std::memset(in_v, 0, v_bytes);
            for (int64_t ikv = 0; ikv < kv_len; ++ikv) {
                float * v_row = in_v + static_cast<size_t>(ikv * value_dim);
                const char * src_v_row = static_cast<const char *>(v_data) +
                                         ikv * v->nb[1] + k_head * v->nb[2] + k_batch * v->nb[3];
                if (v->type == GGML_TYPE_F32) {
                    const float * src = reinterpret_cast<const float *>(src_v_row);
                    for (int64_t d = 0; d < value_dim; ++d) {
                        v_row[d] = src[d];
                    }
                } else {
                    const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(src_v_row);
                    for (int64_t d = 0; d < value_dim; ++d) {
                        v_row[d] = GGML_FP16_TO_FP32(src[d]);
                    }
                }
            }

            // MASK: [q_bucket, kv_bucket]
            std::memset(in_mask, 0, m_bytes);
            const float slope = (cfg.max_bias > 0.0f)
                ? (ih < n_head_log2 ? std::pow(m0, static_cast<float>(ih + 1))
                                    : std::pow(m1, static_cast<float>(2 * (ih - n_head_log2) + 1)))
                : 1.0f;
            for (int64_t iq = 0; iq < q_bucket; ++iq) {
                float * m_row = in_mask + static_cast<size_t>(iq * kv_bucket);
                for (int64_t ikv = 0; ikv < kv_bucket; ++ikv) {
                    float mv;
                    if (ikv >= kv_len) {
                        mv = k_mask_neg_inf;
                    } else if (iq >= q_len || !mask) {
                        mv = 0.0f;
                    } else {
                        mv = ggml_fmsh_read_mask_f32(mask, ikv, iq, mask_head, mask_batch);
                        // ZG330 softmax path is unstable with +/-inf or NaN mask values.
                        // Normalize mask sent to NPU to finite numbers while preserving intent.
                        if (!std::isfinite(mv)) {
                            mv = (std::isnan(mv) || mv > 0.0f) ? 0.0f : k_mask_neg_inf;
                        }
                        if (cfg.max_bias > 0.0f) {
                            mv *= slope;
                        }
                    }
                    m_row[ikv] = mv;
                }
            }

            float scale = cfg.scale;
            if (cfg.use_logit_softcap) {
                scale /= cfg.logit_softcap;
            }
            in_scale[0] = scale;
            if (in_softcap != nullptr) {
                in_softcap[0] = cfg.logit_softcap;
            }

            std::vector<Tensor> inputs = entry->input_tensors;

            // AXI 同步修复（参考 flash-attn-netmake-cpu-test.cpp）：
            // forward() 后 ready_=true 永久置位，后续 waitForReady() 立即返回读到上帧数据。
            // 修复：forward() 后立即 setReady(false) + 安装新 check_func_，再 waitForReady()。
            const bool use_axi_sync = entry->zg_device.defined();
            uint32_t layer_before = 0;
            if (use_axi_sync) {
                layer_before = entry->zg_device.layerCount();
            }

            auto outputs = entry->session.forward(inputs);
            if (outputs.empty()) {
                if (err) *err = "FLASH_ATTN_EXT session.forward returned empty output";
                return false;
            }

            if (use_axi_sync) {
                if (entry->layer_increment == 0) {
                    // 第一次 forward：forward() 内部的 check_func_ 可能使用 apply() 时的绝对
                    // layerCount 目标，在完整推理中该目标早已超过，导致 waitForReady 立即返回。
                    // 修复：直接 spin-poll layerCount，等待其稳定（NPU 完成计算）。
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
                    // 等待 NPU 启动（layerCount 变化）
                    while (entry->zg_device.layerCount() == layer_before) {
                        if (std::chrono::steady_clock::now() > deadline) {
                            if (err) *err = "flash_attn_ext NPU layerCount sync timeout (start)";
                            return false;
                        }
                    }
                    // 等待 layerCount 稳定（NPU 完成所有层）：5ms 无变化则认为完成
                    uint32_t stable_count = entry->zg_device.layerCount();
                    auto stable_start = std::chrono::steady_clock::now();
                    while (std::chrono::steady_clock::now() < deadline) {
                        const uint32_t cur = entry->zg_device.layerCount();
                        if (cur != stable_count) {
                            stable_count = cur;
                            stable_start = std::chrono::steady_clock::now();
                        } else {
                            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - stable_start).count();
                            if (elapsed_ms >= 5) {
                                break;
                            }
                        }
                    }
                    entry->layer_increment = (stable_count > layer_before)
                        ? (stable_count - layer_before) : 1;
                    ggml_fmsh_log_locked(ctx, 1,
                        "axi_sync_first ib=" + std::to_string(ib) +
                        " ih=" + std::to_string(ih) +
                        " layer_before=" + std::to_string(layer_before) +
                        " stable_count=" + std::to_string(stable_count) +
                        " layer_increment=" + std::to_string(entry->layer_increment));
                } else {
                    // 后续 forward：ready_=true，需重置并安装指向本次目标的 check_func_。
                    // 旧 check_func_ 的 layer_target 已过时，waitForReady 会立即返回读到上帧数据。
                    const uint32_t layer_target = layer_before + entry->layer_increment;
                    ZG330Device zg_dev = entry->zg_device;
                    outputs[0].setReady(false);
                    outputs[0].setCheckFunc([zg_dev, layer_target](const Device &) -> bool {
                        return zg_dev.layerCount() >= layer_target;
                    });
                    if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "flash_attn_ext")) {
                        return false;
                    }
                    ggml_fmsh_log_locked(ctx, 0,
                        "axi_sync_sub ib=" + std::to_string(ib) +
                        " ih=" + std::to_string(ih) +
                        " layer_before=" + std::to_string(layer_before) +
                        " layer_target=" + std::to_string(layer_target) +
                        " layer_actual=" + std::to_string(entry->zg_device.layerCount()) +
                        " incr=" + std::to_string(entry->layer_increment));
                }
            } else {
                if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "flash_attn_ext")) {
                    return false;
                }
            }
            outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, y_bytes);
            // Debug: log first few output values for first 2 heads of first batch.
            if (use_axi_sync && ib == 0 && ih < 2) {
                static std::atomic<int> s_out_log_budget{16};
                if (s_out_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    std::string vals;
                    for (int64_t _d = 0; _d < std::min<int64_t>(4, value_dim); ++_d) {
                        vals += std::to_string(out_tmp[static_cast<size_t>(_d)]) + " ";
                    }
                    ggml_fmsh_log_locked(ctx, 1,
                        "axi_out ib=0 ih=" + std::to_string(ih) +
                        " iq0_vals=[" + vals + "]");
                }
            }

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            const auto out_stats = calc_f32_stats_local(out_tmp.data(), static_cast<size_t>(q_bucket * value_dim));
            if ((out_stats.inf_cnt > 0 || out_stats.nan_cnt > 0) &&
                s_flash_inf_trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                // Input tensors are fp32; Y output is also fp32.
                ggml_fmsh_log_locked(
                    ctx, 2,
                    "flash_inf_trace batch=" + std::to_string(ib) +
                    " head=" + std::to_string(ih) +
                    " k_head=" + std::to_string(k_head) +
                    " mask_head=" + std::to_string(mask_head) +
                    " q_len=" + std::to_string(q_len) +
                    " kv_len=" + std::to_string(kv_len) +
                    " q_bucket=" + std::to_string(q_bucket) +
                    " kv_bucket=" + std::to_string(kv_bucket) +
                    " scale=" + std::to_string(scale) +
                    " Y[min,max,inf,nan]=[" + std::to_string(out_stats.min_v) + "," +
                    std::to_string(out_stats.max_v) + "," +
                    std::to_string(out_stats.inf_cnt) + "," +
                    std::to_string(out_stats.nan_cnt) + "]");
            }
#endif

            for (int64_t iq = 0; iq < q_len; ++iq) {
                const float * out_row = out_tmp.data() + static_cast<size_t>(iq * value_dim);
                for (int64_t d = 0; d < value_dim; ++d) {
                    ggml_fmsh_write_f32_indexed(node, d, ih, iq, ib, out_row[d]);
                }
            }
        }
    }

    // Restore original data pointers if they were redirected.
    const_cast<ggml_tensor *>(k)->data = k_orig_data;
    const_cast<ggml_tensor *>(v)->data = v_orig_data;

    return true;
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

static std::string ggml_fmsh_op_name_from_id(uint32_t op_id) {
    if (op_id < static_cast<uint32_t>(GGML_OP_COUNT)) {
        return std::string(ggml_op_name(static_cast<enum ggml_op>(op_id)));
    }
    return "unknown(" + std::to_string(op_id) + ")";
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

// Write a device-resident tensor to a safe host location, handling ZG buffer tensors
// (whose t->data is a physical address) by redirecting via a staging buffer.
// Used in output-chunk-conflict eviction paths where we have the Tensor but not going
// through materialize_if_device's map lookup.
static void ggml_fmsh_evict_to_host(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    const Tensor & device_tensor) {
    if (!t) return;
    const size_t bytes = ggml_nbytes(t);
    if (bytes == 0) return;
    MemChunk zg_chunk;
    size_t   zg_off = 0;
    if (ggml_fmsh_is_zg_buffer_tensor(t, &zg_chunk, &zg_off)) {
        const ggml_tensor * root = t;
        while (root->view_src) root = root->view_src;
        auto & staging = ctx->zg_staging_bufs[root];
        if (staging.empty()) {
            staging.resize(bytes);
            device_tensor.read(staging.data(), 0, bytes);
        }
        // Do not redirect t->data here.
    } else if (t->data != nullptr) {
        device_tensor.read(reinterpret_cast<char *>(t->data), 0, bytes);
    }
}

static void ggml_fmsh_materialize_if_device(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    std::unordered_map<const ggml_tensor *, Tensor> & device_tensor_map) {
    if (!t) return;
    auto it = device_tensor_map.find(t);
    if (it == device_tensor_map.end()) return;

    const size_t bytes = ggml_nbytes(t);
    if (bytes == 0) { device_tensor_map.erase(it); return; }

    MemChunk zg_chunk;
    size_t   zg_off = 0;
    if (ggml_fmsh_is_zg_buffer_tensor(t, &zg_chunk, &zg_off)) {
        // t->data is a ZG DDR physical address — D2H into staging buffer keyed by root tensor.
        // We do NOT modify t->data here; callers that need host access should
        // check zg_staging_bufs and swap root->data around the CPU call.
        const ggml_tensor * root = t;
        while (root->view_src) root = root->view_src;
        auto & staging = ctx->zg_staging_bufs[root];
        // Only D2H the full root tensor once per graph_compute; subsequent views share it.
        if (staging.empty()) {
            const size_t root_bytes = ggml_nbytes(root);
            staging.resize(root_bytes);
            MemChunk rc; size_t ro = 0;
            ggml_fmsh_is_zg_buffer_tensor(root, &rc, &ro);
            rc.read(staging.data(), ro, root_bytes);
            ggml_fmsh_log_locked(ctx, 1,
                "materialize_zg_to_staging bytes=" + std::to_string(root_bytes) +
                " op_src=" + std::string(ggml_op_name(t->op)));
        }
    } else if (t->data != nullptr) {
        // t->data is a real host pointer (compute buffer): write directly.
        it->second.read(reinterpret_cast<char *>(t->data), 0, bytes);
        ggml_fmsh_log_locked(ctx, 1,
            "materialize_to_host bytes=" + std::to_string(bytes) +
            " op_src=" + std::string(ggml_op_name(t->op)));
    }
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
            " elementwise_cache_size=" + std::to_string(ctx->elementwise_session_cache.size()) +
            " flash_cache_size=" + std::to_string(ctx->flash_attn_session_cache.size()));
        for (const auto & kv : ctx->perf) {
            const auto & p = kv.second;
            ggml_fmsh_log_locked(
                ctx, 1,
                "op_summary op=" + ggml_fmsh_op_name_from_id(kv.first) +
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
        const double flash_hit = ctx->flash_attn_total == 0 ? 0.0 : (100.0 * static_cast<double>(ctx->flash_attn_offloaded) / static_cast<double>(ctx->flash_attn_total));
        ggml_fmsh_log_locked(
            ctx, 1,
            "offload_summary op=FLASH_ATTN_EXT total=" + std::to_string(ctx->flash_attn_total) +
            " offloaded=" + std::to_string(ctx->flash_attn_offloaded) +
            " fallback=" + std::to_string(ctx->flash_attn_fallback) +
            " hit_rate_pct=" + std::to_string(flash_hit));
        ggml_fmsh_log_locked(
            ctx, 1,
            "offload_summary op=ELEMENTWISE output_chunk_hits=" + std::to_string(ctx->elementwise_output_chunk_hits) +
            " output_chunk_misses=" + std::to_string(ctx->elementwise_output_chunk_misses));
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
        // Device is reference-counted; don't force-close here.
        // The underlying connection closes when the last handle (dev_ctx->memory_device) drops.
        ctx->zg_device = Device();
        ctx->device_opened = false;
    }
    // Disconnect buft_ctx from this backend so future alloc_buffer calls fall back to CPU.
    if (backend->device) {
        auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(backend->device->context);
        if (dev_ctx && dev_ctx->buft_ctx) {
            std::lock_guard<std::mutex> lock(dev_ctx->mu);
            dev_ctx->buft_ctx->backend_ctx = nullptr;
        }
    }
    if (ctx && ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }
    delete ctx;
    delete backend;
}

// Returns true if t's data lives in a ZG DDR buffer (allocated by our custom buffer type).
// Fills out_chunk with the buffer's MemChunk and out_offset with the byte offset of t
// within that chunk. Views are resolved: offset accumulates view_offs up the chain.
static bool ggml_fmsh_is_zg_buffer_tensor(
        const ggml_tensor * t, MemChunk * out_chunk, size_t * out_offset) {
    if (!t) return false;
    // Walk to root of the view chain, accumulating view_offs.
    size_t total_off = 0;
    const ggml_tensor * root = t;
    while (root->view_src) {
        total_off += static_cast<size_t>(root->view_offs);
        root = root->view_src;
    }
    if (!root->buffer || !root->buffer->buft ||
        root->buffer->buft->iface.alloc_buffer != ggml_fmsh_zg330_buffer_type_alloc_buffer) {
        return false;
    }
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(root->buffer->context);
    if (!buf_ctx || !buf_ctx->chunk.defined()) return false;
    // Offset of the root tensor's data within the chunk.
    const uintptr_t base = reinterpret_cast<uintptr_t>(root->buffer->iface.get_base(root->buffer));
    const size_t root_off = static_cast<size_t>(reinterpret_cast<uintptr_t>(root->data) - base);
    *out_chunk  = buf_ctx->chunk;
    *out_offset = root_off + total_off;
    return true;
}

// Build an icraft Tensor pointing to the ZG DDR region for ggml tensor t.
// The TensorType shape is flat (1-D, total elements) with F32 element type —
// sufficient for materialize_if_device (D2H read) and for ops that look up
// device_tensor_map to feed into session.forward.
static Tensor ggml_fmsh_make_zg_tensor(const ggml_tensor * t) {
    MemChunk chunk;
    size_t offset = 0;
    if (!ggml_fmsh_is_zg_buffer_tensor(t, &chunk, &offset)) {
        return Tensor();
    }
    // Choose element ScalarType based on ggml type.
    icraft::xir::ScalarType elem;
    switch (t->type) {
        case GGML_TYPE_F16:
            elem = icraft::xir::FloatType::FP16();
            break;
        case GGML_TYPE_BF16:
            elem = icraft::xir::FloatType::BF16();
            break;
        default:
            elem = icraft::xir::FloatType::FP32();
            break;
    }
    const int64_t n_elem = ggml_nelements(t);
    icraft::xir::TensorType ttype(elem,
                                   icraft::xir::Array<int64_t>{n_elem},
                                   icraft::xir::Layout());
    return Tensor(ttype, chunk, static_cast<uint64_t>(offset));
}

// Execute a single CPU node with ZG buffer tensors temporarily redirected to their staging buffers.
// After execution, writes staging data back to ZG DDR for any src that is a ZG buffer
// and has a staging buffer (meaning it may have been written by the CPU op, e.g. SET_ROWS into KV cache).
static enum ggml_status ggml_fmsh_compute_cpu_node_with_zg_redirect(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_cgraph * cgraph,
    int node_index) {
    ggml_tensor * node = cgraph->nodes[node_index];

    // Save original data pointers for ZG buffer tensors and redirect to staging.
    struct redirect_entry { ggml_tensor * t; void * orig_data; };
    std::vector<redirect_entry> redirects;
    auto redirect_tensor = [&](ggml_tensor * t) {
        if (!t) return;
        // Walk to root tensor.
        ggml_tensor * root = t;
        while (root->view_src) root = root->view_src;
        MemChunk zg_chunk; size_t zg_off = 0;
        if (!ggml_fmsh_is_zg_buffer_tensor(root, &zg_chunk, &zg_off)) return;
        auto it = ctx->zg_staging_bufs.find(root);
        if (it == ctx->zg_staging_bufs.end() || it->second.empty()) return;
        char * staging_base = it->second.data();
        // root may already be redirected; use saved orig_data as ZG base.
        uintptr_t root_zg = reinterpret_cast<uintptr_t>(root->data);
        for (const auto & r : redirects) {
            if (r.t == root) { root_zg = reinterpret_cast<uintptr_t>(r.orig_data); break; }
        }
        // Redirect root first (if not already done).
        if (![&]{ for (const auto & r : redirects) if (r.t == root) return true; return false; }()) {
            redirects.push_back({root, root->data});
            root->data = staging_base + (root_zg - root_zg); // = staging_base
        }
        // Redirect t if it differs from root (view: host_ptr = staging + (t_zg - root_zg)).
        if (t != root && ![&]{ for (const auto & r : redirects) if (r.t == t) return true; return false; }()) {
            // t->data was set during view init: root_zg_at_init + view_offs_chain.
            // Use the saved root ZG address to compute offset correctly.
            const uintptr_t t_zg = reinterpret_cast<uintptr_t>(t->data);
            redirects.push_back({t, t->data});
            t->data = staging_base + (t_zg - root_zg);
        }
    };
    for (int s = 0; s < GGML_MAX_SRC; ++s) redirect_tensor(node->src[s]);
    redirect_tensor(node); // output tensor may also be ZG buffer (unusual but safe)

    const enum ggml_status st = ggml_fmsh_compute_cpu_node(ctx, cgraph, node_index);

    // Restore original data pointers.
    for (const auto & r : redirects) r.t->data = r.orig_data;

    if (st != GGML_STATUS_SUCCESS) return st;

    // Write back: for any src ZG buffer that has a staging buffer, sync staging → ZG DDR.
    // This handles ops like SET_ROWS that write into a ZG buffer (e.g. KV cache).
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = node->src[s];
        if (!src) break;
        const ggml_tensor * root = src;
        while (root->view_src) root = root->view_src;
        auto sit = ctx->zg_staging_bufs.find(root);
        if (sit == ctx->zg_staging_bufs.end() || sit->second.empty()) continue;
        MemChunk zg_chunk; size_t zg_off = 0;
        if (!ggml_fmsh_is_zg_buffer_tensor(root, &zg_chunk, &zg_off)) continue;
        zg_chunk.write(zg_off, sit->second.data(), sit->second.size());
        ggml_fmsh_log_locked(ctx, 1,
            "staging_writeback bytes=" + std::to_string(sit->second.size()) +
            " op=" + std::string(ggml_op_name(node->op)) +
            " src=" + std::to_string(s));
    }
    return GGML_STATUS_SUCCESS;
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
    // Staging buffers for ZG buffer tensors that need CPU access this compute pass.
    // Cleared at the top of each graph_compute so redirected t->data pointers from the
    // previous pass are invalidated before new ZG allocations are registered.
    ctx->zg_staging_bufs.clear();

    // Pre-populate device_tensor_map with any tensor that lives in ZG DDR
    // (model weights, KV cache, etc.) so graph_compute can route ops correctly.
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        auto pre_register = [&](const ggml_tensor * t) {
            if (!t || device_tensor_map.count(t)) return;
            Tensor zt = ggml_fmsh_make_zg_tensor(t);
            if (zt.defined()) {
                device_tensor_map[t] = std::move(zt);
            }
        };
        pre_register(node);
        for (int s = 0; s < GGML_MAX_SRC; ++s) pre_register(node->src[s]);
    }

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
            // ROPE can consume device-resident src[0].
            if (consumer->op == GGML_OP_ROPE && src_idx == 0 &&
                consumer->type == GGML_TYPE_F32 && consumer->src[0] != nullptr &&
                consumer->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            // Elementwise ops can consume a device tensor at src[0].
            // RMS_NORM uses two-phase PL+scalar-sqrt execution so it also cannot use input_override here.
            if (src_idx == 0 && ggml_fmsh_should_run_elementwise_zg(ctx, consumer)) {
                ggml::fmsh::netmake::ElementwiseZgOp ek;
                int64_t er = 0, ec = 0;
                if (!ggml_fmsh_validate_elementwise(consumer, &ek, &er, &ec)) {
                    return false;
                }
                return ek != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM;
            }
            return false;
        };

    // Greedy chain extension: starting at from_idx, extend as far as possible into a linear
    // same-cols elementwise chain. Returns true only if chain has >=2 ops.
    auto try_extend_ew_chain = [&](int from_idx, ggml_fmsh_zg330_fused_ew_chain * out_chain) -> bool {
        ggml_tensor * node0 = cgraph->nodes[from_idx];
        ggml::fmsh::netmake::ElementwiseZgOp kind0;
        int64_t rows0 = 0, cols0 = 0;
        if (!ggml_fmsh_validate_elementwise(node0, &kind0, &rows0, &cols0)) return false;
        if (!ggml_fmsh_should_run_elementwise_zg(ctx, node0)) return false;
        if (kind0 == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX) return false;

        out_chain->nodes.clear();
        out_chain->ops.clear();
        out_chain->cgraph_indices.clear();
        out_chain->nodes.push_back(node0);
        out_chain->ops.push_back(kind0);
        out_chain->cgraph_indices.push_back(from_idx);
        out_chain->cols = cols0;
        out_chain->rows = rows0;

        int     cur_idx  = from_idx;
        ggml_tensor * cur_node = node0;

        while (static_cast<int>(out_chain->ops.size()) < 16) {
            ggml_tensor * next_node    = nullptr;
            int           next_src_idx = -1;
            if (!find_single_future_consumer(cur_idx, cur_node, &next_node, &next_src_idx)) break;
            if (next_src_idx != 0) break;            // secondary consumer, not linear
            if (next_node->src[0] != cur_node) break; // not direct linear dependency

            ggml::fmsh::netmake::ElementwiseZgOp next_kind;
            int64_t next_rows = 0, next_cols = 0;
            if (!ggml_fmsh_validate_elementwise(next_node, &next_kind, &next_rows, &next_cols)) break;
            if (next_cols != out_chain->cols) break;
            if (!ggml_fmsh_should_run_elementwise_zg(ctx, next_node)) break;
            if (next_kind == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX) break;
            // icraft ZG330 codegen only supports Transpose/ReduceSum on named-axis layouts
            // (FD). Intermediate tensors get *C layout → crash. RMS_NORM must be first.
            if (next_kind == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) break;

            // Find next_node's cgraph index (linear scan from cur_idx+1).
            int next_idx = -1;
            for (int j = cur_idx + 1; j < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j] == next_node) { next_idx = j; break; }
            }
            if (next_idx < 0) break;

            out_chain->nodes.push_back(next_node);
            out_chain->ops.push_back(next_kind);
            out_chain->cgraph_indices.push_back(next_idx);
            cur_idx  = next_idx;
            cur_node = next_node;
        }

        if (static_cast<int>(out_chain->ops.size()) < 2) return false;

        out_chain->compiled_rows =
            ggml_fmsh_elementwise_compile_rows(out_chain->ops[0], out_chain->rows);
        return true;
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

        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            ctx->flash_attn_total++;
            if (!ctx->offload_flash_attn_ext) {
                ggml_fmsh_log_locked(ctx, 1, "fallback op=FLASH_ATTN_EXT reason=disabled_by_env");
                ctx->flash_attn_fallback++;
                log_dispatch_boundary(false, node, "flash_disabled");
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[3], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[4], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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

            log_dispatch_boundary(true, node, "dispatch_switch");
            bool created = false;
            std::string err;
            ggml_fmsh_zg330_flash_attn_session_entry * entry =
                ggml_fmsh_get_or_create_flash_attn_session(ctx, node, &created, &err);
            if (!entry) {
                ggml_fmsh_log_locked(ctx, 2, "fallback op=FLASH_ATTN_EXT reason=" + err);
                ctx->flash_attn_fallback++;
                if (ctx->strict_mode) {
                    return GGML_STATUS_FAILED;
                }
                log_dispatch_boundary(false, node, "flash_session_create_failed");
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[3], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[4], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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
                std::string(created ? "session_create" : "session_hit") +
                " op=FLASH_ATTN_EXT" +
                " q_len=" + std::to_string(entry->info.q_len) +
                " kv_len=" + std::to_string(entry->info.kv_len) +
                " q_bucket=" + std::to_string(entry->info.q_bucket) +
                " kv_bucket=" + std::to_string(entry->info.kv_bucket) +
                " softmax_cols=" + std::to_string(entry->info.softmax_cols) +
                " n_head=" + std::to_string(entry->info.n_head) +
                " n_head_kv=" + std::to_string(entry->info.n_head_kv) +
                " mask=" + std::to_string(entry->info.has_mask ? 1 : 0) +
                " causal=" + std::to_string(entry->info.causal ? 1 : 0) +
                " net=" + entry->bundle.net_name +
                " net_cache=" + std::string(entry->bundle.ram_cache_hit ? "HIT" : "MISS") +
                " compile_now=" + std::string((created && entry->bundle.compiled_now) ? "YES" : "NO"));

            const auto t0 = std::chrono::high_resolution_clock::now();
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            std::vector<float> cpu_ref_out;
            if (ctx->debug_compare  == 1) {
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[3], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[4], device_tensor_map);
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
                    !backup_if_alias(node->src[2]) ||
                    !backup_if_alias(node->src[3]) ||
                    !backup_if_alias(node->src[4])) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_backup_failed op=FLASH_ATTN_EXT reason=" + dbg_err);
                }
                const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                if (cpu_st != GGML_STATUS_SUCCESS) {
                    return cpu_st;
                }
                std::string snap_err;
                if (!ggml_fmsh_tensor_snapshot_f32(node, cpu_ref_out, &snap_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=FLASH_ATTN_EXT reason=" + snap_err);
                }
                if (!debug_saved.empty()) {
                    std::string restore_err;
                    if (!ggml_fmsh_debug_restore_tensors(debug_saved, &restore_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "debug_compare_restore_failed op=FLASH_ATTN_EXT reason=" + restore_err);
                        return GGML_STATUS_FAILED;
                    }
                }
            }
#endif
            // Materialize any device-resident Q/K/V inputs before executing flash attention.
            // These reads are no-ops if the tensors are already on host (the common case for batch 1).
            ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[3], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[4], device_tensor_map);
            if (!ggml_fmsh_execute_flash_attn_ext(ctx, node, entry, &err)) {
                ggml_fmsh_log_locked(ctx, 2, "fallback op=FLASH_ATTN_EXT reason=" + err);
                ctx->flash_attn_fallback++;
                if (ctx->strict_mode) {
                    return GGML_STATUS_FAILED;
                }
                log_dispatch_boundary(false, node, "flash_exec_failed");
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[3], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[4], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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
                ggml_fmsh_materialize_if_device(ctx, node, device_tensor_map);
                ggml_fmsh_debug_compare_and_log(ctx, node, cpu_ref_out, i);
            }
#endif
            ctx->flash_attn_offloaded++;
            has_prev_device_output = false;
            prev_device_output_node = nullptr;
            device_tensor_map.erase(node);
            last_dispatched_valid = true;
            last_dispatched_zg = true;
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT) {
            ctx->mul_mat_total++;
            if (ggml_fmsh_is_batched_mul_mat(node)) {
                ctx->batched_mul_mat_total++;
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
            if (ctx->debug_compare == 1) {
                const bool has_device_src0 = node->src[0] != nullptr && device_tensor_map.find(node->src[0]) != device_tensor_map.end();
                const bool has_device_src1 = node->src[1] != nullptr && device_tensor_map.find(node->src[1]) != device_tensor_map.end();
                const bool has_device_src2 = node->src[2] != nullptr && device_tensor_map.find(node->src[2]) != device_tensor_map.end();
                if (has_device_src0 || has_device_src1 || has_device_src2) {
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "debug_compare_skip op=" + std::string(ggml_op_name(node->op)) +
                        " reason=device_input_chain");
                } else {
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
                const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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
            }
            }
            const bool keep_device_output_effective = keep_device_output_only;
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
                ggml_fmsh_materialize_if_device(ctx, node, device_tensor_map);
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

        if (node->op == GGML_OP_ROPE) {
            if (!ctx->offload_rope) {
                // Fall through to CPU.
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                const enum ggml_status st_cpu = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                if (st_cpu != GGML_STATUS_SUCCESS) { return st_cpu; }
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = false;
                continue;
            }
            bool rope_created = false;
            std::string rope_err;
            ggml_fmsh_zg330_rope_session_entry * rope_entry =
                ggml_fmsh_get_or_create_rope_session(ctx, node, &rope_created, &rope_err);
            if (rope_entry) {
                ggml_fmsh_log_locked(
                    ctx, 1,
                    std::string(rope_created ? "session_create" : "session_hit") +
                    " op=ROPE hits=" + std::to_string(rope_entry->hit_count) +
                    " net=" + rope_entry->bundle.net_name +
                    " net_cache=" + std::string(rope_entry->bundle.ram_cache_hit ? "HIT" : "MISS"));
                // Materialize any device-resident src before execute.
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                const auto t0_r = std::chrono::high_resolution_clock::now();
                if (!ggml_fmsh_execute_rope(ctx, node, rope_entry, &rope_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "fallback op=ROPE reason=" + rope_err);
                    if (ctx->strict_mode) { return GGML_STATUS_FAILED; }
                    const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                    if (st != GGML_STATUS_SUCCESS) { return st; }
                } else {
                    ggml_fmsh_accumulate_profile(ctx, node, rope_entry->session,
                        std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0_r).count());
                }
                has_prev_device_output = false;
                prev_device_output_node = nullptr;
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = true;
                continue;
            }
            // Fall back to host CPU for ROPE if ZG session creation failed.
            ggml_fmsh_log_locked(ctx, 1, "rope_fallback_host reason=" + rope_err);
            ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
            const enum ggml_status st_r = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
            if (st_r != GGML_STATUS_SUCCESS) { return st_r; }
            has_prev_device_output = false;
            prev_device_output_node = nullptr;
            device_tensor_map.erase(node);
            last_dispatched_valid = true;
            last_dispatched_zg = false;
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
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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

            // ── RMS_NORM two-phase split path (no Sqrt on PL) ─────────────────
            if (ek == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM && !ctx->rms_norm_native_sqrt) {
                bool split_created = false;
                std::string split_err;
                ggml_fmsh_zg330_rmsnorm_split_session_entry * split_entry =
                    ggml_fmsh_get_or_create_rmsnorm_split_session(ctx, node, &split_created, &split_err);
                if (split_entry) {
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        std::string(split_created ? "session_create" : "session_hit") +
                        " op=RMS_NORM_SPLIT hits=" + std::to_string(split_entry->hit_count) +
                        " pre=" + split_entry->bundle.net_name_pre +
                        " post=" + split_entry->bundle.net_name_post);

                    // Check if upstream produced a device-resident tensor we can use as X.
                    const Tensor * split_input_override = nullptr;
                    const ggml_tensor * split_input_src = nullptr;
                    if (node->src[0] != nullptr) {
                        auto it_dev = device_tensor_map.find(node->src[0]);
                        if (it_dev != device_tensor_map.end()) {
                            split_input_override = &it_dev->second;
                            split_input_src = node->src[0];
                        } else if (has_prev_device_output && prev_device_output_node == node->src[0]) {
                            split_input_override = &prev_device_output;
                            split_input_src = node->src[0];
                        }
                    }

                    // Determine if next consumer (MUL) can consume device output.
                    ggml_tensor * split_next_consumer = nullptr;
                    int split_next_src_idx = -1;
                    const bool split_keep_device =
                        split_entry->post_output_chunk.defined() &&
                        find_single_future_consumer(i, node, &split_next_consumer, &split_next_src_idx) &&
                        can_consume_from_device(split_next_consumer, split_next_src_idx);

                    // Evict stale tensors sharing the post output chunk.
                    if (split_entry->post_output_chunk.defined()) {
                        for (auto it_map = device_tensor_map.begin(); it_map != device_tensor_map.end();) {
                            if (it_map->second.defined() && it_map->second.chunk() == split_entry->post_output_chunk) {
                                const ggml_tensor * held = it_map->first;
                                if (held && held->data) {
                                    const size_t hb = ggml_nbytes(held);
                                    if (hb > 0) ggml_fmsh_evict_to_host(ctx, held, it_map->second);
                                }
                                if (has_prev_device_output && prev_device_output_node == held) {
                                    has_prev_device_output = false; prev_device_output_node = nullptr;
                                }
                                it_map = device_tensor_map.erase(it_map);
                            } else { ++it_map; }
                        }
                    }

                    const auto t0_s = std::chrono::high_resolution_clock::now();
                    // If no device input override, src[0] must be host-accessible.
                    if (split_input_override == nullptr) {
                        ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                    }
                    Tensor split_chained_output;
                    if (!ggml_fmsh_execute_rmsnorm_split(ctx, node, split_entry,
                            split_input_override, split_keep_device, &split_chained_output, &split_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "fallback op=RMS_NORM reason=" + split_err);
                        if (ctx->strict_mode) { return GGML_STATUS_FAILED; }
                        const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                        if (st != GGML_STATUS_SUCCESS) { return st; }
                        has_prev_device_output = false;
                        prev_device_output_node = nullptr;
                    } else {
                        ggml_fmsh_accumulate_profile(ctx, node, split_entry->session_post,
                            std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0_s).count());
                        // Remove consumed input from device map.
                        if (split_input_src) {
                            device_tensor_map.erase(split_input_src);
                            if (has_prev_device_output && prev_device_output_node == split_input_src) {
                                has_prev_device_output = false; prev_device_output_node = nullptr;
                            }
                        }
                        has_prev_device_output = false;
                        prev_device_output_node = nullptr;
                        if (split_keep_device && split_chained_output.defined()) {
                            device_tensor_map[node] = split_chained_output;
                            ggml_fmsh_log_locked(ctx, 0, "rmsnorm_split keep_device_output=1");
                        } else {
                            device_tensor_map.erase(node);
                        }
                    }
                    last_dispatched_valid = true;
                    last_dispatched_zg = true;
                    continue;
                }
                // Fall through to single-network path (uses Sqrt, may fail on ZG).
                ggml_fmsh_log_locked(ctx, 1, "rmsnorm_split_session_create_failed reason=" + split_err + " falling back to single-network");
            }

            // ── Greedy fused chain attempt ────────────────────────────────────
            {
                ggml_fmsh_zg330_fused_ew_chain fchain;
                if (try_extend_ew_chain(i, &fchain)) {
                    bool fcreated = false;
                    std::string ferr;
                    ggml_fmsh_zg330_fused_ew_entry * fentry =
                        ggml_fmsh_get_or_create_fused_ew_session(ctx, fchain, &fcreated, &ferr);
                    if (fentry) {
                        const auto t0_f = std::chrono::high_resolution_clock::now();

                        // Determine PLDDR input for the chain's first op.
                        const Tensor *   fchain_input_ptr = nullptr;
                        const ggml_tensor * fchain_input_src = nullptr;
                        if (fchain.ops[0] != ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX &&
                            fchain.nodes[0]->src[0] != nullptr) {
                            auto it_dev = device_tensor_map.find(fchain.nodes[0]->src[0]);
                            if (it_dev != device_tensor_map.end()) {
                                fchain_input_ptr = &it_dev->second;
                                fchain_input_src = fchain.nodes[0]->src[0];
                            } else if (has_prev_device_output &&
                                       prev_device_output_node == fchain.nodes[0]->src[0]) {
                                fchain_input_ptr = &prev_device_output;
                                fchain_input_src = fchain.nodes[0]->src[0];
                            }
                        }

                        // Determine whether to keep chain output on device.
                        const int fchain_last_idx = fchain.cgraph_indices.back();
                        ggml_tensor * fchain_next_consumer = nullptr;
                        int           fchain_next_src_idx  = -1;
                        const bool fkeep_device =
                            find_single_future_consumer(
                                fchain_last_idx, fchain.nodes.back(),
                                &fchain_next_consumer, &fchain_next_src_idx) &&
                            can_consume_from_device(fchain_next_consumer, fchain_next_src_idx);

                        // Evict any device tensors that alias fentry->output_chunk.
                        if (fentry->output_chunk.defined()) {
                            for (auto it_map = device_tensor_map.begin();
                                 it_map != device_tensor_map.end();) {
                                if (it_map->second.defined() &&
                                    it_map->second.chunk() == fentry->output_chunk) {
                                    const ggml_tensor * held = it_map->first;
                                    if (held && held->data) {
                                        const size_t hb = ggml_nbytes(held);
                                        if (hb > 0) {
                                            it_map->second.read(
                                                reinterpret_cast<char *>(held->data), 0, hb);
                                        }
                                    }
                                    if (has_prev_device_output &&
                                        prev_device_output_node == held) {
                                        has_prev_device_output = false;
                                        prev_device_output_node = nullptr;
                                    }
                                    it_map = device_tensor_map.erase(it_map);
                                } else {
                                    ++it_map;
                                }
                            }
                        }

                        // Materialize secondary inputs (src[1] of each chain node, e.g. ADD bias).
                        if (fchain_input_ptr == nullptr) {
                            ggml_fmsh_materialize_if_device(ctx, fchain.nodes[0]->src[0], device_tensor_map);
                        }
                        for (size_t fk = 0; fk < fchain.nodes.size(); ++fk) {
                            ggml_fmsh_materialize_if_device(ctx, fchain.nodes[fk]->src[1], device_tensor_map);
                        }
                        Tensor fchained_output;
                        if (ggml_fmsh_execute_fused_ew(
                                ctx, fchain, fentry, fchain_input_ptr,
                                fkeep_device, &fchained_output, &ferr)) {
                            // Consume PLDDR input.
                            if (fchain_input_src != nullptr) {
                                device_tensor_map.erase(fchain_input_src);
                                if (has_prev_device_output &&
                                    prev_device_output_node == fchain_input_src) {
                                    has_prev_device_output = false;
                                    prev_device_output_node = nullptr;
                                }
                            }

                            has_prev_device_output = false;
                            prev_device_output_node = nullptr;
                            if (fkeep_device && fchained_output.defined()) {
                                device_tensor_map[fchain.nodes.back()] = fchained_output;
                            } else {
                                device_tensor_map.erase(fchain.nodes.back());
                            }

                            const auto t1_f = std::chrono::high_resolution_clock::now();
                            const double ms_f = std::chrono::duration<double, std::milli>(
                                t1_f - t0_f).count();
                            ggml_fmsh_accumulate_profile(
                                ctx, fchain.nodes.back(), fentry->session, ms_f);

                            i = fchain_last_idx; // advance to last chain node (loop ++i moves past it)
                            last_dispatched_valid = true;
                            last_dispatched_zg    = true;
                            continue;
                        }
                        // Fused execution failed – fall through to single-op path.
                        ggml_fmsh_log_locked(ctx, 2, "fused_ew_fallback reason=" + ferr);
                    }
                }
            }
            // ── Single-op elementwise path (unchanged) ────────────────────────

            log_dispatch_boundary(true, node, "dispatch_switch");
            bool created = false;
            std::string err;
            ggml_fmsh_log_locked(ctx, 1, "elementwise_get_session op=" + std::string(ggml_op_name(node->op)));
            ggml_fmsh_zg330_elementwise_session_entry * entry =
                ggml_fmsh_get_or_create_elementwise_session(ctx, node, &created, &err);
            if (!entry) {
                ggml_fmsh_log_locked(ctx, 2, "fallback op=" + std::string(ggml_op_name(node->op)) + " reason=" + err);
                if (err.rfind("fatal:", 0) == 0) {
                    return GGML_STATUS_FAILED;
                }
                if (ctx->strict_mode) {
                    return GGML_STATUS_FAILED;
                }
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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

            // Check if src[0] has a device-resident result we can chain directly.
            // RMS_NORM uses two-phase PL+scalar-sqrt execution so it cannot use input_override.
            const Tensor * chained_input_ptr = nullptr;
            const ggml_tensor * ew_chained_input_src = nullptr;
            if (ek != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM &&
                node->src[0] != nullptr) {
                auto it_dev = device_tensor_map.find(node->src[0]);
                if (it_dev != device_tensor_map.end()) {
                    chained_input_ptr = &it_dev->second;
                    ew_chained_input_src = node->src[0];
                } else if (has_prev_device_output && prev_device_output_node == node->src[0]) {
                    chained_input_ptr = &prev_device_output;
                    ew_chained_input_src = node->src[0];
                }
            }

            // Determine whether to keep output on device for the next consumer.
            ggml_tensor * ew_next_consumer = nullptr;
            int ew_next_src_idx = -1;
            const bool keep_device_output_only =
                find_single_future_consumer(i, node, &ew_next_consumer, &ew_next_src_idx) &&
                can_consume_from_device(ew_next_consumer, ew_next_src_idx);

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            std::vector<float> cpu_ref_out;
            if (ctx->debug_compare == 1) {
                const bool has_device_src0 = node->src[0] != nullptr && device_tensor_map.find(node->src[0]) != device_tensor_map.end();
                const bool has_device_src1 = node->src[1] != nullptr && device_tensor_map.find(node->src[1]) != device_tensor_map.end();
                const bool has_device_src2 = node->src[2] != nullptr && device_tensor_map.find(node->src[2]) != device_tensor_map.end();
                if (has_device_src0 || has_device_src1 || has_device_src2) {
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "debug_compare_skip op=" + std::string(ggml_op_name(node->op)) +
                        " reason=device_input_chain");
                } else {
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
                const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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
            }
            const bool keep_device_output_effective = keep_device_output_only;
#else
            const bool keep_device_output_effective = keep_device_output_only;
#endif
            if (entry->output_chunk.defined()) {
                for (auto it_map = device_tensor_map.begin(); it_map != device_tensor_map.end();) {
                    if (it_map->second.defined() && it_map->second.chunk() == entry->output_chunk) {
                        const ggml_tensor * held = it_map->first;
                        if (held != nullptr && held->data != nullptr) {
                            const size_t held_bytes = ggml_nbytes(held);
                            if (held_bytes > 0) {
                                ggml_fmsh_evict_to_host(ctx, held, it_map->second);
                            }
                        }
                        ggml_fmsh_log_locked(
                            ctx, 1,
                            "elementwise_output_chunk_conflict_materialize op=" + std::string(ggml_op_name(node->op)));
                        if (has_prev_device_output && prev_device_output_node == held) {
                            has_prev_device_output = false;
                            prev_device_output_node = nullptr;
                        }
                        it_map = device_tensor_map.erase(it_map);
                    } else {
                        ++it_map;
                    }
                }
            }
            // Ensure src tensors are host-accessible before elementwise execute.
            if (chained_input_ptr == nullptr) {
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
            }
            ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
            ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
            Tensor chained_output;
            if (!ggml_fmsh_execute_elementwise(ctx, node, entry, chained_input_ptr, keep_device_output_effective, &chained_output, &err)) {
                ggml_fmsh_log_locked(ctx, 2, "fallback op=" + std::string(ggml_op_name(node->op)) + " reason=" + err);
                if (err.rfind("fatal:", 0) == 0) {
                    return GGML_STATUS_FAILED;
                }
                if (ctx->strict_mode) {
                    return GGML_STATUS_FAILED;
                }
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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

            // Remove consumed input from device_tensor_map.
            if (ew_chained_input_src != nullptr) {
                device_tensor_map.erase(ew_chained_input_src);
                if (has_prev_device_output && prev_device_output_node == ew_chained_input_src) {
                    has_prev_device_output = false;
                    prev_device_output_node = nullptr;
                }
            }

            has_prev_device_output = false;
            prev_device_output_node = nullptr;
            if (keep_device_output_effective && chained_output.defined()) {
                Tensor map_output = chained_output;
                if (ew_next_consumer != nullptr && ew_next_src_idx == 0 &&
                    ggml_fmsh_should_run_elementwise_zg(ctx, ew_next_consumer) &&
                    map_output.chunk().defined()) {
                    ggml::fmsh::netmake::ElementwiseZgOp next_kind;
                    int64_t next_rows = 0;
                    int64_t next_cols = 0;
                    if (ggml_fmsh_validate_elementwise(ew_next_consumer, &next_kind, &next_rows, &next_cols) &&
                        next_cols == entry->cols) {
                        const int64_t next_compile_rows = ggml_fmsh_elementwise_compile_rows(next_kind, next_rows);
                        if (next_compile_rows > 0 && next_compile_rows <= entry->compiled_rows &&
                            next_compile_rows != entry->compiled_rows) {
                            TensorType map_type = map_output.dtype().clone();
                            map_type.setShape(icraft::xir::Array<int64_t>{next_compile_rows, next_cols});
                            map_output = Tensor(map_type, map_output.chunk(), map_output.offset());
                            ggml_fmsh_log_locked(
                                ctx, 1,
                                "elementwise_output_retype_for_consumer producer=" +
                                    std::string(ggml_op_name(node->op)) +
                                    " consumer=" + std::string(ggml_op_name(ew_next_consumer->op)) +
                                    " rows=" + std::to_string(entry->compiled_rows) +
                                    "->" + std::to_string(next_compile_rows) +
                                    " cols=" + std::to_string(next_cols));
                        }
                    }
                }
                device_tensor_map[node] = map_output;
                ggml_fmsh_log_locked(ctx, 0, "dispatch op=" + std::string(ggml_op_name(node->op)) + " keep_device_output=1");
            } else {
                device_tensor_map.erase(node);
            }
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
            const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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
        const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
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

// Forward declaration (defined after buffer callbacks).
static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(
        ggml_fmsh_zg330_device_context * dev_ctx);

static ggml_backend_t ggml_backend_fmsh_zg330_init_impl(ggml_fmsh_zg330_device_context * dev_ctx = nullptr) {
    auto * ctx = new ggml_backend_fmsh_zg330_context;
    ctx->cpu_backend = ggml_backend_cpu_init();
    if (!ctx->cpu_backend) {
        delete ctx;
        return nullptr;
    }


    if (dev_ctx) {
        std::lock_guard<std::mutex> lock(dev_ctx->mu);
        if (dev_ctx->memory_device_opened) {
            // Share the device handle (reference-counted); do not consume/clear it
            // so subsequent init_backend calls (e.g. probe cycles) can reuse it.
            ctx->zg_device = dev_ctx->memory_device;
            ctx->device_opened = true;
            ctx->device_url = dev_ctx->memory_device_url;
        }
    }

    ctx->strict_mode = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_STRICT", false);
    ctx->enable_log = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_LOG", true);
    ctx->log_level = ggml_fmsh_get_log_level();
    ctx->offload_cpy_dup = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_CPY_DUP", false);
    ctx->offload_soft_max = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_SOFT_MAX", true);
    ctx->offload_rms_norm = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_RMS_NORM", true);
    ctx->rms_norm_native_sqrt = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_RMS_NORM_NATIVE_SQRT", false);
    ctx->offload_rope = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_ROPE", true);
    ctx->offload_flash_attn_ext = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_FLASH_ATTN_EXT", true);
    ctx->flash_softmax_cu = std::max<int64_t>(1, static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_FLASH_SOFTMAX_CU", 8)));
    ctx->flash_precompile_kv_depth = static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_FLASH_PRECOMPILE_KV_DEPTH", 1));
    ctx->flash_kv_bucket_max = std::max<int64_t>(1, static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_FLASH_KV_BUCKET_MAX", 8192)));
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
            " rms_norm_native_sqrt=" + std::to_string(ctx->rms_norm_native_sqrt ? 1 : 0) +
            " offload_flash_attn_ext=" + std::to_string(ctx->offload_flash_attn_ext ? 1 : 0) +
            " flash_softmax_cu=" + std::to_string(ctx->flash_softmax_cu) +
            " flash_precompile_kv_depth=" + std::to_string(ctx->flash_precompile_kv_depth) +
            " flash_kv_bucket_max=" + std::to_string(ctx->flash_kv_bucket_max) +
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            " debug_compare=" + std::to_string(ctx->debug_compare ? 1 : 0) +
            " debug_compare_atol=" + std::to_string(ctx->debug_compare_atol) +
            " debug_compare_rtol=" + std::to_string(ctx->debug_compare_rtol) +
#endif
            " cache_dir=" + ctx->cache_dir.string() +
            " device_opened=" + std::to_string(ctx->device_opened ? 1 : 0));
    }

    // Bind this backend to the device's buft_ctx so alloc_buffer can use the device.
    if (dev_ctx) {
        std::lock_guard<std::mutex> lock(dev_ctx->mu);
        // Ensure buft_ctx exists (created lazily in buffer_type_impl).
        if (!dev_ctx->buft_ctx) {
            dev_ctx->buft_ctx = new ggml_fmsh_zg330_buft_ctx{};
            dev_ctx->buft_ctx->dev_ctx_ptr = dev_ctx;
        }
        dev_ctx->buft_ctx->backend_ctx = ctx;
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
    auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(dev ? dev->context : nullptr);
    if (dev_ctx) {
        std::lock_guard<std::mutex> lock(dev_ctx->mu);
        if (dev_ctx->memory_cache_valid) {
            *free = dev_ctx->memory_free;
            *total = dev_ctx->memory_total;
            return;
        }

        try {
            if (!dev_ctx->memory_device_opened) {
                dev_ctx->memory_device_url = ggml_fmsh_get_device_url();
                dev_ctx->memory_device = Device::Open(dev_ctx->memory_device_url);
                dev_ctx->memory_device_opened = true;
            }

            std::string err;
            const bool ok = ggml_fmsh_query_pl_memory(dev_ctx->memory_device, free, total, &err);
            if (ok) {
                dev_ctx->memory_free = *free;
                dev_ctx->memory_total = *total;
                dev_ctx->memory_cache_valid = true;
                return;
            }
            GGML_LOG_WARN("%s: failed to query PL memory: %s\n", __func__, err.c_str());
        } catch (const std::exception & e) {
            GGML_LOG_WARN("%s: failed to open zg330 device for PL memory query: %s\n", __func__, e.what());
        } catch (...) {
            GGML_LOG_WARN("%s: failed to open zg330 device for PL memory query\n", __func__);
        }
    }

    // Last-resort fallback so llama_params_fit does not silently reassign
    // layers to CPU when the device is unavailable during backend enumeration.
    const size_t fallback = size_t(1) << 50; // 1 PiB, well under INT64_MAX
    *free = fallback;
    *total = fallback;
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
    GGML_UNUSED(params);
    return ggml_backend_fmsh_zg330_init_impl(static_cast<ggml_fmsh_zg330_device_context *>(dev ? dev->context : nullptr));
}

static void ggml_fmsh_zg330_buffer_free(ggml_backend_buffer_t buffer) {
    delete static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
}

// get_base returns the ZG DDR physical address as a fake host pointer.
// This pointer is NEVER dereferenced on the host; it is only used as a unique
// identifier for tensor->data. The address is 4096-aligned (from ZG malloc),
// which satisfies the ggml allocator alignment requirement.
static void * ggml_fmsh_zg330_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    const uint64_t addr = ctx->chunk->begin.addr();
    return reinterpret_cast<void *>(static_cast<uintptr_t>(addr));
}

static enum ggml_status ggml_fmsh_zg330_buffer_init_tensor(
        ggml_backend_buffer_t /*buffer*/, struct ggml_tensor * /*tensor*/) {
    return GGML_STATUS_SUCCESS;
}

static size_t ggml_fmsh_zg330_tensor_offset(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_fmsh_zg330_buffer_get_base(buffer));
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(tensor->data) - base);
}

static void ggml_fmsh_zg330_buffer_set_tensor(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    const size_t toff = ggml_fmsh_zg330_tensor_offset(buffer, tensor);
    buf_ctx->chunk.write(toff + offset,
                         const_cast<char *>(static_cast<const char *>(data)),
                         size);
}

static void ggml_fmsh_zg330_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    const size_t toff = ggml_fmsh_zg330_tensor_offset(buffer, tensor);
    buf_ctx->chunk.read(static_cast<char *>(data), toff + offset, size);
}

static void ggml_fmsh_zg330_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    if (!buf_ctx || !buf_ctx->chunk.defined()) return;
    // Zero-fill the entire ZG DDR buffer so that staged D2H reads get deterministic data.
    // Done in 4 KB chunks to stay within socket RPC payload limits.
    constexpr size_t CHUNK = 4096;
    std::vector<char> zero(CHUNK, static_cast<char>(value));
    for (size_t done = 0; done < buf_ctx->size; done += CHUNK) {
        const size_t blk = std::min(CHUNK, buf_ctx->size - done);
        buf_ctx->chunk.write(done, zero.data(), blk);
    }
}

static ggml_backend_buffer_t ggml_fmsh_zg330_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = static_cast<ggml_fmsh_zg330_buft_ctx *>(buft->context);
    if (!buft_ctx || !buft_ctx->backend_ctx || !buft_ctx->backend_ctx->device_opened) {
        return ggml_backend_cpu_buffer_type()->iface.alloc_buffer(
                   ggml_backend_cpu_buffer_type(), size);
    }
    auto * bctx = buft_ctx->backend_ctx;
    Device zg_dev = bctx->zg_device;
    MemChunk chunk;
    try {
        chunk = zg_dev.defaultMemRegion().malloc(size, /*auto_free=*/true, /*alignment=*/4096);
    } catch (const std::exception & e) {
        ggml_fmsh_log_locked(bctx, 2,
            std::string("zg_alloc_buffer_fail size=") + std::to_string(size) + " " + e.what());
        return ggml_backend_cpu_buffer_type()->iface.alloc_buffer(
                   ggml_backend_cpu_buffer_type(), size);
    }
    if (!chunk.defined()) {
        return ggml_backend_cpu_buffer_type()->iface.alloc_buffer(
                   ggml_backend_cpu_buffer_type(), size);
    }
    ggml_fmsh_log_locked(bctx, 1, "zg_alloc_buffer size=" + std::to_string(size));
    auto * buf_ctx = new ggml_fmsh_zg330_buffer_ctx{std::move(chunk), size};
    static const struct ggml_backend_buffer_i zg_buffer_iface = {
        /* .free_buffer     = */ ggml_fmsh_zg330_buffer_free,
        /* .get_base        = */ ggml_fmsh_zg330_buffer_get_base,
        /* .init_tensor     = */ ggml_fmsh_zg330_buffer_init_tensor,
        /* .memset_tensor   = */ nullptr,
        /* .set_tensor      = */ ggml_fmsh_zg330_buffer_set_tensor,
        /* .get_tensor      = */ ggml_fmsh_zg330_buffer_get_tensor,
        /* .set_tensor_2d   = */ nullptr,
        /* .get_tensor_2d   = */ nullptr,
        /* .cpy_tensor      = */ nullptr,
        /* .clear           = */ ggml_fmsh_zg330_buffer_clear,
        /* .reset           = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, zg_buffer_iface, buf_ctx, size);
}

// Returns the per-device ZG buffer type. Called from device_get_buffer_type;
// the buft object is cached on dev_ctx (created once, never freed).
static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(
        ggml_fmsh_zg330_device_context * dev_ctx) {
    if (!dev_ctx->cached_buft) {
        if (!dev_ctx->buft_ctx) {
            dev_ctx->buft_ctx = new ggml_fmsh_zg330_buft_ctx{};
            dev_ctx->buft_ctx->dev_ctx_ptr = dev_ctx;
        }
        static struct ggml_backend_buffer_type_i iface = {
            /* .get_name      = */ [](ggml_backend_buffer_type_t) -> const char * { return "FMSH_ZG330"; },
            /* .alloc_buffer  = */ ggml_fmsh_zg330_buffer_type_alloc_buffer,
            /* .get_alignment = */ [](ggml_backend_buffer_type_t) -> size_t { return 4096; },
            /* .get_max_size  = */ nullptr,
            /* .get_alloc_size= */ nullptr,
            /* .is_host       = */ [](ggml_backend_buffer_type_t) -> bool { return false; },
        };
        dev_ctx->cached_buft = new ggml_backend_buffer_type{iface, /*device=*/nullptr, /*context=*/dev_ctx->buft_ctx};
    }
    return dev_ctx->cached_buft;
}

// Legacy zero-arg overload: fallback to CPU buffer type.
static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(void) {
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // Compute buffer and weights stay on CPU host.
    // KV cache uses ZG DDR via the dedicated get_kv_buft proc_address interface.
    // Model weights are uploaded to ZG DDR manually inside graph_compute via session weight_tensors.
    return ggml_backend_cpu_buffer_type();
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
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }
    // Accept ZG DDR buffer types created by ggml_fmsh_zg330_buffer_type_impl(bctx).
    if (buft && buft->iface.alloc_buffer == ggml_fmsh_zg330_buffer_type_alloc_buffer) {
        return true;
    }
    return false;
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
    static auto * dev_ctx = new ggml_fmsh_zg330_device_context;
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_fmsh_zg330_device_i,
        /* .reg     = */ reg,
        /* .context = */ dev_ctx,
    };
    return &dev;
}

// Returns the ZG DDR buffer type for KV cache tensors.
// Called via proc_address by llama-kv-cache.cpp when offloading KV cache.
static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_get_kv_buft(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(dev->context);
    if (!dev_ctx) return nullptr;
    return ggml_backend_fmsh_zg330_buffer_type_impl(dev_ctx);
}

static void * ggml_backend_fmsh_zg330_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return reinterpret_cast<void *>(ggml_backend_fmsh_zg330_set_n_threads);
    }
    if (std::strcmp(name, "ggml_backend_fmsh_zg330_get_kv_buft") == 0) {
        return reinterpret_cast<void *>(ggml_backend_fmsh_zg330_get_kv_buft);
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
