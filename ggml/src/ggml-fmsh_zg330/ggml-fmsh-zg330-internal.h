#pragma once

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
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>


namespace ggml_fmsh_zg330_impl {

using namespace icraft::xrt;
namespace zg330 = icraft::xrt::zg330;

// custom_op_hub register offsets for single-pass RMSNorm hardware accelerator.
constexpr uint32_t kCopBase            = 0x400C0000U;
constexpr uint32_t kCopRegStart        = 0x0000U;
constexpr uint32_t kCopRegReadCfg      = 0x0004U;
constexpr uint32_t kCopRegWriteCfg     = 0x0008U;
constexpr uint32_t kCopRegCfg0         = 0x000CU;
constexpr uint32_t kCopRegShape0       = 0x0010U;
constexpr uint32_t kCopRegEps          = 0x0014U;
constexpr uint32_t kCopRegAddrHi       = 0x0018U;
constexpr uint32_t kCopRegHubSelW      = 0x0034U;
constexpr uint32_t kCopRegDone         = 0x0080U;
constexpr uint32_t kCopRegStatus       = 0x0090U;
constexpr uint32_t kCopRegHubSelR      = 0x00ACU;
constexpr uint32_t kCopRegVersion      = 0x00C0U;
constexpr uint32_t kCopRegReset        = 0x01DCU;
constexpr uint32_t kCopHubSelRmsnorm   = 1U;
constexpr uint32_t kCopRmsnormVersion  = 0x20260611U;
constexpr uint32_t kCopDoneMask        = 0x1U;
constexpr uint32_t kCopBusyMask        = 0x2U;
constexpr uint32_t kCopErrorMask       = 0x4U;
constexpr size_t   kCopBeatBytes       = 64UL;
constexpr size_t   kCopBf16PerBeat     = 32UL;
constexpr int      kCopTimeoutMs       = 10000;

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
    ZG330Device zg_device;
    uint32_t layer_increment = 0;
};

// Quantized weight matmul (Q4_K/Q6_K projections, FFN, lm_head) offloaded to NPU.
// The quantized src0 is dequantized once to F32 (tf32 NPU path) and kept resident on
// device; the activation A and the per-(k,n) network run in F32, output Y stays F32.
// M is bucketed (next pow2, capped) so the compiled matmul_MxKxN shape set is bounded.
// Large N (e.g. lm_head vocab) is split into column chunks, each its own compiled
// network — handled internally so the dispatch sees one node.
struct ggml_fmsh_zg330_qweight_session_entry {
    struct weight_slice {
        uintptr_t key = 0;   // src0 host pointer (identifies the static weight)
        Tensor    tensor;    // F32 [k, n_this], resident on device
    };
    struct sub_net {
        int64_t col_off = 0; // output-feature offset this chunk covers
        int64_t n_this  = 0; // number of output features in this chunk
        ggml::fmsh::netmake::MatmulZgNetworkBundle bundle;
        Session    session;
        TensorType input_type_a; // F32 [m_bucket, k]
        TensorType input_type_b; // F32 [k, n_this]
        Tensor     input_tensor_a;
        bool       input_tensor_a_ready = false;
        std::vector<weight_slice> weights;
        // AXI 1-frame-delay sync (see flash-attn): reused session keeps ready_=true,
        // so on ARM/AXI we must re-arm a layerCount check_func_ after each forward,
        // otherwise waitForReady() returns the PREVIOUS token's stale output.
        uint32_t layer_increment = 0;
    };
    uint64_t hit_count = 0;
    int64_t  m_bucket = 0;
    int64_t  k = 0;
    int64_t  n_full = 0;
    ZG330Device zg_device;   // valid in ARM/AXI mode; empty in socket mode
    std::vector<sub_net> subnets;
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
    ZG330Device zg_device;
    uint32_t layer_increment = 0;
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
    ZG330Device zg_device;
    uint32_t layer_increment_pre  = 0;
    uint32_t layer_increment_post = 0;
};

struct ggml_fmsh_zg330_rmsnorm_customop_entry {
    int64_t  cols        = 0;
    int64_t  alloc_rows  = 0;
    MemChunk mem_chunk;
    MemChunk output_chunk;
    uint64_t hit_count   = 0;
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
    ZG330Device zg_device;
    uint32_t layer_increment = 0;
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
    ZG330Device zg_device;
    uint32_t layer_increment = 0;
};

struct ggml_fmsh_zg330_bf16_bridge_session_entry {
    int64_t rows = 0;
    int64_t cols = 0;
    uint64_t hit_count = 0;
    ggml::fmsh::netmake::ElementwiseZgNetworkBundle bundle;
    Session session;
    TensorType input_type;
    Tensor input_tensor;
    ZG330Device zg_device;
    uint32_t layer_increment = 0;
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

struct ggml_fmsh_chain_tensor_id {
    ggml_op op = GGML_OP_NONE;
    uintptr_t data = 0;
    int64_t ne[GGML_MAX_DIMS] = {0, 0, 0, 0};
    ggml_op src_op[GGML_MAX_SRC] = {};
    uintptr_t src_data[GGML_MAX_SRC] = {};
    int64_t src_ne[GGML_MAX_SRC][GGML_MAX_DIMS] = {};
    bool valid = false;

    bool operator==(const ggml_fmsh_chain_tensor_id & other) const {
        return valid == other.valid &&
               op == other.op &&
               data == other.data &&
               std::memcmp(ne, other.ne, sizeof(ne)) == 0 &&
               std::memcmp(src_op, other.src_op, sizeof(src_op)) == 0 &&
               std::memcmp(src_data, other.src_data, sizeof(src_data)) == 0 &&
               std::memcmp(src_ne, other.src_ne, sizeof(src_ne)) == 0;
    }
};

struct ggml_fmsh_pending_chain_tensor_entry {
    ggml_fmsh_chain_tensor_id id;
    Tensor tensor;
    int remaining_uses = 0;
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

struct ggml_fmsh_zg330_shared_device_entry {
    Device device;
    size_t ref_count = 0;
};

struct ggml_backend_fmsh_zg330_context {
    ggml_backend_t cpu_backend = nullptr;

    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_session_entry>> session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_qweight_session_entry>> qweight_session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_elementwise_session_entry>, ggml_fmsh_zg330_op_signature_hash> elementwise_session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_rmsnorm_split_session_entry>, ggml_fmsh_zg330_op_signature_hash> rmsnorm_split_session_cache;
    std::unordered_map<ggml_fmsh_zg330_op_signature, std::unique_ptr<ggml_fmsh_zg330_rope_session_entry>, ggml_fmsh_zg330_op_signature_hash> rope_session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_flash_attn_session_entry>> flash_attn_session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_bf16_bridge_session_entry>> bridge_session_cache;
    std::unordered_map<std::string, std::unique_ptr<ggml_fmsh_zg330_fused_ew_entry>> fused_ew_session_cache;
    std::unordered_map<int64_t, std::unique_ptr<ggml_fmsh_zg330_rmsnorm_customop_entry>>
        rmsnorm_customop_cache;

    bool strict_mode = false;
    bool enable_log = true;
    int log_level = 1; // 0 debug, 1 info, 2 warn, 3 error
    bool offload_cpy_dup = true;
    bool offload_soft_max = true;
    bool offload_rms_norm = true;
    bool offload_flash_attn_ext = true;
    bool offload_rope = true;
    bool disable_device_input_chain = false;
    bool rms_norm_native_sqrt = false; // false=split (pre+host_sqrt+post), true=fused+Sqrt@hostt
    bool rms_norm_custom_op = false;
    bool rms_norm_custom_op_verified = false;
    int64_t flash_softmax_cu = 8;
    int64_t flash_precompile_kv_depth = 1;
    int64_t flash_kv_bucket_max = 8192;

    // Quantized weight-matmul offload (Q4_K/Q6_K projections, FFN, lm_head) onto NPU.
    // Weights are dequantized once to F32 (tf32 NPU path) and kept resident on device;
    // M is bucketed (next pow2, capped) so the set of compiled matmul_MxKxN shapes is bounded.
    bool    offload_quant_weights = true;
    int64_t mul_mat_m_bucket_max = 512;        // cap for M bucketing (1 = decode-only)
    int64_t mul_mat_n_chunk = 16384;           // split N into chunks (0 = no split)
    int64_t mul_mat_precompile_m_depth = 0;    // speculatively precompile next pow2 M buckets
    uint64_t mul_mat_qweight_offloaded = 0;
    uint64_t mul_mat_qweight_bytes_resident = 0;
    // Persistent F32 dequant buffers for qweight chunks, keyed by (quant_data_ptr + col_byte_offset).
    // Stable address → fake_src0.view_src=nullptr → src0_is_static=true in execute_mul_mat
    // → weight uploaded to ZG DDR once on first forward, cached in entry->weight_tensors thereafter.
    std::unordered_map<uintptr_t, std::vector<float>> qweight_f32_bufs;

    std::filesystem::path cache_dir;
    std::filesystem::path log_file;

    Device zg_device;
    bool device_opened = false;
    std::string device_url;

    // Per-graph-compute staging buffers for ZG buffer tensors that need CPU access.
    // When a ZG buffer tensor (weight/KV) is materialized, D2H data goes here and
    // tensor->data is redirected to the staging buffer for the duration of graph_compute.
    std::unordered_map<const ggml_tensor *, std::vector<char>> zg_staging_bufs;

    // Carry one chainable device output across graph_compute calls. Decode often splits
    // a logical producer -> qweight consumer pair across adjacent graph invocations.
    bool pending_prev_device_output_valid = false;
    Tensor pending_prev_device_output;
    ggml_fmsh_chain_tensor_id pending_prev_device_output_id;
    std::vector<ggml_fmsh_pending_chain_tensor_entry> pending_chainable_device_tensors;

    struct op_perf {
        uint64_t calls = 0;
        double total_ms = 0.0;
        double profile_total_ms = 0.0;
        double memcpy_ms = 0.0;
        double hard_ms = 0.0;
        double other_ms = 0.0;
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
bool ggml_fmsh_is_zg_buffer_tensor(
        const ggml_tensor * t, MemChunk * out_chunk, size_t * out_offset);
ggml_backend_buffer_t ggml_fmsh_zg330_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size);
bool ggml_fmsh_ensure_zg_staging(
        ggml_backend_fmsh_zg330_context * ctx,
        const ggml_tensor * t,
        bool read_existing,
        std::string * err);
bool ggml_fmsh_sync_zg_staging_to_device(
        ggml_backend_fmsh_zg330_context * ctx,
        const ggml_tensor * t,
        const char * reason,
        std::string * err);
const char * ggml_fmsh_host_data_for_tensor(
        ggml_backend_fmsh_zg330_context * ctx,
        const ggml_tensor * t);
inline void ggml_fmsh_write_f32_indexed(
        ggml_backend_fmsh_zg330_context * ctx,
        ggml_tensor * t,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3,
        float v);

struct ggml_fmsh_zg330_device_context {
    std::mutex mu;
    bool memory_cache_valid = false;
    size_t memory_free = 0;
    size_t memory_total = 0;
    Device zg_device;
    bool device_opened = false;
    std::string device_url;
    // Owned ZG buffer type for this device; backend_ctx is updated on init/free.
    ggml_fmsh_zg330_buft_ctx * buft_ctx = nullptr;
    ggml_backend_buffer_type * cached_buft = nullptr; // owns the single buft object
};

// ─── Debug support struct (conditionally compiled) ────────────────────────
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
struct ggml_fmsh_debug_saved_tensor {
    ggml_tensor * t = nullptr;
    std::vector<float> data;
};
#endif

// ─── Forward declarations (must precede inline helpers that call them) ────
// ─── Forward declarations ─────────────────────────────────────────────────
// utils
ggml_fmsh_zg330_op_signature ggml_fmsh_make_signature(const ggml_tensor * node);
int64_t ggml_fmsh_next_pow2_i64(int64_t v);
int64_t ggml_fmsh_align_up_i64(int64_t v, int64_t a);
bool ggml_fmsh_rmsnorm_customop_cols_ok(ggml_backend_fmsh_zg330_context * ctx, int64_t cols);
int64_t ggml_fmsh_node_rows(const ggml_tensor * t);
ggml_fmsh_chain_tensor_id ggml_fmsh_make_chain_tensor_id(const ggml_tensor * t);
bool ggml_fmsh_wait_tensor_ready(const Tensor & t, std::string * err, const char * where, int64_t timeout_ms = 30000);
int64_t ggml_fmsh_elementwise_compile_rows(ggml::fmsh::netmake::ElementwiseZgOp kind, int64_t rows);
ggml_fmsh_zg330_op_signature ggml_fmsh_make_elementwise_signature(const ggml_tensor * node, ggml::fmsh::netmake::ElementwiseZgOp kind, int64_t rows, int64_t cols);
bool ggml_fmsh_get_env_bool(const char * key, bool def);
double ggml_fmsh_get_env_double(const char * key, double def);
uint64_t ggml_fmsh_get_env_u64(const char * key, uint64_t def);
int64_t ggml_fmsh_flash_bucket_len(int64_t len);
int ggml_fmsh_get_log_level(void);
std::mutex & ggml_fmsh_shared_device_registry_mu();
std::unordered_map<std::string, ggml_fmsh_zg330_shared_device_entry> & ggml_fmsh_shared_device_registry();
bool ggml_fmsh_acquire_shared_device(const std::string & device_url, Device * out_device, std::string * err);
void ggml_fmsh_release_shared_device(const std::string & device_url, Device * device, bool * device_opened);
const char * ggml_fmsh_log_tag(int level);
void ggml_fmsh_log_locked(ggml_backend_fmsh_zg330_context * ctx, int level, const std::string & msg);
bool ggml_fmsh_is_meta_op(const ggml_tensor * op);
bool ggml_fmsh_is_chain_alias_op(const ggml_tensor * op);
bool ggml_fmsh_is_host_dispatch_op(const ggml_tensor * op);
bool ggml_fmsh_is_elementwise_zg_op(const ggml_tensor * op);
bool ggml_fmsh_is_cpy_dup_op(const ggml_tensor * op);
bool ggml_fmsh_map_elementwise_op(ggml_op op, ggml::fmsh::netmake::ElementwiseZgOp * out);
bool ggml_fmsh_src0_quant_ok(const ggml_tensor * src0);
bool ggml_fmsh_quant_weights_enabled();
int64_t ggml_fmsh_mul_mat_m_bucket_max_env();
int64_t ggml_fmsh_mul_mat_n_max_env();
bool ggml_fmsh_is_quant_weight_mul_mat(const ggml_tensor * op);
bool ggml_fmsh_is_supported_op(const ggml_tensor * op);
bool ggml_fmsh_should_run_elementwise_zg(const ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * op);
bool ggml_fmsh_can_run_mul_mat_zg(const ggml_tensor * node);
std::string ggml_fmsh_get_device_url();
bool ggml_fmsh_validate_mul_mat(const ggml_tensor * node, int64_t * m, int64_t * k, int64_t * n);
int64_t ggml_fmsh_mul_mat_m_bucket(int64_t m, int64_t bucket_max);
std::string ggml_fmsh_make_mul_mat_cache_key(const ggml_tensor * node, int64_t m, int64_t k, int64_t n);
bool ggml_fmsh_validate_elementwise(const ggml_tensor * node, ggml::fmsh::netmake::ElementwiseZgOp * op_kind, int64_t * rows, int64_t * cols);
float ggml_fmsh_read_mask_f32(const ggml_tensor * mask, int64_t i0, int64_t i1, int64_t i2, int64_t i3);
float ggml_fmsh_read_mask_f32(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * mask, int64_t i0, int64_t i1, int64_t i2, int64_t i3);
bool ggml_fmsh_is_probably_causal_mask(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * mask, int64_t q_len, int64_t kv_len, int64_t mask_head, int64_t mask_batch);
bool ggml_fmsh_validate_flash_attn_ext(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, ggml_fmsh_flash_attn_validation * out, std::string * err);
std::string ggml_fmsh_make_flash_attn_cache_key(const ggml_tensor * node, const ggml_fmsh_flash_attn_validation & v);
bool ggml_fmsh_open_device_if_needed(ggml_backend_fmsh_zg330_context * ctx, std::string * err);
bool ggml_fmsh_open_device_context_if_needed(ggml_fmsh_zg330_device_context * dev_ctx, Device * out_device, std::string * err);
bool ggml_fmsh_query_pl_memory(Device & device, size_t * free_bytes, size_t * total_bytes, std::string * err);
// matmul
ggml_fmsh_zg330_session_entry * ggml_fmsh_get_or_create_mul_mat_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, bool * created, std::string * err);
ggml_fmsh_zg330_qweight_session_entry * ggml_fmsh_get_or_create_qweight_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, bool * created, std::string * err);
bool ggml_fmsh_forward_synced(ggml_backend_fmsh_zg330_context * ctx, Session & session, const std::vector<Tensor> & inputs, ZG330Device & zg_device, uint32_t & layer_increment, Tensor * out, const char * where, std::string * err);
bool ggml_fmsh_execute_qweight_mul_mat(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, ggml_fmsh_zg330_qweight_session_entry * entry, const Tensor * input_override, std::string * err);
// elementwise
ggml_fmsh_zg330_elementwise_session_entry * ggml_fmsh_get_or_create_elementwise_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, bool * created, std::string * err);
ggml_fmsh_zg330_rmsnorm_split_session_entry * ggml_fmsh_get_or_create_rmsnorm_split_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, bool * created, std::string * err);
ggml_fmsh_zg330_rope_session_entry * ggml_fmsh_get_or_create_rope_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, bool * created, std::string * err);
ggml_fmsh_zg330_fused_ew_entry * ggml_fmsh_get_or_create_fused_ew_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_fmsh_zg330_fused_ew_chain & chain, bool * created, std::string * err);
ggml_fmsh_zg330_flash_attn_session_entry * ggml_fmsh_get_or_create_flash_attn_session(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, bool * created, std::string * err);
std::string ggml_fmsh_make_bf16_bridge_cache_key(int64_t rows, int64_t cols);
ggml_fmsh_zg330_bf16_bridge_session_entry * ggml_fmsh_get_or_create_bf16_bridge_session(ggml_backend_fmsh_zg330_context * ctx, int64_t rows, int64_t cols, bool * created, std::string * err);
bool ggml_fmsh_execute_elementwise(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, ggml_fmsh_zg330_elementwise_session_entry * entry, const Tensor * input_override, int input_override_src_idx, bool keep_device_output_only, bool write_host_output, Tensor * chained_output, std::string * err);
bool ggml_fmsh_execute_rmsnorm_customop(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, int64_t rows, int64_t cols, const Tensor * input_override, bool keep_device_output, bool write_host_output, Tensor * chained_output, std::string * err);
bool ggml_fmsh_execute_rmsnorm_split(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, ggml_fmsh_zg330_rmsnorm_split_session_entry * entry, const Tensor * input_override, bool keep_device_output, bool write_host_output, Tensor * chained_output, std::string * err);
bool ggml_fmsh_execute_rope(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, ggml_fmsh_zg330_rope_session_entry * entry, std::string * err);
bool ggml_fmsh_execute_fused_ew(ggml_backend_fmsh_zg330_context * ctx, const ggml_fmsh_zg330_fused_ew_chain & chain, ggml_fmsh_zg330_fused_ew_entry * entry, const Tensor * input_override, bool keep_device_output_only, bool write_host_output, Tensor * chained_output, std::string * err);
const char * ggml_fmsh_host_data_for_tensor(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t);
bool ggml_fmsh_write_f32_flat_to_staging(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * t, const float * src, size_t n, std::string * err);
// flash-attn / execute
bool ggml_fmsh_execute_qweight_via_f32(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, std::string * err);
bool ggml_fmsh_execute_mul_mat(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, ggml_fmsh_zg330_session_entry * entry, const Tensor * input_override, bool keep_device_output_only, bool write_host_output, Tensor * chained_output, std::string * err);
bool ggml_fmsh_execute_flash_attn_ext(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, ggml_fmsh_zg330_flash_attn_session_entry * entry, std::string * err);
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
bool ggml_fmsh_tensor_snapshot_f32(const ggml_tensor * t, std::vector<float> & out, std::string * err);
bool ggml_fmsh_tensor_snapshot_f32(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, std::vector<float> & out, std::string * err);
bool ggml_fmsh_tensor_restore_f32(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * t, const std::vector<float> & in, std::string * err);
bool ggml_fmsh_debug_backup_tensor(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * t, std::vector<ggml_fmsh_debug_saved_tensor> & saved, std::unordered_set<ggml_tensor *> & seen, std::string * err);
bool ggml_fmsh_debug_restore_tensors(ggml_backend_fmsh_zg330_context * ctx, std::vector<ggml_fmsh_debug_saved_tensor> & saved, std::string * err);
void ggml_fmsh_debug_compare_and_log(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * node, const std::vector<float> & ref, int node_index);
#endif
// graph
void ggml_fmsh_accumulate_profile(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * node, const Session & session, double total_ms);
std::string ggml_fmsh_op_name_from_id(uint32_t op_id);
std::string ggml_fmsh_fmt_ms(double ms);
std::string ggml_fmsh_fmt_pct(double part, double total);
bool ggml_fmsh_is_batched_mul_mat(const ggml_tensor * node);
size_t ggml_fmsh_tensor_live_bytes_f32(const Tensor & t);
std::string ggml_fmsh_tensor_loc_str(ggml_backend_fmsh_zg330_context * ctx, const Tensor & t);
void ggml_fmsh_evict_to_host(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, const Tensor & device_tensor);
void ggml_fmsh_materialize_if_device(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, std::unordered_map<const ggml_tensor *, Tensor> & device_tensor_map);
bool ggml_fmsh_ensure_zg_staging(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, bool read_existing, std::string * err);
bool ggml_fmsh_sync_zg_staging_to_device(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, const char * reason, std::string * err);
int ggml_fmsh_count_future_consumers(ggml_cgraph * cgraph, int from_idx, const ggml_tensor * producer);
enum ggml_status ggml_fmsh_compute_cpu_fallback(ggml_backend_fmsh_zg330_context * ctx, ggml_cgraph * cgraph);
enum ggml_status ggml_fmsh_compute_cpu_node(ggml_backend_fmsh_zg330_context * ctx, ggml_cgraph * cgraph, int node_index);
const char * ggml_backend_fmsh_zg330_get_name(ggml_backend_t backend);
void ggml_backend_fmsh_zg330_free(ggml_backend_t backend);
bool ggml_fmsh_is_zg_buffer_tensor(const ggml_tensor * t, MemChunk * out_chunk, size_t * out_offset);
Tensor ggml_fmsh_make_zg_tensor(const ggml_tensor * t);
Tensor ggml_fmsh_clone_chain_tensor(ggml_backend_fmsh_zg330_context * ctx, const Tensor & src, std::string * err);
enum ggml_status ggml_fmsh_compute_cpu_node_with_zg_redirect(ggml_backend_fmsh_zg330_context * ctx, ggml_cgraph * cgraph, int node_index);
enum ggml_status ggml_fmsh_execute_set_rows_on_device(ggml_backend_fmsh_zg330_context * ctx, ggml_cgraph * cgraph, int node_idx, std::unordered_map<const ggml_tensor *, Tensor> & device_tensor_map);
enum ggml_status ggml_backend_fmsh_zg330_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);
// backend
ggml_guid_t ggml_backend_fmsh_zg330_guid(void);
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(ggml_fmsh_zg330_device_context * dev_ctx, ggml_backend_dev_t dev);
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(void);
ggml_backend_t ggml_backend_fmsh_zg330_init_impl(ggml_fmsh_zg330_device_context * dev_ctx = nullptr);
bool ggml_backend_is_fmsh_zg330_impl(ggml_backend_t backend);
void ggml_backend_fmsh_zg330_set_n_threads(ggml_backend_t backend, int n_threads);
const char * ggml_backend_fmsh_zg330_device_get_name(ggml_backend_dev_t dev);
const char * ggml_backend_fmsh_zg330_device_get_description(ggml_backend_dev_t dev);
void ggml_backend_fmsh_zg330_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total);
enum ggml_backend_dev_type ggml_backend_fmsh_zg330_device_get_type(ggml_backend_dev_t dev);
void ggml_backend_fmsh_zg330_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props);
ggml_backend_t ggml_backend_fmsh_zg330_device_init_backend(ggml_backend_dev_t dev, const char * params);
void ggml_fmsh_zg330_buffer_free(ggml_backend_buffer_t buffer);
void * ggml_fmsh_zg330_buffer_get_base(ggml_backend_buffer_t buffer);
enum ggml_status ggml_fmsh_zg330_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
size_t ggml_fmsh_zg330_tensor_offset(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);
void ggml_fmsh_zg330_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_fmsh_zg330_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_fmsh_zg330_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
void ggml_fmsh_zg330_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);
ggml_backend_buffer_t ggml_fmsh_zg330_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_device_get_buffer_type(ggml_backend_dev_t dev);
ggml_backend_buffer_t ggml_backend_fmsh_zg330_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);
bool ggml_backend_fmsh_zg330_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);
bool ggml_backend_fmsh_zg330_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
const char * ggml_backend_fmsh_zg330_reg_get_name(ggml_backend_reg_t reg);
size_t ggml_backend_fmsh_zg330_reg_get_device_count(ggml_backend_reg_t reg);
ggml_backend_dev_t ggml_backend_fmsh_zg330_reg_get_device(ggml_backend_reg_t reg, size_t index);
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_get_kv_buft(ggml_backend_dev_t dev);
void * ggml_backend_fmsh_zg330_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);
extern const ggml_backend_reg_i ggml_backend_fmsh_zg330_reg_i;

// ─── Inline helper functions ──────────────────────────────────────────────
inline uint16_t ggml_fmsh_f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

inline uint32_t ggml_fmsh_float_to_u32(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

inline float ggml_fmsh_read_f32_broadcast(
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


inline float ggml_fmsh_read_f32_broadcast(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const int64_t x0 = t->ne[0] == 1 ? 0 : i0;
    const int64_t x1 = t->ne[1] == 1 ? 0 : i1;
    const int64_t x2 = t->ne[2] == 1 ? 0 : i2;
    const int64_t x3 = t->ne[3] == 1 ? 0 : i3;
    const char * base = ggml_fmsh_host_data_for_tensor(ctx, t);
    const char * p = base + x0 * t->nb[0] + x1 * t->nb[1] + x2 * t->nb[2] + x3 * t->nb[3];
    return *reinterpret_cast<const float *>(p);
}

inline void ggml_fmsh_write_f32_indexed(
    ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3,
    float v) {
    char * p = static_cast<char *>(t->data) + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    *reinterpret_cast<float *>(p) = v;
}

inline void ggml_fmsh_write_f32_indexed(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3,
    float v) {
    if (!ctx || !t) {
        ggml_fmsh_write_f32_indexed(t, i0, i1, i2, i3, v);
        return;
    }
    MemChunk chunk;
    size_t off = 0;
    if (!ggml_fmsh_is_zg_buffer_tensor(t, &chunk, &off)) {
        ggml_fmsh_write_f32_indexed(t, i0, i1, i2, i3, v);
        return;
    }
    std::string err;
    if (!ggml_fmsh_ensure_zg_staging(ctx, t, true, &err)) {
        ggml_fmsh_log_locked(ctx, 3, "write_staging_prepare_failed op=" + std::string(ggml_op_name(t->op)) + " reason=" + err);
        std::exit(EXIT_FAILURE);
    }
    char * base = const_cast<char *>(ggml_fmsh_host_data_for_tensor(ctx, t));
    char * p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    *reinterpret_cast<float *>(p) = v;
}

inline float ggml_fmsh_read_f32_indexed(
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const char * p = static_cast<const char *>(t->data) + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    return *reinterpret_cast<const float *>(p);
}

inline float ggml_fmsh_read_f32_indexed(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const char * base = ggml_fmsh_host_data_for_tensor(ctx, t);
    const char * p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
    return *reinterpret_cast<const float *>(p);
}

inline float ggml_fmsh_read_f32_linear(const ggml_tensor * t, size_t idx) {
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

inline float ggml_fmsh_read_f32_linear(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, size_t idx) {
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
    return ggml_fmsh_read_f32_broadcast(
        ctx,
        t,
        static_cast<int64_t>(i0),
        static_cast<int64_t>(i1),
        static_cast<int64_t>(i2),
        static_cast<int64_t>(i3));
}


} // namespace ggml_fmsh_zg330_impl
