#include "ggml-fmsh-zg330-internal.h"

namespace ggml_fmsh_zg330_impl {

ggml_fmsh_zg330_op_signature ggml_fmsh_make_signature(const ggml_tensor * node) {
    ggml_fmsh_zg330_op_signature sig = {};
    sig.op = static_cast<uint32_t>(node->op);
    sig.dtype = static_cast<uint32_t>(node->type);
    std::memcpy(sig.shape, node->ne, sizeof(sig.shape));
    return sig;
}

int64_t ggml_fmsh_next_pow2_i64(int64_t v) {
    if (v <= 1) {
        return 1;
    }
    int64_t p = 1;
    while (p < v && p < (1LL << 30)) {
        p <<= 1;
    }
    return p < v ? v : p;
}

int64_t ggml_fmsh_align_up_i64(int64_t v, int64_t a) {
    if (a <= 1) {
        return v;
    }
    const int64_t r = v % a;
    return r == 0 ? v : (v + (a - r));
}



void ggml_fmsh_log_locked(ggml_backend_fmsh_zg330_context * ctx, int level, const std::string & msg);

bool ggml_fmsh_rmsnorm_customop_cols_ok(ggml_backend_fmsh_zg330_context * ctx, int64_t cols) {
    if (cols == 1024 || cols == 256) {
        return true;
    }
    ggml_fmsh_log_locked(ctx, 2, "rmsnorm_customop_skip reason=unsupported_cols cols=" + std::to_string(cols));
    return false;
}

int64_t ggml_fmsh_node_rows(const ggml_tensor * t) {
    if (t == nullptr) {
        return 0;
    }
    return t->ne[1] * t->ne[2] * t->ne[3];
}

ggml_fmsh_chain_tensor_id ggml_fmsh_make_chain_tensor_id(const ggml_tensor * t) {
    ggml_fmsh_chain_tensor_id id;
    if (t == nullptr) {
        return id;
    }
    id.op = t->op;
    id.data = reinterpret_cast<uintptr_t>(t->data);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id.ne[i] = t->ne[i];
    }
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = t->src[s];
        if (src == nullptr) {
            continue;
        }
        id.src_op[s] = src->op;
        id.src_data[s] = reinterpret_cast<uintptr_t>(src->data);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            id.src_ne[s][d] = src->ne[d];
        }
    }
    id.valid = true;
    return id;
}

bool ggml_fmsh_wait_tensor_ready(
    const Tensor & t,
    std::string * err,
    const char * where,
    int64_t timeout_ms) {
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

int64_t ggml_fmsh_elementwise_compile_rows(ggml::fmsh::netmake::ElementwiseZgOp kind, int64_t rows) {
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

ggml_fmsh_zg330_op_signature ggml_fmsh_make_elementwise_signature(
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

bool ggml_fmsh_get_env_bool(const char * key, bool def) {
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

double ggml_fmsh_get_env_double(const char * key, double def) {
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

uint64_t ggml_fmsh_get_env_u64(const char * key, uint64_t def) {
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

int64_t ggml_fmsh_flash_bucket_len(int64_t len) {
    if (len <= 1) {
        return 1;
    }
    return ggml_fmsh_next_pow2_i64(len);
}

int ggml_fmsh_get_log_level(void) {
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

std::mutex & ggml_fmsh_shared_device_registry_mu() {
    static std::mutex mu;
    return mu;
}

std::unordered_map<std::string, ggml_fmsh_zg330_shared_device_entry> & ggml_fmsh_shared_device_registry() {
    static std::unordered_map<std::string, ggml_fmsh_zg330_shared_device_entry> registry;
    return registry;
}

bool ggml_fmsh_acquire_shared_device(
    const std::string & device_url,
    Device * out_device,
    std::string * err) {
    if (out_device == nullptr) {
        if (err) {
            *err = "shared device acquire called with null out_device";
        }
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ggml_fmsh_shared_device_registry_mu());
        auto & registry = ggml_fmsh_shared_device_registry();
        auto it = registry.find(device_url);
        if (it != registry.end()) {
            it->second.ref_count++;
            *out_device = it->second.device;
            return true;
        }

        ggml_fmsh_zg330_shared_device_entry entry;
        entry.device = Device::Open(device_url);
        entry.ref_count = 1;
        *out_device = entry.device;
        registry.emplace(device_url, std::move(entry));
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    } catch (...) {
        if (err) {
            *err = "unknown error";
        }
        return false;
    }
}

void ggml_fmsh_release_shared_device(
    const std::string & device_url,
    Device * device,
    bool * device_opened) {
    if (device == nullptr || device_opened == nullptr || !*device_opened) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ggml_fmsh_shared_device_registry_mu());
        auto & registry = ggml_fmsh_shared_device_registry();
        auto it = registry.find(device_url);
        if (it != registry.end()) {
            if (it->second.ref_count > 0) {
                it->second.ref_count--;
            }
            if (it->second.ref_count == 0) {
                try {
                    Device::Close(it->second.device);
                } catch (...) {
                }
                registry.erase(it);
            }
        }
    }

    *device = Device();
    *device_opened = false;
}

const char * ggml_fmsh_log_tag(int level) {
    switch (level) {
        case 0: return "DEBUG";
        case 1: return "INFO";
        case 2: return "WARN";
        default: return "ERROR";
    }
}

void ggml_fmsh_log_locked(ggml_backend_fmsh_zg330_context * ctx, int level, const std::string & msg) {
    if (ctx == nullptr) {
        // Some validation helpers (e.g. can_run_mul_mat_zg, invoked from can_consume_from_device
        // during planning) log without a context. Drop those messages rather than crash.
        return;
    }
    if (!ctx->enable_log || level < ctx->log_level) {
        return;
    }
    std::ofstream ofs(ctx->log_file, std::ios::app);
    if (ofs) {
        ofs << "[" << ggml_fmsh_log_tag(level) << "] " << msg << "\n";
    }
}

bool ggml_fmsh_is_meta_op(const ggml_tensor * op) {
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

bool ggml_fmsh_is_chain_alias_op(const ggml_tensor * op) {
    if (!op) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
            return true;
        default:
            return false;
    }
}

bool ggml_fmsh_is_host_dispatch_op(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_SET:
        case GGML_OP_MUL_MAT_ID:
            return true;
        default:
            return false;
    }
}

bool ggml_fmsh_is_elementwise_zg_op(const ggml_tensor * op) {
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

bool ggml_fmsh_is_cpy_dup_op(const ggml_tensor * op) {
    return op->op == GGML_OP_CPY || op->op == GGML_OP_DUP;
}

bool ggml_fmsh_map_elementwise_op(
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

// A quantized src0 (model weight) is offloadable iff ggml provides a dequant kernel.
bool ggml_fmsh_src0_quant_ok(const ggml_tensor * src0) {
    if (src0 == nullptr || !ggml_is_quantized(src0->type)) {
        return false;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(src0->type);
    return tt != nullptr && tt->to_float != nullptr;
}

// Env gate cached once; mirrors ctx->offload_quant_weights for the context-less
// supports_op path used by the scheduler during graph planning.
bool ggml_fmsh_quant_weights_enabled() {
    static const bool en = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_QUANT_WEIGHTS", true);
    return en;
}

// M bucket cap, cached once for the context-less planning path.
int64_t ggml_fmsh_mul_mat_m_bucket_max_env() {
    static const int64_t cap = std::max<int64_t>(1,
        static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_MUL_MAT_M_BUCKET_MAX", 1)));
    return cap;
}

// Optional N (output-feature) cap: matmuls with n beyond this stay on CPU.
// HOST dynamic weight DMA is unreliable for large n on ZG330 (produces wrong results).
// Static weight (ZG DDR resident) would require too much DDR for all layers simultaneously.
// Default 512: known-good threshold from empirical testing.
// Set to 0 to disable (for debugging only — will produce incorrect output for large n).
int64_t ggml_fmsh_mul_mat_n_max_env() {
    static const int64_t cap =
        static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_MUL_MAT_N_MAX", 512));
    return cap;
}

// Quantized weight matmul: 2D only (no batched ne12/ne13), src1 plain F32, and M
// representable within the bucket cap. We deliberately only claim ops we will actually
// run on the NPU — ops whose M exceeds the cap stay on the CPU backend (the well-tested
// native path), so they never enter the FMSH graph and need a fragile in-graph fallback.
bool ggml_fmsh_is_quant_weight_mul_mat(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    if (!ggml_fmsh_quant_weights_enabled() || !ggml_fmsh_src0_quant_ok(op->src[0])) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[1]->ne[2] != 1 || op->src[1]->ne[3] != 1) {
        return false;
    }
    // Optional N cap: keep very-large-N matmuls (e.g. lm_head) on CPU.
    const int64_t n_max = ggml_fmsh_mul_mat_n_max_env();
    if (n_max > 0 && op->src[0]->ne[1] > n_max) {
        return false;
    }
    // M-bucket feasibility (next pow2 within cap); keep in sync with ggml_fmsh_mul_mat_m_bucket.
    const int64_t m = op->src[1]->ne[1];
    if (m > 1) {
        int64_t b = 1;
        while (b < m) {
            b <<= 1;
        }
        if (b > ggml_fmsh_mul_mat_m_bucket_max_env()) {
            return false;
        }
    }
    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

bool ggml_fmsh_is_supported_op(const ggml_tensor * op) {
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
    if (ggml_fmsh_is_cpy_dup_op(op)) {
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
    // Quantized weight matmul (Q4_K/Q6_K projections, FFN, lm_head): handled by the
    // dedicated dequant→BF16 NPU path. src0 stays quantized; dst F32, src1 F32, 2D.
    if (ggml_fmsh_is_quant_weight_mul_mat(op)) {
        return true;
    }
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[1]->type != GGML_TYPE_F32 && op->src[1]->type != GGML_TYPE_BF16) {
        return false;
    }
    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

bool ggml_fmsh_should_run_elementwise_zg(
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

bool ggml_fmsh_can_run_mul_mat_zg(const ggml_tensor * node) {
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

std::string ggml_fmsh_get_device_url() {
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

bool ggml_fmsh_validate_mul_mat(const ggml_tensor * node, int64_t * m, int64_t * k, int64_t * n) {
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

// Bucket M to the next power of two (capped). Returns <=0 when m exceeds the cap,
// signalling the caller to fall back to CPU (we won't compile unbounded shapes).
int64_t ggml_fmsh_mul_mat_m_bucket(int64_t m, int64_t bucket_max) {
    if (m <= 1) {
        return 1;
    }
    int64_t b = 1;
    while (b < m) {
        b <<= 1;
    }
    if (b > std::max<int64_t>(1, bucket_max)) {
        return -1;
    }
    return b;
}

std::string ggml_fmsh_make_mul_mat_cache_key(const ggml_tensor * node, int64_t m, int64_t k, int64_t n) {
    return std::to_string(static_cast<uint32_t>(node->op)) + "|" +
           std::to_string(static_cast<uint32_t>(node->type)) + "|" +
           std::to_string(node->ne[0]) + "," +
           std::to_string(node->ne[1]) + "," +
           std::to_string(node->ne[2]) + "," +
           std::to_string(node->ne[3]) + "|" +
           std::to_string(m) + "x" + std::to_string(k) + "x" + std::to_string(n);
}

bool ggml_fmsh_validate_elementwise(
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

float ggml_fmsh_read_mask_f32(
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

float ggml_fmsh_read_mask_f32(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * mask,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3);

bool ggml_fmsh_is_probably_causal_mask(
    ggml_backend_fmsh_zg330_context * ctx,
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
            const float mv = ggml_fmsh_read_mask_f32(ctx, mask, c, q_row, mask_head, mask_batch);
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

bool ggml_fmsh_validate_flash_attn_ext(
    ggml_backend_fmsh_zg330_context * ctx,
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
    if ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32 && v->type != GGML_TYPE_BF16)) {
        if (err) *err = "FLASH_ATTN_EXT requires k/v type F16, BF16, or F32";
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
        std::string staging_err;
        if (!ggml_fmsh_ensure_zg_staging(ctx, mask, true, &staging_err)) {
            if (err) *err = "FLASH_ATTN_EXT mask staging failed: " + staging_err;
            return false;
        }
        causal = ggml_fmsh_is_probably_causal_mask(ctx, mask, q_len, kv_len, 0, 0);
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

std::string ggml_fmsh_make_flash_attn_cache_key(
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

bool ggml_fmsh_open_device_if_needed(ggml_backend_fmsh_zg330_context * ctx, std::string * err) {
    if (ctx->device_opened) {
        return true;
    }
    try {
        ctx->device_url = ggml_fmsh_get_device_url();
        if (!ggml_fmsh_acquire_shared_device(ctx->device_url, &ctx->zg_device, err)) {
            GGML_LOG_ERROR("%s: failed to open zg330 device: %s\n", __func__, err ? err->c_str() : "unknown error");
            std::exit(EXIT_FAILURE);
        }
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

bool ggml_fmsh_open_device_context_if_needed(
    ggml_fmsh_zg330_device_context * dev_ctx,
    Device * out_device,
    std::string * err) {
    if (!dev_ctx || !out_device) {
        if (err) {
            *err = "device context is null";
        }
        return false;
    }
    if (dev_ctx->device_opened) {
        *out_device = dev_ctx->zg_device;
        return true;
    }

    dev_ctx->device_url = ggml_fmsh_get_device_url();
    if (!ggml_fmsh_acquire_shared_device(dev_ctx->device_url, &dev_ctx->zg_device, err)) {
        return false;
    }
    dev_ctx->device_opened = true;
    *out_device = dev_ctx->zg_device;
    return true;
}

bool ggml_fmsh_query_pl_memory(Device & device, size_t * free_bytes, size_t * total_bytes, std::string * err) {
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


} // namespace ggml_fmsh_zg330_impl
