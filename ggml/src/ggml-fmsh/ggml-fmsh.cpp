#include "ggml-fmsh.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-xir/core/compile_target.h>
#include <icraft-xir/core/network.h>
#include <icraft-xir/core/reflection.h>
#include <icraft-xir/ops/abs.h>
#include <icraft-xir/ops/add.h>
#include <icraft-xir/ops/cast.h>
#include <icraft-xir/ops/elu.h>
#include <icraft-xir/ops/gelu.h>
#include <icraft-xir/ops/hardsigmoid.h>
#include <icraft-xir/ops/hardswish.h>
#include <icraft-xir/ops/matmul.h>
#include <icraft-xir/ops/mean.h>
#include <icraft-xir/ops/multiply.h>
#include <icraft-xir/ops/neg.h>
#include <icraft-xir/ops/relu.h>
#include <icraft-xir/ops/sigmoid.h>
#include <icraft-xir/ops/silu.h>
#include <icraft-xir/ops/softmax.h>
#include <icraft-xir/ops/sqrt.h>
#include <icraft-xir/ops/sum.h>
#include <icraft-xir/ops/tanh.h>
#include <icraft-xir/ops/transpose.h>
#include <icraft-xrt/core/device.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

struct ggml_backend_fmsh_context {
    ggml_backend_t cpu_backend = nullptr;
    bool enable_npu = false;
    bool runtime_ready = false;
    bool verbose = false;
    bool cache_enabled = true;
    bool warned_fallback = false;
    bool warned_unsupported[GGML_OP_COUNT] = { false };
    bool warned_npu_unavailable[GGML_OP_COUNT] = { false };
    std::string device_url;
    std::string cache_dir;
    bool npu_device_open = false;
    icraft::xrt::Device npu_device;
    std::unordered_map<std::string, icraft::xrt::Session> session_cache;
    std::mutex session_cache_mutex;
};

struct ggml_backend_fmsh_device_context {
    bool iCraft_enabled = false;
};

using ggml_fmsh_cpu_init_fn = ggml_backend_t (*)();
using ggml_fmsh_cpu_set_n_threads_fn = void (*)(ggml_backend_t, int);
using ggml_fmsh_cpu_buffer_type_fn = ggml_backend_buffer_type_t (*)();
using ggml_fmsh_cpu_buffer_from_ptr_fn = ggml_backend_buffer_t (*)(void *, size_t);

static ggml_fmsh_cpu_init_fn s_cpu_init = nullptr;
static ggml_fmsh_cpu_set_n_threads_fn s_cpu_set_n_threads = nullptr;
static ggml_fmsh_cpu_buffer_type_fn s_cpu_buffer_type = nullptr;
static ggml_fmsh_cpu_buffer_from_ptr_fn s_cpu_buffer_from_ptr = nullptr;
static void * s_cpu_lib_handle = nullptr;

static void ggml_fmsh_resolve_cpu_symbols(void) {
    if (s_cpu_init && s_cpu_set_n_threads && s_cpu_buffer_type && s_cpu_buffer_from_ptr) {
        return;
    }

    s_cpu_init = (ggml_fmsh_cpu_init_fn) dlsym(RTLD_DEFAULT, "ggml_backend_cpu_init");
    s_cpu_set_n_threads = (ggml_fmsh_cpu_set_n_threads_fn) dlsym(RTLD_DEFAULT, "ggml_backend_cpu_set_n_threads");
    s_cpu_buffer_type = (ggml_fmsh_cpu_buffer_type_fn) dlsym(RTLD_DEFAULT, "ggml_backend_cpu_buffer_type");
    s_cpu_buffer_from_ptr = (ggml_fmsh_cpu_buffer_from_ptr_fn) dlsym(RTLD_DEFAULT, "ggml_backend_cpu_buffer_from_ptr");

    if (s_cpu_init && s_cpu_set_n_threads && s_cpu_buffer_type && s_cpu_buffer_from_ptr) {
        return;
    }

    if (s_cpu_lib_handle == nullptr) {
        s_cpu_lib_handle = dlopen("libggml-cpu.so", RTLD_NOW | RTLD_LOCAL);
        if (s_cpu_lib_handle == nullptr) {
            s_cpu_lib_handle = dlopen("libggml-cpu.so.0", RTLD_NOW | RTLD_LOCAL);
        }
        if (s_cpu_lib_handle == nullptr) {
            s_cpu_lib_handle = dlopen("libggml-cpu.so.0.9.8", RTLD_NOW | RTLD_LOCAL);
        }
    }

    if (s_cpu_lib_handle != nullptr) {
        s_cpu_init = (ggml_fmsh_cpu_init_fn) dlsym(s_cpu_lib_handle, "ggml_backend_cpu_init");
        s_cpu_set_n_threads = (ggml_fmsh_cpu_set_n_threads_fn) dlsym(s_cpu_lib_handle, "ggml_backend_cpu_set_n_threads");
        s_cpu_buffer_type = (ggml_fmsh_cpu_buffer_type_fn) dlsym(s_cpu_lib_handle, "ggml_backend_cpu_buffer_type");
        s_cpu_buffer_from_ptr = (ggml_fmsh_cpu_buffer_from_ptr_fn) dlsym(s_cpu_lib_handle, "ggml_backend_cpu_buffer_from_ptr");
    }
}

static bool ggml_fmsh_parse_env_bool(const char * key, bool default_value) {
    const char * v = std::getenv(key);
    if (v == nullptr || *v == '\0') {
        return default_value;
    }

    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });

    if (s == "0" || s == "false" || s == "off" || s == "no") {
        return false;
    }
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
        return true;
    }

    return default_value;
}

static bool ggml_fmsh_probe_runtime(bool verbose) {
    const char * url_env = std::getenv("GGML_FMSH_DEVICE_URL");

#if defined(__linux__)
    const char * default_url = "axi://ql100aiu?npu=0x40000000&dma=0x80000000";
#else
    const char * default_url = "socket://ql100aiu@127.0.0.1:9981?npu=0x40000000&dma=0x80000000";
#endif

    const std::string device_url = (url_env != nullptr && *url_env != '\0') ? std::string(url_env) : std::string(default_url);

    try {
        auto dev = icraft::xrt::Device::Open(device_url);
        const bool ok = dev.check(0);
        icraft::xrt::Device::Close(dev);

        if (verbose) {
            GGML_LOG_INFO("ggml-fmsh: runtime probe url=%s ready=%d\n", device_url.c_str(), (int) ok);
        }

        return ok;
    } catch (const std::exception & e) {
        if (verbose) {
            GGML_LOG_WARN("ggml-fmsh: runtime probe failed for url=%s, reason=%s\n", device_url.c_str(), e.what());
        }
        return false;
    } catch (...) {
        if (verbose) {
            GGML_LOG_WARN("ggml-fmsh: runtime probe failed for url=%s, reason=unknown\n", device_url.c_str());
        }
        return false;
    }
}

static std::string ggml_fmsh_get_device_url(void) {
    const char * url_env = std::getenv("GGML_FMSH_DEVICE_URL");
    if (url_env != nullptr && *url_env != '\0') {
        return std::string(url_env);
    }

#if defined(__linux__)
    return "axi://ql100aiu?npu=0x40000000&dma=0x80000000";
#else
    return "socket://ql100aiu@127.0.0.1:9981?npu=0x40000000&dma=0x80000000";
#endif
}

static bool ggml_fmsh_is_contiguous_binary(const ggml_tensor * t) {
    return t != nullptr && ggml_is_contiguous(t);
}

static icraft::xir::ScalarType ggml_fmsh_to_icraft_scalar(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
            return icraft::xir::FloatType::FP32();
        case GGML_TYPE_F16:
            return icraft::xir::FloatType::FP16();
        case GGML_TYPE_BF16:
            return icraft::xir::FloatType::BF16();
        default:
            return {};
    }
}

static icraft::xir::TensorType ggml_fmsh_make_tensor_type(const ggml_tensor * t) {
    auto scalar = ggml_fmsh_to_icraft_scalar(t->type);
    const icraft::xir::Array<int64_t> shape {
        std::max<int64_t>(1, t->ne[3]),
        std::max<int64_t>(1, t->ne[2]),
        std::max<int64_t>(1, t->ne[1]),
        std::max<int64_t>(1, t->ne[0]),
    };
    return icraft::xir::TensorType(scalar, shape, icraft::xir::Layout::NCHW());
}

static icraft::xir::TensorType ggml_fmsh_make_tensor_type_with_ggml_type(const ggml_tensor * t, ggml_type forced_type) {
    auto scalar = ggml_fmsh_to_icraft_scalar(forced_type);
    const icraft::xir::Array<int64_t> shape {
        std::max<int64_t>(1, t->ne[3]),
        std::max<int64_t>(1, t->ne[2]),
        std::max<int64_t>(1, t->ne[1]),
        std::max<int64_t>(1, t->ne[0]),
    };
    return icraft::xir::TensorType(scalar, shape, icraft::xir::Layout::NCHW());
}

// Helper to centralize network creation + logging when verbose is enabled.
static icraft::xir::Network ggml_fmsh_make_network(const std::string &net_name, ggml_backend_fmsh_context * ctx) {
    if (ctx != nullptr && ctx->verbose) {
        GGML_LOG_INFO("ggml-fmsh: compiling network %s\n", net_name.c_str());
    }
    return icraft::xir::Network(net_name, icraft::xir::Framework::ONNX, "1");
}

// Make op outputs explicit from ggml dst type/shape.
// This avoids relying on iCraft TypeInfer registration for every op/target pair.
static icraft::xir::Output ggml_fmsh_make_output_from_dst(icraft::xir::Operation op, const ggml_tensor * dst) {
    icraft::xir::Value outv(ggml_fmsh_make_tensor_type(dst));
    op.setOutputs({ outv });
    return icraft::xir::Output(outv);
}

static icraft::xir::QuantizedScaleArray ggml_fmsh_default_cut_scale(void) {
    return icraft::xir::QuantizedScaleArray(icraft::xir::Array<double> { 1.0 }, -1);
}

static void ggml_fmsh_set_cut_scale_if_present(icraft::xir::Operation & op) {
    if (op->hasAttr("cut_scale")) {
        op.setAttr("cut_scale", ggml_fmsh_default_cut_scale());
    }
}

static bool ggml_fmsh_tensor_to_fp32(const ggml_tensor * src, std::vector<float> & out) {
    if (src == nullptr || src->data == nullptr) {
        return false;
    }

    const int64_t n = ggml_nelements(src);
    if (n <= 0) {
        return false;
    }

    out.resize((size_t) n);

    if (src->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src->data, (size_t) n * sizeof(float));
        return true;
    }

    if (src->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src->data, out.data(), n);
        return true;
    }

    if (src->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) src->data, out.data(), n);
        return true;
    }

    const ggml_type_traits * tt = ggml_get_type_traits(src->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        return false;
    }

    tt->to_float(src->data, out.data(), n);
    return true;
}

static bool ggml_fmsh_tensor_convert_to(const ggml_tensor * src, ggml_type dst_type, std::vector<uint8_t> & out) {
    if (src == nullptr || src->data == nullptr) {
        return false;
    }

    if (src->type == dst_type) {
        const size_t nb = ggml_nbytes(src);
        out.resize(nb);
        std::memcpy(out.data(), src->data, nb);
        return true;
    }

    std::vector<float> f32;
    if (!ggml_fmsh_tensor_to_fp32(src, f32)) {
        return false;
    }

    const int64_t n = (int64_t) f32.size();
    if (dst_type == GGML_TYPE_F32) {
        out.resize((size_t) n * sizeof(float));
        std::memcpy(out.data(), f32.data(), out.size());
        return true;
    }

    if (dst_type == GGML_TYPE_F16) {
        out.resize((size_t) n * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) out.data(), n);
        return true;
    }

    if (dst_type == GGML_TYPE_BF16) {
        out.resize((size_t) n * sizeof(ggml_bf16_t));
        ggml_fp32_to_bf16_row_ref(f32.data(), (ggml_bf16_t *) out.data(), n);
        return true;
    }

    return false;
}

static size_t ggml_fmsh_type_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
            return sizeof(float);
        case GGML_TYPE_F16:
            return sizeof(ggml_fp16_t);
        case GGML_TYPE_BF16:
            return sizeof(ggml_bf16_t);
        default:
            return 0;
    }
}

static bool ggml_fmsh_prepare_binary_rhs_broadcast(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    std::vector<uint8_t> & rhs_out,
    const void *& rhs_data_out,
    uint64_t & rhs_nbytes_out) {

    rhs_data_out = src1->data;
    rhs_nbytes_out = (uint64_t) ggml_nbytes(src1);

    bool same_shape = true;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (src0->ne[d] != src1->ne[d]) {
            same_shape = false;
            break;
        }
    }

    if (same_shape) {
        return true;
    }

    const size_t ts = ggml_fmsh_type_size(src1->type);
    if (ts == 0) {
        return false;
    }

    const int64_t ne0 = src0->ne[0], ne1 = src0->ne[1], ne2 = src0->ne[2], ne3 = src0->ne[3];
    const int64_t se0 = src1->ne[0], se1 = src1->ne[1], se2 = src1->ne[2], se3 = src1->ne[3];

    rhs_out.resize((size_t) ggml_nbytes(src0));
    const char * sbase = (const char *) src1->data;
    char * dbase = (char *) rhs_out.data();

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        const int64_t j3 = (se3 == 1 ? 0 : i3);
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            const int64_t j2 = (se2 == 1 ? 0 : i2);
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                const int64_t j1 = (se1 == 1 ? 0 : i1);
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const int64_t j0 = (se0 == 1 ? 0 : i0);

                    const size_t src_index = (size_t) (((j3 * se2 + j2) * se1 + j1) * se0 + j0);
                    const size_t dst_index = (size_t) (((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0);
                    std::memcpy(dbase + dst_index * ts, sbase + src_index * ts, ts);
                }
            }
        }
    }

    rhs_data_out = rhs_out.data();
    rhs_nbytes_out = (uint64_t) rhs_out.size();
    return true;
}

// iCraft runtime currently misses BuyiTarget TypeInfer registrations for a few ops.
// Register lightweight fallbacks locally so session creation can proceed.
static void ggml_fmsh_register_buyi_typeinfer_workarounds(void) {
    static std::once_flag once;
    std::call_once(once, []() {
        using namespace icraft::xir;

        const auto tinfer_identity_1in = [](const auto * op) {
            if (op == nullptr || op->inputs.empty()) {
                return Array<TensorType> {};
            }
            return Array<TensorType> { op->inputs[0].tensorType() };
        };

        ReflectionTable::Global()._register<MultiplyNode>()
            .set_tinfer<MultiplyNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<AddNode>()
            .set_tinfer<AddNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<MatmulNode>()
            .set_tinfer<MatmulNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<TransposeNode>()
            .set_tinfer<TransposeNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<AbsNode>()
            .set_tinfer<AbsNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<NegNode>()
            .set_tinfer<NegNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<TanhNode>()
            .set_tinfer<TanhNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<EluNode>()
            .set_tinfer<EluNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<ReluNode>()
            .set_tinfer<ReluNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<SigmoidNode>()
            .set_tinfer<SigmoidNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<GeluNode>()
            .set_tinfer<GeluNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<HardswishNode>()
            .set_tinfer<HardswishNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<HardsigmoidNode>()
            .set_tinfer<HardsigmoidNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<SqrtNode>()
            .set_tinfer<SqrtNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<SoftmaxNode>()
            .set_tinfer<SoftmaxNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<SiluNode>()
            .set_tinfer<SiluNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<MeanNode>()
            .set_tinfer<MeanNode, BuyiTarget>(tinfer_identity_1in);

        ReflectionTable::Global()._register<SumNode>()
            .set_tinfer<SumNode, BuyiTarget>(tinfer_identity_1in);
    });
}

static void ggml_fmsh_append_tensor_sig(std::string & key, const ggml_tensor * t) {
    if (t == nullptr) {
        key += "|null";
        return;
    }

    key += "|t=" + std::to_string((int) t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        key += ",n" + std::to_string(d) + "=" + std::to_string((long long) t->ne[d]);
    }
}

static std::string ggml_fmsh_hash_fnv1a_64(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= (uint64_t) c;
        h *= 1099511628211ULL;
    }

    std::ostringstream oss;
    oss << std::hex << h;
    return oss.str();
}

static std::string ggml_fmsh_cache_file_path(const ggml_backend_fmsh_context * ctx, const std::string & cache_id) {
    return (std::filesystem::path(ctx->cache_dir) / (cache_id + ".snapshot")).string();
}

static std::string ggml_fmsh_default_cache_dir(void) {
    const char * cache_dir_env = std::getenv("GGML_FMSH_CACHE_DIR");
    if (cache_dir_env != nullptr && *cache_dir_env != '\0') {
        return std::string(cache_dir_env);
    }

    const char * home = std::getenv("HOME");
    if (home != nullptr && *home != '\0') {
        return (std::filesystem::path(home) / ".cache" / "ggml-fmsh").string();
    }

    return ".ggml-fmsh-cache";
}

static bool ggml_fmsh_open_device(ggml_backend_fmsh_context * ctx) {
    if (ctx == nullptr || !ctx->enable_npu || !ctx->runtime_ready) {
        return false;
    }

    if (ctx->npu_device_open) {
        return true;
    }

    try {
        ctx->npu_device = icraft::xrt::Device::Open(ctx->device_url);
        ctx->npu_device_open = true;
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: open device failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: open device failed: unknown error\n");
        }
        return false;
    }
}

static void ggml_fmsh_preload_session_cache(ggml_backend_fmsh_context * ctx) {
    if (ctx == nullptr || !ctx->cache_enabled || !ctx->enable_npu) {
        return;
    }

    namespace fs = std::filesystem;
    try {
        fs::create_directories(ctx->cache_dir);
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: failed to create cache directory: %s\n", ctx->cache_dir.c_str());
        }
        return;
    }

    size_t loaded = 0;
    for (const auto & entry : fs::directory_iterator(ctx->cache_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path p = entry.path();
        if (p.extension() != ".snapshot") {
            continue;
        }

        const std::string cache_id = p.stem().string();

        try {
            icraft::xrt::Session sess = icraft::xrt::Session::CreateWithSnapshot(
                p,
                { icraft::xrt::BuyiBackend::Init(), icraft::xrt::HostBackend::Init() },
                { ctx->npu_device, icraft::xrt::HostDevice::Default() });

            std::lock_guard<std::mutex> lock(ctx->session_cache_mutex);
            ctx->session_cache[cache_id] = sess;
            ++loaded;
        } catch (const std::exception & e) {
            if (ctx->verbose) {
                GGML_LOG_WARN("ggml-fmsh: failed to load snapshot %s: %s\n", p.c_str(), e.what());
            }
        } catch (...) {
            if (ctx->verbose) {
                GGML_LOG_WARN("ggml-fmsh: failed to load snapshot %s: unknown error\n", p.c_str());
            }
        }
    }

    GGML_LOG_INFO("ggml-fmsh: preloaded %zu cached session(s) from %s\n", loaded, ctx->cache_dir.c_str());

}

static bool ggml_fmsh_get_or_create_session(
    ggml_backend_fmsh_context * ctx,
    const std::string & cache_key,
    const std::function<icraft::xrt::Session(void)> & creator,
    icraft::xrt::Session & sess_out) {

    const std::string cache_id = ggml_fmsh_hash_fnv1a_64(cache_key);

    {
        std::lock_guard<std::mutex> lock(ctx->session_cache_mutex);
        auto it = ctx->session_cache.find(cache_id);
        if (it != ctx->session_cache.end()) {
            sess_out = it->second;
            GGML_LOG_INFO("ggml-fmsh: successfully found session in cache for key %s \n", cache_key.c_str());
            return true;
        }
    }

    if (ctx->cache_enabled) {
        const std::string snapshot_path = ggml_fmsh_cache_file_path(ctx, cache_id);
        if (std::filesystem::exists(snapshot_path)) {
            try {
                icraft::xrt::Session sess = icraft::xrt::Session::CreateWithSnapshot(
                    snapshot_path,
                    { icraft::xrt::BuyiBackend::Init(), icraft::xrt::HostBackend::Init() },
                    { ctx->npu_device, icraft::xrt::HostDevice::Default() });

                {
                    std::lock_guard<std::mutex> lock(ctx->session_cache_mutex);
                    ctx->session_cache[cache_id] = sess;
                }
                sess_out = sess;

                GGML_LOG_INFO("ggml-fmsh: loaded session from cache %s\n", snapshot_path.c_str());
                return true;
            } catch (const std::exception & e) {
                // Fall through to compile if disk cache cannot be loaded.
                GGML_LOG_WARN("ggml-fmsh: failed to load session from cache %s: %s\n", snapshot_path.c_str(), e.what());
            } catch (...) {
                // Fall through to compile if disk cache cannot be loaded.
                GGML_LOG_WARN("ggml-fmsh: failed to load session from cache %s: unknown error\n", snapshot_path.c_str());
            }
        }
        else{
            GGML_LOG_INFO("ggml-fmsh: no cache found for session %s at %s\n", cache_key.c_str(), snapshot_path.c_str());
        }
    }

    icraft::xrt::Session sess;
    try {
        sess = creator();
    } catch (const std::exception & e) {
        GGML_LOG_WARN("ggml-fmsh: failed to create session for key %s: %s\n", cache_key.c_str(), e.what());
        return false;
    } catch (...) {
        GGML_LOG_WARN("ggml-fmsh: failed to create session for key %s: unknown error\n", cache_key.c_str());
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(ctx->session_cache_mutex);
        ctx->session_cache[cache_id] = sess;
    }

    if (ctx->cache_enabled) {
        const std::string snapshot_path = ggml_fmsh_cache_file_path(ctx, cache_id);
        try {
            std::filesystem::create_directories(std::filesystem::path(snapshot_path).parent_path());
            GGML_LOG_INFO("ggml-fmsh: saving session %s to cache %s\n", cache_id.c_str(), snapshot_path.c_str());
            sess.dumpSnapshot(snapshot_path);
        } catch (const std::exception & e) {
            GGML_LOG_WARN("ggml-fmsh: failed to save snapshot %s: %s\n", snapshot_path.c_str(), e.what());
        } catch (...) {
            GGML_LOG_WARN("ggml-fmsh: failed to save snapshot %s: unknown error\n", snapshot_path.c_str());
        }
    }

    sess_out = sess;
    return true;
}

static bool ggml_fmsh_tensor_compatible_binary(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        GGML_LOG_INFO("ggml-fmsh: %s:%d returning false\n", __func__, __LINE__);
        return false;
    }

    if (src0->type != src1->type || src0->type != dst->type) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const int64_t n0 = src0->ne[d];
        const int64_t n1 = src1->ne[d];
        if (!(n1 == 1 || n1 == n0)) {
            return false;
        }
    }

    if (ggml_nbytes(src1) == 0) {
        return false;
    }

    return ggml_fmsh_is_contiguous_binary(src0) && ggml_fmsh_is_contiguous_binary(src1) && ggml_fmsh_is_contiguous_binary(dst);
}

static bool ggml_fmsh_tensor_compatible_softmax(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || src1 != nullptr) {
        return false;
    }

    if (src0->type != dst->type) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    return ggml_is_contiguous(src0) && ggml_is_contiguous(dst);
}

static bool ggml_fmsh_tensor_compatible_unary(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || src1 != nullptr) {
        return false;
    }

    if (src0->type != dst->type) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }

    if (!ggml_are_same_shape(src0, dst)) {
        return false;
    }

    return ggml_is_contiguous(src0) && ggml_is_contiguous(dst);
}

static bool ggml_fmsh_tensor_compatible_mul_mat(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        return false;
    }

    // src0 can be quantized (weights); src1/dst are runtime activations.
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_BF16) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_BF16) {
        return false;
    }

    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    if (dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }

    return ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
}

static bool ggml_fmsh_tensor_compatible_sum(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || src1 != nullptr) {
        GGML_LOG_WARN("ggml-fmsh: sum NPU execution failed: incompatible tensor\n");
        return false;
    }

    if (src0->type != dst->type) {
        GGML_LOG_WARN("ggml-fmsh: sum NPU execution failed: incompatible tensor types\n");
        return false;
    }

    if (ggml_nelements(dst) != 1) {
        GGML_LOG_WARN("ggml-fmsh: sum NPU execution failed: destination tensor has more than one element\n");
        return false;
    }

    return ggml_is_contiguous(src0) && ggml_is_contiguous(dst);
}

static bool ggml_fmsh_tensor_compatible_mean(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || src1 != nullptr) {
        GGML_LOG_WARN("ggml-fmsh: mean NPU execution failed: incompatible tensor\n");
        return false;
    }

    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        GGML_LOG_WARN("ggml-fmsh: mean NPU execution failed: incompatible tensor types\n");
        return false;
    }

    if (dst->ne[0] != 1 || dst->ne[1] != src0->ne[1] || dst->ne[2] != src0->ne[2] || dst->ne[3] != src0->ne[3]) {
        GGML_LOG_WARN("ggml-fmsh: mean NPU execution failed: incompatible tensor dimensions\n");
        return false;
    }

    return ggml_is_contiguous(src0) && ggml_is_contiguous(dst);
}

enum ggml_fmsh_binop_kind {
    GGML_FMSH_BINOP_ADD,
    GGML_FMSH_BINOP_SUB,
    GGML_FMSH_BINOP_MUL,
};

static bool ggml_fmsh_supports_unary_npu(ggml_unary_op unary_op) {
    switch (unary_op) {
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_HARDSIGMOID:
            return true;
        default:
            return false;
    }
}

static bool ggml_fmsh_exec_binary_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst, ggml_fmsh_binop_kind kind) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: binary NPU execution skipped: NPU disabled or runtime not ready\n");
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        GGML_LOG_WARN("ggml-fmsh: binary NPU execution failed: unable to open device\n");
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_binary(dst)) {
        GGML_LOG_WARN("ggml-fmsh: binary NPU execution failed: incompatible tensor\n");
        return false;
    }

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);
        const bool rhs_needs_broadcast = !ggml_are_same_shape(src0, src1);
        xir::TensorType t1 = rhs_needs_broadcast ? ggml_fmsh_make_tensor_type(src0) : ggml_fmsh_make_tensor_type(src1);

        std::string cache_key = "binop|kind=" + std::to_string((int) kind);
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        cache_key += rhs_needs_broadcast ? "|rhs=broadcast" : "|rhs=direct";
        ggml_fmsh_append_tensor_sig(cache_key, rhs_needs_broadcast ? src0 : src1);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            const xir::Array<xir::TensorType> inputs_t { t0, t1 };
            xir::Input in(inputs_t);
            in.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid;
            if (kind == GGML_FMSH_BINOP_ADD) {
                op_mid = xir::Add(in[0], in[1]);
            } else if (kind == GGML_FMSH_BINOP_SUB) {
                op_mid = xir::Add(in[0], in[1], 1.0, -1.0);
            } else {
                op_mid = xir::Multiply(in[0], in[1]);
            }
            ggml_fmsh_set_cut_scale_if_present(op_mid);

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_binop";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        xrt::Tensor tx1(t1);
        tx0.mallocOn(xrt::HostDevice::MemRegion());
        tx1.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        std::vector<uint8_t> rhs_broadcast;
        const void * rhs_data = src1->data;
        uint64_t b1 = (uint64_t) ggml_nbytes(src1);
        if (!ggml_fmsh_prepare_binary_rhs_broadcast(src0, src1, rhs_broadcast, rhs_data, b1)) {
            return false;
        }

        tx0.write(0, (char *) src0->data, b0);
        tx1.write(0, const_cast<char *>(reinterpret_cast<const char *>(rhs_data)), b1);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0, tx1 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: binary NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: binary NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_tensor_compatible_transpose(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || src1 != nullptr) {
        return false;
    }

    if (src0->type != dst->type) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) {
        return false;
    }

    // ggml_transpose swaps dim0 <-> dim1 and keeps dim2/dim3.
    if (dst->ne[0] != src0->ne[1] || dst->ne[1] != src0->ne[0] || dst->ne[2] != src0->ne[2] || dst->ne[3] != src0->ne[3]) {
        return false;
    }

    return ggml_is_contiguous(src0) && ggml_is_contiguous(dst);
}
static bool ggml_fmsh_exec_transpose_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_transpose(dst)) {
        return false;
    }

    ggml_tensor * src0 = dst->src[0];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);

        std::string cache_key = "transpose|dims=0,1,3,2";
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            xir::Input in0(t0);
            in0.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid = xir::Transpose(in0[0], xir::Array<int64_t> { 0, 1, 3, 2 }, xir::Layout::NCHW());

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_transpose";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in0);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        tx0.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0->data, b0);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: transpose NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: transpose NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_exec_softmax_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: softmax NPU execution skipped: NPU disabled or runtime not ready\n");
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        GGML_LOG_WARN("ggml-fmsh: softmax NPU execution failed: unable to open device\n");  
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_softmax(dst)) {
        GGML_LOG_WARN("ggml-fmsh: softmax NPU execution failed: incompatible tensor\n");
        return false;
    }

    ggml_tensor * src0 = dst->src[0];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);

        std::string cache_key = "softmax|axis=-1";
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            xir::Input in0(t0);
            in0.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid = xir::Softmax(in0[0], -1);

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_softmax";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in0);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        tx0.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0->data, b0);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: softmax NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: softmax NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_exec_sqrt_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: sqrt NPU execution skipped: NPU disabled or runtime not ready\n");
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        GGML_LOG_WARN("ggml-fmsh: sqrt NPU execution failed: unable to open device\n");
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_unary(dst)) {
        GGML_LOG_WARN("ggml-fmsh: sqrt NPU execution failed: incompatible tensor\n");
        return false;
    }

    ggml_tensor * src0 = dst->src[0];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);

        std::string cache_key = "sqrt";
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            xir::Input in0(t0);
            in0.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid = xir::Sqrt(in0[0]);

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_sqrt";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in0);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        tx0.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0->data, b0);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: sqrt NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: sqrt NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_exec_unary_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst, ggml_unary_op unary_op) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_unary(dst)) {
        return false;
    }

    ggml_tensor * src0 = dst->src[0];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);

        std::string cache_key = "unary|u=" + std::to_string((int) unary_op);
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            xir::Input in0(t0);
            in0.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid;
            switch (unary_op) {
                case GGML_UNARY_OP_ABS:
                    op_mid = xir::Abs(in0[0]);
                    break;
                case GGML_UNARY_OP_NEG:
                    op_mid = xir::Neg(in0[0]);
                    break;
                case GGML_UNARY_OP_TANH:
                    op_mid = xir::Tanh(in0[0]);
                    break;
                case GGML_UNARY_OP_ELU:
                    op_mid = xir::Elu(in0[0]);
                    break;
                case GGML_UNARY_OP_RELU:
                    op_mid = xir::Relu(in0[0]);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    op_mid = xir::Sigmoid(in0[0]);
                    break;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_GELU_ERF:
                    op_mid = xir::Gelu(in0[0]);
                    break;
                case GGML_UNARY_OP_SILU:
                    {
                        // Use explicit Sigmoid + Multiply to provide Buyi cut_scale attr on multiply node.
                        xir::Operation op_sig = xir::Sigmoid(in0[0]);
                        xir::Operation op_mul = xir::Multiply(in0[0], op_sig[0]);
                        ggml_fmsh_set_cut_scale_if_present(op_mul);

                        xir::Output out = ggml_fmsh_make_output_from_dst(op_mul, dst);
                        out.setCompileTarget(xir::HostTarget::Init());
                        std::string net_name = "ggml_fmsh_unary";
                        xir::Network net = ggml_fmsh_make_network(net_name, ctx);
                        net.addOp(in0);
                        net.addOp(op_sig);
                        net.addOp(op_mul);
                        net.addOp(out);

                        xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                            net.view(),
                            { ctx->npu_device, xrt::HostDevice::Default() });
                        created.apply();
                        return created;
                    }
                case GGML_UNARY_OP_HARDSWISH:
                    op_mid = xir::Hardswish(in0[0]);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    op_mid = xir::Hardsigmoid(in0[0]);
                    break;
                default:
                    throw std::runtime_error("unsupported unary op");
            }


            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());
            std::string net_name = "ggml_fmsh_unary";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in0);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        tx0.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0->data, b0);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: unary NPU execution failed (%s): %s\n", ggml_unary_op_name(unary_op), e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: unary NPU execution failed (%s): unknown error\n", ggml_unary_op_name(unary_op));
        }
        return false;
    }
}

static bool ggml_fmsh_exec_mul_mat_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution skipped: NPU not enabled or runtime not ready\n");
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution skipped: failed to open NPU device\n");
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_mul_mat(dst)) {
        GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution skipped: incompatible tensor types\n");
        return false;
    }

    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    try {
        using namespace icraft;

        const ggml_type npu_compute_type = (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16) ? src1->type : GGML_TYPE_F16;

        std::vector<uint8_t> src0_conv;
        std::vector<uint8_t> src1_conv;
        if (!ggml_fmsh_tensor_convert_to(src0, npu_compute_type, src0_conv)) {
            GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution skipped: failed to convert src0 type %d\n", (int) src0->type);
            return false;
        }
        if (!ggml_fmsh_tensor_convert_to(src1, npu_compute_type, src1_conv)) {
            GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution skipped: failed to convert src1 type %d\n", (int) src1->type);
            return false;
        }

        xir::TensorType t0 = ggml_fmsh_make_tensor_type_with_ggml_type(src0, npu_compute_type);
        xir::TensorType t1 = ggml_fmsh_make_tensor_type_with_ggml_type(src1, npu_compute_type);

        std::string cache_key = "mul_mat|ct=" + std::to_string((int) npu_compute_type);
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, src1);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            const xir::Array<xir::TensorType> inputs_t { t0, t1 };
            xir::Input in(inputs_t);
            in.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid = xir::Matmul(in[0], in[1], xir::NullOpt);
            ggml_fmsh_set_cut_scale_if_present(op_mid);

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_mul_mat";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        xrt::Tensor tx1(t1);
        tx0.mallocOn(xrt::HostDevice::MemRegion());
        tx1.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) src0_conv.size();
        const uint64_t b1 = (uint64_t) src1_conv.size();
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0_conv.data(), b0);
        tx1.write(0, (char *) src1_conv.data(), b1);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0, tx1 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: mul_mat NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_exec_sum_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: sum NPU execution skipped: NPU not enabled or runtime not ready\n");
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        GGML_LOG_WARN("ggml-fmsh: sum NPU execution skipped: failed to open NPU device\n");
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_sum(dst)) {
        GGML_LOG_WARN("ggml-fmsh: sum NPU execution skipped: incompatible tensor types\n");
        return false;
    }

    ggml_tensor * src0 = dst->src[0];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);

        std::string cache_key = "sum";
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            xir::Input in0(t0);
            in0.setCompileTarget(xir::HostTarget::Init());

            xir::Operation op_mid = xir::Sum(in0[0]);

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_sum";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in0);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        tx0.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0->data, b0);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: sum NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: sum NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_exec_mean_npu(ggml_backend_fmsh_context * ctx, ggml_tensor * dst) {
    if (!ctx->enable_npu || !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: mean NPU execution skipped: NPU not enabled or runtime not ready\n");
        return false;
    }

    if (!ggml_fmsh_open_device(ctx)) {
        GGML_LOG_WARN("ggml-fmsh: mean NPU execution skipped: failed to open NPU device\n");
        return false;
    }

    if (!ggml_fmsh_tensor_compatible_mean(dst)) {
        GGML_LOG_WARN("ggml-fmsh: mean NPU execution skipped: incompatible tensor types\n");
        return false;
    }

    ggml_tensor * src0 = dst->src[0];

    try {
        using namespace icraft;

        xir::TensorType t0 = ggml_fmsh_make_tensor_type(src0);

        std::string cache_key = "mean|axis=3|keep=1";
        ggml_fmsh_append_tensor_sig(cache_key, src0);
        ggml_fmsh_append_tensor_sig(cache_key, dst);

        xrt::Session sess;
        if (!ggml_fmsh_get_or_create_session(ctx, cache_key, [&]() {
            xir::Input in0(t0);
            in0.setCompileTarget(xir::HostTarget::Init());

            // ggml_mean reduces dim0 and keeps reduced dimension.
            xir::Operation op_mid = xir::Mean(in0[0], 3, true);

            xir::Output out = ggml_fmsh_make_output_from_dst(op_mid, dst);
            out.setCompileTarget(xir::HostTarget::Init());

            std::string net_name = "ggml_fmsh_mean";
            xir::Network net = ggml_fmsh_make_network(net_name, ctx);
            net.addOp(in0);
            net.addOp(op_mid);
            net.addOp(out);

            xrt::Session created = xrt::Session::Create<xrt::BuyiBackend, xrt::HostBackend>(
                net.view(),
                { ctx->npu_device, xrt::HostDevice::Default() });
            created.apply();
            return created;
        }, sess)) {
            return false;
        }

        xrt::Tensor tx0(t0);
        tx0.mallocOn(xrt::HostDevice::MemRegion());

        const uint64_t b0 = (uint64_t) ggml_nbytes(src0);
        const uint64_t bd = (uint64_t) ggml_nbytes(dst);

        tx0.write(0, (char *) src0->data, b0);

        std::vector<xrt::Tensor> outputs = sess.forward({ tx0 });
        if (outputs.empty()) {
            return false;
        }

        outputs[0].read((char *) dst->data, 0, bd);
        return true;
    } catch (const std::exception & e) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: mean NPU execution failed: %s\n", e.what());
        }
        return false;
    } catch (...) {
        if (ctx->verbose) {
            GGML_LOG_WARN("ggml-fmsh: mean NPU execution failed: unknown error\n");
        }
        return false;
    }
}

static bool ggml_fmsh_supports_llm_op(const struct ggml_tensor * op) {
    // Only advertise ops that currently have a concrete NPU execution path.
    // Other ops are routed to CPU backend by scheduler partitioning.
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_SQRT:
        case GGML_OP_SUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_UNARY:
            return ggml_fmsh_supports_unary_npu(ggml_get_unary_op(op));

        default:
            return false;
    }
}

// backend interface

static const char * ggml_backend_fmsh_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FMSH";
}

static void ggml_backend_fmsh_free(ggml_backend_t backend) {
    ggml_backend_fmsh_context * ctx = (ggml_backend_fmsh_context *) backend->context;
    if (ctx != nullptr && ctx->npu_device_open) {
        try {
            icraft::xrt::Device::Close(ctx->npu_device);
        } catch (...) {
        }
        ctx->npu_device_open = false;
    }

    if (ctx != nullptr && ctx->cpu_backend != nullptr) {
        ggml_backend_free(ctx->cpu_backend);
        ctx->cpu_backend = nullptr;
    }

    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_fmsh_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_fmsh_context * ctx = (ggml_backend_fmsh_context *) backend->context;
    if (ctx == nullptr || ctx->cpu_backend == nullptr) {
        GGML_LOG_ERROR("%s: invalid FMSH backend context\n", __func__);
        return GGML_STATUS_FAILED;
    }

    bool used_cpu_fallback = false;
    std::vector<ggml_tensor *> offloaded_nodes;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        const enum ggml_op op = node->op;
        const bool supported = ggml_fmsh_supports_llm_op(node);

        if (!supported) {
            used_cpu_fallback = true;
            if (op >= 0 && op < GGML_OP_COUNT && !ctx->warned_unsupported[op]) {
                GGML_LOG_WARN("ggml-fmsh: op %s is not supported by FMSH NPU, falling back to CPU\n", ggml_op_name(op));
                ctx->warned_unsupported[op] = true;
            }
            continue;
        }

        GGML_LOG_INFO("ggml-fmsh: attempting to offload op %s to NPU, %d / %d \n", ggml_op_name(op), i+1, cgraph->n_nodes);

        bool offloaded = false;
        switch (op) {
            case GGML_OP_MUL_MAT:
                offloaded = ggml_fmsh_exec_mul_mat_npu(ctx, node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_ADD1:
                offloaded = ggml_fmsh_exec_binary_npu(ctx, node, GGML_FMSH_BINOP_ADD);
                break;
            case GGML_OP_SUB:
                offloaded = ggml_fmsh_exec_binary_npu(ctx, node, GGML_FMSH_BINOP_SUB);
                break;
            case GGML_OP_MUL:
                offloaded = ggml_fmsh_exec_binary_npu(ctx, node, GGML_FMSH_BINOP_MUL);
                break;
            case GGML_OP_SQRT:
                offloaded = ggml_fmsh_exec_sqrt_npu(ctx, node);
                break;
            case GGML_OP_SUM:
                offloaded = ggml_fmsh_exec_sum_npu(ctx, node);
                break;
            case GGML_OP_MEAN:
                offloaded = ggml_fmsh_exec_mean_npu(ctx, node);
                break;
            case GGML_OP_SOFT_MAX:
                offloaded = ggml_fmsh_exec_softmax_npu(ctx, node);
                break;
            case GGML_OP_TRANSPOSE:
                offloaded = ggml_fmsh_exec_transpose_npu(ctx, node);
                break;
            case GGML_OP_UNARY:
                offloaded = ggml_fmsh_exec_unary_npu(ctx, node, ggml_get_unary_op(node));
                break;
            default:
                GGML_LOG_WARN("ggml-fmsh: op %s is not supported by FMSH NPU\n", ggml_op_name(op));
                break;
        }

        if (offloaded) {
            node->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            offloaded_nodes.push_back(node);
            continue;
        }

        used_cpu_fallback = true;
        if (op >= 0 && op < GGML_OP_COUNT && !ctx->warned_npu_unavailable[op]) {
            GGML_LOG_WARN("ggml-fmsh: NPU execution for op %s is unavailable, falling back to CPU\n", ggml_op_name(op));
            ctx->warned_npu_unavailable[op] = true;
        }
    }

    if (used_cpu_fallback && !ctx->warned_fallback) {
        GGML_LOG_WARN("ggml-fmsh: CPU fallback is in use for part of the graph\n");
        ctx->warned_fallback = true;
    }

    const enum ggml_status st = ggml_backend_graph_compute(ctx->cpu_backend, cgraph);
    for (ggml_tensor * node : offloaded_nodes) {
        node->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }

    return st;
}

static struct ggml_backend_i ggml_backend_fmsh_i = {
    /* .get_name                = */ ggml_backend_fmsh_get_name,
    /* .free                    = */ ggml_backend_fmsh_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_fmsh_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_fmsh_guid(void) {
    static ggml_guid guid = { 0x77, 0xda, 0x66, 0x9e, 0x58, 0x4e, 0x40, 0xe0, 0x98, 0xa5, 0x5d, 0x5b, 0x70, 0xf2, 0x32, 0x1c };
    return &guid;
}

ggml_backend_t ggml_backend_fmsh_init(size_t device_index) {
    GGML_ASSERT(device_index == 0);

    ggml_fmsh_register_buyi_typeinfer_workarounds();

    ggml_fmsh_resolve_cpu_symbols();
    if (s_cpu_init == nullptr) {
        GGML_LOG_ERROR("ggml-fmsh: failed to resolve ggml CPU backend symbols (libggml-cpu.so not loadable?)\n");
        return nullptr;
    }

    ggml_backend_fmsh_context * ctx = new ggml_backend_fmsh_context;
    ctx->cpu_backend = s_cpu_init();
    if (ctx->cpu_backend == nullptr) {
        delete ctx;
        return nullptr;
    }

    ctx->verbose = ggml_fmsh_parse_env_bool("GGML_FMSH_VERBOSE", false);
    ctx->cache_enabled = ggml_fmsh_parse_env_bool("GGML_FMSH_CACHE", true);
    ctx->cache_dir = ggml_fmsh_default_cache_dir();

    const bool iCraft_in_build = true;
    ctx->device_url = ggml_fmsh_get_device_url();
    ctx->runtime_ready = ggml_fmsh_probe_runtime(ctx->verbose);

    const bool force_cpu = ggml_fmsh_parse_env_bool("GGML_FMSH_FORCE_CPU", false);
    const bool request_npu = ggml_fmsh_parse_env_bool("GGML_FMSH_ENABLE_NPU", true);
    ctx->enable_npu = iCraft_in_build && !force_cpu && request_npu && ctx->runtime_ready;

    if (request_npu && !ctx->runtime_ready) {
        GGML_LOG_WARN("ggml-fmsh: iCraft runtime/device is not ready, CPU fallback will be used\n");
    }

    if (ctx->enable_npu && ggml_fmsh_open_device(ctx)) {
        ggml_fmsh_preload_session_cache(ctx);
    }

    GGML_LOG_INFO("ggml-fmsh: init device=%zu, iCraft_build=%d, runtime_ready=%d, enable_npu=%d, cache=%d\n",
                    device_index,
                    (int) iCraft_in_build,
                    (int) ctx->runtime_ready,
                    (int) ctx->enable_npu,
                    (int) ctx->cache_enabled);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_fmsh_guid(),
        /* .iface   = */ ggml_backend_fmsh_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_fmsh_reg(), device_index),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_fmsh(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_fmsh_guid());
}

static void ggml_backend_fmsh_set_n_threads(ggml_backend_t backend_fmsh, int n_threads) {
    GGML_ASSERT(ggml_backend_is_fmsh(backend_fmsh));

    ggml_backend_fmsh_context * ctx = (ggml_backend_fmsh_context *) backend_fmsh->context;
    ggml_fmsh_resolve_cpu_symbols();
    if (ctx != nullptr && ctx->cpu_backend != nullptr && s_cpu_set_n_threads != nullptr) {
        s_cpu_set_n_threads(ctx->cpu_backend, n_threads);
    }
}

// device interface

static const char * ggml_backend_fmsh_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH";
}

static const char * ggml_backend_fmsh_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH ARM + Buyi NPU (iCraft runtime, CPU fallback enabled)";
}

static void ggml_backend_fmsh_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    // Memory is currently provisioned via host buffer path.
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_fmsh_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_fmsh_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_fmsh_device_get_name(dev);
    props->description = ggml_backend_fmsh_device_get_description(dev);
    props->type        = ggml_backend_fmsh_device_get_type(dev);
    ggml_backend_fmsh_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_fmsh_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_fmsh_init(0);
}

static ggml_backend_buffer_type_t ggml_backend_fmsh_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    ggml_fmsh_resolve_cpu_symbols();
    return s_cpu_buffer_type != nullptr ? s_cpu_buffer_type() : nullptr;
}

static ggml_backend_buffer_t ggml_backend_fmsh_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    ggml_fmsh_resolve_cpu_symbols();
    return s_cpu_buffer_from_ptr != nullptr ? s_cpu_buffer_from_ptr(ptr, size) : nullptr;
}

static bool ggml_backend_fmsh_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (op == nullptr) {
        return false;
    }

    // Conservative dtype gate for current MVP.
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) {
        return false;
    }

    return ggml_fmsh_supports_llm_op(op);
}

static bool ggml_backend_fmsh_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_fmsh_device_i = {
    /* .get_name             = */ ggml_backend_fmsh_device_get_name,
    /* .get_description      = */ ggml_backend_fmsh_device_get_description,
    /* .get_memory           = */ ggml_backend_fmsh_device_get_memory,
    /* .get_type             = */ ggml_backend_fmsh_device_get_type,
    /* .get_props            = */ ggml_backend_fmsh_device_get_props,
    /* .init_backend         = */ ggml_backend_fmsh_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_fmsh_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_fmsh_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_fmsh_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fmsh_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// backend reg interface

static const char * ggml_backend_fmsh_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FMSH";
}

static size_t ggml_backend_fmsh_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_fmsh_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_fmsh_device_context dev_ctx {
        /* .iCraft_enabled = */ true,
    };

    static ggml_backend_device ggml_backend_fmsh_device = {
        /* .iface   = */ ggml_backend_fmsh_device_i,
        /* .reg     = */ reg,
        /* .context = */ &dev_ctx,
    };

    return &ggml_backend_fmsh_device;
}

static void * ggml_backend_fmsh_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_fmsh_set_n_threads;
    }

    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_fmsh_reg_i = {
    /* .get_name         = */ ggml_backend_fmsh_reg_get_name,
    /* .get_device_count = */ ggml_backend_fmsh_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fmsh_reg_get_device,
    /* .get_proc_address = */ ggml_backend_fmsh_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_fmsh_reg(void) {
    static struct ggml_backend_reg ggml_backend_fmsh_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fmsh_reg_i,
        /* .context     = */ nullptr,
    };

    return &ggml_backend_fmsh_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fmsh_reg)
