#include "ggml-fmsh-zg330.h"
#include "fmsh-zg330-netmake.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xir/core/network.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using namespace icraft::xrt;
using namespace icraft::xir;

namespace {

struct fmsh_zg330_compiled_session {
    std::string key;
    Network network;
    Session session;
};

struct ggml_backend_fmsh_zg330_context {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<fmsh_zg330_compiled_session>> compiled;
    uint64_t cache_hit_count = 0;
    uint64_t cache_miss_count = 0;
    Device device;
    bool device_open = false;
    std::string device_url;
    fs::path cache_root;
};

static bool fmsh_log_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * env_cstr = std::getenv("GGML_FMSH_ZG330_LOG");
        const std::string env = env_cstr != nullptr ? env_cstr : "";
        cached = (!env.empty() && env != "0" && env != "false" && env != "FALSE") ? 1 : 0;
    }
    return cached != 0;
}

static std::string fmsh_getenv_str(const char * name, const std::string & fallback = {}) {
    const char * value = std::getenv(name);
    return value != nullptr ? value : fallback;
}

static fs::path fmsh_cache_root(void) {
    const std::string env = fmsh_getenv_str("GGML_FMSH_ZG330_CACHE_DIR");
    if (!env.empty()) {
        return fs::path(env);
    }
    return fs::current_path() / ".cache" / "ggml-fmsh_zg330";
}

static std::string fmsh_default_url(void) {
    const std::string explicit_url = fmsh_getenv_str("GGML_FMSH_ZG330_URL");
    if (!explicit_url.empty()) {
        return explicit_url;
    }

#if defined(FMSH_ZG330_USE_AXI)
    return "axi://zg330aiu?npu=0x40000000&dma=0x80000000";
#elif defined(FMSH_ZG330_USE_SOCKET)
    {
        const std::string ip = fmsh_getenv_str("GGML_FMSH_ZG330_IP", "192.168.110.157");
        const std::string port = fmsh_getenv_str("GGML_FMSH_ZG330_PORT", "9981");
        return "socket://zg330aiu@" + ip + ":" + port;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "axi://zg330aiu?npu=0x40000000&dma=0x80000000";
#endif

    const std::string ip = fmsh_getenv_str("GGML_FMSH_ZG330_IP", "192.168.110.157");
    const std::string port = fmsh_getenv_str("GGML_FMSH_ZG330_PORT", "9981");
    return "socket://zg330aiu@" + ip + ":" + port;
}

static std::vector<float> fmsh_tensor_to_float_2d(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->ne[2] == 1 && tensor->ne[3] == 1);

    const int64_t cols = tensor->ne[0];
    const int64_t rows = tensor->ne[1];

    std::vector<float> out((size_t) cols * (size_t) rows);

    if (tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor)) {
        std::memcpy(out.data(), tensor->data, out.size() * sizeof(float));
        return out;
    }

    if (tensor->type == GGML_TYPE_F32) {
        for (int64_t row = 0; row < rows; ++row) {
            const char * src = (const char *) tensor->data + row * tensor->nb[1];
            std::memcpy(out.data() + row * cols, src, (size_t) cols * sizeof(float));
        }
        return out;
    }

    const ggml_to_float_t to_float = ggml_get_type_traits(tensor->type)->to_float;
    GGML_ASSERT(to_float != nullptr);

    for (int64_t row = 0; row < rows; ++row) {
        const char * src = (const char *) tensor->data + row * tensor->nb[1];
        to_float(src, out.data() + row * cols, cols);
    }

    return out;
}

static std::vector<float> fmsh_transpose_row_major(const std::vector<float> & src, int64_t rows, int64_t cols) {
    std::vector<float> dst((size_t) rows * (size_t) cols);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            dst[(size_t) col * (size_t) rows + (size_t) row] = src[(size_t) row * (size_t) cols + (size_t) col];
        }
    }
    return dst;
}

static std::vector<ggml_bf16_t> fmsh_transpose_row_major_to_bf16(const std::vector<float> & src, int64_t rows, int64_t cols) {
    std::vector<ggml_bf16_t> dst((size_t) rows * (size_t) cols);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            dst[(size_t) col * (size_t) rows + (size_t) row] = ggml_fp32_to_bf16(src[(size_t) row * (size_t) cols + (size_t) col]);
        }
    }
    return dst;
}

static std::string fmsh_make_key(int64_t k, int64_t n) {
    std::ostringstream oss;
    oss << "matmul_" << k << "x" << n << "_bf16";
    return oss.str();
}

static Device & fmsh_open_device(ggml_backend_fmsh_zg330_context * ctx) {
    if (!ctx->device_open) {
        ctx->device_url = fmsh_default_url();
        if (fmsh_log_enabled()) {
            GGML_LOG_INFO("\rfmsh_zg330: open device: %s\n", ctx->device_url.c_str());
        }
        ctx->device = Device::Open(ctx->device_url);
        ctx->device_open = true;
    }
    return ctx->device;
}

static std::shared_ptr<fmsh_zg330_compiled_session> fmsh_get_or_create_session(
        ggml_backend_fmsh_zg330_context * ctx,
        int64_t k,
        int64_t n) {
    const std::string key = fmsh_make_key(k, n);

    auto it = ctx->compiled.find(key);
    if (it != ctx->compiled.end()) {
        ctx->cache_hit_count++;
        if (fmsh_log_enabled()) {
            GGML_LOG_INFO("\rfmsh_zg330: session cache hit #%llu | key=%s | k=%lld n=%lld | map_size=%zu\n",
                          (unsigned long long) ctx->cache_hit_count,
                          key.c_str(),
                          (long long) k, (long long) n,
                          ctx->compiled.size());
        }
        return it->second;
    }

    ctx->cache_miss_count++;
    auto compiled = std::make_shared<fmsh_zg330_compiled_session>();
    compiled->key = key;
    if (fmsh_log_enabled()) {
        GGML_LOG_INFO("\rfmsh_zg330: session cache miss #%llu | key=%s | k=%lld n=%lld | map_size=%zu | fallback=disk-cache-or-compile\n",
                      (unsigned long long) ctx->cache_miss_count,
                      key.c_str(),
                      (long long) k, (long long) n,
                      ctx->compiled.size());
    }
    compiled->network = fmsh_zg330_compile_network(ctx->cache_root, key, k, n);
    compiled->session = Session::Create<zg330::ZG330Backend, HostBackend>(
            compiled->network.view(0),
            { fmsh_open_device(ctx), HostDevice::Default() });

    auto zg_backend = compiled->session->backends[0].cast<zg330::ZG330Backend>();
    zg_backend.ocmOptimize(zg330::OcmOpt::BEST_SCORE);
    compiled->session.apply();

    ctx->compiled[key] = compiled;
    return compiled;
}

static std::vector<float> fmsh_tensor_slice_to_float_2d(const ggml_tensor * tensor, int64_t i2, int64_t i3) {
    GGML_ASSERT(i2 >= 0 && i3 >= 0);
    GGML_ASSERT(i2 < tensor->ne[2] && i3 < tensor->ne[3]);

    const int64_t cols = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    std::vector<float> out((size_t) cols * (size_t) rows);

    const char * base = (const char *) tensor->data + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    if (tensor->type == GGML_TYPE_F32) {
        for (int64_t row = 0; row < rows; ++row) {
            const char * src = base + row * tensor->nb[1];
            std::memcpy(out.data() + row * cols, src, (size_t) cols * sizeof(float));
        }
        return out;
    }

    const ggml_to_float_t to_float = ggml_get_type_traits(tensor->type)->to_float;
    GGML_ASSERT(to_float != nullptr);

    for (int64_t row = 0; row < rows; ++row) {
        const char * src = base + row * tensor->nb[1];
        to_float(src, out.data() + row * cols, cols);
    }

    return out;
}

static void fmsh_tensor_slice_from_float_2d(ggml_tensor * tensor, int64_t i2, int64_t i3, const float * src) {
    GGML_ASSERT(i2 >= 0 && i3 >= 0);
    GGML_ASSERT(i2 < tensor->ne[2] && i3 < tensor->ne[3]);

    const int64_t cols = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    char * base = (char *) tensor->data + i2 * tensor->nb[2] + i3 * tensor->nb[3];

    if (tensor->type == GGML_TYPE_F32) {
        for (int64_t row = 0; row < rows; ++row) {
            char * dst = base + row * tensor->nb[1];
            std::memcpy(dst, src + row * cols, (size_t) cols * sizeof(float));
        }
        return;
    }

    const ggml_from_float_t from_float = ggml_get_type_traits(tensor->type)->from_float_ref;
    GGML_ASSERT(from_float != nullptr);
    for (int64_t row = 0; row < rows; ++row) {
        char * dst = base + row * tensor->nb[1];
        from_float(src + row * cols, dst, cols);
    }
}

static void fmsh_run_mul_mat(ggml_backend_fmsh_zg330_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(dst->ne[0] == src0->ne[1]);
    GGML_ASSERT(dst->ne[1] == src1->ne[1]);

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int64_t batches2 = dst->ne[2];
    const int64_t batches3 = dst->ne[3];

    std::vector<float> weight_nk = fmsh_tensor_to_float_2d(src0);
    auto compiled = fmsh_get_or_create_session(ctx, k, n);

    if (compiled->network.inputs().size() < 2) {
        throw std::runtime_error("fmsh_zg330: compiled matmul network must have two inputs (A, W)");
    }

    if (fmsh_log_enabled()) {
        GGML_LOG_INFO("\rfmsh_zg330: run mul_mat: src0=[%lld,%lld,1,1] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld] dst_type=%s\n",
                      (long long) src0->ne[0], (long long) src0->ne[1],
                      (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
                      (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
                      ggml_type_name(dst->type));
    }

    Tensor weight_tensor(compiled->network.inputs()[1].tensorType().clone());
    weight_tensor.mallocOn(HostDevice::MemRegion());
    const size_t weight_elems = (size_t) k * (size_t) n;
    const size_t weight_bytes = (size_t) compiled->network.inputs()[1].tensorType().bytes();
    if (weight_bytes == weight_elems * sizeof(ggml_bf16_t)) {
        std::vector<ggml_bf16_t> weight_kxn_bf16 = fmsh_transpose_row_major_to_bf16(weight_nk, n, k);
        std::memcpy(weight_tensor.data().cptr(), weight_kxn_bf16.data(), weight_kxn_bf16.size() * sizeof(ggml_bf16_t));
    } else if (weight_bytes == weight_elems * sizeof(float)) {
        std::vector<float> weight_kxn = fmsh_transpose_row_major(weight_nk, n, k);
        std::memcpy(weight_tensor.data().cptr(), weight_kxn.data(), weight_kxn.size() * sizeof(float));
    } else {
        throw std::runtime_error("fmsh_zg330: unsupported weight input tensor byte size");
    }

    const size_t input_bytes = (size_t) compiled->network.inputs()[0].tensorType().bytes();
    const size_t row_input_bytes = input_bytes;
    const size_t row_output_bytes = (size_t) n * sizeof(float);

    for (int64_t i3 = 0; i3 < batches3; ++i3) {
        const int64_t src1_i3 = src1->ne[3] == 1 ? 0 : i3;
        for (int64_t i2 = 0; i2 < batches2; ++i2) {
            const int64_t src1_i2 = src1->ne[2] == 1 ? 0 : i2;

            std::vector<float> input_mk = fmsh_tensor_slice_to_float_2d(src1, src1_i2, src1_i3);
            std::vector<float> output_mn((size_t) m * (size_t) n);
            std::vector<ggml_bf16_t> input_row_bf16((size_t) k);
            for (int64_t row = 0; row < m; ++row) {
                Tensor input_tensor(compiled->network.inputs()[0].tensorType().clone());
                input_tensor.mallocOn(HostDevice::MemRegion());
                const float * row_src = input_mk.data() + (size_t) row * (size_t) k;
                if (row_input_bytes == (size_t) k * sizeof(ggml_bf16_t)) {
                    ggml_fp32_to_bf16_row(row_src, input_row_bf16.data(), k);
                    std::memcpy(input_tensor.data().cptr(), input_row_bf16.data(), row_input_bytes);
                } else if (row_input_bytes == (size_t) k * sizeof(float)) {
                    std::memcpy(input_tensor.data().cptr(), row_src, row_input_bytes);
                } else {
                    throw std::runtime_error("fmsh_zg330: unsupported activation input tensor byte size");
                }

                auto outputs = compiled->session.forward({ input_tensor, weight_tensor });
                if (outputs.empty()) {
                    throw std::runtime_error("fmsh_zg330: session.forward returned no outputs");
                }

                outputs[0].read((char *) (output_mn.data() + (size_t) row * (size_t) n), 0, row_output_bytes);
            }
            fmsh_tensor_slice_from_float_2d(dst, i2, i3, output_mn.data());
        }
    }
}

static const char * ggml_backend_fmsh_zg330_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FMSH_ZG330";
}

static void ggml_backend_fmsh_zg330_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_fmsh_zg330_context *) backend->context;
    if (ctx != nullptr && ctx->device_open) {
        Device::Close(ctx->device);
    }
    delete ctx;
    delete backend;
}

static ggml_status ggml_backend_fmsh_zg330_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_fmsh_zg330_context *) backend->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    int mul_mat_count = 0;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ++mul_mat_count;
                fmsh_run_mul_mat(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    if (fmsh_log_enabled()) {
        GGML_LOG_INFO("\rfmsh_zg330: graph_compute finished, executed MUL_MAT nodes: %d\n", mul_mat_count);
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_fmsh_zg330_i = {
    /* .get_name                = */ ggml_backend_fmsh_zg330_get_name,
    /* .free                    = */ ggml_backend_fmsh_zg330_free,
    /* .set_tensor_async        = */ nullptr,
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
    static ggml_guid guid = { 0x65, 0x27, 0x90, 0x43, 0x40, 0x07, 0x4e, 0x5b, 0xa2, 0x0f, 0xe7, 0xca, 0x08, 0x4e, 0x52, 0x11 };
    return &guid;
}

} // namespace

ggml_backend_t ggml_backend_fmsh_zg330_init(void) {
    auto * ctx = new ggml_backend_fmsh_zg330_context;
    ctx->cache_root = fmsh_cache_root();

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_fmsh_zg330_guid(),
        /* .iface   = */ ggml_backend_fmsh_zg330_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_fmsh_zg330_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_fmsh_zg330(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_fmsh_zg330_guid());
}

static const char * ggml_backend_fmsh_zg330_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH_ZG330";
}

static const char * ggml_backend_fmsh_zg330_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH ZG330 via Icraft";
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
    return ggml_backend_fmsh_zg330_init();
}

static ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_fmsh_zg330_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_fmsh_zg330_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    static std::atomic<int> logged_rejects { 0 };
    auto reject = [&](const char * why) {
        if (fmsh_log_enabled()) {
            const int idx = logged_rejects.fetch_add(1);
            if (idx < 64) {
                GGML_LOG_INFO("\rfmsh_zg330: reject op %s: %s | src0=%s src1=%s dst=%s | dst=[%lld,%lld,%lld,%lld]\n",
                              ggml_op_desc(op),
                              why,
                              op->src[0] ? ggml_type_name(op->src[0]->type) : "null",
                              op->src[1] ? ggml_type_name(op->src[1]->type) : "null",
                              ggml_type_name(op->type),
                              (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3]);
            }
        }
        return false;
    };

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT:
            break;
        default:
            return reject("unsupported op");
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return reject("missing inputs");
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return reject("src0 with batched dims is not supported");
    }

    if (src0->ne[0] != src1->ne[0]) {
        return reject("incompatible K dimension");
    }

    if (op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) {
        return reject("unexpected output shape");
    }

    if (op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3]) {
        return reject("src1/dst batch dims mismatch");
    }

    if (src0->type != GGML_TYPE_F32 && ggml_get_type_traits(src0->type)->to_float == nullptr) {
        return reject("src0 cannot convert to float");
    }

    if (src1->type != GGML_TYPE_F32 && ggml_get_type_traits(src1->type)->to_float == nullptr) {
        return reject("src1 cannot convert to float");
    }

    if (ggml_is_quantized(op->type)) {
        return reject("quantized dst is not supported");
    }

    if (op->type != GGML_TYPE_F32 && ggml_get_type_traits(op->type)->from_float_ref == nullptr) {
        return reject("dst cannot convert from float");
    }

    if (fmsh_log_enabled()) {
        GGML_LOG_INFO("\rfmsh_zg330: accept MUL_MAT | src0=%s src1=%s dst=%s | src0=[%lld,%lld,%lld,%lld] src1=[%lld,%lld,%lld,%lld] dst=[%lld,%lld,%lld,%lld]\n",
                      ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(op->type),
                      (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
                      (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
                      (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3]);
    }

    return true;
}

static bool ggml_backend_fmsh_zg330_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static ggml_backend_device_i ggml_backend_fmsh_zg330_device_i = {
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

static ggml_backend_reg_i ggml_backend_fmsh_zg330_reg_i = {
    /* .get_name         = */ ggml_backend_fmsh_zg330_reg_get_name,
    /* .get_device_count = */ ggml_backend_fmsh_zg330_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fmsh_zg330_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_fmsh_zg330_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fmsh_zg330_reg_i,
        /* .context     = */ nullptr,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fmsh_zg330_reg)
