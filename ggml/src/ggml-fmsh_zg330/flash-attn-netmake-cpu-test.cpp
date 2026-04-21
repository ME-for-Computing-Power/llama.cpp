#include "ggml-fmsh-zg330-netmake.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/zg330backend/zg330backend.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/zg330_device.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace icraft::xrt;
namespace zg330 = icraft::xrt::zg330;

constexpr float kMaskNegInf = -1e9f;

struct Config {
    int64_t head_dim = 128;
    int64_t value_dim = 128;
    int64_t q_len = 97;
    int64_t kv_len = 193;
    int64_t softmax_cu = 8;

    int iters = 5;
    int seed = 20260420;

    bool use_mask = true;
    bool causal_mask = true;

    ggml_type kv_type = GGML_TYPE_F16;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;

    int n_threads = 1;
    float input_scale = 1.0f;

    std::filesystem::path cache_dir = ".cache/deploy/flash_attn_standalone";
    std::string device_url;
};

struct Inputs {
    std::vector<float> q;    // [q_len, head_dim]
    std::vector<float> k;    // [kv_len, head_dim]
    std::vector<float> v;    // [kv_len, value_dim]
    std::vector<float> mask; // [q_len, kv_len]
};

struct Stats {
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    size_t inf_cnt = 0;
    size_t nan_cnt = 0;
};

struct DiffStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double nmse = 0.0;
    size_t both_finite_cnt = 0;
    size_t finite_mismatch_cnt = 0;
    size_t cpu_inf_or_nan = 0;
    size_t npu_inf_or_nan = 0;
};

static int64_t next_pow2_i64(int64_t v) {
    if (v <= 1) {
        return 1;
    }
    int64_t p = 1;
    while (p < v && p < (1LL << 30)) {
        p <<= 1;
    }
    return p < v ? v : p;
}

static int64_t align_up_i64(int64_t v, int64_t a) {
    if (a <= 1) {
        return v;
    }
    const int64_t r = v % a;
    return r == 0 ? v : (v + (a - r));
}

static std::string default_device_url() {
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

static void print_usage(const char * argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --head-dim N\n"
        << "  --value-dim N\n"
        << "  --q-len N\n"
        << "  --kv-len N\n"
        << "  --softmax-cu N\n"
        << "  --iters N\n"
        << "  --seed N\n"
        << "  --threads N\n"
        << "  --input-scale F\n"
        << "  --cache-dir PATH\n"
        << "  --device-url URL\n"
        << "  --kv-type f16|f32\n"
        << "  --max-bias F\n"
        << "  --logit-softcap F\n"
        << "  --no-mask\n"
        << "  --random-mask\n"
        << "  --help\n";
}

static bool parse_i64(const char * s, int64_t * out) {
    if (!s || !out) {
        return false;
    }
    char * end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    *out = static_cast<int64_t>(v);
    return true;
}

static bool parse_i32(const char * s, int * out) {
    int64_t v = 0;
    if (!parse_i64(s, &v)) {
        return false;
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

static bool parse_f32(const char * s, float * out) {
    if (!s || !out) {
        return false;
    }
    char * end = nullptr;
    const float v = std::strtof(s, &end);
    if (!end || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

static Config parse_args(int argc, char ** argv) {
    Config cfg;
    cfg.device_url = default_device_url();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--head-dim") {
            if (!parse_i64(need_value("--head-dim"), &cfg.head_dim)) throw std::runtime_error("invalid --head-dim");
        } else if (arg == "--value-dim") {
            if (!parse_i64(need_value("--value-dim"), &cfg.value_dim)) throw std::runtime_error("invalid --value-dim");
        } else if (arg == "--q-len") {
            if (!parse_i64(need_value("--q-len"), &cfg.q_len)) throw std::runtime_error("invalid --q-len");
        } else if (arg == "--kv-len") {
            if (!parse_i64(need_value("--kv-len"), &cfg.kv_len)) throw std::runtime_error("invalid --kv-len");
        } else if (arg == "--softmax-cu") {
            if (!parse_i64(need_value("--softmax-cu"), &cfg.softmax_cu)) throw std::runtime_error("invalid --softmax-cu");
        } else if (arg == "--iters") {
            if (!parse_i32(need_value("--iters"), &cfg.iters)) throw std::runtime_error("invalid --iters");
        } else if (arg == "--seed") {
            if (!parse_i32(need_value("--seed"), &cfg.seed)) throw std::runtime_error("invalid --seed");
        } else if (arg == "--threads") {
            if (!parse_i32(need_value("--threads"), &cfg.n_threads)) throw std::runtime_error("invalid --threads");
        } else if (arg == "--input-scale") {
            if (!parse_f32(need_value("--input-scale"), &cfg.input_scale)) throw std::runtime_error("invalid --input-scale");
        } else if (arg == "--cache-dir") {
            cfg.cache_dir = need_value("--cache-dir");
        } else if (arg == "--device-url") {
            cfg.device_url = need_value("--device-url");
        } else if (arg == "--kv-type") {
            const std::string t = need_value("--kv-type");
            if (t == "f16") {
                cfg.kv_type = GGML_TYPE_F16;
            } else if (t == "f32") {
                cfg.kv_type = GGML_TYPE_F32;
            } else {
                throw std::runtime_error("--kv-type must be f16 or f32");
            }
        } else if (arg == "--max-bias") {
            if (!parse_f32(need_value("--max-bias"), &cfg.max_bias)) throw std::runtime_error("invalid --max-bias");
        } else if (arg == "--logit-softcap") {
            if (!parse_f32(need_value("--logit-softcap"), &cfg.logit_softcap)) throw std::runtime_error("invalid --logit-softcap");
        } else if (arg == "--no-mask") {
            cfg.use_mask = false;
        } else if (arg == "--random-mask") {
            cfg.causal_mask = false;
            cfg.use_mask = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (cfg.head_dim <= 0 || cfg.value_dim <= 0 || cfg.q_len <= 0 || cfg.kv_len <= 0 || cfg.softmax_cu <= 0) {
        throw std::runtime_error("invalid shape args");
    }
    if (cfg.iters <= 0) {
        throw std::runtime_error("--iters must be > 0");
    }
    if (cfg.n_threads <= 0) {
        throw std::runtime_error("--threads must be > 0");
    }
    if (cfg.logit_softcap < 0.0f) {
        throw std::runtime_error("--logit-softcap must be >= 0");
    }
    if (cfg.max_bias > 0.0f && !cfg.use_mask) {
        throw std::runtime_error("--max-bias > 0 requires mask");
    }
    if (!(cfg.input_scale > 0.0f)) {
        throw std::runtime_error("--input-scale must be > 0");
    }
    return cfg;
}

static Inputs make_random_inputs(const Config & cfg, int iter) {
    std::mt19937 rng(static_cast<uint32_t>(cfg.seed + iter * 9973));
    std::uniform_real_distribution<float> qkv_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mask_dist(-1.5f, 0.0f);
    std::uniform_real_distribution<float> p_dist(0.0f, 1.0f);

    Inputs in;
    in.q.resize(static_cast<size_t>(cfg.q_len * cfg.head_dim));
    in.k.resize(static_cast<size_t>(cfg.kv_len * cfg.head_dim));
    in.v.resize(static_cast<size_t>(cfg.kv_len * cfg.value_dim));
    in.mask.resize(static_cast<size_t>(cfg.q_len * cfg.kv_len), 0.0f);

    for (float & x : in.q) x = qkv_dist(rng) * cfg.input_scale;
    for (float & x : in.k) x = qkv_dist(rng) * cfg.input_scale;
    for (float & x : in.v) x = qkv_dist(rng) * cfg.input_scale;

    if (cfg.use_mask) {
        for (int64_t iq = 0; iq < cfg.q_len; ++iq) {
            const int64_t causal_limit = cfg.kv_len - cfg.q_len + iq;
            for (int64_t ikv = 0; ikv < cfg.kv_len; ++ikv) {
                float mv = 0.0f;
                if (cfg.causal_mask) {
                    mv = (ikv > causal_limit) ? kMaskNegInf : mask_dist(rng);
                } else {
                    const float p = p_dist(rng);
                    mv = (p < 0.08f) ? kMaskNegInf : mask_dist(rng);
                }
                in.mask[static_cast<size_t>(iq * cfg.kv_len + ikv)] = mv;
            }
        }
    }

    return in;
}

static Stats calc_stats(const std::vector<float> & v) {
    Stats s;
    if (v.empty()) {
        s.min_v = 0.0f;
        s.max_v = 0.0f;
        return s;
    }
    for (float x : v) {
        if (std::isnan(x)) {
            s.nan_cnt++;
            continue;
        }
        if (std::isinf(x)) {
            s.inf_cnt++;
            continue;
        }
        s.min_v = std::min(s.min_v, x);
        s.max_v = std::max(s.max_v, x);
    }
    if (!std::isfinite(s.min_v)) s.min_v = 0.0f;
    if (!std::isfinite(s.max_v)) s.max_v = 0.0f;
    return s;
}

static DiffStats calc_diff_stats(const std::vector<float> & cpu, const std::vector<float> & npu) {
    if (cpu.size() != npu.size()) {
        throw std::runtime_error("diff size mismatch");
    }

    DiffStats ds;
    double abs_sum = 0.0;
    double num = 0.0;
    double den = 0.0;

    for (size_t i = 0; i < cpu.size(); ++i) {
        const float a = cpu[i];
        const float b = npu[i];

        const bool a_f = std::isfinite(a);
        const bool b_f = std::isfinite(b);
        if (!a_f) ds.cpu_inf_or_nan++;
        if (!b_f) ds.npu_inf_or_nan++;

        if (!a_f || !b_f) {
            continue;
        }

        const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
        ds.max_abs = std::max(ds.max_abs, d);
        abs_sum += d;

        const double e = static_cast<double>(a) - static_cast<double>(b);
        num += e * e;
        den += static_cast<double>(a) * static_cast<double>(a);
        ds.both_finite_cnt++;
    }

    if (ds.both_finite_cnt > 0) {
        ds.mean_abs = abs_sum / static_cast<double>(ds.both_finite_cnt);
    }
    ds.nmse = num / std::max(den, 1e-20);

    for (size_t i = 0; i < cpu.size(); ++i) {
        if (std::isfinite(cpu[i]) && std::isfinite(npu[i])) {
            const double d = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(npu[i]));
            if (d > 5e-3) {
                ds.finite_mismatch_cnt++;
            }
        }
    }

    return ds;
}

class NetmakeRunner {
public:
    struct ShapeInfo {
        int64_t q_bucket = 0;
        int64_t kv_bucket = 0;
        int64_t softmax_cols = 0;
    };

    NetmakeRunner(const Config & cfg, const ShapeInfo & shape)
        : cfg_(cfg), shape_(shape) {
        bundle_ = ggml::fmsh::netmake::get_or_compile_flash_attn_zg_network(
            cfg.cache_dir,
            cfg.head_dim,
            cfg.value_dim,
            shape_.q_bucket,
            shape_.kv_bucket,
            shape_.softmax_cols,
            cfg.logit_softcap > 0.0f);

        device_ = Device::Open(cfg.device_url);
        device_opened_ = true;

        session_ = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle_.network.view(0), {device_, HostDevice::Default()});
        session_.enableTimeProfile(true);
        session_.apply();

        for (const auto & in : bundle_.network.inputs()) {
            input_types_.push_back(in.tensorType().clone());
        }
        input_tensors_.resize(input_types_.size());
        for (size_t i = 0; i < input_tensors_.size(); ++i) {
            input_tensors_[i] = Tensor(input_types_[i].clone());
            input_tensors_[i].mallocOn(HostDevice::MemRegion());
        }

        if (input_tensors_.size() < 5 || input_tensors_.size() > 6) {
            throw std::runtime_error("unexpected netmake flash input count");
        }

        // 在 AXI 模式下，session_.forward(inputs) 返回的 Tensor 的 check_func_
        // 使用绝对 layerCount 目标值做同步。第一次 forward 后 layerCount 达到目标值 N，
        // 第二次 forward 时 check_func_ 捕获的目标仍是 N，而 layerCount 已经是 N，
        // 导致 waitForReady 立即返回，读到上一帧的结果（1帧延迟 bug）。
        //
        // 修复：使用 ZG330Device::layerCount() 做边沿触发同步：
        // 在 forward 之前记录 layerCount，forward 之后等待 layerCount 增加，
        // 确保新的计算真正完成后再读取结果。
        if (device_.is<ZG330Device>()) {
            zg_device_ = device_.cast<ZG330Device>();
        }
    }

    ~NetmakeRunner() {
        if (device_opened_) {
            try {
                Device::Close(device_);
            } catch (...) {
            }
            device_opened_ = false;
        }
    }

    std::vector<float> run(const Inputs & in) {
        float * in_q = reinterpret_cast<float *>(input_tensors_[0].data().cptr());
        float * in_k = reinterpret_cast<float *>(input_tensors_[1].data().cptr());
        float * in_v = reinterpret_cast<float *>(input_tensors_[2].data().cptr());
        float * in_scale = reinterpret_cast<float *>(input_tensors_[3].data().cptr());
        float * in_mask = reinterpret_cast<float *>(input_tensors_[4].data().cptr());
        float * in_softcap = nullptr;
        if (input_tensors_.size() > 5) {
            in_softcap = reinterpret_cast<float *>(input_tensors_[5].data().cptr());
        }

        const size_t q_bytes = static_cast<size_t>(shape_.q_bucket * cfg_.head_dim) * sizeof(float);
        const size_t k_bytes = static_cast<size_t>(cfg_.head_dim * shape_.kv_bucket) * sizeof(float);
        const size_t v_bytes = static_cast<size_t>(shape_.kv_bucket * cfg_.value_dim) * sizeof(float);
        const size_t m_bytes = static_cast<size_t>(shape_.q_bucket * shape_.kv_bucket) * sizeof(float);

        std::memset(in_q, 0, q_bytes);
        std::memset(in_k, 0, k_bytes);
        std::memset(in_v, 0, v_bytes);
        std::memset(in_mask, 0, m_bytes);

        for (int64_t iq = 0; iq < cfg_.q_len; ++iq) {
            float * q_row = in_q + static_cast<size_t>(iq * cfg_.head_dim);
            const float * src = in.q.data() + static_cast<size_t>(iq * cfg_.head_dim);
            std::memcpy(q_row, src, static_cast<size_t>(cfg_.head_dim) * sizeof(float));
        }

        for (int64_t ikv = 0; ikv < cfg_.kv_len; ++ikv) {
            const float * k_row = in.k.data() + static_cast<size_t>(ikv * cfg_.head_dim);
            for (int64_t d = 0; d < cfg_.head_dim; ++d) {
                in_k[static_cast<size_t>(d * shape_.kv_bucket + ikv)] = k_row[d];
            }

            float * v_row = in_v + static_cast<size_t>(ikv * cfg_.value_dim);
            const float * src_v = in.v.data() + static_cast<size_t>(ikv * cfg_.value_dim);
            std::memcpy(v_row, src_v, static_cast<size_t>(cfg_.value_dim) * sizeof(float));
        }

        for (int64_t iq = 0; iq < shape_.q_bucket; ++iq) {
            float * m_row = in_mask + static_cast<size_t>(iq * shape_.kv_bucket);
            for (int64_t ikv = 0; ikv < shape_.kv_bucket; ++ikv) {
                if (ikv >= cfg_.kv_len) {
                    m_row[ikv] = kMaskNegInf;
                } else if (iq >= cfg_.q_len || !cfg_.use_mask) {
                    m_row[ikv] = 0.0f;
                } else {
                    m_row[ikv] = in.mask[static_cast<size_t>(iq * cfg_.kv_len + ikv)];
                }
            }
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
        if (cfg_.logit_softcap > 0.0f) {
            scale /= cfg_.logit_softcap;
        }
        in_scale[0] = scale;
        if (in_softcap != nullptr) {
            in_softcap[0] = cfg_.logit_softcap;
        }

        // AXI 模式同步修复：
        //
        // 根本原因：Session 内部的 tmap_ 会复用同一个 Tensor 对象（同一个 TensorNode）。
        // 第一次 forward 后 waitForReady 成功，将 TensorNode::ready_ 置为 true。
        // 此后每次 forward 调用 waitForReady 时，因 ready_=true 立即返回，
        // 不等待 NPU 完成，导致读到上一帧数据（1帧延迟 bug）。
        //
        // 修复方案：
        // 1. forward 前记录 layerCount（layer_before）
        // 2. 调用 session_.forward() — 内部会触发 NPU 计算，但 waitForReady 立即返回
        // 3. 对输出 Tensor 调用 setReady(false) 重置 ready_ 标志
        // 4. 安装新的 check_func_，等待 layerCount >= layer_before + layer_increment_
        //    （layer_increment_ 在第一次 forward 时学习得到）
        // 5. 调用 waitForReady() 真正等待 NPU 完成
        bool use_layer_count_sync = zg_device_.defined();
        uint32_t layer_before = 0;
        if (use_layer_count_sync) {
            layer_before = zg_device_.layerCount();
        }

        auto outputs = session_.forward(input_tensors_);
        if (outputs.empty()) {
            throw std::runtime_error("session_.forward returned empty outputs");
        }

        if (use_layer_count_sync) {
            // 学习每次 forward 的 layerCount 增量（第一次 forward 时）
            if (layer_increment_ == 0) {
                // 第一次：等待 layerCount 变化，学习增量
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                while (zg_device_.layerCount() == layer_before) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        throw std::runtime_error("NPU layerCount sync timeout (learning increment)");
                    }
                }
                // 继续等待直到 layerCount 稳定（NPU 完成所有层）
                // 用 waitForReady 完成第一次同步（此时 ready_=false，check_func_ 有效）
                if (!outputs[0].waitForReady(std::chrono::milliseconds(30000))) {
                    throw std::runtime_error("Waiting for netmake output ready timed out (first forward)");
                }
                layer_increment_ = zg_device_.layerCount() - layer_before;
                std::cout << "[axi-sync] learned layer_increment=" << layer_increment_ << std::endl;
            } else {
                // 后续 forward：重置 ready_ 并安装正确的 check_func_
                const uint32_t layer_target = layer_before + layer_increment_;
                ZG330Device zg_dev = zg_device_;  // capture by value for lambda
                outputs[0].setReady(false);
                outputs[0].setCheckFunc([zg_dev, layer_target](const Device &) -> bool {
                    return zg_dev.layerCount() >= layer_target;
                });
                if (!outputs[0].waitForReady(std::chrono::milliseconds(30000))) {
                    throw std::runtime_error("Waiting for netmake output ready timed out (AXI sync)");
                }
            }
        } else {
            // socket 模式：使用原来的 waitForReady
            if (!outputs[0].waitForReady(std::chrono::milliseconds(30000))) {
                throw std::runtime_error("Waiting for netmake output ready timed out");
            }
        }

        std::vector<float> out(static_cast<size_t>(shape_.q_bucket * cfg_.value_dim), 0.0f);
        outputs[0].read(reinterpret_cast<char *>(out.data()), 0, out.size() * sizeof(float));
        return out;
    }

    const ggml::fmsh::netmake::FlashAttnZgNetworkBundle & bundle() const {
        return bundle_;
    }

private:
    Config cfg_;
    ShapeInfo shape_;

    Device device_;
    bool device_opened_ = false;

    ggml::fmsh::netmake::FlashAttnZgNetworkBundle bundle_;
    Session session_;

    std::vector<TensorType> input_types_;
    std::vector<Tensor> input_tensors_;
    ZG330Device zg_device_;      // valid only in AXI mode; used for layerCount() edge-triggered sync
    uint32_t layer_increment_ = 0;  // learned on first forward: how many layers per forward call
};

static std::vector<float> run_cpu_reference(const Config & cfg, const Inputs & in) {
    ggml_cpu_init();

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        throw std::runtime_error("ggml_backend_cpu_init failed");
    }
    ggml_backend_cpu_set_n_threads(backend, cfg.n_threads);

    const size_t graph_size = 2u * 1024u * 1024u;
    ggml_init_params params = {
        graph_size,
        nullptr,
        true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.head_dim, cfg.q_len, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, cfg.kv_type, cfg.head_dim, cfg.kv_len, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, cfg.kv_type, cfg.value_dim, cfg.kv_len, 1, 1);
    ggml_tensor * m = cfg.use_mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.kv_len, cfg.q_len, 1, 1) : nullptr;

    const float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, cfg.max_bias, cfg.logit_softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    }

    ggml_backend_tensor_set(q, in.q.data(), 0, in.q.size() * sizeof(float));

    if (cfg.kv_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(k, in.k.data(), 0, in.k.size() * sizeof(float));
        ggml_backend_tensor_set(v, in.v.data(), 0, in.v.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> k_f16(in.k.size());
        std::vector<ggml_fp16_t> v_f16(in.v.size());
        ggml_fp32_to_fp16_row(in.k.data(), k_f16.data(), static_cast<int64_t>(in.k.size()));
        ggml_fp32_to_fp16_row(in.v.data(), v_f16.data(), static_cast<int64_t>(in.v.size()));
        ggml_backend_tensor_set(k, k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));
    }

    if (m) {
        std::vector<ggml_fp16_t> m_f16(in.mask.size());
        ggml_fp32_to_fp16_row(in.mask.data(), m_f16.data(), static_cast<int64_t>(in.mask.size()));
        ggml_backend_tensor_set(m, m_f16.data(), 0, m_f16.size() * sizeof(ggml_fp16_t));
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        throw std::runtime_error("ggml_backend_graph_compute failed");
    }

    std::vector<float> y(static_cast<size_t>(cfg.q_len * cfg.value_dim), 0.0f);
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return y;
}

static void print_config(const Config & cfg, int64_t q_bucket, int64_t kv_bucket, int64_t softmax_cols) {
    std::cout
        << "config"
        << " head_dim=" << cfg.head_dim
        << " value_dim=" << cfg.value_dim
        << " q_len=" << cfg.q_len
        << " kv_len=" << cfg.kv_len
        << " q_bucket=" << q_bucket
        << " kv_bucket=" << kv_bucket
        << " softmax_cols=" << softmax_cols
        << " kv_type=" << (cfg.kv_type == GGML_TYPE_F16 ? "f16" : "f32")
        << " iters=" << cfg.iters
        << " seed=" << cfg.seed
        << " mask=" << (cfg.use_mask ? (cfg.causal_mask ? "causal" : "random") : "none")
        << " max_bias=" << cfg.max_bias
        << " logit_softcap=" << cfg.logit_softcap
        << " threads=" << cfg.n_threads
        << " input_scale=" << cfg.input_scale
        << " cache_dir=" << cfg.cache_dir.string()
        << " device_url=" << cfg.device_url
        << "\n";
}

static std::vector<float> crop_netmake_output(const Config & cfg, int64_t q_bucket, const std::vector<float> & y_bucket) {
    std::vector<float> y(static_cast<size_t>(cfg.q_len * cfg.value_dim), 0.0f);
    for (int64_t iq = 0; iq < cfg.q_len; ++iq) {
        const float * src = y_bucket.data() + static_cast<size_t>(iq * cfg.value_dim);
        float * dst = y.data() + static_cast<size_t>(iq * cfg.value_dim);
        std::memcpy(dst, src, static_cast<size_t>(cfg.value_dim) * sizeof(float));
    }
    GGML_UNUSED(q_bucket);
    return y;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Config cfg = parse_args(argc, argv);

        const int64_t q_bucket = next_pow2_i64(cfg.q_len);
        const int64_t kv_bucket = next_pow2_i64(cfg.kv_len);
        const int64_t softmax_cols = align_up_i64(q_bucket, std::max<int64_t>(1, cfg.softmax_cu));

        print_config(cfg, q_bucket, kv_bucket, softmax_cols);

        NetmakeRunner::ShapeInfo shape_info;
        shape_info.q_bucket = q_bucket;
        shape_info.kv_bucket = kv_bucket;
        shape_info.softmax_cols = softmax_cols;

        NetmakeRunner runner(cfg, shape_info);

        std::cout
            << "netmake"
            << " net=" << runner.bundle().net_name
            << " ram_cache_hit=" << (runner.bundle().ram_cache_hit ? 1 : 0)
            << " compiled_now=" << (runner.bundle().compiled_now ? 1 : 0)
            << "\n";

        bool has_fail = false;

        for (int it = 0; it < cfg.iters; ++it) {
            const Inputs in = make_random_inputs(cfg, it);

            const std::vector<float> cpu_y = run_cpu_reference(cfg, in);
            const Stats cpu_stats = calc_stats(cpu_y);

            const std::vector<float> npu_y_bucket = runner.run(in);
            const std::vector<float> npu_y = crop_netmake_output(cfg, q_bucket, npu_y_bucket);
            const Stats npu_stats = calc_stats(npu_y);

            const DiffStats diff = calc_diff_stats(cpu_y, npu_y);

            const bool bad = (npu_stats.inf_cnt > 0) || (npu_stats.nan_cnt > 0) ||
                             (diff.cpu_inf_or_nan == 0 && diff.npu_inf_or_nan > 0) ||
                             (diff.max_abs > 1e-2 && diff.nmse > 1e-4);
            if (bad) {
                has_fail = true;
            }

            std::cout
                << "iter=" << it
                << " cpu[min,max,inf,nan]=[" << cpu_stats.min_v << "," << cpu_stats.max_v << ","
                << cpu_stats.inf_cnt << "," << cpu_stats.nan_cnt << "]"
                << " npu[min,max,inf,nan]=[" << npu_stats.min_v << "," << npu_stats.max_v << ","
                << npu_stats.inf_cnt << "," << npu_stats.nan_cnt << "]"
                << " diff[max_abs,mean_abs,nmse]=[" << std::setprecision(8)
                << diff.max_abs << "," << diff.mean_abs << "," << diff.nmse << "]"
                << " finite_mismatch=" << diff.finite_mismatch_cnt
                << " status=" << (bad ? "FAIL" : "OK")
                << "\n";

            if (bad) {
                size_t printed = 0;
                for (int64_t iq = 0; iq < cfg.q_len && printed < 8; ++iq) {
                    for (int64_t d = 0; d < cfg.value_dim && printed < 8; ++d) {
                        const size_t idx = static_cast<size_t>(iq * cfg.value_dim + d);
                        const float a = cpu_y[idx];
                        const float b = npu_y[idx];
                        const bool bad_elem = (!std::isfinite(a) || !std::isfinite(b) ||
                                               std::fabs(static_cast<double>(a) - static_cast<double>(b)) > 1e-2);
                        if (bad_elem) {
                            std::cout << "  mismatch[q=" << iq << ",d=" << d << "] cpu=" << a << " npu=" << b << "\n";
                            printed++;
                        }
                    }
                }
            }
        }

        if (has_fail) {
            std::cout << "result=FAIL\n";
            return 1;
        }

        std::cout << "result=PASS\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
