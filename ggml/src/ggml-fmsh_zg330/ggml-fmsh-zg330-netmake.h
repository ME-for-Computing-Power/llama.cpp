#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <icraft-xir/core/network.h>

namespace ggml::fmsh::netmake {

struct OnnxMatmulModel {
    std::filesystem::path onnx_path;
    std::string input_name;
    std::string weight_name;
    std::string output_name;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
};

struct IcraftArtifacts {
    std::filesystem::path work_dir;
    std::filesystem::path toml_path;
    std::filesystem::path json_path;
    std::filesystem::path raw_path;
};

struct MatmulZgNetworkBundle {
    std::string net_name;
    icraft::xir::Network network;
    bool ram_cache_hit = false;
    bool compiled_now = false;
};

enum class ElementwiseZgOp : uint32_t {
    ADD = 0,
    MUL = 1,
    SCALE = 2,
    CPY = 3,
    DUP = 4,
    SOFT_MAX = 5,
    RMS_NORM = 6,
    BF16_BRIDGE = 7,
};

struct ElementwiseZgNetworkBundle {
    std::string net_name;
    icraft::xir::Network network;
    bool ram_cache_hit = false;
    bool compiled_now = false;
};

struct FlashAttnZgNetworkBundle {
    std::string net_name;
    icraft::xir::Network network;
    bool ram_cache_hit = false;
    bool compiled_now = false;
};

// Two-phase RMS_NORM bundle: pre computes mean(x²)+eps per row, post applies X * r_inv.
struct RmsNormSplitNetworkBundle {
    std::string net_name_pre;   // rmsnorm_pre: inputs=[X[R,C], eps[1,1]], output=[r_sq[R,1]]
    std::string net_name_post;  // rmsnorm_post: inputs=[X[R,C], r_inv[R,1]], output=[Y[R,C]]
    icraft::xir::Network network_pre;
    icraft::xir::Network network_post;
    bool ram_cache_hit = false;
    bool compiled_now = false;
};

// ROPE NeoX bundle: inputs=[X[N,d], theta[N,d/2]], output=[Y[N,d]]
// theta contains the rotation angles (computed on host per position×dim-pair).
struct RopeZgNetworkBundle {
    std::string net_name;
    icraft::xir::Network network;
    bool ram_cache_hit = false;
    bool compiled_now = false;
};

void preload_zg_cache(const std::filesystem::path & work_root);

MatmulZgNetworkBundle get_or_compile_matmul_zg_network(
    const std::filesystem::path & work_root,
    int64_t m,
    int64_t k,
    int64_t n);

ElementwiseZgNetworkBundle get_or_compile_elementwise_zg_network(
    const std::filesystem::path & work_root,
    ElementwiseZgOp op,
    int64_t rows,
    int64_t cols);

FlashAttnZgNetworkBundle get_or_compile_flash_attn_zg_network(
    const std::filesystem::path & work_root,
    int64_t head_dim,
    int64_t value_dim,
    int64_t q_len,
    int64_t kv_len,
    int64_t softmax_cols,
    bool use_logit_softcap);

ElementwiseZgNetworkBundle get_or_compile_bf16_bridge_zg_network(
    const std::filesystem::path & work_root,
    int64_t rows,
    int64_t cols);

ElementwiseZgNetworkBundle get_or_compile_fused_ew_zg_network(
    const std::filesystem::path & work_root,
    const std::vector<ElementwiseZgOp> & ops,
    int64_t rows,
    int64_t cols);

// Two-phase RMS_NORM: avoids the ZG-unsupported Sqrt op by splitting into a pre network
// (reduces X→r²) and a post network (multiplies X by host-computed r_inv = 1/sqrt(r²)).
RmsNormSplitNetworkBundle get_or_compile_rmsnorm_split_zg_networks(
    const std::filesystem::path & work_root,
    int64_t rows,
    int64_t cols);

// ROPE NeoX: the network computes Sin/Cos of theta and applies the rotation.
// rows=N (seq*n_head), cols=d (head_dim), half_cols=d/2.
RopeZgNetworkBundle get_or_compile_rope_zg_network(
    const std::filesystem::path & work_root,
    int64_t rows,
    int64_t cols);

} // namespace ggml::fmsh::netmake
