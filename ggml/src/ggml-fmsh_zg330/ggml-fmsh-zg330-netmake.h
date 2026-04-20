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

void preload_matmul_zg_cache(const std::filesystem::path & work_root);

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

} // namespace ggml::fmsh::netmake
