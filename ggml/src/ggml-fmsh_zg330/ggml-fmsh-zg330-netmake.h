#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
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

// Bundle for a load-only, pre-baked (offline-compiled) constant-weight matmul network,
// e.g. the LM-head/unembedding matmul. Unlike MatmulZgNetworkBundle, this is never
// compiled at runtime: if the artifact is missing, `found` is false and callers must
// fall back (e.g. to CPU) rather than invoking `icraft compile`.
struct ConstMatmulZgNetworkBundle {
    std::string net_name;
    icraft::xir::Network network;
    std::filesystem::path raw_path; // for resident_summary logging (stat() proxy, not host RSS)
    bool found = false;
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

// Recursively scans `work_dir/.cache` (and `work_dir/.cache` itself) for
// `<net_name>_ZG.json` / `<net_name>_ZG.raw`, matching the layout produced by
// `icraft compile`. Throws if not found.
std::pair<std::filesystem::path, std::filesystem::path> find_generated_zg_json_raw(
    const std::filesystem::path & work_dir,
    const std::string & net_name);

// Load-only lookup for a pre-baked constant-weight matmul network (e.g. LM head).
// Never invokes `icraft compile`. Net name is `<net_name_prefix>_<m>x<k>x<n>`,
// expected under `<work_root>/<net_name>/.cache/...`. On any miss or load
// failure returns a bundle with `found == false` instead of throwing.
// Uses `Network::lazyLoadParamsFromFile` so the raw weight blob is not fully
// materialized in host RAM.
ConstMatmulZgNetworkBundle load_prebaked_const_matmul_zg_network(
    const std::filesystem::path & work_root,
    int64_t m,
    int64_t k,
    int64_t n,
    const std::string & net_name_prefix);

} // namespace ggml::fmsh::netmake
