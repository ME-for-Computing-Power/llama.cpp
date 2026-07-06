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

    // Manually mmap'd backing for `network`'s lazily-loaded params (see
    // load_prebaked_const_matmul_zg_network). Ownership of the *mapping*
    // itself still belongs to `network` (its destructor munmap()s it); this
    // is only exposed so evict_prebaked_const_matmul_host_cache() can advise
    // the kernel to drop the resident pages once Session::apply() has copied
    // them to device memory, instead of paying host RAM for the mapping's
    // full lifetime.
    void * mmap_addr = nullptr;
    uint64_t mmap_size = 0;
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

// Drops the host page-cache pages backing `bundle.mmap_addr` via
// madvise(MADV_DONTNEED). Call only after the network has been fully
// deployed (Session::apply() returned), since that is the last point the
// backend reads from the mmap'd file on the host side. Safe no-op if the
// bundle wasn't produced by load_prebaked_const_matmul_zg_network. Returns
// true if madvise() was actually issued and succeeded.
bool evict_prebaked_const_matmul_host_cache(ConstMatmulZgNetworkBundle & bundle);

// Discovers the (k, n) shape of a pre-baked constant-weight matmul artifact
// for a given `net_name_prefix` and `m`, by scanning `work_root` for a
// directory named `<net_name_prefix>_<m>x<k>x<n>` (the layout produced by
// load_prebaked_const_matmul_zg_network / the offline bake script). Used to
// eagerly deploy the network (Session::apply()) at backend-init time —
// before m/k/n are known from a live MUL_MAT node — so the one-time host
// page-cache spike from apply()'s DMA upload lands before the model's own
// weights get faulted in by the first graph_compute, instead of stacking on
// top of them. Returns false if no matching directory is found.
bool find_prebaked_const_matmul_dims(
    const std::filesystem::path & work_root,
    const std::string & net_name_prefix,
    int64_t m,
    int64_t * out_k,
    int64_t * out_n);

} // namespace ggml::fmsh::netmake
