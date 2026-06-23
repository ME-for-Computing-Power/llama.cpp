#include "ggml-fmsh-zg330-internal.h"

namespace ggml_fmsh_zg330_impl {

void ggml_fmsh_accumulate_profile(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    const Session & session,
    double total_ms) {
    auto & p = ctx->perf[static_cast<uint32_t>(node->op)];
    p.calls++;
    p.total_ms += total_ms;

    const auto profile = session.timeProfileResults();
    double profile_total_ms = 0.0;
    double memcpy_ms = 0.0;
    double hard_ms = 0.0;
    double other_ms = 0.0;
    for (const auto & kv : profile) {
        const auto & t = kv.second;
        profile_total_ms += std::get<0>(t);
        memcpy_ms += std::get<1>(t);
        hard_ms += std::get<2>(t);
        other_ms += std::get<3>(t);
    }
    p.profile_total_ms += profile_total_ms;
    p.memcpy_ms += memcpy_ms;
    p.hard_ms += hard_ms;
    p.other_ms += other_ms;
}

std::string ggml_fmsh_op_name_from_id(uint32_t op_id) {
    if (op_id < static_cast<uint32_t>(GGML_OP_COUNT)) {
        return std::string(ggml_op_name(static_cast<enum ggml_op>(op_id)));
    }
    return "unknown(" + std::to_string(op_id) + ")";
}

std::string ggml_fmsh_fmt_ms(double ms) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << ms;
    return oss.str();
}

std::string ggml_fmsh_fmt_pct(double part, double total) {
    const double pct = total > 0.0 ? (100.0 * part / total) : 0.0;
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << pct;
    return oss.str();
}

bool ggml_fmsh_is_batched_mul_mat(const ggml_tensor * node) {
    return node != nullptr && node->op == GGML_OP_MUL_MAT && (node->ne[2] > 1 || node->ne[3] > 1);
}

size_t ggml_fmsh_tensor_live_bytes_f32(const Tensor & t) {
    if (!t.defined()) {
        return 0;
    }
    const int64_t elems = std::max<int64_t>(0, t.dtype().numElements());
    return static_cast<size_t>(elems) * sizeof(float);
}

std::string ggml_fmsh_tensor_loc_str(ggml_backend_fmsh_zg330_context * ctx, const Tensor & t) {
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
void ggml_fmsh_evict_to_host(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    const Tensor & device_tensor) {
    if (!t) return;
    const size_t bytes = ggml_nbytes(t);
    const size_t live_bytes = ggml_fmsh_tensor_live_bytes_f32(device_tensor);
    const size_t read_bytes = std::min(bytes, live_bytes);
    if (bytes == 0) return;
    MemChunk zg_chunk;
    size_t   zg_off = 0;
    if (ggml_fmsh_is_zg_buffer_tensor(t, &zg_chunk, &zg_off)) {
        const ggml_tensor * root = t;
        while (root->view_src) root = root->view_src;
        auto & staging = ctx->zg_staging_bufs[root];
        if (staging.empty()) {
            staging.resize(read_bytes);
            if (read_bytes > 0) {
                device_tensor.read(staging.data(), 0, read_bytes);
            }
            if (read_bytes != bytes) {
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "evict_to_host_clamp op=" +
                        std::string(ggml_op_name(t->op)) +
                        " logical_bytes=" + std::to_string(bytes) +
                        " live_bytes=" + std::to_string(live_bytes));
            }
        }
        // Do not redirect t->data here.
    } else if (t->data != nullptr) {
        if (read_bytes > 0) {
            device_tensor.read(reinterpret_cast<char *>(t->data), 0, read_bytes);
        }
        if (read_bytes != bytes) {
            ggml_fmsh_log_locked(
                ctx, 1,
                "evict_to_host_clamp op=" +
                    std::string(ggml_op_name(t->op)) +
                    " logical_bytes=" + std::to_string(bytes) +
                    " live_bytes=" + std::to_string(live_bytes));
        }
    }
}

void ggml_fmsh_materialize_if_device(
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
        const ggml_tensor * root = t;
        while (root->view_src) root = root->view_src;
        auto & staging = ctx->zg_staging_bufs[root];
        if (staging.empty()) {
            const size_t root_bytes = ggml_nbytes(root);
            staging.resize(root_bytes);
            if (t == root && it->second.defined()) {
                const size_t live_bytes = ggml_fmsh_tensor_live_bytes_f32(it->second);
                const size_t read_bytes = std::min(root_bytes, live_bytes);
                if (read_bytes > 0) {
                    it->second.read(staging.data(), 0, read_bytes);
                }
                if (read_bytes < root_bytes) {
                    std::memset(staging.data() + read_bytes, 0, root_bytes - read_bytes);
                }
                ggml_fmsh_log_locked(ctx, 1,
                    "materialize_zg_from_clone bytes=" + std::to_string(read_bytes) +
                    " root_bytes=" + std::to_string(root_bytes) +
                    " op_src=" + std::string(ggml_op_name(t->op)));
            } else {
                MemChunk rc; size_t ro = 0;
                ggml_fmsh_is_zg_buffer_tensor(root, &rc, &ro);
                rc.read(staging.data(), ro, root_bytes);
                ggml_fmsh_log_locked(ctx, 1,
                    "materialize_zg_to_staging bytes=" + std::to_string(root_bytes) +
                    " op_src=" + std::string(ggml_op_name(t->op)));
            }
        }
    } else if (t->data != nullptr) {
        // t->data is a real host pointer (compute buffer): write directly.
        const size_t live_bytes = ggml_fmsh_tensor_live_bytes_f32(it->second);
        const size_t read_bytes = std::min(bytes, live_bytes);
        if (read_bytes > 0) {
            it->second.read(reinterpret_cast<char *>(t->data), 0, read_bytes);
        }
        if (read_bytes != bytes) {
            ggml_fmsh_log_locked(ctx, 1,
                "materialize_to_host_clamp bytes=" + std::to_string(bytes) +
                " live_bytes=" + std::to_string(live_bytes) +
                " op_src=" + std::string(ggml_op_name(t->op)));
        }
        ggml_fmsh_log_locked(ctx, 1,
            "materialize_to_host bytes=" + std::to_string(bytes) +
            " op_src=" + std::string(ggml_op_name(t->op)));
    }
    device_tensor_map.erase(it);
}

bool ggml_fmsh_ensure_zg_staging(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    bool read_existing,
    std::string * err) {
    if (!t) {
        return true;
    }
    const ggml_tensor * root = t;
    while (root->view_src) {
        root = root->view_src;
    }

    MemChunk zg_chunk;
    size_t zg_off = 0;
    if (!ggml_fmsh_is_zg_buffer_tensor(root, &zg_chunk, &zg_off)) {
        return true;
    }

    try {
        auto & staging = ctx->zg_staging_bufs[root];
        const size_t root_bytes = ggml_nbytes(root);
        if (staging.empty()) {
            staging.resize(root_bytes);
            if (read_existing && root_bytes > 0) {
                zg_chunk.read(staging.data(), zg_off, root_bytes);
                ggml_fmsh_log_locked(ctx, 1,
                    "staging_d2h bytes=" + std::to_string(root_bytes) +
                    " op_src=" + std::string(ggml_op_name(t->op)));
            } else {
                ggml_fmsh_log_locked(ctx, 1,
                    "staging_alloc bytes=" + std::to_string(root_bytes) +
                    " op_src=" + std::string(ggml_op_name(t->op)));
            }
        }
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

bool ggml_fmsh_sync_zg_staging_to_device(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    const char * reason,
    std::string * err) {
    if (!t) {
        return true;
    }
    const ggml_tensor * root = t;
    while (root->view_src) {
        root = root->view_src;
    }
    auto sit = ctx->zg_staging_bufs.find(root);
    if (sit == ctx->zg_staging_bufs.end() || sit->second.empty()) {
        return true;
    }

    MemChunk zg_chunk;
    size_t zg_off = 0;
    if (!ggml_fmsh_is_zg_buffer_tensor(root, &zg_chunk, &zg_off)) {
        return true;
    }

    try {
        zg_chunk.write(zg_off, sit->second.data(), sit->second.size());
        ggml_fmsh_log_locked(ctx, 1,
            "staging_h2d bytes=" + std::to_string(sit->second.size()) +
            " op=" + std::string(ggml_op_name(t->op)) +
            " reason=" + (reason ? reason : "sync"));
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

int ggml_fmsh_count_future_consumers(ggml_cgraph * cgraph, int from_idx, const ggml_tensor * producer) {
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

enum ggml_status ggml_fmsh_compute_cpu_fallback(ggml_backend_fmsh_zg330_context * ctx, ggml_cgraph * cgraph) {
    if (!ctx->cpu_backend) {
        return GGML_STATUS_FAILED;
    }
    return ggml_backend_graph_compute(ctx->cpu_backend, cgraph);
}

enum ggml_status ggml_fmsh_compute_cpu_node(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_cgraph * cgraph,
    int node_index) {
    if (!ctx->cpu_backend) {
        return GGML_STATUS_FAILED;
    }
    ggml_cgraph node_view = ggml_graph_view(cgraph, node_index, node_index + 1);
    return ggml_backend_graph_compute(ctx->cpu_backend, &node_view);
}

const char * ggml_backend_fmsh_zg330_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FMSH_ZG330";
}

void ggml_backend_fmsh_zg330_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_fmsh_zg330_context *>(backend->context);
    bool release_device = false;
    std::string device_url;
    if (ctx) {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ggml_fmsh_log_locked(
            ctx, 1,
            "backend free: cache_size=" + std::to_string(ctx->session_cache.size()) +
            " elementwise_cache_size=" + std::to_string(ctx->elementwise_session_cache.size()) +
            " flash_cache_size=" + std::to_string(ctx->flash_attn_session_cache.size()));
        double perf_total_ms = 0.0;
        for (const auto & kv : ctx->perf) {
            perf_total_ms += kv.second.total_ms;
        }
        for (const auto & kv : ctx->perf) {
            const auto & p = kv.second;
            const double avg_ms = p.calls > 0 ? (p.total_ms / static_cast<double>(p.calls)) : 0.0;
            const double not_in_profile_ms = std::max(0.0, p.total_ms - p.profile_total_ms);
            ggml_fmsh_log_locked(
                ctx, 1,
                "op_summary op=" + ggml_fmsh_op_name_from_id(kv.first) +
                " calls=" + std::to_string(p.calls) +
                " total_ms=" + ggml_fmsh_fmt_ms(p.total_ms) +
                " avg_ms=" + ggml_fmsh_fmt_ms(avg_ms) +
                " profile_total_ms=" + ggml_fmsh_fmt_ms(p.profile_total_ms) +
                " memcpy_ms=" + ggml_fmsh_fmt_ms(p.memcpy_ms) +
                " hard_ms=" + ggml_fmsh_fmt_ms(p.hard_ms) +
                " other_ms=" + ggml_fmsh_fmt_ms(p.other_ms) +
                " not_in_profile_ms=" + ggml_fmsh_fmt_ms(not_in_profile_ms) +
                "\n  pct:"
                " profile_total=" + ggml_fmsh_fmt_pct(p.profile_total_ms, p.total_ms) +
                " memcpy=" + ggml_fmsh_fmt_pct(p.memcpy_ms, p.total_ms) +
                " hard=" + ggml_fmsh_fmt_pct(p.hard_ms, p.total_ms) +
                " other=" + ggml_fmsh_fmt_pct(p.other_ms, p.total_ms) +
                " not_in_profile=" + ggml_fmsh_fmt_pct(not_in_profile_ms, p.total_ms) +
                "\n  global_pct:"
                " total=" + ggml_fmsh_fmt_pct(p.total_ms, perf_total_ms) +
                " profile_total=" + ggml_fmsh_fmt_pct(p.profile_total_ms, perf_total_ms) +
                " memcpy=" + ggml_fmsh_fmt_pct(p.memcpy_ms, perf_total_ms) +
                " hard=" + ggml_fmsh_fmt_pct(p.hard_ms, perf_total_ms) +
                " other=" + ggml_fmsh_fmt_pct(p.other_ms, perf_total_ms) +
                " not_in_profile=" + ggml_fmsh_fmt_pct(not_in_profile_ms, perf_total_ms));
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
            " qweight_offloaded=" + std::to_string(ctx->mul_mat_qweight_offloaded) +
            " qweight_resident_mib=" + std::to_string(ctx->mul_mat_qweight_bytes_resident / (1024 * 1024)) +
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
        ctx->zg_staging_bufs.clear();
        ctx->session_cache.clear();
        ctx->qweight_session_cache.clear();
        ctx->elementwise_session_cache.clear();
        ctx->rmsnorm_split_session_cache.clear();
        ctx->rmsnorm_customop_cache.clear();
        ctx->rms_norm_custom_op_verified = false;
        ctx->rope_session_cache.clear();
        ctx->flash_attn_session_cache.clear();
        ctx->bridge_session_cache.clear();
        ctx->fused_ew_session_cache.clear();
        ctx->reuse_pools.clear();
        ctx->qweight_f32_bufs.clear();

        release_device = ctx->device_opened;
        device_url = ctx->device_url;
    }

    if (ctx && release_device) {
        ggml_fmsh_release_shared_device(device_url, &ctx->zg_device, &ctx->device_opened);
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
bool ggml_fmsh_is_zg_buffer_tensor(
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
    if (out_chunk) {
        *out_chunk = buf_ctx->chunk;
    }
    if (out_offset) {
        *out_offset = root_off + total_off;
    }
    return true;
}

// Build an icraft Tensor pointing to the ZG DDR region for ggml tensor t.
// The TensorType shape is flat (1-D, total elements) with F32 element type —
// sufficient for materialize_if_device (D2H read) and for ops that look up
// device_tensor_map to feed into session.forward.
Tensor ggml_fmsh_make_zg_tensor(const ggml_tensor * t) {
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

Tensor ggml_fmsh_clone_chain_tensor(
    ggml_backend_fmsh_zg330_context * ctx,
    const Tensor & src,
    std::string * err) {
    if (!src.defined()) {
        return Tensor();
    }
    const size_t bytes = ggml_fmsh_tensor_live_bytes_f32(src);
    if (bytes == 0) {
        return Tensor();
    }
    try {
        if (ctx != nullptr && ctx->device_opened) {
            MemChunk chunk = ctx->zg_device.defaultMemRegion().malloc(bytes, true, 4096);
            Tensor cloned(src.dtype(), chunk, 0);
            std::vector<char> tmp(bytes);
            src.read(tmp.data(), 0, bytes);
            cloned.write(0, tmp.data(), bytes);
            return cloned;
        }
        Tensor cloned(src.dtype().clone());
        cloned.mallocOn(HostDevice::MemRegion());
        std::vector<char> tmp(bytes);
        src.read(tmp.data(), 0, bytes);
        std::memcpy(cloned.data().cptr(), tmp.data(), bytes);
        return cloned;
    } catch (const std::exception & e) {
        if (err) {
            *err = std::string("clone chain tensor failed: ") + e.what();
        }
        return Tensor();
    }
}

// Execute a single CPU node with ZG buffer tensors temporarily redirected to their staging buffers.
// After execution, writes staging data back to ZG DDR for any src that is a ZG buffer
// and has a staging buffer (meaning it may have been written by the CPU op, e.g. SET_ROWS into KV cache).
enum ggml_status ggml_fmsh_compute_cpu_node_with_zg_redirect(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_cgraph * cgraph,
    int node_index) {
    ggml_tensor * node = cgraph->nodes[node_index];
    std::string staging_err;

    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        if (!node->src[s]) {
            continue;
        }
        if (!ggml_fmsh_ensure_zg_staging(ctx, node->src[s], true, &staging_err)) {
            ggml_fmsh_log_locked(ctx, 3,
                "staging_prepare_failed op=" + std::string(ggml_op_name(node->op)) +
                " src=" + std::to_string(s) +
                " reason=" + staging_err);
            return GGML_STATUS_FAILED;
        }
    }
    if (!ggml_fmsh_ensure_zg_staging(ctx, node, true, &staging_err)) {
        ggml_fmsh_log_locked(ctx, 3,
            "staging_prepare_failed op=" + std::string(ggml_op_name(node->op)) +
            " dst reason=" + staging_err);
        return GGML_STATUS_FAILED;
    }

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

    // Write back: dst first, then any source roots that have staging. This preserves
    // unsupported CPU op results in PLDDR and covers in-place src mutation.
    if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "fallback_dst", &staging_err)) {
        ggml_fmsh_log_locked(ctx, 3,
            "staging_writeback_failed op=" + std::string(ggml_op_name(node->op)) +
            " dst reason=" + staging_err);
        return GGML_STATUS_FAILED;
    }
    for (int s = 0; s < GGML_MAX_SRC; ++s) {
        const ggml_tensor * src = node->src[s];
        if (!src) break;
        if (!ggml_fmsh_sync_zg_staging_to_device(ctx, src, "fallback_src", &staging_err)) {
            ggml_fmsh_log_locked(ctx, 3,
                "staging_writeback_failed op=" + std::string(ggml_op_name(node->op)) +
                " src=" + std::to_string(s) +
                " reason=" + staging_err);
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

// Execute SET_ROWS entirely on-device: scatter-write new rows directly into ZG DDR
// without pulling the whole KV cache tensor to host.
// SET_ROWS semantics (see ggml.c ggml_set_rows):
//   src[0] = new row data (F32, shape [nc, nr, ne02, ne03])
//   src[1] = row indices  (I64/I32, shape [nr, ne11, ne12, 1])
//   src[2] = KV cache base (view source, ZG buffer)
//   dst    = view of src[2]  (the node itself)
enum ggml_status ggml_fmsh_execute_set_rows_on_device(
        ggml_backend_fmsh_zg330_context * ctx,
        ggml_cgraph * cgraph,
        int node_idx,
        std::unordered_map<const ggml_tensor *, Tensor> & device_tensor_map) {

    ggml_tensor * node = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = node->src[0]; // new row data, F32
    const ggml_tensor * src1 = node->src[1]; // row indices, I64 or I32

    // --- Resolve dst ZG chunk (node is a view of KV cache in ZG DDR) ---
    MemChunk dst_chunk;
    size_t   dst_off = 0;
    if (!ggml_fmsh_is_zg_buffer_tensor(node, &dst_chunk, &dst_off)) {
        // dst not ZG buffer — safe fallback to CPU path
        ggml_fmsh_materialize_if_device(ctx, src0, device_tensor_map);
        ggml_fmsh_materialize_if_device(ctx, src1, device_tensor_map);
        return ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, node_idx);
    }

    // Only F16, BF16, and F32 dst types handled on-device; others fall back.
    if (node->type != GGML_TYPE_F16 && node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) {
        ggml_fmsh_materialize_if_device(ctx, src0, device_tensor_map);
        ggml_fmsh_materialize_if_device(ctx, src1, device_tensor_map);
        return ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, node_idx);
    }

    // --- Resolve src0 (new row data, F32) to a host-accessible pointer ---
    std::vector<char> src0_staging;
    const float * src0_data = nullptr;
    {
        // Priority 1: zg_staging_bufs may already hold fresh F32 data for src0.
        // rope_readback (and similar readbacks) fill staging via ggml_fmsh_write_f32_indexed
        // but do NOT clear zg_staging_bufs afterward, so the buffer is reusable here.
        // This avoids a redundant D2H when src0 was just written back to PLDDR.
        const char * staged_ptr = ggml_fmsh_host_data_for_tensor(ctx, src0);
        const char * raw_ptr    = static_cast<const char *>(src0->data);
        if (staged_ptr != raw_ptr) {
            // Staging hit: fresh host-side F32 copy available, skip D2H.
            src0_data = reinterpret_cast<const float *>(staged_ptr);
            device_tensor_map.erase(src0);
            ggml_fmsh_log_locked(ctx, 1,
                "set_rows_src0_staging_hit bytes=" + std::to_string(ggml_nbytes(src0)));
        } else {
            // Priority 2: src0 is device-resident in device_tensor_map; D2H needed.
            auto it = device_tensor_map.find(src0);
            if (it != device_tensor_map.end()) {
                const size_t bytes = ggml_nbytes(src0);
                src0_staging.resize(bytes);
                it->second.read(src0_staging.data(), 0, bytes);
                device_tensor_map.erase(it);
                src0_data = reinterpret_cast<const float *>(src0_staging.data());
                ggml_fmsh_log_locked(ctx, 1,
                    "set_rows_src0_d2h bytes=" + std::to_string(bytes));
            } else {
                src0_data = reinterpret_cast<const float *>(src0->data);
            }
        }
    }
    if (!src0_data || !src1->data) {
        return GGML_STATUS_FAILED;
    }

    // --- Scatter write each new row into ZG DDR ---
    const int64_t nc   = src0->ne[0]; // elements per row
    const int64_t nr   = src0->ne[1]; // number of new rows
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const size_t dst_elem_bytes  = ggml_type_size(node->type);
    const size_t dst_row_bytes   = static_cast<size_t>(nc) * dst_elem_bytes;

    // Reusable conversion buffer for F32 → F16/BF16.
    std::vector<char> row_buf;
    if (node->type == GGML_TYPE_F16 || node->type == GGML_TYPE_BF16) {
        row_buf.resize(dst_row_bytes);
    }

    int64_t rows_written = 0;
    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            const int64_t i12 = i03 % src1->ne[2];
            const int64_t i11 = i02 % src1->ne[1];
            for (int64_t i = 0; i < nr; ++i) {
                // Row index into KV cache (matches CPU: i10=i, i11, i12).
                int64_t dst_row_idx;
                if (src1->type == GGML_TYPE_I64) {
                    dst_row_idx = *reinterpret_cast<const int64_t *>(
                        static_cast<const char *>(src1->data) +
                        i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]);
                } else {
                    dst_row_idx = static_cast<int64_t>(*reinterpret_cast<const int32_t *>(
                        static_cast<const char *>(src1->data) +
                        i * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2]));
                }

                const float * src_row = reinterpret_cast<const float *>(
                    reinterpret_cast<const char *>(src0_data) +
                    i * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);

                const size_t write_off = dst_off
                    + static_cast<size_t>(dst_row_idx) * static_cast<size_t>(node->nb[1])
                    + static_cast<size_t>(i02) * static_cast<size_t>(node->nb[2])
                    + static_cast<size_t>(i03) * static_cast<size_t>(node->nb[3]);

                if (node->type == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(src_row,
                        reinterpret_cast<ggml_fp16_t *>(row_buf.data()), nc);
                    dst_chunk.write(write_off, row_buf.data(), dst_row_bytes);
                } else if (node->type == GGML_TYPE_BF16) {
                    ggml_fp32_to_bf16_row(src_row,
                        reinterpret_cast<ggml_bf16_t *>(row_buf.data()), nc);
                    dst_chunk.write(write_off, row_buf.data(), dst_row_bytes);
                } else {
                    // F32 dst: write src row directly
                    dst_chunk.write(write_off,
                        const_cast<char *>(reinterpret_cast<const char *>(src_row)), dst_row_bytes);
                }
                ++rows_written;
            }
        }
    }

    ggml_fmsh_log_locked(ctx, 1,
        "set_rows_on_device rows=" + std::to_string(rows_written) +
        " nc=" + std::to_string(nc) +
        " write_bytes=" + std::to_string(
            static_cast<size_t>(rows_written) * dst_row_bytes));

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_backend_fmsh_zg330_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
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
    ggml_fmsh_chain_tensor_id prev_device_output_id;
    std::unordered_map<const ggml_tensor *, Tensor> device_tensor_map;
    std::unordered_set<const ggml_tensor *> chainable_device_tensors;
    std::unordered_map<const ggml_tensor *, int> chainable_device_remaining_uses;
    std::vector<ggml_fmsh_pending_chain_tensor_entry> unresolved_chainable_device_tensors;
    // Staging buffers for ZG buffer tensors that need CPU access this compute pass.
    // Cleared at the top of each graph_compute so redirected t->data pointers from the
    // previous pass are invalidated before new ZG allocations are registered.
    ctx->zg_staging_bufs.clear();
    const bool pending_prev_device_output_valid = ctx->pending_prev_device_output_valid;
    const Tensor pending_prev_device_output = ctx->pending_prev_device_output;
    const ggml_fmsh_chain_tensor_id pending_prev_device_output_id = ctx->pending_prev_device_output_id;
    ctx->pending_prev_device_output_valid = false;
    ctx->pending_prev_device_output = Tensor();
    ctx->pending_prev_device_output_id = ggml_fmsh_chain_tensor_id();

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
        [&](ggml_tensor * consumer, int src_idx, const ggml_tensor * producer) -> bool {
            if (consumer->op == GGML_OP_MUL_MAT && src_idx == 1 &&
                (ggml_fmsh_can_run_mul_mat_zg(consumer) || ggml_fmsh_is_quant_weight_mul_mat(consumer)) &&
                consumer->ne[2] == 1 && consumer->ne[3] == 1) {
                const int64_t cm = consumer->src[1] ? consumer->src[1]->ne[1] : 1;
                if (cm > 1) {
                    int64_t cb = 1;
                    while (cb < cm) cb <<= 1;
                    if (cb > ctx->mul_mat_m_bucket_max) {
                        return false;
                    }
                }
                return true;
            }
            // ROPE can consume device-resident src[0].
            if (consumer->op == GGML_OP_ROPE && src_idx == 0 &&
                consumer->type == GGML_TYPE_F32 && consumer->src[0] != nullptr &&
                consumer->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            // Elementwise ops can consume a device tensor at src[0]. Binary ADD/MUL can also
            // consume a device tensor at src[1]. RMS_NORM uses two-phase PL+scalar-sqrt execution
            // so it cannot use input_override here.
            if (ggml_fmsh_should_run_elementwise_zg(ctx, consumer)) {
                ggml::fmsh::netmake::ElementwiseZgOp ek;
                int64_t er = 0, ec = 0;
                if (!ggml_fmsh_validate_elementwise(consumer, &ek, &er, &ec)) {
                    return false;
                }
                if (ek == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
                    return false;
                }
                if (src_idx == 0) {
                    return true;
                }
                if (src_idx == 1 &&
                    (ek == ggml::fmsh::netmake::ElementwiseZgOp::ADD ||
                     ek == ggml::fmsh::netmake::ElementwiseZgOp::MUL) &&
                    producer != nullptr &&
                    producer->ne[0] == consumer->ne[0] &&
                    ggml_fmsh_node_rows(producer) == ggml_fmsh_node_rows(consumer)) {
                    return true;
                }
            }
            return false;
        };

    auto next_chain_alias =
        [&](const ggml_tensor * t) -> const ggml_tensor * {
            if (t == nullptr) {
                return nullptr;
            }
            if (t->view_src != nullptr) {
                return t->view_src;
            }
            if (ggml_fmsh_is_chain_alias_op(t) && t->src[0] != nullptr) {
                return t->src[0];
            }
            return nullptr;
        };

    auto resolve_chain_source =
        [&](const ggml_tensor * t) -> const ggml_tensor * {
            const ggml_tensor * cur = t;
            while (cur != nullptr) {
                if (cur->op != GGML_OP_NONE &&
                    !ggml_fmsh_is_chain_alias_op(cur)) {
                    return cur;
                }
                const ggml_tensor * next = next_chain_alias(cur);
                if (next == nullptr || next == cur) {
                    break;
                }
                cur = next;
            }
            return cur;
        };

    auto tensor_matches_chain_id =
        [&](const ggml_tensor * cand, const ggml_fmsh_chain_tensor_id & id) -> bool {
            if (cand == nullptr || !id.valid) {
                return false;
            }
            if (cand->op != id.op || reinterpret_cast<uintptr_t>(cand->data) != id.data) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (cand->ne[d] != id.ne[d]) {
                    return false;
                }
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = cand->src[s];
                const ggml_op src_op = src != nullptr ? src->op : GGML_OP_NONE;
                const uintptr_t src_data = reinterpret_cast<uintptr_t>(src != nullptr ? src->data : nullptr);
                if (src_op != id.src_op[s] || src_data != id.src_data[s]) {
                    return false;
                }
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    const int64_t src_ne = src != nullptr ? src->ne[d] : 0;
                    if (src_ne != id.src_ne[s][d]) {
                        return false;
                    }
                }
            }
            return true;
        };

    auto find_cgraph_tensor_by_id =
        [&](const ggml_fmsh_chain_tensor_id & id) -> ggml_tensor * {
            if (!id.valid) {
                return nullptr;
            }

            auto try_match = [&](ggml_tensor * cand) -> ggml_tensor * {
                if (cand == nullptr) {
                    return nullptr;
                }
                if (tensor_matches_chain_id(cand, id)) {
                    return cand;
                }
                ggml_tensor * resolved = const_cast<ggml_tensor *>(resolve_chain_source(cand));
                if (resolved != nullptr && tensor_matches_chain_id(resolved, id)) {
                    return resolved;
                }
                return nullptr;
            };

            for (int j = 0; j < cgraph->n_nodes; ++j) {
                ggml_tensor * node = cgraph->nodes[j];
                if (ggml_tensor * match = try_match(node)) {
                    return match;
                }
                if (node == nullptr) {
                    continue;
                }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (ggml_tensor * match = try_match(node->src[s])) {
                        return match;
                    }
                }
            }
            return nullptr;
        };

    auto clear_prev_device_output = [&]() {
        has_prev_device_output = false;
        prev_device_output = Tensor();
        prev_device_output_node = nullptr;
        prev_device_output_id = ggml_fmsh_chain_tensor_id();
    };

    auto set_prev_device_output = [&](ggml_tensor * owner, const Tensor & tensor) {
        std::string clone_err;
        Tensor stable_tensor = ggml_fmsh_clone_chain_tensor(ctx, tensor, &clone_err);
        if (tensor.defined() && !stable_tensor.defined()) {
            ggml_fmsh_log_locked(ctx, 2, clone_err);
        }
        has_prev_device_output = stable_tensor.defined();
        prev_device_output = has_prev_device_output ? stable_tensor : Tensor();
        prev_device_output_node = owner;
        prev_device_output_id = owner != nullptr ? ggml_fmsh_make_chain_tensor_id(owner)
                                                 : ggml_fmsh_chain_tensor_id();
    };

    if (pending_prev_device_output_valid) {
        ggml_tensor * carried_owner = find_cgraph_tensor_by_id(pending_prev_device_output_id);
        if (pending_prev_device_output.defined()) {
            has_prev_device_output = true;
            prev_device_output = pending_prev_device_output;
            prev_device_output_node = carried_owner;
            prev_device_output_id = pending_prev_device_output_id;
        }
        if (carried_owner != nullptr) {
            ggml_fmsh_log_locked(
                ctx, 1,
                "graph_compute_prev_carry owner_op=" +
                    std::string(ggml_op_name(carried_owner->op)) +
                    " owner_ptr=" +
                    std::to_string(reinterpret_cast<uintptr_t>(carried_owner)) +
                    " owner_data=" +
                    std::to_string(reinterpret_cast<uintptr_t>(carried_owner->data)));
        } else if (pending_prev_device_output.defined()) {
            ggml_fmsh_log_locked(
                ctx, 1,
                "graph_compute_prev_carry_unresolved owner_op=" +
                    std::string(ggml_op_name(pending_prev_device_output_id.op)) +
                    " owner_data=" + std::to_string(pending_prev_device_output_id.data) +
                    " shape=[" + std::to_string(pending_prev_device_output_id.ne[0]) + "," +
                    std::to_string(pending_prev_device_output_id.ne[1]) + "," +
                    std::to_string(pending_prev_device_output_id.ne[2]) + "," +
                    std::to_string(pending_prev_device_output_id.ne[3]) + "]");
        }
    }

    for (const auto & pending : ctx->pending_chainable_device_tensors) {
        if (!pending.id.valid || !pending.tensor.defined()) {
            continue;
        }
        ggml_tensor * carried_owner = find_cgraph_tensor_by_id(pending.id);
        if (carried_owner == nullptr) {
            unresolved_chainable_device_tensors.push_back(pending);
            continue;
        }
        device_tensor_map[carried_owner] = pending.tensor;
        chainable_device_tensors.insert(carried_owner);
        chainable_device_remaining_uses[carried_owner] = pending.remaining_uses;
    }
    ctx->pending_chainable_device_tensors.clear();

    auto same_chain_tensor =
        [&](const ggml_tensor * a, const ggml_tensor * b) -> bool {
            a = resolve_chain_source(a);
            b = resolve_chain_source(b);
            if (a == b) {
                return true;
            }
            if (a == nullptr || b == nullptr) {
                return false;
            }
            if (a->op != b->op ||
                a->data != b->data ||
                !ggml_are_same_shape(a, b) ||
                !ggml_are_same_stride(a, b)) {
                return false;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * as = resolve_chain_source(a->src[s]);
                const ggml_tensor * bs = resolve_chain_source(b->src[s]);
                if (as != bs) {
                    return false;
                }
            }
            return true;
        };

    auto find_device_chain_tensor =
        [&](const ggml_tensor * t, const Tensor ** out_tensor, const ggml_tensor ** out_owner) -> bool {
            const ggml_tensor * cur = t;
            while (cur != nullptr) {
                auto it_dev = device_tensor_map.find(cur);
                if (it_dev != device_tensor_map.end() &&
                    chainable_device_tensors.count(cur) != 0) {
                    if (out_tensor) {
                        *out_tensor = &it_dev->second;
                    }
                    if (out_owner) {
                        *out_owner = cur;
                    }
                    return true;
                }
                const ggml_tensor * next = next_chain_alias(cur);
                if (next == nullptr || next == cur) {
                    break;
                }
                cur = next;
            }
            const ggml_tensor * resolved = resolve_chain_source(t);
            if (resolved != nullptr) {
                for (const ggml_tensor * held : chainable_device_tensors) {
                    auto it_dev = device_tensor_map.find(held);
                    if (it_dev == device_tensor_map.end()) {
                        continue;
                    }
                    if (!same_chain_tensor(resolved, held)) {
                        continue;
                    }
                    if (out_tensor) {
                        *out_tensor = &it_dev->second;
                    }
                    if (out_owner) {
                        *out_owner = held;
                    }
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "chain_lookup_equiv_match want_op=" +
                            std::string(ggml_op_name(resolved->op)) +
                            " held_op=" + std::string(ggml_op_name(held->op)) +
                            " want_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(resolved)) +
                            " held_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(held)));
                    return true;
                }
                for (const auto & pending : unresolved_chainable_device_tensors) {
                    if (!pending.tensor.defined() || !pending.id.valid) {
                        continue;
                    }
                    if (!tensor_matches_chain_id(resolved, pending.id)) {
                        continue;
                    }
                    if (out_tensor) {
                        *out_tensor = &pending.tensor;
                    }
                    if (out_owner) {
                        *out_owner = resolved;
                    }
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "chain_lookup_pending_id_match want_op=" +
                            std::string(ggml_op_name(resolved->op)) +
                            " want_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(resolved)) +
                            " want_data=" + std::to_string(reinterpret_cast<uintptr_t>(resolved->data)));
                    return true;
                }
            }
            return false;
        };

    auto find_prev_device_chain_tensor =
        [&](const ggml_tensor * t, const Tensor ** out_tensor, const ggml_tensor ** out_owner) -> bool {
            if (!has_prev_device_output) {
                return false;
            }
            const ggml_tensor * resolved = resolve_chain_source(t);
            if (resolved == nullptr) {
                return false;
            }
            const bool match =
                (prev_device_output_node != nullptr && same_chain_tensor(resolved, prev_device_output_node)) ||
                (prev_device_output_id.valid && tensor_matches_chain_id(resolved, prev_device_output_id));
            if (!match) {
                return false;
            }
            if (out_tensor) {
                *out_tensor = &prev_device_output;
            }
            if (out_owner) {
                *out_owner = prev_device_output_node != nullptr ? prev_device_output_node : resolved;
            }
            return true;
        };

    auto count_future_device_consumers =
        [&](int from_idx, const ggml_tensor * producer) -> int {
            producer = resolve_chain_source(producer);
            if (producer == nullptr) {
                return 0;
            }
            int count = 0;
            for (int j = from_idx + 1; j < cgraph->n_nodes; ++j) {
                ggml_tensor * cand = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const ggml_tensor * cand_src = resolve_chain_source(cand->src[s]);
                    const bool same = same_chain_tensor(cand_src, producer);
                    const bool can_consume = same && can_consume_from_device(cand, s, cand_src);
                    if (producer->op == GGML_OP_MUL &&
                        producer->ne[0] == 1024 &&
                        ggml_fmsh_node_rows(producer) == 1 &&
                        cand->op == GGML_OP_MUL_MAT &&
                        ggml_fmsh_is_quant_weight_mul_mat(cand)) {
                        ggml_fmsh_log_locked(
                            ctx, 1,
                            "future_device_probe from_idx=" + std::to_string(from_idx) +
                                " cand_idx=" + std::to_string(j) +
                                " src_idx=" + std::to_string(s) +
                                " producer_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                                " cand_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(cand)) +
                                " cand_src_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(cand_src)) +
                                " same=" + std::to_string(same ? 1 : 0) +
                                " can_consume=" + std::to_string(can_consume ? 1 : 0));
                    }
                    if (can_consume) {
                        if (producer->op == GGML_OP_MUL &&
                            cand->op == GGML_OP_MUL_MAT &&
                            ggml_fmsh_is_quant_weight_mul_mat(cand)) {
                            ggml_fmsh_log_locked(
                                ctx, 1,
                                "future_consumer_equiv_match producer_ptr=" +
                                    std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                                    " cand_src_ptr=" +
                                    std::to_string(reinterpret_cast<uintptr_t>(cand_src)) +
                                    " consumer_ptr=" +
                                    std::to_string(reinterpret_cast<uintptr_t>(cand)));
                        }
                        ++count;
                    }
                }
            }
            return count;
        };

    auto find_future_qweight_consumer_src1 =
        [&](int from_idx, const ggml_tensor * producer, ggml_tensor ** out_node) -> bool {
            producer = resolve_chain_source(producer);
            if (producer == nullptr) {
                return false;
            }
            for (int j = from_idx + 1; j < cgraph->n_nodes; ++j) {
                ggml_tensor * cand = cgraph->nodes[j];
                if (cand->op != GGML_OP_MUL_MAT || !ggml_fmsh_is_quant_weight_mul_mat(cand)) {
                    continue;
                }
                if (cand->ne[2] != 1 || cand->ne[3] != 1) {
                    continue;
                }
                const bool same = same_chain_tensor(cand->src[1], producer);
                if (producer->op == GGML_OP_MUL &&
                    producer->ne[0] == 1024 &&
                    ggml_fmsh_node_rows(producer) == 1) {
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "future_qweight_probe from_idx=" + std::to_string(from_idx) +
                            " cand_idx=" + std::to_string(j) +
                            " producer_ptr=" +
                            std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                            " cand_ptr=" +
                            std::to_string(reinterpret_cast<uintptr_t>(cand)) +
                            " cand_src1_ptr=" +
                            std::to_string(reinterpret_cast<uintptr_t>(cand->src[1])) +
                            " cand_src1_resolved=" +
                            std::to_string(reinterpret_cast<uintptr_t>(resolve_chain_source(cand->src[1]))) +
                            " same=" + std::to_string(same ? 1 : 0));
                }
                if (!same) {
                    continue;
                }
                if (out_node != nullptr) {
                    *out_node = cand;
                }
                return true;
            }
            return false;
        };

    auto count_future_consumers_any =
        [&](int from_idx, const ggml_tensor * producer) -> int {
            producer = resolve_chain_source(producer);
            if (producer == nullptr) {
                return 0;
            }
            int count = 0;
            for (int j = from_idx + 1; j < cgraph->n_nodes; ++j) {
                ggml_tensor * cand = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const ggml_tensor * cand_src = resolve_chain_source(cand->src[s]);
                    const bool same = same_chain_tensor(cand_src, producer);
                    if (producer->op == GGML_OP_MUL &&
                        producer->ne[0] == 1024 &&
                        ggml_fmsh_node_rows(producer) == 1 &&
                        cand->op == GGML_OP_MUL_MAT &&
                        ggml_fmsh_is_quant_weight_mul_mat(cand)) {
                        ggml_fmsh_log_locked(
                            ctx, 1,
                            "future_any_probe from_idx=" + std::to_string(from_idx) +
                                " cand_idx=" + std::to_string(j) +
                                " src_idx=" + std::to_string(s) +
                                " producer_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                                " cand_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(cand)) +
                                " cand_src_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(cand_src)) +
                                " same=" + std::to_string(same ? 1 : 0));
                    }
                    if (same) {
                        ++count;
                    }
                }
            }
            return count;
        };

    auto log_future_device_consumers =
        [&](int from_idx, const ggml_tensor * producer, const char * tag) {
            producer = resolve_chain_source(producer);
            if (!producer) return;
            int count = 0;
            for (int j = from_idx + 1; j < cgraph->n_nodes; ++j) {
                ggml_tensor * cand = cgraph->nodes[j];
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    const ggml_tensor * cand_src = resolve_chain_source(cand->src[s]);
                    if (same_chain_tensor(cand_src, producer)) {
                        const bool can_consume = can_consume_from_device(cand, s, cand_src);
                        ggml_fmsh_log_locked(
                            ctx, 1,
                            std::string("future_consumer tag=") + tag +
                                " producer=" + ggml_op_name(producer->op) +
                                " consumer=" + ggml_op_name(cand->op) +
                                " src_idx=" + std::to_string(s) +
                                " can_consume=" + std::to_string(can_consume ? 1 : 0) +
                                " node_idx=" + std::to_string(j));
                        if (can_consume) {
                            ++count;
                        }
                    }
                }
            }
            ggml_fmsh_log_locked(
                ctx, 1,
                std::string("future_consumer_summary tag=") + tag +
                    " producer=" + ggml_op_name(producer->op) +
                    " count=" + std::to_string(count));
        };

    auto drop_chainable_tensor = [&](const ggml_tensor * t) {
        t = resolve_chain_source(t);
        if (!t) return;
        device_tensor_map.erase(t);
        chainable_device_tensors.erase(t);
        chainable_device_remaining_uses.erase(t);
        const ggml_fmsh_chain_tensor_id id = ggml_fmsh_make_chain_tensor_id(t);
        unresolved_chainable_device_tensors.erase(
            std::remove_if(
                unresolved_chainable_device_tensors.begin(),
                unresolved_chainable_device_tensors.end(),
                [&](const ggml_fmsh_pending_chain_tensor_entry & entry) {
                    return entry.id == id;
                }),
            unresolved_chainable_device_tensors.end());
    };

    auto should_retain_cross_graph_chain =
        [&](const ggml_tensor * producer) -> bool {
            producer = resolve_chain_source(producer);
            if (producer == nullptr) {
                return false;
            }
            // Decode often splits the final per-token MUL activation and its qweight
            // projections across adjacent graph_compute calls. Keep only that narrow
            // producer class resident on device; broader retention breaks session-shared
            // compiled row shapes such as RMS_NORM_SPLIT post outputs.
            return producer->op == GGML_OP_MUL &&
                   producer->type == GGML_TYPE_F32 &&
                   producer->ne[2] == 1 &&
                   producer->ne[3] == 1 &&
                   ggml_fmsh_node_rows(producer) == 1;
        };

    auto retain_chainable_tensor = [&](int from_idx, const ggml_tensor * producer, const Tensor & device_tensor) -> int {
        producer = resolve_chain_source(producer);
        if (!producer || !device_tensor.defined()) {
            return 0;
        }
        std::string clone_err;
        Tensor stable_tensor = ggml_fmsh_clone_chain_tensor(ctx, device_tensor, &clone_err);
        if (!stable_tensor.defined()) {
            ggml_fmsh_log_locked(ctx, 2, clone_err);
            drop_chainable_tensor(producer);
            return 0;
        }
        const int future_uses = count_future_device_consumers(from_idx, producer);
        if (future_uses > 0) {
            device_tensor_map[producer] = stable_tensor;
            chainable_device_tensors.insert(producer);
            chainable_device_remaining_uses[producer] = future_uses;
            if (producer->op == GGML_OP_CPY) {
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "chain_retain producer=CPY future_uses=" + std::to_string(future_uses) +
                        " shape=[" + std::to_string(producer->ne[0]) + "," +
                        std::to_string(producer->ne[1]) + "," +
                        std::to_string(producer->ne[2]) + "," +
                        std::to_string(producer->ne[3]) + "]");
            }
        } else if (should_retain_cross_graph_chain(producer)) {
            device_tensor_map[producer] = stable_tensor;
            chainable_device_tensors.insert(producer);
            chainable_device_remaining_uses[producer] = -1;
            ggml_fmsh_log_locked(
                ctx, 1,
                "chain_retain_cross_graph producer=" +
                    std::string(ggml_op_name(producer->op)) +
                    " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                    " data=" + std::to_string(reinterpret_cast<uintptr_t>(producer->data)) +
                    " shape=[" + std::to_string(producer->ne[0]) + "," +
                    std::to_string(producer->ne[1]) + "," +
                    std::to_string(producer->ne[2]) + "," +
                    std::to_string(producer->ne[3]) + "]");
        } else {
            drop_chainable_tensor(producer);
        }
        return future_uses;
    };

    auto release_chainable_consumer = [&](int from_idx, const ggml_tensor * producer) {
        producer = resolve_chain_source(producer);
        if (!producer) return;
        auto it = chainable_device_remaining_uses.find(producer);
        const int remaining_local_uses = count_future_device_consumers(from_idx, producer);
        if (it == chainable_device_remaining_uses.end()) {
            const ggml_fmsh_chain_tensor_id id = ggml_fmsh_make_chain_tensor_id(producer);
            auto it_unresolved = std::find_if(
                unresolved_chainable_device_tensors.begin(),
                unresolved_chainable_device_tensors.end(),
                [&](const ggml_fmsh_pending_chain_tensor_entry & entry) {
                    return entry.id == id;
                });
            if (it_unresolved == unresolved_chainable_device_tensors.end()) {
                drop_chainable_tensor(producer);
                return;
            }
            if (remaining_local_uses > 0) {
                it_unresolved->remaining_uses = remaining_local_uses;
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "chain_release_keep_unresolved producer=" +
                        std::string(ggml_op_name(producer->op)) +
                        " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                        " remaining_local_uses=" + std::to_string(remaining_local_uses));
                return;
            }
            if (should_retain_cross_graph_chain(producer)) {
                it_unresolved->remaining_uses = -1;
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "chain_release_hold_cross_graph_unresolved producer=" +
                        std::string(ggml_op_name(producer->op)) +
                        " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)));
                return;
            }
            unresolved_chainable_device_tensors.erase(it_unresolved);
            ggml_fmsh_log_locked(
                ctx, 1,
                "chain_release_drop_unresolved producer=" +
                    std::string(ggml_op_name(producer->op)) +
                    " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)));
            return;
        }
        if (remaining_local_uses > 0) {
            it->second = remaining_local_uses;
            ggml_fmsh_log_locked(
                ctx, 1,
                "chain_release_keep producer=" +
                    std::string(ggml_op_name(producer->op)) +
                    " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)) +
                    " remaining_local_uses=" + std::to_string(remaining_local_uses));
            return;
        }
        if (should_retain_cross_graph_chain(producer)) {
            it->second = -1;
            ggml_fmsh_log_locked(
                ctx, 1,
                "chain_release_hold_cross_graph producer=" +
                    std::string(ggml_op_name(producer->op)) +
                    " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)));
            return;
        }
        ggml_fmsh_log_locked(
            ctx, 1,
            "chain_release_drop producer=" +
                std::string(ggml_op_name(producer->op)) +
                " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(producer)));
        if (it->second <= 0) {
            drop_chainable_tensor(producer);
        }
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
                if (has_prev_device_output && same_chain_tensor(prev_device_output_node, node)) {
                    clear_prev_device_output();
                }
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
                if (has_prev_device_output && same_chain_tensor(prev_device_output_node, node)) {
                    clear_prev_device_output();
                }
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
                    return ggml_fmsh_debug_backup_tensor(ctx, s, debug_saved, debug_seen, &dbg_err);
                };
                if (!ggml_fmsh_debug_backup_tensor(ctx, node, debug_saved, debug_seen, &dbg_err) ||
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
                if (!ggml_fmsh_tensor_snapshot_f32(ctx, node, cpu_ref_out, &snap_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=FLASH_ATTN_EXT reason=" + snap_err);
                }
                if (!debug_saved.empty()) {
                    std::string restore_err;
                    if (!ggml_fmsh_debug_restore_tensors(ctx, debug_saved, &restore_err)) {
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
                if (has_prev_device_output && same_chain_tensor(prev_device_output_node, node)) {
                    clear_prev_device_output();
                }
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
            clear_prev_device_output();
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

            // Quantized weight matmul (Q4_K/Q6_K projections, FFN, lm_head): dequant→BF16
            // NPU path. Best-effort — any failure falls back to CPU, never fatal.
            if (ctx->offload_quant_weights && ggml_fmsh_is_quant_weight_mul_mat(node)) {
                bool qw_created = false;
                std::string qw_err;
                ggml_fmsh_zg330_qweight_session_entry * qw_entry =
                    ggml_fmsh_get_or_create_qweight_session(ctx, node, &qw_created, &qw_err);
                bool qw_ok = (qw_entry != nullptr);
                if (qw_ok) {
                    const Tensor * qw_chained_input_ptr = nullptr;
                    const ggml_tensor * qw_chained_input_src = nullptr;
                    const ggml_tensor * qw_src1_resolved = resolve_chain_source(node->src[1]);
                    if (!ctx->disable_device_input_chain &&
                        find_device_chain_tensor(node->src[1], &qw_chained_input_ptr, &qw_chained_input_src) &&
                        node->ne[2] == 1 && node->ne[3] == 1) {
                    } else if (!ctx->disable_device_input_chain &&
                               find_prev_device_chain_tensor(node->src[1], &qw_chained_input_ptr, &qw_chained_input_src) &&
                               node->ne[2] == 1 && node->ne[3] == 1) {
                    }
                    if (node->ne[2] == 1 && node->ne[3] == 1) {
                        ggml_fmsh_log_locked(
                            ctx, 1,
                            "qweight_chain_probe src1_op=" +
                                std::string(node->src[1] ? ggml_op_name(node->src[1]->op) : "null") +
                                " resolved_op=" +
                                std::string(qw_src1_resolved ? ggml_op_name(qw_src1_resolved->op) : "null") +
                                " view_src_op=" +
                                std::string((node->src[1] && node->src[1]->view_src) ? ggml_op_name(node->src[1]->view_src->op) : "null") +
                                " src1_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(node->src[1])) +
                                " resolved_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(qw_src1_resolved)) +
                                " src1_data=" +
                                std::to_string(reinterpret_cast<uintptr_t>(node->src[1] ? node->src[1]->data : nullptr)) +
                                " prev_op=" +
                                std::string(
                                    prev_device_output_node ? ggml_op_name(prev_device_output_node->op)
                                                            : ggml_op_name(prev_device_output_id.op)) +
                                " prev_ptr=" +
                                std::to_string(reinterpret_cast<uintptr_t>(prev_device_output_node)) +
                                " prev_data=" +
                                std::to_string(
                                    prev_device_output_node != nullptr
                                        ? reinterpret_cast<uintptr_t>(prev_device_output_node->data)
                                        : prev_device_output_id.data) +
                                " has_prev=" + std::to_string(has_prev_device_output ? 1 : 0) +
                                " chain_hit=" + std::to_string(qw_chained_input_ptr != nullptr ? 1 : 0));
                    }
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        std::string(qw_created ? "session_create" : "session_hit") +
                        " op=MUL_MAT kind=qweight shape=[" + std::to_string(node->ne[0]) + "," +
                        std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," +
                        std::to_string(node->ne[3]) + "]" +
                        " src0_type=" + std::to_string(static_cast<int>(node->src[0]->type)) +
                        " m_bucket=" + std::to_string(qw_entry->m_bucket) +
                        " k=" + std::to_string(qw_entry->k) +
                        " n=" + std::to_string(qw_entry->n_full) +
                        " chunks=" + std::to_string(qw_entry->subnets.size()) +
                        " net=" + (qw_entry->subnets.empty() ? std::string("none") : qw_entry->subnets[0].bundle.net_name) +
                        " net_cache=" + std::string((!qw_entry->subnets.empty() && qw_entry->subnets[0].bundle.ram_cache_hit) ? "HIT" : "MISS") +
                        " compile_now=" + std::string((qw_created && !qw_entry->subnets.empty() && qw_entry->subnets[0].bundle.compiled_now) ? "YES" : "NO"));
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
                    std::vector<float> qw_cpu_ref;
                    if (ctx->debug_compare == 1) {
                        const bool has_device_src0 =
                            node->src[0] != nullptr && device_tensor_map.find(node->src[0]) != device_tensor_map.end();
                        const bool has_device_src1 =
                            node->src[1] != nullptr && device_tensor_map.find(node->src[1]) != device_tensor_map.end();
                        const bool has_device_src2 =
                            node->src[2] != nullptr && device_tensor_map.find(node->src[2]) != device_tensor_map.end();
                        if (!ctx->disable_device_input_chain &&
                            (qw_chained_input_ptr != nullptr || has_device_src0 || has_device_src1 || has_device_src2)) {
                            ggml_fmsh_log_locked(
                                ctx, 1,
                                "debug_compare_skip op=" + std::string(ggml_op_name(node->op)) +
                                " reason=device_input_chain");
                        } else {
                            std::vector<ggml_fmsh_debug_saved_tensor> dbg_saved;
                            std::unordered_set<ggml_tensor *> dbg_seen;
                            std::string dbg_err;
                            if (ggml_fmsh_debug_backup_tensor(ctx, node, dbg_saved, dbg_seen, &dbg_err)) {
                                const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                                if (cpu_st == GGML_STATUS_SUCCESS) {
                                    std::string snap_err;
                                    ggml_fmsh_tensor_snapshot_f32(ctx, node, qw_cpu_ref, &snap_err);
                                }
                                std::string restore_err;
                                ggml_fmsh_debug_restore_tensors(ctx, dbg_saved, &restore_err);
                            }
                        }
                    }
#endif
                    qw_ok = ggml_fmsh_execute_qweight_mul_mat(
                        ctx, node, qw_entry, qw_chained_input_ptr, &qw_err);
                    if (qw_ok) {
                        ctx->mul_mat_offloaded++;
                        if (qw_chained_input_ptr != nullptr) {
                            ctx->mul_mat_device_input_chain++;
                            ggml_fmsh_log_locked(
                                ctx, 0,
                                "dispatch op=MUL_MAT kind=qweight chained_input=1");
                        }
                        if (qw_chained_input_src != nullptr) {
                            release_chainable_consumer(i, qw_chained_input_src);
                            if (has_prev_device_output && prev_device_output_node == qw_chained_input_src) {
                                clear_prev_device_output();
                            }
                        }
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
                        if (!qw_cpu_ref.empty()) {
                            ggml_fmsh_debug_compare_and_log(ctx, node, qw_cpu_ref, i);
                        }
#endif
                    }
                }
                if (!qw_ok) {
                    ggml_fmsh_log_locked(ctx, 2, "fallback op=MUL_MAT kind=qweight reason=" + qw_err);
                    ctx->mul_mat_fallback++;
                    ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                    ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                    const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                    if (st != GGML_STATUS_SUCCESS) {
                        return st;
                    }
                }
                clear_prev_device_output();
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = qw_ok;
                continue;
            }

            if (!ggml_fmsh_can_run_mul_mat_zg(node)) {
                // F32 matmul we can't handle (layout/dtype) — graceful CPU fallback.
                ggml_fmsh_log_locked(
                    ctx, 2,
                    "fallback op=MUL_MAT reason=unsupported_layout_or_dtype dst_type=" +
                    std::to_string(static_cast<int>(node->type)) +
                    " src0_type=" + std::to_string(static_cast<int>(node->src[0]->type)) +
                    " src1_type=" + std::to_string(static_cast<int>(node->src[1]->type)));
                ctx->mul_mat_fallback++;
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
                const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                if (st != GGML_STATUS_SUCCESS) {
                    return st;
                }
                clear_prev_device_output();
                device_tensor_map.erase(node);
                last_dispatched_valid = true;
                last_dispatched_zg = false;
                continue;
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
            if (!ctx->disable_device_input_chain &&
                find_device_chain_tensor(node->src[1], &chained_input_ptr, &chained_input_src) &&
                node->ne[2] == 1 && node->ne[3] == 1) {
            } else if (!ctx->disable_device_input_chain &&
                       find_prev_device_chain_tensor(node->src[1], &chained_input_ptr, &chained_input_src) &&
                       node->ne[2] == 1 && node->ne[3] == 1) {
            }
            const int future_device_consumers =
                (!ctx->disable_device_input_chain && node->ne[2] == 1 && node->ne[3] == 1)
                    ? count_future_device_consumers(i, node)
                    : 0;
            const int future_consumers_any =
                (!ctx->disable_device_input_chain && node->ne[2] == 1 && node->ne[3] == 1)
                    ? count_future_consumers_any(i, node)
                    : 0;
            const bool keep_device_output_only = future_device_consumers > 0;
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            std::vector<float> cpu_ref_out;
            if (ctx->debug_compare == 1) {
                const bool has_device_src0 = node->src[0] != nullptr && device_tensor_map.find(node->src[0]) != device_tensor_map.end();
                const bool has_device_src1 = node->src[1] != nullptr && device_tensor_map.find(node->src[1]) != device_tensor_map.end();
                const bool has_device_src2 = node->src[2] != nullptr && device_tensor_map.find(node->src[2]) != device_tensor_map.end();
                if (!ctx->disable_device_input_chain && (has_device_src0 || has_device_src1 || has_device_src2)) {
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
                    return ggml_fmsh_debug_backup_tensor(ctx, s, debug_saved, debug_seen, &dbg_err);
                };
                if (!ggml_fmsh_debug_backup_tensor(ctx, node, debug_saved, debug_seen, &dbg_err) ||
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
                if (!ggml_fmsh_tensor_snapshot_f32(ctx, node, cpu_ref_out, &snap_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=MUL_MAT reason=" + snap_err);
                }
                if (!debug_saved.empty()) {
                    std::string restore_err;
                    if (!ggml_fmsh_debug_restore_tensors(ctx, debug_saved, &restore_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "debug_compare_restore_failed op=MUL_MAT reason=" + restore_err);
                        return GGML_STATUS_FAILED;
                    }
                }
            }
            }
            const bool keep_device_output_effective = keep_device_output_only;
            const bool write_host_output_effective =
                !keep_device_output_only ||
                !cpu_ref_out.empty() ||
                future_consumers_any > future_device_consumers;
#else
            const bool keep_device_output_effective = keep_device_output_only;
            const bool write_host_output_effective =
                !keep_device_output_only ||
                future_consumers_any > future_device_consumers;
#endif
            Tensor chained_output;
            if (!ggml_fmsh_execute_mul_mat(
                    ctx, node, entry, chained_input_ptr,
                    keep_device_output_effective, write_host_output_effective,
                    &chained_output, &err)) {
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
                    release_chainable_consumer(i, chained_input_src);
                }
            }
            if (!ctx->disable_device_input_chain && node->ne[2] == 1 && node->ne[3] == 1) {
                const ggml_tensor * mm_chain_owner = resolve_chain_source(node);
                set_prev_device_output(const_cast<ggml_tensor *>(mm_chain_owner), chained_output);
                if (keep_device_output_effective) {
                    const int retained_uses = retain_chainable_tensor(i, node, chained_output);
                    ggml_fmsh_log_locked(
                        ctx, 0,
                        "dispatch op=MUL_MAT keep_device_output=1 future_device_consumers=" +
                            std::to_string(retained_uses));
                } else {
                    drop_chainable_tensor(node);
                }
            } else {
                clear_prev_device_output();
                drop_chainable_tensor(node);
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
                clear_prev_device_output();
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
                clear_prev_device_output();
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
            clear_prev_device_output();
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
                if (has_prev_device_output && same_chain_tensor(prev_device_output_node, node)) {
                    clear_prev_device_output();
                }
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

            // ── RMS_NORM: custom_op_hub single-pass path ─────────────────────
            if (ek == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM && !ctx->rms_norm_native_sqrt) {
                if (ctx->rms_norm_custom_op &&
                    ggml_fmsh_rmsnorm_customop_cols_ok(ctx, ecols)) {

                    // Input chaining: check for device-resident src[0].
                    const Tensor * cop_input_override = nullptr;
                    const ggml_tensor * cop_input_src = nullptr;
                    if (!ctx->disable_device_input_chain && node->src[0] != nullptr) {
                        if (find_device_chain_tensor(node->src[0], &cop_input_override, &cop_input_src)) {
                        } else if (find_prev_device_chain_tensor(node->src[0], &cop_input_override, &cop_input_src)) {
                        }
                    }

                    // Output chaining: check future consumers.
                    const int cop_future_device = !ctx->disable_device_input_chain
                        ? count_future_device_consumers(i, node) : 0;
                    const int cop_future_any = !ctx->disable_device_input_chain
                        ? count_future_consumers_any(i, node) : 0;
                    const bool cop_keep_device = !ctx->disable_device_input_chain && cop_future_device > 0;
                    const bool cop_write_host = !cop_keep_device ||
                        cop_future_any > cop_future_device;

                    // Evict stale tensors sharing the custom-op output chunk.
                    auto & cop_cache_ptr = ctx->rmsnorm_customop_cache[0];
                    if (cop_cache_ptr && cop_cache_ptr->output_chunk.defined()) {
                        for (auto it_map = device_tensor_map.begin(); it_map != device_tensor_map.end();) {
                            if (it_map->second.defined() && it_map->second.chunk() == cop_cache_ptr->output_chunk) {
                                const ggml_tensor * held = it_map->first;
                                if (held && held->data) {
                                    const size_t hb = ggml_nbytes(held);
                                    if (hb > 0) ggml_fmsh_evict_to_host(ctx, held, it_map->second);
                                }
                                if (has_prev_device_output && prev_device_output_node == held) {
                                    clear_prev_device_output();
                                }
                                chainable_device_tensors.erase(held);
                                chainable_device_remaining_uses.erase(held);
                                it_map = device_tensor_map.erase(it_map);
                            } else { ++it_map; }
                        }
                    }

                    // Materialize src[0] only if no input chain.
                    if (cop_input_override == nullptr) {
                        ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
                    }

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
                    std::vector<float> cop_cpu_ref;
                    if (ctx->debug_compare) {
                        std::vector<ggml_fmsh_debug_saved_tensor> cop_debug_saved;
                        std::unordered_set<ggml_tensor *> cop_debug_seen;
                        std::string cop_dbg_err;
                        const auto cop_backup_if_alias = [&](ggml_tensor * s) -> bool {
                            if (s == nullptr || s->data == nullptr || node->data == nullptr) return true;
                            if (s != node && s->data != node->data) return true;
                            return ggml_fmsh_debug_backup_tensor(ctx, s, cop_debug_saved, cop_debug_seen, &cop_dbg_err);
                        };
                        if (!ggml_fmsh_debug_backup_tensor(ctx, node, cop_debug_saved, cop_debug_seen, &cop_dbg_err) ||
                            !cop_backup_if_alias(node->src[0])) {
                            ggml_fmsh_log_locked(ctx, 2, "debug_compare_backup_failed op=RMS_NORM_CUSTOMOP reason=" + cop_dbg_err);
                        }
                        const enum ggml_status cpu_st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                        if (cpu_st != GGML_STATUS_SUCCESS) { return cpu_st; }
                        std::string cop_snap_err;
                        if (!ggml_fmsh_tensor_snapshot_f32(ctx, node, cop_cpu_ref, &cop_snap_err)) {
                            ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=RMS_NORM_CUSTOMOP reason=" + cop_snap_err);
                        }
                        if (!cop_debug_saved.empty()) {
                            std::string cop_restore_err;
                            if (!ggml_fmsh_debug_restore_tensors(ctx, cop_debug_saved, &cop_restore_err)) {
                                ggml_fmsh_log_locked(ctx, 2, "debug_compare_restore_failed op=RMS_NORM_CUSTOMOP reason=" + cop_restore_err);
                                return GGML_STATUS_FAILED;
                            }
                        }
                    }
#endif

                    Tensor cop_chained_output;
                    std::string cop_err;
                    if (!ggml_fmsh_execute_rmsnorm_customop(ctx, node, erows, ecols,
                            cop_input_override, cop_keep_device, cop_write_host,
                            &cop_chained_output, &cop_err)) {
                        ggml_fmsh_log_locked(ctx, 3, "fatal op=RMS_NORM_CUSTOMOP reason=" + cop_err);
                        return GGML_STATUS_FAILED;
                    }

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
                    if (!cop_cpu_ref.empty()) {
                        ggml_fmsh_debug_compare_and_log(ctx, node, cop_cpu_ref, i);
                    }
#endif

                    // Consume input chain.
                    if (cop_input_src != nullptr) {
                        release_chainable_consumer(i, cop_input_src);
                        if (has_prev_device_output && prev_device_output_node == cop_input_src) {
                            clear_prev_device_output();
                        }
                    }

                    // Register output chain.
                    if (cop_chained_output.defined()) {
                        set_prev_device_output(
                            const_cast<ggml_tensor *>(resolve_chain_source(node)),
                            cop_chained_output);
                    } else {
                        clear_prev_device_output();
                    }
                    if (cop_keep_device && cop_chained_output.defined()) {
                        const int retained = retain_chainable_tensor(i, node, cop_chained_output);
                        ggml_fmsh_log_locked(ctx, 0,
                            "rmsnorm_customop keep_device_output=1 future_device_consumers=" +
                            std::to_string(retained));
                    } else {
                        drop_chainable_tensor(node);
                    }
                    last_dispatched_valid = true;
                    last_dispatched_zg = true;
                    continue;
                }

                // ── RMS_NORM two-phase split path (no Sqrt on PL) ────────────
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
                    if (!ctx->disable_device_input_chain && node->src[0] != nullptr) {
                        if (find_device_chain_tensor(node->src[0], &split_input_override, &split_input_src)) {
                        } else if (find_prev_device_chain_tensor(node->src[0], &split_input_override, &split_input_src)) {
                        }
                    }

                    // Determine if next consumer (MUL) can consume device output.
                    const int split_future_device_consumers =
                        !ctx->disable_device_input_chain ? count_future_device_consumers(i, node) : 0;
                    const int split_future_consumers_any =
                        !ctx->disable_device_input_chain ? count_future_consumers_any(i, node) : 0;
                    const bool split_keep_device =
                        !ctx->disable_device_input_chain &&
                        split_entry->post_output_chunk.defined() &&
                        split_future_device_consumers > 0;
                    const bool split_write_host_output =
                        !split_keep_device ||
                        split_future_consumers_any > split_future_device_consumers;

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
                                        clear_prev_device_output();
                                    }
                                chainable_device_tensors.erase(held);
                                chainable_device_remaining_uses.erase(held);
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
                            split_input_override, split_keep_device, split_write_host_output,
                            &split_chained_output, &split_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "fallback op=RMS_NORM reason=" + split_err);
                        if (ctx->strict_mode) { return GGML_STATUS_FAILED; }
                        const enum ggml_status st = ggml_fmsh_compute_cpu_node_with_zg_redirect(ctx, cgraph, i);
                        if (st != GGML_STATUS_SUCCESS) { return st; }
                        clear_prev_device_output();
                    } else {
                        if (split_keep_device) {
                            log_future_device_consumers(i, node, "rmsnorm_split");
                        }
                        ggml_fmsh_accumulate_profile(ctx, node, split_entry->session_post,
                            std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0_s).count());
                        // Remove consumed input from device map.
                        if (split_input_src) {
                            release_chainable_consumer(i, split_input_src);
                            if (has_prev_device_output && prev_device_output_node == split_input_src) {
                                clear_prev_device_output();
                            }
                        }
                        if (split_chained_output.defined()) {
                            set_prev_device_output(
                                const_cast<ggml_tensor *>(resolve_chain_source(node)),
                                split_chained_output);
                        } else {
                            clear_prev_device_output();
                        }
                        if (split_keep_device && split_chained_output.defined()) {
                            const int retained_uses = retain_chainable_tensor(i, node, split_chained_output);
                            ggml_fmsh_log_locked(
                                ctx, 0,
                                "rmsnorm_split keep_device_output=1 future_device_consumers=" +
                                    std::to_string(retained_uses));
                        } else {
                            drop_chainable_tensor(node);
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
            if (!ctx->disable_device_input_chain)
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
                            if (find_device_chain_tensor(
                                    fchain.nodes[0]->src[0], &fchain_input_ptr, &fchain_input_src)) {
                            } else if (find_prev_device_chain_tensor(
                                           fchain.nodes[0]->src[0], &fchain_input_ptr, &fchain_input_src)) {
                            }
                        }

                        // Determine whether to keep chain output on device.
                        const int fchain_last_idx = fchain.cgraph_indices.back();
                        const int f_future_device_consumers =
                            count_future_device_consumers(fchain_last_idx, fchain.nodes.back());
                        const int f_future_consumers_any =
                            count_future_consumers_any(fchain_last_idx, fchain.nodes.back());
                        const bool fkeep_device = f_future_device_consumers > 0;
                        const bool fwrite_host_output =
                            !fkeep_device || f_future_consumers_any > f_future_device_consumers;

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
                                            ggml_fmsh_evict_to_host(ctx, held, it_map->second);
                                        }
                                    }
                                    if (has_prev_device_output &&
                                        prev_device_output_node == held) {
                                        clear_prev_device_output();
                                    }
                                    chainable_device_tensors.erase(held);
                                    chainable_device_remaining_uses.erase(held);
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
                            if (!ggml_fmsh_is_cpy_dup_op(fchain.nodes[fk])) {
                                ggml_fmsh_materialize_if_device(ctx, fchain.nodes[fk]->src[1], device_tensor_map);
                            }
                        }
                        Tensor fchained_output;
                        if (ggml_fmsh_execute_fused_ew(
                                ctx, fchain, fentry, fchain_input_ptr,
                                fkeep_device, fwrite_host_output, &fchained_output, &ferr)) {
                            // Consume PLDDR input.
                            if (fchain_input_src != nullptr) {
                                release_chainable_consumer(fchain_last_idx, fchain_input_src);
                                if (has_prev_device_output &&
                                    prev_device_output_node == fchain_input_src) {
                                    clear_prev_device_output();
                                }
                            }

                            if (fchained_output.defined()) {
                                set_prev_device_output(
                                    const_cast<ggml_tensor *>(resolve_chain_source(fchain.nodes.back())),
                                    fchained_output);
                            } else {
                                clear_prev_device_output();
                            }
                            if (fkeep_device && fchained_output.defined()) {
                                retain_chainable_tensor(fchain_last_idx, fchain.nodes.back(), fchained_output);
                            } else {
                                drop_chainable_tensor(fchain.nodes.back());
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
                if (has_prev_device_output && same_chain_tensor(prev_device_output_node, node)) {
                    clear_prev_device_output();
                }
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
            int ew_chained_input_src_idx = -1;
            if (!ctx->disable_device_input_chain &&
                ek != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
                auto try_pick_chain_input = [&](const ggml_tensor * src, int src_idx) -> bool {
                    if (src == nullptr || !can_consume_from_device(node, src_idx, resolve_chain_source(src))) {
                        return false;
                    }
                    const Tensor * device_chain_tensor = nullptr;
                    const ggml_tensor * device_chain_owner = nullptr;
                    if (find_device_chain_tensor(src, &device_chain_tensor, &device_chain_owner)) {
                        chained_input_ptr = device_chain_tensor;
                        ew_chained_input_src = device_chain_owner;
                        ew_chained_input_src_idx = src_idx;
                        return true;
                    }
                    if (find_prev_device_chain_tensor(src, &device_chain_tensor, &device_chain_owner)) {
                        chained_input_ptr = device_chain_tensor;
                        ew_chained_input_src = device_chain_owner;
                        ew_chained_input_src_idx = src_idx;
                        return true;
                    }
                    return false;
                };
                (void) try_pick_chain_input(node->src[0], 0);
                if (chained_input_ptr == nullptr) {
                    (void) try_pick_chain_input(node->src[1], 1);
                }
            }

            // Determine whether to keep output on device for the next consumer.
            ggml_tensor * ew_next_consumer = nullptr;
            int ew_next_src_idx = -1;
            int ew_future_device_consumers = 0;
            int ew_future_consumers_any = 0;
            ggml_tensor * ew_qweight_consumer = nullptr;
            const bool keep_device_output_only =
                !ctx->disable_device_input_chain &&
                (((ew_future_device_consumers = count_future_device_consumers(i, node)) > 0) ||
                 find_future_qweight_consumer_src1(i, node, &ew_qweight_consumer));
            ew_future_consumers_any =
                !ctx->disable_device_input_chain ? count_future_consumers_any(i, node) : 0;
            if (ew_qweight_consumer != nullptr && ew_future_device_consumers == 0) {
                ew_future_device_consumers = 1;
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "future_qweight_consumer_match producer=" +
                        std::string(ggml_op_name(node->op)) +
                        " producer_ptr=" +
                        std::to_string(reinterpret_cast<uintptr_t>(node)) +
                        " consumer_ptr=" +
                        std::to_string(reinterpret_cast<uintptr_t>(ew_qweight_consumer)));
            }
            if (node->op == GGML_OP_CPY) {
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "cpy_chain_probe future_device=" + std::to_string(ew_future_device_consumers) +
                        " future_any=" + std::to_string(ew_future_consumers_any) +
                        " src0_op=" + std::string(node->src[0] ? ggml_op_name(node->src[0]->op) : "null") +
                        " src1_op=" + std::string(node->src[1] ? ggml_op_name(node->src[1]->op) : "null") +
                        " resolved_owner_op=" +
                        std::string(resolve_chain_source(node) ? ggml_op_name(resolve_chain_source(node)->op) : "null"));
            }
            if (node->op == GGML_OP_MUL && node->ne[0] == 1024 && node->ne[1] == 1 &&
                node->ne[2] == 1 && node->ne[3] == 1) {
                ggml_fmsh_log_locked(
                    ctx, 1,
                    "mul_chain_probe shape=1x1024 node_ptr=" +
                        std::to_string(reinterpret_cast<uintptr_t>(node)) +
                        " node_data=" +
                        std::to_string(reinterpret_cast<uintptr_t>(node->data)) +
                        " future_device=" + std::to_string(ew_future_device_consumers) +
                        " future_any=" + std::to_string(ew_future_consumers_any));
                for (int probe_j = i + 1; probe_j < std::min(i + 8, cgraph->n_nodes); ++probe_j) {
                    ggml_tensor * probe = cgraph->nodes[probe_j];
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "mul_chain_probe_next idx=" + std::to_string(probe_j) +
                            " op=" + std::string(ggml_op_name(probe->op)) +
                            " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(probe)) +
                            " src0_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(probe->src[0])) +
                            " src1_ptr=" + std::to_string(reinterpret_cast<uintptr_t>(probe->src[1])) +
                            " src0_resolved=" +
                            std::to_string(reinterpret_cast<uintptr_t>(resolve_chain_source(probe->src[0]))) +
                            " src1_resolved=" +
                            std::to_string(reinterpret_cast<uintptr_t>(resolve_chain_source(probe->src[1]))));
                }
            }
            if (keep_device_output_only) {
                log_future_device_consumers(i, node, "elementwise");
                (void) find_single_future_consumer(i, node, &ew_next_consumer, &ew_next_src_idx);
            }

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            std::vector<float> cpu_ref_out;
            if (ctx->debug_compare == 1) {
                const bool has_device_src0 = node->src[0] != nullptr && device_tensor_map.find(node->src[0]) != device_tensor_map.end();
                const bool has_device_src1 = node->src[1] != nullptr && device_tensor_map.find(node->src[1]) != device_tensor_map.end();
                const bool has_device_src2 = node->src[2] != nullptr && device_tensor_map.find(node->src[2]) != device_tensor_map.end();
                if (!ctx->disable_device_input_chain && (has_device_src0 || has_device_src1 || has_device_src2)) {
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
                    return ggml_fmsh_debug_backup_tensor(ctx, s, debug_saved, debug_seen, &dbg_err);
                };
                if (!ggml_fmsh_debug_backup_tensor(ctx, node, debug_saved, debug_seen, &dbg_err) ||
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
                if (!ggml_fmsh_tensor_snapshot_f32(ctx, node, cpu_ref_out, &snap_err)) {
                    ggml_fmsh_log_locked(ctx, 2, "debug_compare_snapshot_failed op=" + std::string(ggml_op_name(node->op)) + " reason=" + snap_err);
                }
                if (!debug_saved.empty()) {
                    std::string restore_err;
                    if (!ggml_fmsh_debug_restore_tensors(ctx, debug_saved, &restore_err)) {
                        ggml_fmsh_log_locked(ctx, 2, "debug_compare_restore_failed op=" + std::string(ggml_op_name(node->op)) + " reason=" + restore_err);
                        return GGML_STATUS_FAILED;
                    }
                }
                }
            }
            const bool keep_device_output_effective =
                keep_device_output_only || should_retain_cross_graph_chain(node);
            const bool write_host_output_effective =
                !keep_device_output_only ||
                !cpu_ref_out.empty() ||
                ew_future_consumers_any > ew_future_device_consumers;
#else
            const bool keep_device_output_effective =
                keep_device_output_only || should_retain_cross_graph_chain(node);
            const bool write_host_output_effective =
                !keep_device_output_only ||
                ew_future_consumers_any > ew_future_device_consumers;
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
                            clear_prev_device_output();
                        }
                        chainable_device_tensors.erase(held);
                        chainable_device_remaining_uses.erase(held);
                        it_map = device_tensor_map.erase(it_map);
                    } else {
                        ++it_map;
                    }
                }
            }
            // Ensure src tensors are host-accessible before elementwise execute.
            if (chained_input_ptr == nullptr || ew_chained_input_src_idx != 0) {
                ggml_fmsh_materialize_if_device(ctx, node->src[0], device_tensor_map);
            }
            if (!ggml_fmsh_is_cpy_dup_op(node) &&
                (chained_input_ptr == nullptr || ew_chained_input_src_idx != 1)) {
                ggml_fmsh_materialize_if_device(ctx, node->src[1], device_tensor_map);
            }
            ggml_fmsh_materialize_if_device(ctx, node->src[2], device_tensor_map);
            Tensor chained_output;
            if (!ggml_fmsh_execute_elementwise(
                    ctx, node, entry, chained_input_ptr, ew_chained_input_src_idx,
                    keep_device_output_effective, write_host_output_effective,
                    &chained_output, &err)) {
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
                release_chainable_consumer(i, ew_chained_input_src);
                if (has_prev_device_output && prev_device_output_node == ew_chained_input_src) {
                    clear_prev_device_output();
                }
            }

            if (chained_output.defined()) {
                set_prev_device_output(
                    const_cast<ggml_tensor *>(resolve_chain_source(node)),
                    chained_output);
                if (node->op == GGML_OP_MUL && node->ne[0] == 1024 && node->ne[1] == 1 &&
                    node->ne[2] == 1 && node->ne[3] == 1) {
                    ggml_fmsh_log_locked(
                        ctx, 1,
                        "elementwise_prev_set op=MUL ptr=" +
                            std::to_string(reinterpret_cast<uintptr_t>(prev_device_output_node)) +
                            " data=" +
                            std::to_string(reinterpret_cast<uintptr_t>(
                                prev_device_output_node != nullptr ? prev_device_output_node->data : nullptr)));
                }
            } else {
                clear_prev_device_output();
            }
            if (keep_device_output_effective && chained_output.defined()) {
                retain_chainable_tensor(i, node, chained_output);
                ggml_fmsh_log_locked(
                    ctx, 0,
                    "dispatch op=" + std::string(ggml_op_name(node->op)) +
                    " keep_device_output=1 future_device_consumers=" +
                    std::to_string(ew_future_device_consumers));
            } else {
                drop_chainable_tensor(node);
            }
            last_dispatched_valid = true;
            last_dispatched_zg = true;
            continue;
        }

        if (node->op == GGML_OP_SET_ROWS) {
            ggml_fmsh_log_locked(ctx, 1,
                "set_rows_on_device shape=[" +
                std::to_string(node->ne[0]) + "," +
                std::to_string(node->ne[1]) + "," +
                std::to_string(node->ne[2]) + "," +
                std::to_string(node->ne[3]) + "]");
            const enum ggml_status st = ggml_fmsh_execute_set_rows_on_device(
                ctx, cgraph, i, device_tensor_map);
            if (st != GGML_STATUS_SUCCESS) return st;
            device_tensor_map.erase(node);
            clear_prev_device_output();
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
        device_tensor_map.erase(node);
        last_dispatched_valid = true;
        last_dispatched_zg = false;
    }

    ctx->pending_prev_device_output_valid = has_prev_device_output;
    ctx->pending_prev_device_output = has_prev_device_output ? prev_device_output : Tensor();
    ctx->pending_prev_device_output_id =
        !has_prev_device_output ? ggml_fmsh_chain_tensor_id()
        : prev_device_output_node != nullptr ? ggml_fmsh_make_chain_tensor_id(prev_device_output_node)
                                             : prev_device_output_id;
    ctx->pending_chainable_device_tensors.clear();
    auto append_pending_chain_tensor =
        [&](const ggml_fmsh_pending_chain_tensor_entry & entry) {
            if (!entry.id.valid || !entry.tensor.defined() || entry.remaining_uses == 0) {
                return;
            }
            auto it_existing = std::find_if(
                ctx->pending_chainable_device_tensors.begin(),
                ctx->pending_chainable_device_tensors.end(),
                [&](const ggml_fmsh_pending_chain_tensor_entry & cur) {
                    return cur.id == entry.id;
                });
            if (it_existing != ctx->pending_chainable_device_tensors.end()) {
                *it_existing = entry;
            } else {
                ctx->pending_chainable_device_tensors.push_back(entry);
            }
        };
    for (const ggml_tensor * held : chainable_device_tensors) {
        auto it_dev = device_tensor_map.find(held);
        auto it_uses = chainable_device_remaining_uses.find(held);
        if (it_dev == device_tensor_map.end() || !it_dev->second.defined()) {
            continue;
        }
        if (it_uses == chainable_device_remaining_uses.end() || it_uses->second == 0) {
            continue;
        }
        append_pending_chain_tensor(
            ggml_fmsh_pending_chain_tensor_entry{
                ggml_fmsh_make_chain_tensor_id(held),
                it_dev->second,
                it_uses->second,
            });
    }
    for (const auto & pending : unresolved_chainable_device_tensors) {
        append_pending_chain_tensor(pending);
    }
    if (has_prev_device_output) {
        ggml_fmsh_log_locked(
            ctx, 1,
            "graph_compute_prev_store owner_op=" +
                std::string(
                    prev_device_output_node ? ggml_op_name(prev_device_output_node->op)
                                            : ggml_op_name(ctx->pending_prev_device_output_id.op)) +
                " owner_ptr=" +
                std::to_string(reinterpret_cast<uintptr_t>(prev_device_output_node)) +
                " owner_data=" +
                std::to_string(
                    prev_device_output_node != nullptr
                        ? reinterpret_cast<uintptr_t>(prev_device_output_node->data)
                        : ctx->pending_prev_device_output_id.data));
    }

    return GGML_STATUS_SUCCESS;
}


} // namespace ggml_fmsh_zg330_impl
