#include "ggml-fmsh-zg330-internal.h"

namespace ggml_fmsh_zg330_impl {

ggml_fmsh_zg330_session_entry * ggml_fmsh_get_or_create_mul_mat_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    int64_t m = 0, k = 0, n = 0;
    if (!ggml_fmsh_validate_mul_mat(node, &m, &k, &n)) {
        if (err) *err = "invalid MUL_MAT shape";
        return nullptr;
    }
    const ggml_fmsh_zg330_op_signature sig = ggml_fmsh_make_signature(node);
    const std::string cache_key = ggml_fmsh_make_mul_mat_cache_key(node, m, k, n);
    auto it = ctx->session_cache.find(cache_key);
    if (it != ctx->session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_matmul_zg_network(ctx->cache_dir, m, k, n);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);

        ggml_fmsh_log_locked(
            ctx, 0,
            "session_opt_config net=" + bundle.net_name + " reuse_ok=0 connect_ok=0 instr_bytes=0 weight_bytes=0 ftmp_bytes=0 in_v_id=-1 out_v_id=-1");
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_session_entry>();
        entry->signature = sig;
        entry->hit_count = 1;
        entry->m = m;
        entry->k = k;
        entry->n = n;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        entry->input_type_a = entry->bundle.network.inputs()[0].tensorType().clone();
        entry->input_type_b = entry->bundle.network.inputs()[1].tensorType().clone();
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }

        ggml_fmsh_zg330_session_entry * ptr = entry.get();
        ctx->session_cache.emplace(cache_key, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
}

// Build (or fetch) the session(s) for a quantized weight matmul. M is bucketed and
// N is split into column chunks; each chunk gets its own BF16 matmul network.
ggml_fmsh_zg330_qweight_session_entry * ggml_fmsh_get_or_create_qweight_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    int64_t m = 0, k = 0, n = 0;
    if (!ggml_fmsh_validate_mul_mat(node, &m, &k, &n)) {
        if (err) *err = "invalid MUL_MAT shape";
        return nullptr;
    }

    const int64_t m_bucket = ggml_fmsh_mul_mat_m_bucket(m, ctx->mul_mat_m_bucket_max);
    if (m_bucket <= 0) {
        if (err) *err = "M exceeds mul_mat_m_bucket_max; CPU fallback";
        return nullptr;
    }

    const int64_t n_chunk = (ctx->mul_mat_n_chunk > 0 && ctx->mul_mat_n_chunk < n)
        ? ctx->mul_mat_n_chunk : n;

    // Key by (dtype, k, n_full, m_bucket): networks depend only on these, and per-weight
    // Share sessions by shape (11 sessions for this model). The weight is passed as a
    // HOST-side tensor on every call so icraft DMAs it fresh each time — device-resident
    // inputs are not rebound by forward(), but host inputs are always re-read.
    // The layerCount edge-triggered sync in forward_synced ensures the NPU finishes
    // before we read the output, avoiding the stale-output bug.
    const std::string cache_key =
        "qw|" + std::to_string(static_cast<int>(node->src[0]->type)) +
        "|" + std::to_string(k) + "x" + std::to_string(n) + "x" + std::to_string(m_bucket);
    auto it = ctx->qweight_session_cache.find(cache_key);
    if (it != ctx->qweight_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto entry = std::make_unique<ggml_fmsh_zg330_qweight_session_entry>();
        entry->hit_count = 1;
        entry->m_bucket = m_bucket;
        entry->k = k;
        entry->n_full = n;
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }

        for (int64_t col = 0; col < n; col += n_chunk) {
            const int64_t n_this = std::min(n_chunk, n - col);
            ggml_fmsh_zg330_qweight_session_entry::sub_net sn;
            sn.col_off = col;
            sn.n_this = n_this;
            // NOTE: icraft-adapt segfaults on BF16 MatMul networks (toolchain bug; BF16
            // elementwise is fine, BF16 MatMul is not). Use the working tf32 path — same
            // as the attention matmuls. Weights are therefore resident as F32 on device.
            sn.bundle = ggml::fmsh::netmake::get_or_compile_matmul_zg_network(
                ctx->cache_dir, m_bucket, k, n_this, /*bf16=*/false);
            sn.session = Session::Create<zg330::ZG330Backend, HostBackend>(
                sn.bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
            sn.session.enableTimeProfile(true);
            sn.session.apply();
            sn.input_type_a = sn.bundle.network.inputs()[0].tensorType().clone();
            sn.input_type_b = sn.bundle.network.inputs()[1].tensorType().clone();
            entry->subnets.push_back(std::move(sn));
        }

        // Speculatively precompile the next power-of-two M buckets so the ARM cache is
        // warm for prefill (compilation only succeeds on x86; best-effort).
        if (ctx->mul_mat_precompile_m_depth > 0) {
            int64_t pre_m = m_bucket;
            for (int64_t d = 0; d < ctx->mul_mat_precompile_m_depth; ++d) {
                pre_m <<= 1;
                if (pre_m > ctx->mul_mat_m_bucket_max) {
                    break;
                }
                bool stop = false;
                for (int64_t col = 0; col < n && !stop; col += n_chunk) {
                    const int64_t n_this = std::min(n_chunk, n - col);
                    try {
                        (void) ggml::fmsh::netmake::get_or_compile_matmul_zg_network(
                            ctx->cache_dir, pre_m, k, n_this, /*bf16=*/false);
                    } catch (...) {
                        stop = true;
                    }
                }
                if (stop) {
                    break;
                }
            }
        }

        ggml_fmsh_zg330_qweight_session_entry * ptr = entry.get();
        ctx->qweight_session_cache.emplace(cache_key, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

// Run one session.forward and wait for the result with correct AXI synchronization.
// On ARM/AXI a reused session keeps ready_=true after the first forward, so waitForReady()
// would return the PREVIOUS frame's output. We re-arm a layerCount-based check_func_ each
// call (learning the per-forward increment on the first call). On socket/x86 layerCount is
// not a reliable completion signal, so we fall back to plain waitForReady.
bool ggml_fmsh_forward_synced(
    ggml_backend_fmsh_zg330_context * ctx,
    Session & session,
    const std::vector<Tensor> & inputs,
    ZG330Device & zg_device,
    uint32_t & layer_increment,
    Tensor * out,
    const char * where,
    std::string * err) {
    // A reused icraft session returns the SAME output Tensor; its network check_func_ uses an
    // ABSOLUTE layerCount target fixed at apply() time, which is long exceeded mid-inference,
    // so waitForReady() returns immediately reading the PREVIOUS forward's stale output.
    // For large matmul networks (async on AXI), we must wait for the NPU to finish.
    // Use edge-triggered layerCount: record layer_before, call forward(), then spin until
    // layerCount advances past layer_before. This works for every call regardless of increment.
    const bool use_layer_sync = zg_device.defined();
    uint32_t layer_before = 0;
    if (use_layer_sync) {
        layer_before = zg_device.layerCount();
    }

    auto outputs = session.forward(inputs);
    if (outputs.empty()) {
        if (err) *err = std::string(where) + " session.forward returned empty output";
        return false;
    }

    if (use_layer_sync) {
        if (layer_increment == 0) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
            while (zg_device.layerCount() == layer_before) {
                if (std::chrono::steady_clock::now() > deadline) {
                    if (err) *err = std::string(where) + " NPU layerCount sync timeout (start)";
                    return false;
                }
            }

            uint32_t stable_count = zg_device.layerCount();
            auto stable_start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() < deadline) {
                const uint32_t cur = zg_device.layerCount();
                if (cur != stable_count) {
                    stable_count = cur;
                    stable_start = std::chrono::steady_clock::now();
                } else {
                    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - stable_start).count();
                    if (elapsed_ms >= 50) {
                        break;
                    }
                }
            }

            layer_increment = (stable_count > layer_before) ? (stable_count - layer_before) : 1;
            ggml_fmsh_log_locked(ctx, 1, std::string(where) + " layer_sync_first before=" +
                std::to_string(layer_before) + " after=" + std::to_string(stable_count) +
                " increment=" + std::to_string(layer_increment));
        } else {
            const uint32_t layer_target = layer_before + layer_increment;
            ZG330Device zg_dev = zg_device;
            outputs[0].setReady(false);
            outputs[0].setCheckFunc([zg_dev, layer_target](const Device &) -> bool {
                return zg_dev.layerCount() >= layer_target;
            });
            if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, where)) {
                return false;
            }
            ggml_fmsh_log_locked(ctx, 0, std::string(where) + " layer_sync_sub before=" +
                std::to_string(layer_before) + " target=" + std::to_string(layer_target) +
                " actual=" + std::to_string(zg_device.layerCount()) +
                " increment=" + std::to_string(layer_increment));
        }
    } else {
        outputs[0].setReady(false);
        if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, where)) {
            return false;
        }
    }
    *out = outputs[0];
    return true;
}

// Execute a quantized weight matmul on the NPU: dequant src0 → F32 (once, cached on
// device), upload the activation with M padding, run each N-chunk network with AXI sync,
// and scatter the F32 output back into dst.
bool ggml_fmsh_execute_qweight_mul_mat(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_qweight_session_entry * entry,
    const Tensor * input_override,
    std::string * err) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1 || src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        if (err) *err = "qweight MUL_MAT tensor data is null";
        return false;
    }

    const int64_t actual_m = src1->ne[1];
    const int64_t m_bucket = entry->m_bucket;
    const int64_t k = entry->k;
    if (actual_m > m_bucket) {
        if (err) *err = "actual M exceeds bucket";
        return false;
    }
    if (src1->nb[0] != sizeof(float)) {
        if (err) *err = "qweight src1 not contiguous F32";
        return false;
    }

    const struct ggml_type_traits * tt = ggml_get_type_traits(src0->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        if (err) *err = "no dequant kernel for src0 type";
        return false;
    }
    std::string staging_err;
    if (!ggml_fmsh_ensure_zg_staging(ctx, src0, true, &staging_err)) {
        if (err) *err = "qweight src0 staging failed: " + staging_err;
        return false;
    }
    const char * src0_data = ggml_fmsh_host_data_for_tensor(ctx, src0);

    // src1 (activation) may be ZG DDR-resident (fake pointer, not dereferenceable on host).
    // When an upstream op retained its output on device, consume that tensor directly here.
    // Otherwise D2H src1 into a staging buffer and redirect src1->data for the duration
    // of this call.
    std::vector<char> src1_staging;
    void * src1_orig_data = src1->data;
    if (input_override == nullptr) {
        MemChunk zg_chunk; size_t zg_off = 0;
        const ggml_tensor * root = src1;
        while (root->view_src) root = root->view_src;
        if (ggml_fmsh_is_zg_buffer_tensor(root, &zg_chunk, &zg_off)) {
            const size_t root_bytes = ggml_nbytes(root);
            src1_staging.resize(root_bytes);
            zg_chunk.read(src1_staging.data(), zg_off, root_bytes);
            // Redirect src1->data: host_ptr = staging_base + (src1_zg - root_zg)
            const uintptr_t root_zg = reinterpret_cast<uintptr_t>(root->data);
            const uintptr_t src1_zg = reinterpret_cast<uintptr_t>(src1->data);
            const_cast<ggml_tensor *>(src1)->data = src1_staging.data() + (src1_zg - root_zg);
            ggml_fmsh_log_locked(ctx, 1, "qweight_src1_d2h bytes=" + std::to_string(root_bytes));
        }
    }
    struct Src1Restore {
        const ggml_tensor * t; void * orig;
        ~Src1Restore() { const_cast<ggml_tensor *>(t)->data = orig; }
    } src1_restore{src1, src1_orig_data};

    try {
        std::vector<float>      row_buf(static_cast<size_t>(k));
        std::vector<float>      a_host(static_cast<size_t>(m_bucket) * static_cast<size_t>(k));
        std::vector<float>      out_tmp;
        std::vector<float>      chained_src1_host;

        if (input_override != nullptr) {
            const size_t chained_elems = static_cast<size_t>(actual_m) * static_cast<size_t>(k);
            chained_src1_host.resize(chained_elems);
            input_override->read(
                reinterpret_cast<char *>(chained_src1_host.data()),
                0,
                chained_elems * sizeof(float));
            ggml_fmsh_log_locked(
                ctx, 1,
                "qweight_src1_chain_read bytes=" +
                    std::to_string(chained_elems * sizeof(float)));
        }

        for (auto & sn : entry->subnets) {
            // --- activation A: F32 [m_bucket, k], rows>=actual_m zero-padded ---
            // Allocate a FRESH input tensor per forward. A shared session runs back-to-back
            // for many same-shape nodes within one graph_compute; reusing one host buffer let
            // the next node clobber the previous node's still-in-flight input (async forward),
            // producing a pipeline-lagged / corrupted result.
            Tensor input_a(sn.input_type_a.clone());
            input_a.mallocOn(HostDevice::MemRegion());
            std::fill(a_host.begin(), a_host.end(), 0.0f);
            if (input_override != nullptr) {
                for (int64_t r = 0; r < actual_m; ++r) {
                    const float * a_src_row =
                        chained_src1_host.data() + static_cast<size_t>(r) * static_cast<size_t>(k);
                    std::memcpy(
                        a_host.data() + r * k,
                        a_src_row,
                        static_cast<size_t>(k) * sizeof(float));
                }
            } else {
                for (int64_t r = 0; r < actual_m; ++r) {
                    const float * a_src_row = reinterpret_cast<const float *>(
                        static_cast<const char *>(src1->data) + r * src1->nb[1]);
                    std::memcpy(a_host.data() + r * k, a_src_row, static_cast<size_t>(k) * sizeof(float));
                }
            }
            std::memcpy(input_a.data().cptr(), a_host.data(),
                        a_host.size() * sizeof(float));

            // --- weight B: F32 [k, n_this], dequant+transpose, cached per src0 pointer ---
            // Session is shared by shape, but each layer's dequantized weight is cached
            // on device DDR keyed by (src0_data + col_off). After the first call for a
            // given layer, no dequant or DMA is needed — forward() uses the cached tensor.
            Tensor dynamic_weight;
            Tensor * weight_tensor = nullptr;
            const uintptr_t w_key = reinterpret_cast<uintptr_t>(src0_data + sn.col_off * src0->nb[1]);
            for (auto & cached : sn.weights) {
                if (cached.key == w_key) {
                    weight_tensor = &cached.tensor;
                    break;
                }
            }
            if (!weight_tensor) {
                const size_t wbytes =
                    static_cast<size_t>(k) * static_cast<size_t>(sn.n_this) * sizeof(float);
                std::vector<float> host_w(static_cast<size_t>(k) * static_cast<size_t>(sn.n_this));
                for (int64_t i = 0; i < sn.n_this; ++i) {
                    const char * w_src_row =
                        src0_data + (sn.col_off + i) * src0->nb[1];
                    tt->to_float(w_src_row, row_buf.data(), k);
                    for (int64_t j = 0; j < k; ++j) {
                        host_w[static_cast<size_t>(j) * sn.n_this + i] = row_buf[j];
                    }
                }
                Tensor new_weight(sn.input_type_b.clone());
                if (ctx->device_opened) {
                    try {
                        new_weight.mallocOn(ctx->zg_device.defaultMemRegion());
                        new_weight.write(0, reinterpret_cast<char *>(host_w.data()), wbytes);
                    } catch (...) {
                        ggml_fmsh_log_locked(ctx, 2, "qweight_cache_oom k=" + std::to_string(k) +
                            " n=" + std::to_string(sn.n_this) +
                            " resident=" + std::to_string(ctx->mul_mat_qweight_bytes_resident / (1024 * 1024)) + "MiB");
                        new_weight = Tensor(sn.input_type_b.clone());
                        new_weight.mallocOn(HostDevice::MemRegion());
                        std::memcpy(new_weight.data().cptr(), host_w.data(), wbytes);
                    }
                } else {
                    new_weight.mallocOn(HostDevice::MemRegion());
                    std::memcpy(new_weight.data().cptr(), host_w.data(), wbytes);
                }
                sn.weights.push_back({w_key, std::move(new_weight)});
                weight_tensor = &sn.weights.back().tensor;
                ctx->mul_mat_qweight_bytes_resident += wbytes;
                ggml_fmsh_log_locked(ctx, 1, "qweight_cache_miss src0=" +
                    std::to_string(reinterpret_cast<uintptr_t>(src0->data)) +
                    " k=" + std::to_string(k) + " n=" + std::to_string(sn.n_this) +
                    " resident=" + std::to_string(ctx->mul_mat_qweight_bytes_resident / (1024 * 1024)) + "MiB");
            } else {
                ggml_fmsh_log_locked(ctx, 1, "qweight_cache_hit src0=" +
                    std::to_string(reinterpret_cast<uintptr_t>(src0->data)) +
                    " k=" + std::to_string(k) + " n=" + std::to_string(sn.n_this));
            }

            // --- forward + scatter F32 output back into dst ---
            // With device-resident weight (bound at session creation), forward() is
            // synchronous — the NPU completes before forward() returns. waitForReady()
            // returns immediately (ready_=true) but the output IS valid.
            Tensor out_dev;
            if (!ggml_fmsh_forward_synced(ctx, sn.session, {input_a, *weight_tensor},
                                          entry->zg_device, sn.layer_increment, &out_dev,
                                          "qweight_mul_mat", err)) {
                return false;
            }
            out_tmp.resize(static_cast<size_t>(m_bucket) * static_cast<size_t>(sn.n_this));
            out_dev.read(reinterpret_cast<char *>(out_tmp.data()), 0,
                         out_tmp.size() * sizeof(float));
            // Debug: log first few output values to verify NPU result.
            {
                static std::atomic<int> s_qw_out_log{8};
                if (s_qw_out_log.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    std::string vals;
                    for (int64_t _d = 0; _d < std::min<int64_t>(4, sn.n_this); ++_d) {
                        vals += std::to_string(out_tmp[static_cast<size_t>(_d)]) + " ";
                    }
                    ggml_fmsh_log_locked(ctx, 1, "qweight_out_sample k=" + std::to_string(k) +
                        " n=" + std::to_string(sn.n_this) + " vals=[" + vals + "]");
                }
            }
            for (int64_t r = 0; r < actual_m; ++r) {
                const float * src_row = out_tmp.data() + r * sn.n_this;
                for (int64_t c = 0; c < sn.n_this; ++c) {
                    ggml_fmsh_write_f32_indexed(ctx, node, sn.col_off + c, r, 0, 0, src_row[c]);
                }
            }
        }
        if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "qweight_readback", &staging_err)) {
            if (err) *err = staging_err;
            return false;
        }
        ctx->mul_mat_qweight_offloaded++;
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}


} // namespace ggml_fmsh_zg330_impl
