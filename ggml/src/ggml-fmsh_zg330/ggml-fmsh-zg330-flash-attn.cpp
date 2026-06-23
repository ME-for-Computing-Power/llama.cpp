#include "ggml-fmsh-zg330-internal.h"

namespace ggml_fmsh_zg330_impl {

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
bool ggml_fmsh_tensor_snapshot_f32(const ggml_tensor * t, std::vector<float> & out, std::string * err) {
    if (!t || !t->data) {
        if (err) *err = "tensor data null";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) *err = "tensor type is not F32";
        return false;
    }
    const size_t n = ggml_nelements(t);
    out.resize(n);
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    out[idx++] = ggml_fmsh_read_f32_indexed(t, i0, i1, i2, i3);
                }
            }
        }
    }
    return true;
}

bool ggml_fmsh_tensor_snapshot_f32(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    std::vector<float> & out,
    std::string * err) {
    if (!t || !t->data) {
        if (err) *err = "tensor data null";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) *err = "tensor type is not F32";
        return false;
    }
    if (!ggml_fmsh_ensure_zg_staging(ctx, t, true, err)) {
        return false;
    }
    const size_t n = ggml_nelements(t);
    out.resize(n);
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    out[idx++] = ggml_fmsh_read_f32_indexed(ctx, t, i0, i1, i2, i3);
                }
            }
        }
    }
    return true;
}

bool ggml_fmsh_tensor_restore_f32(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * t,
    const std::vector<float> & in,
    std::string * err) {
    if (!t || !t->data) {
        if (err) *err = "tensor data null";
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) *err = "tensor type is not F32";
        return false;
    }
    const size_t n = ggml_nelements(t);
    if (in.size() != n) {
        if (err) *err = "tensor restore size mismatch";
        return false;
    }
    size_t idx = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                    ggml_fmsh_write_f32_indexed(ctx, t, i0, i1, i2, i3, in[idx++]);
                }
            }
        }
    }
    if (!ggml_fmsh_sync_zg_staging_to_device(ctx, t, "debug_restore", err)) {
        return false;
    }
    return true;
}


bool ggml_fmsh_debug_backup_tensor(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * t,
    std::vector<ggml_fmsh_debug_saved_tensor> & saved,
    std::unordered_set<ggml_tensor *> & seen,
    std::string * err) {
    if (!t || !t->data || t->type != GGML_TYPE_F32) {
        return true;
    }
    if (!seen.insert(t).second) {
        return true;
    }
    ggml_fmsh_debug_saved_tensor s;
    s.t = t;
    if (!ggml_fmsh_tensor_snapshot_f32(ctx, t, s.data, err)) {
        return false;
    }
    saved.push_back(std::move(s));
    return true;
}

bool ggml_fmsh_debug_restore_tensors(
    ggml_backend_fmsh_zg330_context * ctx,
    std::vector<ggml_fmsh_debug_saved_tensor> & saved,
    std::string * err) {
    for (auto & s : saved) {
        if (!ggml_fmsh_tensor_restore_f32(ctx, s.t, s.data, err)) {
            return false;
        }
    }
    return true;
}

void ggml_fmsh_debug_compare_and_log(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    const std::vector<float> & ref,
    int node_index) {
    if (!ctx->debug_compare || !node || node->type != GGML_TYPE_F32 || !node->data) {
        return;
    }
    const size_t n = ggml_nelements(node);
    if (ref.size() != n || n == 0) {
        return;
    }

    size_t mismatches = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t first_bad = static_cast<size_t>(-1);
    float first_ref = 0.0f;
    float first_npu = 0.0f;

    size_t idx = 0;
    for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                    const float got = ggml_fmsh_read_f32_indexed(ctx, node, i0, i1, i2, i3);
                    const float exp = ref[idx];
                    const double abs_err = std::fabs(static_cast<double>(got) - static_cast<double>(exp));
                    const double rel_err = abs_err / (std::fabs(static_cast<double>(exp)) + 1e-12);
                    if (abs_err > max_abs) max_abs = abs_err;
                    if (rel_err > max_rel) max_rel = rel_err;
                    if (abs_err > ctx->debug_compare_atol + ctx->debug_compare_rtol * std::fabs(static_cast<double>(exp))) {
                        if (first_bad == static_cast<size_t>(-1)) {
                            first_bad = idx;
                            first_ref = exp;
                            first_npu = got;
                        }
                        ++mismatches;
                    }
                    ++idx;
                }
            }
        }
    }

    ctx->debug_compare_total++;
    if (mismatches > 0) {
        ctx->debug_compare_mismatch++;
        ctx->debug_compare_logged++;
        ggml_fmsh_log_locked(
            ctx, 2,
            "debug_compare_mismatch op=" + std::string(ggml_op_name(node->op)) +
            " node_idx=" + std::to_string(node_index) +
            " shape=[" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," +
            std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + "]" +
            " n=" + std::to_string(n) +
            " mismatches=" + std::to_string(mismatches) +
            " max_abs=" + std::to_string(max_abs) +
            " max_rel=" + std::to_string(max_rel) +
            " first_bad_idx=" + std::to_string(first_bad) +
            " first_ref=" + std::to_string(first_ref) +
            " first_npu=" + std::to_string(first_npu));
    }
}
#endif

// Forward declaration — defined after execute_qweight_via_f32.
bool ggml_fmsh_execute_mul_mat(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_session_entry * entry,
    const Tensor * input_override,
    bool keep_device_output_only,
    bool write_host_output,
    Tensor * chained_output,
    std::string * err);

// Execute a quantized weight matmul by dequantizing src0 to F32 and routing through
// the existing execute_mul_mat path. This reuses the proven attention matmul mechanism
// (shared session, HOST weight, synchronous forward) without any custom qweight code.
bool ggml_fmsh_execute_qweight_via_f32(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    std::string * err) {
    const ggml_tensor * src0 = node->src[0];
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = node->src[1]->ne[1];
    const struct ggml_type_traits * tt = ggml_get_type_traits(src0->type);
    if (!tt || !tt->to_float) {
        if (err) *err = "no dequant kernel";
        return false;
    }
    std::string staging_err;
    if (!ggml_fmsh_ensure_zg_staging(ctx, src0, true, &staging_err)) {
        if (err) *err = "qweight src0 staging failed: " + staging_err;
        return false;
    }
    const char * src0_data = ggml_fmsh_host_data_for_tensor(ctx, src0);

    // N-chunk size: icraft adapt crashes when the weight input is too large
    // (internal representation exceeds 2GB). Chunk N to stay within limits.
    const int64_t n_chunk = (ctx->mul_mat_n_chunk > 0 && ctx->mul_mat_n_chunk < n)
                            ? ctx->mul_mat_n_chunk : n;

    ggml_tensor * orig_src0 = node->src[0];
    bool all_ok = true;

    for (int64_t col = 0; col < n && all_ok; col += n_chunk) {
        const int64_t n_this = std::min(n_chunk, n - col);

        // Dequantize the [n_this, k] slice of src0 for this chunk.
        std::vector<float> f32_buf(static_cast<size_t>(n_this) * static_cast<size_t>(k));
        std::vector<float> row_buf(static_cast<size_t>(k));
        for (int64_t i = 0; i < n_this; ++i) {
            const char * src_row = src0_data + (col + i) * src0->nb[1];
            tt->to_float(src_row, row_buf.data(), k);
            std::memcpy(f32_buf.data() + i * k, row_buf.data(), static_cast<size_t>(k) * sizeof(float));
        }

        // Fake F32 src0 for this chunk: ne[1] = n_this.
        // Mark it as an input tensor so execute_mul_mat treats it as dynamic HOST weight.
        // Do not fake view_src: helper code walks view chains and expects valid tensors.
        // For small n (controlled by MUL_MAT_N_MAX), the HOST→ZG DMA is reliable.
        const size_t chunk_f32_bytes = static_cast<size_t>(n_this) * static_cast<size_t>(k) * sizeof(float);
        ggml_tensor fake_src0{};
        fake_src0.type   = GGML_TYPE_F32;
        fake_src0.ne[0]  = k;
        fake_src0.ne[1]  = n_this;
        fake_src0.ne[2]  = 1;
        fake_src0.ne[3]  = 1;
        fake_src0.nb[0]  = sizeof(float);
        fake_src0.nb[1]  = static_cast<size_t>(k) * sizeof(float);
        fake_src0.nb[2]  = chunk_f32_bytes;
        fake_src0.nb[3]  = chunk_f32_bytes;
        fake_src0.data   = f32_buf.data();
        fake_src0.view_src = nullptr;
        fake_src0.op     = GGML_OP_NONE;
        fake_src0.flags  = GGML_TENSOR_FLAG_INPUT;

        // Redirect dst to a temporary buffer for this chunk [m, n_this], then
        // scatter into the real dst at the correct column offset afterwards.
        std::vector<float> chunk_dst(static_cast<size_t>(m) * static_cast<size_t>(n_this));
        ggml_tensor fake_dst{};
        fake_dst.type   = GGML_TYPE_F32;
        fake_dst.ne[0]  = n_this;
        fake_dst.ne[1]  = m;
        fake_dst.ne[2]  = 1;
        fake_dst.ne[3]  = 1;
        fake_dst.nb[0]  = sizeof(float);
        fake_dst.nb[1]  = static_cast<size_t>(n_this) * sizeof(float);
        fake_dst.nb[2]  = static_cast<size_t>(m) * static_cast<size_t>(n_this) * sizeof(float);
        fake_dst.nb[3]  = fake_dst.nb[2];
        fake_dst.data   = chunk_dst.data();
        fake_dst.op     = GGML_OP_MUL_MAT;

        void  * orig_dst_data = node->data;
        size_t  orig_nb0      = node->nb[0];
        size_t  orig_nb1      = node->nb[1];
        int64_t orig_ne0      = node->ne[0];
        node->src[0] = &fake_src0;
        node->data   = fake_dst.data;
        node->nb[0]  = fake_dst.nb[0];
        node->nb[1]  = fake_dst.nb[1];
        node->ne[0]  = n_this;

        bool ok = false;
        bool created = false;
        ggml_fmsh_zg330_session_entry * entry =
            ggml_fmsh_get_or_create_mul_mat_session(ctx, node, &created, err);
        if (entry) {
            ok = ggml_fmsh_execute_mul_mat(ctx, node, entry, nullptr, false, true, nullptr, err);
        }

        // Restore node fields before scatter.
        node->src[0] = orig_src0;
        node->data   = orig_dst_data;
        node->nb[0]  = orig_nb0;
        node->nb[1]  = orig_nb1;
        node->ne[0]  = orig_ne0;

        if (!ok) {
            all_ok = false;
            break;
        }

        // Scatter chunk result into the real dst: dst[row, col+c] = chunk_dst[row, c].
        for (int64_t row = 0; row < m; ++row) {
            const float * src_row = chunk_dst.data() + row * n_this;
            for (int64_t c = 0; c < n_this; ++c) {
                ggml_fmsh_write_f32_indexed(ctx, node, col + c, row, 0, 0, src_row[c]);
            }
        }
    }

    if (all_ok) {
        if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "qweight_readback", &staging_err)) {
            if (err) *err = staging_err;
            return false;
        }
        ctx->mul_mat_qweight_offloaded++;
    }
    return all_ok;
}

bool ggml_fmsh_execute_mul_mat(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_session_entry * entry,
    const Tensor * input_override,
    bool keep_device_output_only,
    bool write_host_output,
    Tensor * chained_output,
    std::string * err) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1 || src0->data == nullptr || src1->data == nullptr || node->data == nullptr) {
        if (err) *err = "MUL_MAT tensor data is null";
        return false;
    }

    try {
        struct ggml_fmsh_f32_stats {
            float min_v = 0.0f;
            float max_v = 0.0f;
            size_t inf_cnt = 0;
            size_t nan_cnt = 0;
        };
        const auto calc_f32_stats = [](const float * data, size_t n) -> ggml_fmsh_f32_stats {
            ggml_fmsh_f32_stats s;
            if (!data || n == 0) {
                return s;
            }
            s.min_v = data[0];
            s.max_v = data[0];
            for (size_t i = 0; i < n; ++i) {
                const float v = data[i];
                if (std::isnan(v)) {
                    s.nan_cnt++;
                    continue;
                }
                if (std::isinf(v)) {
                    s.inf_cnt++;
                    continue;
                }
                if (v < s.min_v) s.min_v = v;
                if (v > s.max_v) s.max_v = v;
            }
            return s;
        };
        if (entry->bundle.network.inputs().size() < 2) {
            if (err) *err = "compiled network input count < 2";
            return false;
        }

        if (!entry->input_tensor_a_ready) {
            entry->input_tensor_a = Tensor(entry->input_type_a.clone());
            entry->input_tensor_a.mallocOn(HostDevice::MemRegion());
            entry->input_tensor_a_ready = true;
        }

        const int64_t ne02 = src0->ne[2];
        const int64_t ne03 = src0->ne[3];
        const int64_t ne12 = src1->ne[2];
        const int64_t ne13 = src1->ne[3];
        const int64_t r2 = ne12 / ne02;
        const int64_t r3 = ne13 / ne03;

        const size_t out_bytes = static_cast<size_t>(entry->m * entry->n * static_cast<int64_t>(sizeof(float)));
        std::vector<float> out_tmp;
        if (write_host_output) {
            out_tmp.resize(static_cast<size_t>(entry->m * entry->n));
        }

        const bool src1_is_bf16 = (src1->type == GGML_TYPE_BF16);
        const bool allow_input_chain = input_override != nullptr && ne12 == 1 && ne13 == 1;
        if (chained_output != nullptr) {
            *chained_output = Tensor();
        }

        ggml_fmsh_log_locked(
            ctx, 1,
            "mul_mat_exec_enter m=" + std::to_string(entry->m) +
            " k=" + std::to_string(entry->k) +
            " n=" + std::to_string(entry->n) +
            " src1_type=" + std::to_string(static_cast<int>(src1->type)) +
            " src1_is_bf16=" + std::to_string(src1_is_bf16 ? 1 : 0) +
            " allow_input_chain=" + std::to_string(allow_input_chain ? 1 : 0) +
            " ne12=" + std::to_string(ne12) + " ne13=" + std::to_string(ne13));

        ggml_fmsh_zg330_bf16_bridge_session_entry * bridge_entry = nullptr;
        if (src1_is_bf16 && !allow_input_chain) {
            ggml_fmsh_log_locked(ctx, 0, "bf16_bridge_enter rows=" + std::to_string(entry->m) + " cols=" + std::to_string(entry->k));
            std::string bridge_err;
            bool bridge_created = false;
            bridge_entry = ggml_fmsh_get_or_create_bf16_bridge_session(ctx, entry->m, entry->k, &bridge_created, &bridge_err);
            if (!bridge_entry) {
                ggml_fmsh_log_locked(ctx, 3, "bf16_bridge_session_create_failed: " + bridge_err);
                if (err) *err = "BF16 bridge session creation failed: " + bridge_err;
                return false;
            }
            if (!bridge_entry->input_tensor.defined()) {
                bridge_entry->input_tensor = Tensor(bridge_entry->input_type.clone());
                bridge_entry->input_tensor.mallocOn(HostDevice::MemRegion());
            }
            ggml_fmsh_log_locked(
                ctx, 1,
                std::string(bridge_created ? "bf16_bridge_session_create" : "bf16_bridge_session_hit") +
                " rows=" + std::to_string(entry->m) + " cols=" + std::to_string(entry->k) +
                " net=" + bridge_entry->bundle.net_name +
                " net_cache=" + std::string(bridge_entry->bundle.ram_cache_hit ? "HIT" : "MISS") +
                " compiled_now=" + std::string(bridge_entry->bundle.compiled_now ? "YES" : "NO"));
        } else if (src1_is_bf16 && allow_input_chain) {
            ggml_fmsh_log_locked(ctx, 1, "bf16_bridge_skip_reason=input_chain_already_provided");
        } else {
            ggml_fmsh_log_locked(ctx, 1, "bf16_bridge_skip_reason=src1_not_bf16 src1_type=" + std::to_string(static_cast<int>(src1->type)));
        }

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            const int64_t i03 = i13 / r3;
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                const int64_t i02 = i12 / r2;

                if (ggml_fmsh_is_zg_buffer_tensor(src0, nullptr, nullptr) ||
                    ggml_fmsh_is_zg_buffer_tensor(src1, nullptr, nullptr)) {
                    std::string staging_err;
                    if (!ggml_fmsh_ensure_zg_staging(ctx, src0, true, &staging_err) ||
                        !ggml_fmsh_ensure_zg_staging(ctx, src1, true, &staging_err)) {
                        if (err) *err = staging_err;
                        return false;
                    }
                }
                const char * src1_slice = ggml_fmsh_host_data_for_tensor(ctx, src1) + i12 * src1->nb[2] + i13 * src1->nb[3];
                const char * src0_slice = ggml_fmsh_host_data_for_tensor(ctx, src0) + i02 * src0->nb[2] + i03 * src0->nb[3];
                if (ggml_fmsh_is_zg_buffer_tensor(node, nullptr, nullptr)) {
                    std::string staging_err;
                    if (!ggml_fmsh_ensure_zg_staging(ctx, node, true, &staging_err)) {
                        if (err) *err = staging_err;
                        return false;
                    }
                }
                char * dst_slice = const_cast<char *>(ggml_fmsh_host_data_for_tensor(ctx, node)) + i12 * node->nb[2] + i13 * node->nb[3];

                Tensor input_a = entry->input_tensor_a;
                if (allow_input_chain && i12 == 0 && i13 == 0) {
                    input_a = *input_override;
                } else if (src1_is_bf16) {
                    // BF16 bridge path: copy BF16 data directly (2 bytes per element), no CPU conversion
                    ggml_fmsh_log_locked(ctx, 0, "bf16_bridge_fwd_enter i12=" + std::to_string(i12) + " i13=" + std::to_string(i13));
                    uint16_t * bridge_dst = reinterpret_cast<uint16_t *>(bridge_entry->input_tensor.data().cptr());
                    for (int64_t row = 0; row < entry->m; ++row) {
                        const char * a_src_row = src1_slice + row * src1->nb[1];
                        uint16_t * bridge_row_dst = bridge_dst + row * entry->k;
                        if (src1->nb[0] == static_cast<size_t>(sizeof(uint16_t))) {
                            std::memcpy(bridge_row_dst, a_src_row, static_cast<size_t>(entry->k) * sizeof(uint16_t));
                        } else {
                            for (int64_t col = 0; col < entry->k; ++col) {
                                uint16_t v = 0;
                                std::memcpy(&v, a_src_row + col * src1->nb[0], sizeof(uint16_t));
                                bridge_row_dst[col] = v;
                            }
                        }
                    }
                    Tensor bridge_output;
                    if (!ggml_fmsh_forward_synced(
                            ctx, bridge_entry->session, {bridge_entry->input_tensor},
                            bridge_entry->zg_device, bridge_entry->layer_increment,
                            &bridge_output, "bf16_bridge", err)) {
                        ggml_fmsh_log_locked(ctx, 3, "bf16_bridge_wait_ready_failed");
                        return false;
                    }
                    input_a = bridge_output;
                    ctx->mul_mat_device_input_chain++;
                    ggml_fmsh_log_locked(ctx, 0, "bf16_bridge_fwd_done chaining_to_matmul=1");
                } else {
                    // A = src1 slice : [m, k]
                    float * a_dst = reinterpret_cast<float *>(entry->input_tensor_a.data().cptr());
                    for (int64_t row = 0; row < entry->m; ++row) {
                        const char * a_src_row = src1_slice + row * src1->nb[1];
                        float * a_row_dst = a_dst + row * entry->k;
                        if (src1->nb[0] == static_cast<size_t>(sizeof(float))) {
                            const float * src = reinterpret_cast<const float *>(a_src_row);
                            for (int64_t col = 0; col < entry->k; ++col) {
                                a_row_dst[col] = src[col];
                            }
                        } else {
                            for (int64_t col = 0; col < entry->k; ++col) {
                                float v = 0.0f;
                                std::memcpy(&v, a_src_row + col * src1->nb[0], sizeof(float));
                                a_row_dst[col] = v;
                            }
                        }
                    }
                }

                // Only cache transpose for static src0 tensors.
                // Dynamic tensors (e.g. KV-related views) can keep same address but changed content across tokens.
                const bool src0_is_static =
                    (src0->view_src == nullptr) &&
                    (src0->op == GGML_OP_NONE) &&
                    ((src0->flags & GGML_TENSOR_FLAG_INPUT) == 0);

                Tensor dynamic_weight;
                const uintptr_t weight_key = reinterpret_cast<uintptr_t>(src0_slice);
                Tensor * weight_tensor = nullptr;
                if (src0_is_static) {
                    for (auto & cached : entry->weight_tensors) {
                        if (cached.key == weight_key) {
                            weight_tensor = &cached.tensor;
                            break;
                        }
                    }
                }

                if (!weight_tensor) {
                    // Build transposed weight in a host buffer: src0 [n, k] -> B [k, n]
                    const size_t weight_bytes = static_cast<size_t>(entry->k * entry->n) * sizeof(float);
                    std::vector<float> host_w_buf(static_cast<size_t>(entry->k * entry->n));
                    for (int64_t i = 0; i < entry->n; ++i) {
                        const char * w_src_row = src0_slice + i * src0->nb[1];
                        for (int64_t j = 0; j < entry->k; ++j) {
                            float v = 0.0f;
                            std::memcpy(&v, w_src_row + j * src0->nb[0], sizeof(float));
                            host_w_buf[static_cast<size_t>(j * entry->n + i)] = v;
                        }
                    }

                    Tensor new_weight(entry->input_type_b.clone());
                    // Static weights live on NPU PL memory to avoid per-token DMA re-upload.
                    // Dynamic weights (KV views) stay on host since they change each call.
                    if (src0_is_static && ctx->device_opened) {
                        new_weight.mallocOn(ctx->zg_device.defaultMemRegion());
                        new_weight.write(0, reinterpret_cast<char *>(host_w_buf.data()), weight_bytes);
                    } else {
                        new_weight.mallocOn(HostDevice::MemRegion());
                        std::memcpy(reinterpret_cast<float *>(new_weight.data().cptr()), host_w_buf.data(), weight_bytes);
                    }

                    if (src0_is_static) {
                        entry->weight_tensors.push_back({weight_key, std::move(new_weight)});
                        weight_tensor = &entry->weight_tensors.back().tensor;
                    } else {
                        dynamic_weight = std::move(new_weight);
                        weight_tensor = &dynamic_weight;
                    }
                }

                Tensor output;
                if (!ggml_fmsh_forward_synced(ctx, entry->session, {input_a, *weight_tensor},
                                              entry->zg_device, entry->layer_increment,
                                              &output, "mul_mat", err)) {
                    return false;
                }
                if (chained_output != nullptr && i12 == 0 && i13 == 0 && ne12 == 1 && ne13 == 1) {
                    *chained_output = output;
                }

                if (write_host_output) {
                    output.read(reinterpret_cast<char *>(out_tmp.data()), 0, out_bytes);

                    const ggml_fmsh_f32_stats out_stats =
                        calc_f32_stats(out_tmp.data(), static_cast<size_t>(entry->m * entry->n));
                    if (out_stats.inf_cnt > 0 || out_stats.nan_cnt > 0) {
                        const float * w_ptr = reinterpret_cast<const float *>(weight_tensor->data().cptr());
                        const ggml_fmsh_f32_stats w_stats =
                            calc_f32_stats(w_ptr, static_cast<size_t>(entry->k * entry->n));
                        std::string a_trace = "bridge_bf16";
                        if (!src1_is_bf16) {
                            const float * a_ptr = reinterpret_cast<const float *>(input_a.data().cptr());
                            const ggml_fmsh_f32_stats a_stats =
                                calc_f32_stats(a_ptr, static_cast<size_t>(entry->m * entry->k));
                            a_trace = "A[min,max,inf,nan]=[" + std::to_string(a_stats.min_v) + "," +
                                std::to_string(a_stats.max_v) + "," +
                                std::to_string(a_stats.inf_cnt) + "," +
                                std::to_string(a_stats.nan_cnt) + "]";
                        }
                        ggml_fmsh_log_locked(
                            ctx, 3,
                            "mul_mat_inf_trace shape=[" + std::to_string(node->ne[0]) + "," +
                            std::to_string(node->ne[1]) + "," +
                            std::to_string(node->ne[2]) + "," +
                            std::to_string(node->ne[3]) + "]" +
                            " tile_i12=" + std::to_string(i12) +
                            " tile_i13=" + std::to_string(i13) +
                            " src0_nb0=" + std::to_string(src0->nb[0]) +
                            " src1_nb0=" + std::to_string(src1->nb[0]) +
                            " dst_nb0=" + std::to_string(node->nb[0]) +
                            " " + a_trace +
                            " W[min,max,inf,nan]=[" + std::to_string(w_stats.min_v) + "," +
                            std::to_string(w_stats.max_v) + "," +
                            std::to_string(w_stats.inf_cnt) + "," +
                            std::to_string(w_stats.nan_cnt) + "]" +
                            " Y[min,max,inf,nan]=[" + std::to_string(out_stats.min_v) + "," +
                            std::to_string(out_stats.max_v) + "," +
                            std::to_string(out_stats.inf_cnt) + "," +
                            std::to_string(out_stats.nan_cnt) + "]");
                        std::exit(EXIT_FAILURE);
                    }

                    for (int64_t row = 0; row < entry->m; ++row) {
                        char * dst_row = dst_slice + row * node->nb[1];
                        const float * src_row = out_tmp.data() + row * entry->n;
                        if (node->nb[0] == static_cast<size_t>(sizeof(float))) {
                            std::memcpy(dst_row, src_row, static_cast<size_t>(entry->n * sizeof(float)));
                        } else {
                            for (int64_t col = 0; col < entry->n; ++col) {
                                std::memcpy(dst_row + col * node->nb[0], &src_row[col], sizeof(float));
                            }
                        }
                    }
                }
            }
        }
        if (write_host_output) {
            std::string sync_err;
            if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "mul_mat_readback", &sync_err)) {
                if (err) *err = sync_err;
                return false;
            }
        }
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

bool ggml_fmsh_execute_flash_attn_ext(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_flash_attn_session_entry * entry,
    std::string * err) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = node->src[3];
    if (!q || !k || !v || !node->data || !q->data) {
        if (err) *err = "FLASH_ATTN_EXT tensor data is null";
        return false;
    }

    // K and V may live in ZG DDR (fake pointers, not dereferenceable on host).
    // If so, D2H the entire root tensor into a staging buffer and temporarily
    // redirect k->data / v->data so that the existing per-head stride loops work correctly
    // even for permuted/view tensors whose strides differ from contiguous layout.
    // IMPORTANT: use RAII to restore the original pointers on ANY return path — including
    // early error returns — so that callers (e.g. CPU fallback) always see valid pointers.
    std::vector<char> k_root_staging, v_root_staging;
    void * k_orig_data = k->data;
    void * v_orig_data = v->data;
    struct KVRestore {
        const ggml_tensor * k; void * k_orig;
        const ggml_tensor * v; void * v_orig;
        ~KVRestore() {
            const_cast<ggml_tensor *>(k)->data = k_orig;
            const_cast<ggml_tensor *>(v)->data = v_orig;
        }
    } kv_restore{k, k_orig_data, v, v_orig_data};
    {
        auto stage_kv = [&](const ggml_tensor * t, std::vector<char> & staging) -> void * {
            MemChunk chunk; size_t off = 0;
            const ggml_tensor * root = t;
            while (root->view_src) root = root->view_src;
            if (!ggml_fmsh_is_zg_buffer_tensor(root, &chunk, &off)) return t->data;
            const size_t root_bytes = ggml_nbytes(root);
            staging.resize(root_bytes);
            chunk.read(staging.data(), off, root_bytes);
            // Redirect: t's host address = staging_base + (t_zg_addr - root_zg_addr).
            const uintptr_t root_zg = reinterpret_cast<uintptr_t>(root->data);
            const uintptr_t t_zg    = reinterpret_cast<uintptr_t>(t->data);
            return staging.data() + (t_zg - root_zg);
        };
        const_cast<ggml_tensor *>(k)->data = stage_kv(k, k_root_staging);
        const_cast<ggml_tensor *>(v)->data = stage_kv(v, v_root_staging);
    }
    const void * k_data = k->data;
    const void * v_data = v->data;
    std::string staging_err;
    if (!ggml_fmsh_ensure_zg_staging(ctx, q, true, &staging_err)) {
        if (err) *err = "FLASH_ATTN_EXT q staging failed: " + staging_err;
        return false;
    }
    const char * q_data = ggml_fmsh_host_data_for_tensor(ctx, q);


    const auto & cfg = entry->info;
    const int64_t head_dim = cfg.head_dim;
    const int64_t value_dim = cfg.value_dim;
    // Use actual tensor dimensions, not stale session-creation values.
    // Session is keyed by buckets, so actual <= bucket is always guaranteed.
    const int64_t q_len = q->ne[1];
    const int64_t kv_len = k->ne[1];
    const int64_t q_bucket = cfg.q_bucket;
    const int64_t kv_bucket = cfg.kv_bucket;
    const int64_t n_head = cfg.n_head;
    const int64_t n_head_kv = cfg.n_head_kv;
    const int64_t n_batch = cfg.n_batch;
    const int64_t n_batch_kv = k->ne[3];
    const int64_t rk2 = n_head / n_head_kv;
    const int64_t rk3 = n_batch / n_batch_kv;

    const size_t k_bytes = static_cast<size_t>(head_dim * kv_bucket) * sizeof(float);
    const size_t v_bytes = static_cast<size_t>(kv_bucket * value_dim) * sizeof(float);
    const size_t m_bytes = static_cast<size_t>(q_bucket * kv_bucket) * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(q_bucket * value_dim) * sizeof(float);
    const float k_mask_neg_inf = -1e9f;
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
    struct ggml_fmsh_f32_stats_local {
        float min_v = std::numeric_limits<float>::infinity();
        float max_v = -std::numeric_limits<float>::infinity();
        size_t inf_cnt = 0;
        size_t nan_cnt = 0;
    };
    const auto calc_f32_stats_local = [](const float * data, size_t n) -> ggml_fmsh_f32_stats_local {
        ggml_fmsh_f32_stats_local s;
        if (!data || n == 0) {
            s.min_v = 0.0f;
            s.max_v = 0.0f;
            return s;
        }
        for (size_t i = 0; i < n; ++i) {
            const float v = data[i];
            if (std::isnan(v)) {
                s.nan_cnt++;
                continue;
            }
            if (std::isinf(v)) {
                s.inf_cnt++;
                continue;
            }
            s.min_v = std::min(s.min_v, v);
            s.max_v = std::max(s.max_v, v);
        }
        if (!std::isfinite(s.min_v)) s.min_v = 0.0f;
        if (!std::isfinite(s.max_v)) s.max_v = 0.0f;
        return s;
    };
#endif

    for (size_t i = 0; i < entry->input_types.size(); ++i) {
        if (!entry->input_tensors[i].defined()) {
            entry->input_tensors[i] = Tensor(entry->input_types[i].clone());
            entry->input_tensors[i].mallocOn(HostDevice::MemRegion());
        }
    }

    float * in_q     = reinterpret_cast<float *>(entry->input_tensors[0].data().cptr());
    float * in_k     = reinterpret_cast<float *>(entry->input_tensors[1].data().cptr());
    float * in_v     = reinterpret_cast<float *>(entry->input_tensors[2].data().cptr());
    float * in_scale = reinterpret_cast<float *>(entry->input_tensors[3].data().cptr());
    float * in_mask  = reinterpret_cast<float *>(entry->input_tensors[4].data().cptr());
    float * in_softcap = nullptr;
    if (cfg.use_logit_softcap && entry->input_tensors.size() > 5) {
        in_softcap = reinterpret_cast<float *>(entry->input_tensors[5].data().cptr());
    }

    std::vector<float> out_tmp(static_cast<size_t>(q_bucket * value_dim), 0.0f);
    const uint32_t n_head_u32 = static_cast<uint32_t>(std::max<int64_t>(1, n_head));
    const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(n_head_u32))));
    const float m0 = std::pow(2.0f, -(cfg.max_bias) / n_head_log2);
    const float m1 = std::pow(2.0f, -(cfg.max_bias / 2.0f) / n_head_log2);

    for (int64_t ib = 0; ib < n_batch; ++ib) {
        const int64_t k_batch = ib / rk3;
        const int64_t mask_batch = mask ? (ib % mask->ne[3]) : 0;
        for (int64_t ih = 0; ih < n_head; ++ih) {
            const int64_t k_head = ih / rk2;
            const int64_t mask_head = mask ? (ih % mask->ne[2]) : 0;

            // Q: [q_bucket, head_dim]
            for (int64_t iq = 0; iq < q_bucket; ++iq) {
                float * q_row = in_q + static_cast<size_t>(iq * head_dim);
                if (iq < q_len) {
                    const float * src_q_row = reinterpret_cast<const float *>(
                        q_data + iq * q->nb[1] + ih * q->nb[2] + ib * q->nb[3]);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        q_row[d] = src_q_row[d];
                    }
                } else {
                    std::memset(q_row, 0, static_cast<size_t>(head_dim) * sizeof(float));
                }
            }

            // K: [head_dim, kv_bucket]
            std::memset(in_k, 0, k_bytes);
            for (int64_t ikv = 0; ikv < kv_len; ++ikv) {
                const char * src_k_row = static_cast<const char *>(k_data) +
                                         ikv * k->nb[1] + k_head * k->nb[2] + k_batch * k->nb[3];
                if (k->type == GGML_TYPE_F32) {
                    const float * src = reinterpret_cast<const float *>(src_k_row);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        in_k[static_cast<size_t>(d * kv_bucket + ikv)] = src[d];
                    }
                } else if (k->type == GGML_TYPE_BF16) {
                    const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(src_k_row);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        in_k[static_cast<size_t>(d * kv_bucket + ikv)] = GGML_BF16_TO_FP32(src[d]);
                    }
                } else {
                    const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(src_k_row);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        in_k[static_cast<size_t>(d * kv_bucket + ikv)] = GGML_FP16_TO_FP32(src[d]);
                    }
                }
            }

            // V: [kv_bucket, value_dim]
            std::memset(in_v, 0, v_bytes);
            for (int64_t ikv = 0; ikv < kv_len; ++ikv) {
                float * v_row = in_v + static_cast<size_t>(ikv * value_dim);
                const char * src_v_row = static_cast<const char *>(v_data) +
                                         ikv * v->nb[1] + k_head * v->nb[2] + k_batch * v->nb[3];
                if (v->type == GGML_TYPE_F32) {
                    const float * src = reinterpret_cast<const float *>(src_v_row);
                    for (int64_t d = 0; d < value_dim; ++d) {
                        v_row[d] = src[d];
                    }
                } else if (v->type == GGML_TYPE_BF16) {
                    const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(src_v_row);
                    for (int64_t d = 0; d < value_dim; ++d) {
                        v_row[d] = GGML_BF16_TO_FP32(src[d]);
                    }
                } else {
                    const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(src_v_row);
                    for (int64_t d = 0; d < value_dim; ++d) {
                        v_row[d] = GGML_FP16_TO_FP32(src[d]);
                    }
                }
            }

            // MASK: [q_bucket, kv_bucket]
            std::memset(in_mask, 0, m_bytes);
            const float slope = (cfg.max_bias > 0.0f)
                ? (ih < n_head_log2 ? std::pow(m0, static_cast<float>(ih + 1))
                                    : std::pow(m1, static_cast<float>(2 * (ih - n_head_log2) + 1)))
                : 1.0f;
            for (int64_t iq = 0; iq < q_bucket; ++iq) {
                float * m_row = in_mask + static_cast<size_t>(iq * kv_bucket);
                for (int64_t ikv = 0; ikv < kv_bucket; ++ikv) {
                    float mv;
                    if (ikv >= kv_len) {
                        mv = k_mask_neg_inf;
                    } else if (iq >= q_len || !mask) {
                        mv = 0.0f;
                    } else {
                        mv = ggml_fmsh_read_mask_f32(ctx, mask, ikv, iq, mask_head, mask_batch);
                        // ZG330 softmax path is unstable with +/-inf or NaN mask values.
                        // Normalize mask sent to NPU to finite numbers while preserving intent.
                        if (!std::isfinite(mv)) {
                            mv = (std::isnan(mv) || mv > 0.0f) ? 0.0f : k_mask_neg_inf;
                        }
                        if (cfg.max_bias > 0.0f) {
                            mv *= slope;
                        }
                    }
                    m_row[ikv] = mv;
                }
            }

            float scale = cfg.scale;
            if (cfg.use_logit_softcap) {
                scale /= cfg.logit_softcap;
            }
            in_scale[0] = scale;
            if (in_softcap != nullptr) {
                in_softcap[0] = cfg.logit_softcap;
            }

            std::vector<Tensor> inputs = entry->input_tensors;

            // AXI 同步修复（参考 flash-attn-netmake-cpu-test.cpp）：
            // forward() 后 ready_=true 永久置位，后续 waitForReady() 立即返回读到上帧数据。
            // 修复仅适用于 ARM/AXI 直连模式 —— socket 模式下 layerCount() 不是可靠的完成信号，
            // 直接用 waitForReady() 即可（icraft-xrt 在 socket 模式下自己管同步）。
#if defined(__aarch64__) || defined(_M_ARM64)
            const bool use_axi_sync = entry->zg_device.defined();
#else
            const bool use_axi_sync = false;
#endif
            uint32_t layer_before = 0;
            if (use_axi_sync) {
                layer_before = entry->zg_device.layerCount();
            }

            auto outputs = entry->session.forward(inputs);
            if (outputs.empty()) {
                if (err) *err = "FLASH_ATTN_EXT session.forward returned empty output";
                return false;
            }

            if (use_axi_sync) {
                if (entry->layer_increment == 0) {
                    // 第一次 forward：forward() 内部的 check_func_ 可能使用 apply() 时的绝对
                    // layerCount 目标，在完整推理中该目标早已超过，导致 waitForReady 立即返回。
                    // 修复：直接 spin-poll layerCount，等待其稳定（NPU 完成计算）。
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
                    // 等待 NPU 启动（layerCount 变化）
                    while (entry->zg_device.layerCount() == layer_before) {
                        if (std::chrono::steady_clock::now() > deadline) {
                            if (err) *err = "flash_attn_ext NPU layerCount sync timeout (start)";
                            return false;
                        }
                    }
                    // 等待 layerCount 稳定（NPU 完成所有层）：5ms 无变化则认为完成
                    uint32_t stable_count = entry->zg_device.layerCount();
                    auto stable_start = std::chrono::steady_clock::now();
                    while (std::chrono::steady_clock::now() < deadline) {
                        const uint32_t cur = entry->zg_device.layerCount();
                        if (cur != stable_count) {
                            stable_count = cur;
                            stable_start = std::chrono::steady_clock::now();
                        } else {
                            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - stable_start).count();
                            if (elapsed_ms >= 1) {
                                break;
                            }
                        }
                    }
                    entry->layer_increment = (stable_count > layer_before)
                        ? (stable_count - layer_before) : 1;
                    ggml_fmsh_log_locked(ctx, 1,
                        "axi_sync_first ib=" + std::to_string(ib) +
                        " ih=" + std::to_string(ih) +
                        " layer_before=" + std::to_string(layer_before) +
                        " stable_count=" + std::to_string(stable_count) +
                        " layer_increment=" + std::to_string(entry->layer_increment));
                } else {
                    // 后续 forward：ready_=true，需重置并安装指向本次目标的 check_func_。
                    // 旧 check_func_ 的 layer_target 已过时，waitForReady 会立即返回读到上帧数据。
                    const uint32_t layer_target = layer_before + entry->layer_increment;
                    ZG330Device zg_dev = entry->zg_device;
                    outputs[0].setReady(false);
                    outputs[0].setCheckFunc([zg_dev, layer_target](const Device &) -> bool {
                        return zg_dev.layerCount() >= layer_target;
                    });
                    if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "flash_attn_ext")) {
                        return false;
                    }
                    ggml_fmsh_log_locked(ctx, 0,
                        "axi_sync_sub ib=" + std::to_string(ib) +
                        " ih=" + std::to_string(ih) +
                        " layer_before=" + std::to_string(layer_before) +
                        " layer_target=" + std::to_string(layer_target) +
                        " layer_actual=" + std::to_string(entry->zg_device.layerCount()) +
                        " incr=" + std::to_string(entry->layer_increment));
                }
            } else {
                if (!ggml_fmsh_wait_tensor_ready(outputs[0], err, "flash_attn_ext")) {
                    return false;
                }
            }
            outputs[0].read(reinterpret_cast<char *>(out_tmp.data()), 0, y_bytes);
            // Debug: log first few output values for first 2 heads of first batch.
            if (use_axi_sync && ib == 0 && ih < 2) {
                static std::atomic<int> s_out_log_budget{16};
                if (s_out_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    std::string vals;
                    for (int64_t _d = 0; _d < std::min<int64_t>(4, value_dim); ++_d) {
                        vals += std::to_string(out_tmp[static_cast<size_t>(_d)]) + " ";
                    }
                    ggml_fmsh_log_locked(ctx, 1,
                        "axi_out ib=0 ih=" + std::to_string(ih) +
                        " iq0_vals=[" + vals + "]");
                }
            }

#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            const auto out_stats = calc_f32_stats_local(out_tmp.data(), static_cast<size_t>(q_bucket * value_dim));
            if (out_stats.inf_cnt > 0 || out_stats.nan_cnt > 0) {
                // Input tensors are fp32; Y output is also fp32.
                ggml_fmsh_log_locked(
                    ctx, 3,
                    "flash_inf_trace batch=" + std::to_string(ib) +
                    " head=" + std::to_string(ih) +
                    " k_head=" + std::to_string(k_head) +
                    " mask_head=" + std::to_string(mask_head) +
                    " q_len=" + std::to_string(q_len) +
                    " kv_len=" + std::to_string(kv_len) +
                    " q_bucket=" + std::to_string(q_bucket) +
                    " kv_bucket=" + std::to_string(kv_bucket) +
                    " scale=" + std::to_string(scale) +
                    " Y[min,max,inf,nan]=[" + std::to_string(out_stats.min_v) + "," +
                    std::to_string(out_stats.max_v) + "," +
                    std::to_string(out_stats.inf_cnt) + "," +
                    std::to_string(out_stats.nan_cnt) + "]");
                std::exit(EXIT_FAILURE);
            }
#endif

            for (int64_t iq = 0; iq < q_len; ++iq) {
                const float * out_row = out_tmp.data() + static_cast<size_t>(iq * value_dim);
                for (int64_t d = 0; d < value_dim; ++d) {
                    ggml_fmsh_write_f32_indexed(ctx, node, d, ih, iq, ib, out_row[d]);
                }
            }
            std::string sync_err;
            if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "flash_readback", &sync_err)) {
                if (err) *err = sync_err;
                return false;
            }
        }
    }

    return true;
}


} // namespace ggml_fmsh_zg330_impl
