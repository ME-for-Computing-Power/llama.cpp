#include "ggml-fmsh-zg330-internal.h"

namespace ggml_fmsh_zg330_impl {

ggml_fmsh_zg330_elementwise_session_entry * ggml_fmsh_get_or_create_elementwise_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    ggml::fmsh::netmake::ElementwiseZgOp kind;
    int64_t rows = 0;
    int64_t cols = 0;
    if (!ggml_fmsh_validate_elementwise(node, &kind, &rows, &cols)) {
        if (err) *err = "invalid elementwise shape/layout";
        return nullptr;
    }
    const ggml_fmsh_zg330_op_signature sig = ggml_fmsh_make_elementwise_signature(node, kind, rows, cols);
    auto it = ctx->elementwise_session_cache.find(sig);
    if (it != ctx->elementwise_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    int64_t compile_rows = ggml_fmsh_elementwise_compile_rows(kind, rows);

    const bool enable_user_connect =
        kind == ggml::fmsh::netmake::ElementwiseZgOp::MUL ||
        kind == ggml::fmsh::netmake::ElementwiseZgOp::ADD ||
        kind == ggml::fmsh::netmake::ElementwiseZgOp::SCALE ||
        kind == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM;

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_elementwise_zg_network(ctx->cache_dir, kind, compile_rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);

        MemChunk output_chunk;
        TensorType output_type;
        int64_t output_value_id = -1;
        size_t output_bytes = 0;
        if (enable_user_connect) {
            auto zg_backend = session->backends.at(0).cast<zg330::ZG330Backend>();
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork prepare op=" + std::string(ggml_op_name(node->op)));
            const auto & out_values = bundle.network.outputs();
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork outputs op=" + std::string(ggml_op_name(node->op)) +
                " count=" + std::to_string(out_values.size()));
            if (out_values.size() != 1) {
                if (err) *err = "fatal: elementwise outputs size != 1 for userConnectNetwork";
                return nullptr;
            }
            const auto & out_value = out_values[0];
            output_type = out_value.tensorType().clone();
            output_value_id = out_value->v_id;
            output_bytes = out_value.storageBytes();
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork alloc op=" + std::string(ggml_op_name(node->op)) +
                " v_id=" + std::to_string(output_value_id) +
                " bytes=" + std::to_string(output_bytes));
            output_chunk = ctx->zg_device.defaultMemRegion().malloc(output_bytes, true, 4096);
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork alloc_ok op=" + std::string(ggml_op_name(node->op)));

            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork try_backend op=" + std::string(ggml_op_name(node->op)));
            zg_backend.userConnectNetwork(output_chunk, output_value_id);
            ggml_fmsh_log_locked(
                ctx, 1,
                "userConnectNetwork ok op=" + std::string(ggml_op_name(node->op)) +
                " v_id=" + std::to_string(output_value_id) +
                " bytes=" + std::to_string(output_bytes));
        }
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_elementwise_session_entry>();
        entry->signature = sig;
        entry->hit_count = 1;
        entry->op = kind;
        entry->rows = rows;
        entry->cols = cols;
        entry->compiled_rows = compile_rows;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        entry->output_type = output_type;
        entry->output_chunk = output_chunk;
        entry->output_value_id = output_value_id;
        entry->output_bytes = output_bytes;
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }
        for (const auto & in : entry->bundle.network.inputs()) {
            entry->input_types.push_back(in.tensorType().clone());
        }
        entry->input_tensors.resize(entry->input_types.size());

        ggml_fmsh_zg330_elementwise_session_entry * ptr = entry.get();
        ctx->elementwise_session_cache.emplace(sig, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
}

ggml_fmsh_zg330_rmsnorm_split_session_entry * ggml_fmsh_get_or_create_rmsnorm_split_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    ggml::fmsh::netmake::ElementwiseZgOp kind;
    int64_t rows = 0, cols = 0;
    if (!ggml_fmsh_validate_elementwise(node, &kind, &rows, &cols)) {
        if (err) *err = "invalid rmsnorm shape/layout";
        return nullptr;
    }
    if (kind != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
        if (err) *err = "not an RMS_NORM op";
        return nullptr;
    }
    const int64_t compile_rows = ggml_fmsh_elementwise_compile_rows(kind, rows);
    const ggml_fmsh_zg330_op_signature sig = ggml_fmsh_make_elementwise_signature(node, kind, compile_rows, cols);
    auto it = ctx->rmsnorm_split_session_cache.find(sig);
    if (it != ctx->rmsnorm_split_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }
    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }
    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_rmsnorm_split_zg_networks(
            ctx->cache_dir, compile_rows, cols);

        Session session_pre = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network_pre.view(0), {ctx->zg_device, HostDevice::Default()});
        session_pre.enableTimeProfile(true);
        session_pre.apply();

        Session session_post = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network_post.view(0), {ctx->zg_device, HostDevice::Default()});
        session_post.enableTimeProfile(true);

        // Preallocate post output in ZG DDR so the result can stay on device and
        // be consumed directly by the next elementwise op (e.g. MUL) without D2H.
        MemChunk   post_output_chunk;
        TensorType post_output_type;
        int64_t    post_output_value_id = -1;
        size_t     post_output_bytes    = 0;
        {
            auto zg_backend = session_post->backends.at(0).cast<zg330::ZG330Backend>();
            const auto & out_values = bundle.network_post.outputs();
            if (out_values.size() == 1) {
                post_output_type     = out_values[0].tensorType().clone();
                post_output_value_id = out_values[0]->v_id;
                post_output_bytes    = out_values[0].storageBytes();
                post_output_chunk    = ctx->zg_device.defaultMemRegion().malloc(post_output_bytes, true, 4096);
                zg_backend.userConnectNetwork(post_output_chunk, post_output_value_id);
                ggml_fmsh_log_locked(ctx, 1,
                    "rmsnorm_post userConnectNetwork v_id=" + std::to_string(post_output_value_id) +
                    " bytes=" + std::to_string(post_output_bytes));
            }
        }
        session_post.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_rmsnorm_split_session_entry>();
        entry->signature     = sig;
        entry->hit_count     = 1;
        entry->rows          = rows;
        entry->cols          = cols;
        entry->compiled_rows = compile_rows;
        entry->bundle        = std::move(bundle);
        entry->session_pre   = std::move(session_pre);
        entry->session_post  = std::move(session_post);
        entry->post_output_type     = post_output_type;
        entry->post_output_chunk    = post_output_chunk;
        entry->post_output_value_id = post_output_value_id;
        entry->post_output_bytes    = post_output_bytes;
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }
        // Extract input types from both networks; X type from pre[0] (shared with post[0]).
        const auto & pre_inputs = entry->bundle.network_pre.inputs();
        const auto & post_inputs = entry->bundle.network_post.inputs();
        if (pre_inputs.size() < 2 || post_inputs.size() < 2) {
            if (err) *err = "rmsnorm_split: unexpected network input count";
            return nullptr;
        }
        entry->x_input_type    = pre_inputs[0].tensorType().clone();
        entry->eps_input_type  = pre_inputs[1].tensorType().clone();
        entry->r_inv_input_type = post_inputs[1].tensorType().clone();

        ggml_fmsh_zg330_rmsnorm_split_session_entry * ptr = entry.get();
        ctx->rmsnorm_split_session_cache.emplace(sig, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

ggml_fmsh_zg330_rope_session_entry * ggml_fmsh_get_or_create_rope_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    if (!node || node->op != GGML_OP_ROPE) {
        if (err) *err = "not a ROPE op";
        return nullptr;
    }
    if (node->type != GGML_TYPE_F32 || !node->src[0] || node->src[0]->type != GGML_TYPE_F32) {
        if (err) *err = "ROPE: unsupported dtype";
        return nullptr;
    }
    const int32_t rope_mode = ((const int32_t *) node->op_params)[2];
    // Support NeoX (mode 2), IMROPE (mode 40=NEOX|MROPE|8) used by Qwen3.5 text.
    // For text-only IMROPE the 4 position groups (t/h/w/extra) are all equal,
    // so the rotation is identical to plain NeoX — we read pos[i2] (time positions).
    // Pure MROPE mode 8 (vision cross-modal) is not supported.
    const bool is_neox  = (rope_mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool is_imrope = rope_mode == GGML_ROPE_TYPE_IMROPE; // 40
    const bool is_mrope_only = (rope_mode == GGML_ROPE_TYPE_MROPE);   // 8, vision/cross-modal
    if (!is_neox && !is_imrope && rope_mode != 0) {
        if (err) *err = "ROPE: unsupported mode (not NeoX/IMROPE)";
        return nullptr;
    }
    if (is_mrope_only) {
        if (err) *err = "ROPE: pure MROPE (vision) not supported on PL";
        return nullptr;
    }
    const int64_t cols = node->ne[0]; // head_dim
    if (cols <= 0 || cols % 2 != 0) {
        if (err) *err = "ROPE: head_dim must be even";
        return nullptr;
    }
    const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3]; // n_head * seq * batch
    if (rows <= 0) {
        if (err) *err = "ROPE: zero rows";
        return nullptr;
    }

    ggml_fmsh_zg330_op_signature sig = {};
    sig.op = static_cast<uint32_t>(node->op);
    sig.dtype = static_cast<uint32_t>(node->type);
    sig.shape[0] = cols;
    sig.shape[1] = rows;
    sig.shape[2] = 0;
    sig.shape[3] = 0;

    auto it = ctx->rope_session_cache.find(sig);
    if (it != ctx->rope_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_rope_zg_network(ctx->cache_dir, rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_rope_session_entry>();
        entry->signature = sig;
        entry->hit_count = 1;
        entry->rows = rows;
        entry->cols = cols;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }
        for (const auto & in : entry->bundle.network.inputs()) {
            entry->input_types.push_back(in.tensorType().clone());
        }
        entry->input_tensors.resize(entry->input_types.size());

        ggml_fmsh_zg330_rope_session_entry * ptr = entry.get();
        ctx->rope_session_cache.emplace(sig, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

ggml_fmsh_zg330_fused_ew_entry * ggml_fmsh_get_or_create_fused_ew_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_fmsh_zg330_fused_ew_chain & chain,
    bool * created,
    std::string * err) {
    // Build net_name: "fused_add_rmsnorm_mul_bf16_2x1024"
    auto op_name_str = [](ggml::fmsh::netmake::ElementwiseZgOp op) -> std::string {
        switch (op) {
            case ggml::fmsh::netmake::ElementwiseZgOp::ADD:      return "add";
            case ggml::fmsh::netmake::ElementwiseZgOp::MUL:      return "mul";
            case ggml::fmsh::netmake::ElementwiseZgOp::SCALE:    return "scale";
            case ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM: return "rmsnorm";
            default:                                              return "unknown";
        }
    };
    std::string net_name = "fused";
    for (auto op : chain.ops) {
        net_name += "_" + op_name_str(op);
    }
    net_name += "_bf16_" + std::to_string(chain.compiled_rows) + "x" + std::to_string(chain.cols);

    auto it = ctx->fused_ew_session_cache.find(net_name);
    if (it != ctx->fused_ew_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_fused_ew_zg_network(
            ctx->cache_dir, chain.ops, chain.compiled_rows, chain.cols);

        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);

        MemChunk output_chunk;
        int64_t output_value_id = -1;
        size_t output_bytes = 0;
        {
            auto zg_backend = session->backends.at(0).cast<zg330::ZG330Backend>();
            const auto & out_values = bundle.network.outputs();
            if (out_values.size() != 1) {
                if (err) *err = "fatal: fused_ew outputs size != 1";
                return nullptr;
            }
            const auto & out_value = out_values[0];
            output_value_id = out_value->v_id;
            output_bytes = out_value.storageBytes();
            output_chunk = ctx->zg_device.defaultMemRegion().malloc(output_bytes, true, 4096);
            zg_backend.userConnectNetwork(output_chunk, output_value_id);
            ggml_fmsh_log_locked(ctx, 1,
                "fused_ew_userConnectNetwork net=" + net_name +
                " v_id=" + std::to_string(output_value_id) +
                " bytes=" + std::to_string(output_bytes));
        }

        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_fused_ew_entry>();
        entry->net_name      = net_name;
        entry->hit_count     = 1;
        entry->ops           = chain.ops;
        entry->rows          = chain.rows;
        entry->cols          = chain.cols;
        entry->compiled_rows = chain.compiled_rows;
        entry->bundle        = std::move(bundle);
        entry->session       = std::move(session);
        entry->output_chunk  = output_chunk;
        entry->output_value_id = output_value_id;
        entry->output_bytes  = output_bytes;
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }
        for (const auto & in : entry->bundle.network.inputs()) {
            entry->input_types.push_back(in.tensorType().clone());
        }
        entry->input_tensors.resize(entry->input_types.size());

        ggml_fmsh_zg330_fused_ew_entry * ptr = entry.get();
        ctx->fused_ew_session_cache.emplace(net_name, std::move(entry));
        if (created) *created = true;
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_session_create net=" + net_name);
        return ptr;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return nullptr;
    }
}

ggml_fmsh_zg330_flash_attn_session_entry * ggml_fmsh_get_or_create_flash_attn_session(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * node,
    bool * created,
    std::string * err) {
    ggml_fmsh_flash_attn_validation v;
    if (!ggml_fmsh_validate_flash_attn_ext(ctx, node, &v, err)) {
        return nullptr;
    }

    std::string cache_key = ggml_fmsh_make_flash_attn_cache_key(node, v);
    auto it = ctx->flash_attn_session_cache.find(cache_key);
    if (it != ctx->flash_attn_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    std::string last_compile_err = "unknown";
    const int max_softmax_col_retries = 4;
    for (int attempt = 0; attempt <= max_softmax_col_retries; ++attempt) {
        cache_key = ggml_fmsh_make_flash_attn_cache_key(node, v);
        it = ctx->flash_attn_session_cache.find(cache_key);
        if (it != ctx->flash_attn_session_cache.end()) {
            it->second->hit_count++;
            if (created) *created = false;
            return it->second.get();
        }

        try {
            auto bundle = ggml::fmsh::netmake::get_or_compile_flash_attn_zg_network(
                ctx->cache_dir,
                v.head_dim,
                v.value_dim,
                v.q_bucket,
                v.kv_bucket,
                v.softmax_cols,
                v.use_logit_softcap);
            Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
                bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
            session.enableTimeProfile(true);
            session.apply();

            auto entry = std::make_unique<ggml_fmsh_zg330_flash_attn_session_entry>();
            entry->signature = ggml_fmsh_make_signature(node);
            entry->info = v;
            entry->hit_count = 1;
            entry->bundle = std::move(bundle);
            entry->session = std::move(session);
            if (ctx->zg_device.is<ZG330Device>()) {
                entry->zg_device = ctx->zg_device.cast<ZG330Device>();
            }
            for (const auto & in : entry->bundle.network.inputs()) {
                entry->input_types.push_back(in.tensorType().clone());
            }
            entry->input_tensors.resize(entry->input_types.size());

            ggml_fmsh_zg330_flash_attn_session_entry * ptr = entry.get();
            ctx->flash_attn_session_cache.emplace(cache_key, std::move(entry));
            if (created) *created = true;

            int64_t pre_kv = v.kv_bucket;
            for (int64_t d = 0; d < ctx->flash_precompile_kv_depth; ++d) {
                pre_kv *= 2;
                if (pre_kv <= 0 || pre_kv > ctx->flash_kv_bucket_max) {
                    break;
                }
                try {
                    (void) ggml::fmsh::netmake::get_or_compile_flash_attn_zg_network(
                        ctx->cache_dir,
                        v.head_dim,
                        v.value_dim,
                        v.q_bucket,
                        pre_kv,
                        v.softmax_cols,
                        v.use_logit_softcap);
                } catch (...) {
                    break;
                }
            }
            return ptr;
        } catch (const std::exception & e) {
            last_compile_err = e.what();
            v.softmax_cols = ggml_fmsh_next_pow2_i64(v.softmax_cols + 1);
        }
    }

    if (err) {
        *err = "FLASH_ATTN_EXT compile failed: " + last_compile_err;
    }
    return nullptr;
}

std::string ggml_fmsh_make_bf16_bridge_cache_key(int64_t rows, int64_t cols) {
    return "bf16_bridge|" + std::to_string(rows) + "x" + std::to_string(cols);
}

ggml_fmsh_zg330_bf16_bridge_session_entry * ggml_fmsh_get_or_create_bf16_bridge_session(
    ggml_backend_fmsh_zg330_context * ctx,
    int64_t rows,
    int64_t cols,
    bool * created,
    std::string * err) {
    const std::string cache_key = ggml_fmsh_make_bf16_bridge_cache_key(rows, cols);
    auto it = ctx->bridge_session_cache.find(cache_key);
    if (it != ctx->bridge_session_cache.end()) {
        it->second->hit_count++;
        if (created) *created = false;
        return it->second.get();
    }

    if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
        return nullptr;
    }

    try {
        auto bundle = ggml::fmsh::netmake::get_or_compile_bf16_bridge_zg_network(ctx->cache_dir, rows, cols);
        Session session = Session::Create<zg330::ZG330Backend, HostBackend>(
            bundle.network.view(0), {ctx->zg_device, HostDevice::Default()});
        session.enableTimeProfile(true);
        session.apply();

        auto entry = std::make_unique<ggml_fmsh_zg330_bf16_bridge_session_entry>();
        entry->rows = rows;
        entry->cols = cols;
        entry->hit_count = 1;
        entry->bundle = std::move(bundle);
        entry->session = std::move(session);
        entry->input_type = entry->bundle.network.inputs()[0].tensorType().clone();
        if (ctx->zg_device.is<ZG330Device>()) {
            entry->zg_device = ctx->zg_device.cast<ZG330Device>();
        }

        ggml_fmsh_zg330_bf16_bridge_session_entry * ptr = entry.get();
        ctx->bridge_session_cache.emplace(cache_key, std::move(entry));
        if (created) *created = true;
        return ptr;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return nullptr;
    }
}

inline float ggml_fmsh_read_f32_broadcast(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3);

inline void ggml_fmsh_write_f32_indexed(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * t,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3,
    float v);
bool ggml_fmsh_write_f32_flat_to_staging(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * t,
    const float * src,
    size_t n,
    std::string * err);

inline float ggml_fmsh_read_f32_linear(const ggml_tensor * t, size_t idx);
inline float ggml_fmsh_read_f32_linear(ggml_backend_fmsh_zg330_context * ctx, const ggml_tensor * t, size_t idx);
const char * ggml_fmsh_host_data_for_tensor(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t);
float ggml_fmsh_read_mask_f32(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * mask,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3);

bool ggml_fmsh_execute_elementwise(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_elementwise_session_entry * entry,
    const Tensor * input_override,
    int input_override_src_idx,
    bool keep_device_output_only,
    bool write_host_output,
    Tensor * chained_output,
    std::string * err) {
    if (!node || !entry || node->data == nullptr || node->src[0] == nullptr || node->src[0]->data == nullptr) {
        if (err) *err = "elementwise tensor data is null";
        return false;
    }
    try {
        ggml_fmsh_log_locked(ctx, 1, "elementwise_begin op=" + std::string(ggml_op_name(node->op)));
        const int64_t exec_rows = entry->compiled_rows > 0 ? entry->compiled_rows : entry->rows;
        const int64_t actual_rows = ggml_fmsh_node_rows(node);
        const size_t total_elems = static_cast<size_t>(exec_rows * entry->cols);
        const size_t total_bytes = total_elems * sizeof(float);
        for (size_t i = 0; i < entry->input_types.size(); ++i) {
            if (!entry->input_tensors[i].defined()) {
                entry->input_tensors[i] = Tensor(entry->input_types[i].clone());
                entry->input_tensors[i].mallocOn(HostDevice::MemRegion());
            }
        }

        if (chained_output != nullptr) {
            *chained_output = Tensor();
        }
        uint16_t * in0 = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
        const ggml_tensor * src0 = node->src[0];
        const bool can_chain_input0 =
            input_override != nullptr &&
            input_override_src_idx == 0 &&
            entry->op != ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX &&
            entry->op != ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM;
        const bool can_chain_input1 =
            input_override != nullptr &&
            input_override_src_idx == 1 &&
            (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::ADD ||
             entry->op == ggml::fmsh::netmake::ElementwiseZgOp::MUL);
        Tensor input0 = can_chain_input0 ? *input_override : entry->input_tensors[0];

        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX) {
            float scale = 1.0f;
            float max_bias = 0.0f;
            std::memcpy(&scale, (float *) node->op_params + 0, sizeof(float));
            std::memcpy(&max_bias, (float *) node->op_params + 1, sizeof(float));

            const ggml_tensor * src1 = node->src[1];
            const ggml_tensor * src2 = node->src[2];
            if (src2 != nullptr) {
                if (err) *err = "SOFT_MAX with sinks(src2) is not supported";
                return false;
            }

            const int64_t ne12 = src1 ? src1->ne[2] : 1;
            const int64_t ne13 = src1 ? src1->ne[3] : 1;
            const uint32_t n_head = static_cast<uint32_t>(node->ne[2]);
            const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(n_head))));
            const float m0 = std::pow(2.0f, -(max_bias) / n_head_log2);
            const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
            const bool use_f16 = src1 && src1->type == GGML_TYPE_F16;

            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const uint32_t h = static_cast<uint32_t>(i2);
                        const float slope = (max_bias > 0.0f)
                            ? (h < n_head_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_head_log2) + 1))
                            : 1.0f;

                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                        const float * sp = reinterpret_cast<const float *>(
                            static_cast<const char *>(src0->data) + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);

                        const int64_t i11 = i1;
                        const int64_t i12 = i2 % ne12;
                        const int64_t i13 = i3 % ne13;
                        const char * mp_base = src1
                            ? static_cast<const char *>(src1->data) + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]
                            : nullptr;

                        for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                            float v = sp[i0] * scale;
                            if (mp_base) {
                                const float mv = use_f16
                                    ? GGML_FP16_TO_FP32(reinterpret_cast<const ggml_fp16_t *>(mp_base)[i0])
                                    : reinterpret_cast<const float *>(mp_base)[i0];
                                v += slope * mv;
                            }
                            in0[row_off + static_cast<size_t>(i0)] = ggml_fmsh_f32_to_bf16(v);
                        }
                    }
                }
            }
        } else if (!can_chain_input0) {
            if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::CPY ||
                entry->op == ggml::fmsh::netmake::ElementwiseZgOp::DUP) {
                const size_t n = ggml_nelements(node);
                for (size_t idx = 0; idx < n; ++idx) {
                    in0[idx] = ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_linear(ctx, src0, idx));
                }
            } else {
                for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                            const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                            const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                            for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                                in0[row_off + static_cast<size_t>(i0)] = ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src0, i0, i1, i2, i3));
                            }
                        }
                    }
                }
            }
        }
        if (!can_chain_input0 && exec_rows > actual_rows) {
            uint16_t * in0_pad = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
            const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
            for (int64_t r = actual_rows; r < exec_rows; ++r) {
                std::memcpy(
                    in0_pad + static_cast<size_t>(r * entry->cols),
                    in0_pad + static_cast<size_t>((actual_rows - 1) * entry->cols),
                    row_bytes);
            }
        }

        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::ADD || entry->op == ggml::fmsh::netmake::ElementwiseZgOp::MUL) {
            if (!node->src[1] || !node->src[1]->data) {
                if (err) *err = "elementwise binary src1 is null";
                return false;
            }
            uint16_t * in1 = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
            const ggml_tensor * src1 = node->src[1];
            if (!can_chain_input1) {
                for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                            const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                            const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                            for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                                in1[row_off + static_cast<size_t>(i0)] =
                                    ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src1, i0, i1, i2, i3));
                            }
                        }
                    }
                }
            }
            if (exec_rows > actual_rows) {
                uint16_t * in1_pad = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
                const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(
                        in1_pad + static_cast<size_t>(r * entry->cols),
                        in1_pad + static_cast<size_t>((actual_rows - 1) * entry->cols),
                        row_bytes);
                }
            }
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SCALE) {
            float scale = 1.0f;
            if (node->src[1] && node->src[1]->data && ggml_nelements(node->src[1]) >= 1) {
                // SCALE's factor tensor can be a view/strided scalar; always read via ggml strides.
                scale = ggml_fmsh_read_f32_broadcast(ctx, node->src[1], 0, 0, 0, 0);
            } else {
                std::memcpy(&scale, node->op_params, sizeof(float));
            }
            const uint16_t scale_bf16 = ggml_fmsh_f32_to_bf16(scale);
            std::memcpy(entry->input_tensors[1].data().cptr(), &scale_bf16, sizeof(uint16_t));
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::CPY ||
                   entry->op == ggml::fmsh::netmake::ElementwiseZgOp::DUP) {
            const uint16_t one_bf16 = ggml_fmsh_f32_to_bf16(1.0f);
            std::memcpy(entry->input_tensors[1].data().cptr(), &one_bf16, sizeof(uint16_t));
        } else if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
            float eps = 0.0f;
            std::memcpy(&eps, node->op_params, sizeof(float));
            const uint16_t eps_bf16 = ggml_fmsh_f32_to_bf16(eps);
            std::memcpy(entry->input_tensors[1].data().cptr(), &eps_bf16, sizeof(uint16_t));
        }
        if (entry->op == ggml::fmsh::netmake::ElementwiseZgOp::SOFT_MAX &&
            node->src[1] != nullptr &&
            node->src[1]->data != nullptr &&
            entry->input_tensors.size() > 1 &&
            entry->input_tensors[1].defined() &&
            exec_rows > actual_rows) {
            uint16_t * in1_pad = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
            const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
            for (int64_t r = actual_rows; r < exec_rows; ++r) {
                std::memcpy(
                    in1_pad + static_cast<size_t>(r * entry->cols),
                    in1_pad + static_cast<size_t>((actual_rows - 1) * entry->cols),
                    row_bytes);
            }
        }

        if (can_chain_input0) {
            // D2H copy: PLDDR F32 → BF16 host buffer. Read actual_rows only; the PLDDR
            // tensor from the previous op may have fewer rows than exec_rows (e.g. ADD
            // compiled_rows=1 but RMS_NORM compiled_rows=2 for 1-row decode tensors).
            const size_t act_elems =
                static_cast<size_t>(actual_rows) * static_cast<size_t>(entry->cols);
            std::vector<float> d2h_buf(act_elems);
            input0.read(reinterpret_cast<char *>(d2h_buf.data()), 0, act_elems * sizeof(float));
            for (size_t idx = 0; idx < act_elems; ++idx) {
                in0[idx] = ggml_fmsh_f32_to_bf16(d2h_buf[idx]);
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(in0 + r * entry->cols, in0 + (actual_rows - 1) * entry->cols, row_bytes);
                }
            }
        }
        if (can_chain_input1) {
            const size_t act_elems =
                static_cast<size_t>(actual_rows) * static_cast<size_t>(entry->cols);
            std::vector<float> d2h_buf(act_elems);
            input_override->read(reinterpret_cast<char *>(d2h_buf.data()), 0, act_elems * sizeof(float));
            uint16_t * in1 = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
            for (size_t idx = 0; idx < act_elems; ++idx) {
                in1[idx] = ggml_fmsh_f32_to_bf16(d2h_buf[idx]);
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(entry->cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(in1 + r * entry->cols, in1 + (actual_rows - 1) * entry->cols, row_bytes);
                }
            }
        }
        std::vector<Tensor> session_inputs = entry->input_tensors;
        Tensor effective_output;
        if (!ggml_fmsh_forward_synced(
                ctx, entry->session, session_inputs, entry->zg_device,
                entry->layer_increment, &effective_output, "elementwise", err)) {
            return false;
        }
        if (entry->output_chunk.defined()) {
            if (effective_output.chunk() == entry->output_chunk) {
                ctx->elementwise_output_chunk_hits++;
            } else {
                ctx->elementwise_output_chunk_misses++;
                ggml_fmsh_log_locked(
                    ctx, 2,
                    "userConnectNetwork chunk mismatch op=" + std::string(ggml_op_name(node->op)) +
                    " v_id=" + std::to_string(entry->output_value_id) +
                    " bytes=" + std::to_string(entry->output_bytes));
            }
        }
        if (chained_output != nullptr) {
            *chained_output = effective_output;
        }
        ggml_fmsh_log_locked(ctx, 1, "elementwise_forward_done op=" + std::string(ggml_op_name(node->op)));

        // When the dispatcher retains this output for device-input chaining, avoid
        // forcing a readback here. The debug-compare path already skips chained
        // ops, so an unconditional readback only breaks the chain and rewrites the
        // staging buffer without providing any validation value.
        if (write_host_output) {
            std::vector<float> out_tmp(total_elems);
            effective_output.read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);
            ggml_fmsh_log_locked(ctx, 1, "elementwise_read_done op=" + std::string(ggml_op_name(node->op)));
            // Fast path: contiguous ZG-buffer output with matching logical size.
            // Write the flat F32 readback into staging in the node's storage dtype,
            // avoiding the per-element indexed loop for F32/F16/BF16 tensors.
            const bool fast_readback =
                (node->type == GGML_TYPE_F32 || node->type == GGML_TYPE_F16 || node->type == GGML_TYPE_BF16) &&
                ggml_is_contiguous(node) &&
                ggml_nelements(node) == static_cast<int64_t>(total_elems) &&
                ggml_fmsh_is_zg_buffer_tensor(node, nullptr, nullptr);
            if (fast_readback) {
                std::string stg_err;
                if (!ggml_fmsh_write_f32_flat_to_staging(ctx, node, out_tmp.data(), total_elems, &stg_err)) {
                    if (err) *err = stg_err;
                    return false;
                }
                ggml_fmsh_log_locked(ctx, 1, "elementwise_fast_readback op=" + std::string(ggml_op_name(node->op)) +
                    " bytes=" + std::to_string(total_bytes));
            } else {
                for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                            const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                            const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(node->ne[0]);
                            for (int64_t i0 = 0; i0 < node->ne[0]; ++i0) {
                                ggml_fmsh_write_f32_indexed(
                                    ctx, node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                            }
                        }
                    }
                }
            }
            std::string sync_err;
            if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "elementwise_readback", &sync_err)) {
                if (err) *err = sync_err;
                return false;
            }
        } else {
            ggml_fmsh_log_locked(ctx, 0, "elementwise_read_skip op=" + std::string(ggml_op_name(node->op)));
        }
        ggml_fmsh_log_locked(ctx, 1, "elementwise_end op=" + std::string(ggml_op_name(node->op)));
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
}

// Single-pass RMS_NORM via custom_op_hub hardware (bypasses icraft compiler).
bool ggml_fmsh_execute_rmsnorm_customop(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    int64_t rows, int64_t cols,
    const Tensor * input_override,
    bool keep_device_output,
    bool write_host_output,
    Tensor * chained_output,
    std::string * err) {
    if (!node || !node->data || !node->src[0] || !node->src[0]->data) {
        if (err) *err = "rmsnorm_customop: tensor data is null";
        return false;
    }
    try {
        // A. Hardware probe (once, cached in rms_norm_custom_op_verified).
        if (!ctx->rms_norm_custom_op_verified) {
            if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
                return false;
            }
            auto & reg = ctx->zg_device.defaultRegRegion();
            reg.write(kCopBase + kCopRegHubSelW, kCopHubSelRmsnorm, false);
            uint32_t sel_readback = static_cast<uint32_t>(
                reg.read(kCopBase + kCopRegHubSelR, false));
            if (sel_readback != kCopHubSelRmsnorm) {
                if (err) *err = "rmsnorm_customop: hub select readback mismatch sel=" + std::to_string(sel_readback);
                return false;
            }
            uint32_t ver = static_cast<uint32_t>(
                reg.read(kCopBase + kCopRegVersion, false));
            if (ver != kCopRmsnormVersion) {
                if (err) *err = "rmsnorm_customop: version mismatch got=0x" +
                    ([](uint32_t v) { char buf[16]; std::snprintf(buf, sizeof(buf), "%08x", v); return std::string(buf); })(ver) +
                    " expected=0x" +
                    ([](uint32_t v) { char buf[16]; std::snprintf(buf, sizeof(buf), "%08x", v); return std::string(buf); })(kCopRmsnormVersion);
                return false;
            }
            ctx->rms_norm_custom_op_verified = true;
            ggml_fmsh_log_locked(ctx, 1, "rmsnorm_customop_verified version=0x" +
                ([](uint32_t v) { char buf[16]; std::snprintf(buf, sizeof(buf), "%08x", v); return std::string(buf); })(ver));
        }

        // B. PLDDR memory allocation (cached by cols, realloc on row growth).
        const uint32_t rsize_beats = static_cast<uint32_t>(cols / static_cast<int64_t>(kCopBf16PerBeat));
        const uint32_t wsize_beats = rsize_beats; // BF16 output
        const size_t input_bytes  = static_cast<size_t>(rows) * static_cast<size_t>(rsize_beats) * kCopBeatBytes;
        const size_t output_bytes = static_cast<size_t>(rows) * static_cast<size_t>(wsize_beats) * kCopBeatBytes;
        const size_t total_bytes  = input_bytes + output_bytes;

        // Use a single shared chunk (keyed by cols=0) to keep the allocation
        // at the lowest possible PLDDR address and avoid address-dependent
        // hardware DMA issues observed with cols=128 at higher addresses.
        auto & cache_entry_ptr = ctx->rmsnorm_customop_cache[0];
        if (!cache_entry_ptr) {
            cache_entry_ptr = std::make_unique<ggml_fmsh_zg330_rmsnorm_customop_entry>();
            cache_entry_ptr->cols = 0;
        }
        auto * centry = cache_entry_ptr.get();
        centry->hit_count++;

        if (!centry->mem_chunk.defined() || static_cast<size_t>(centry->alloc_rows) < total_bytes) {
            if (!ggml_fmsh_open_device_if_needed(ctx, err)) {
                return false;
            }
            // Pre-allocate for worst case: max supported cols * reasonable max rows.
            const size_t prealloc = std::max(total_bytes, static_cast<size_t>(1024 / 32) * 64 * 64 * 2);
            centry->mem_chunk = ctx->zg_device.defaultMemRegion().malloc(prealloc, true, 4096);
            centry->alloc_rows = static_cast<int64_t>(prealloc);
            ggml_fmsh_log_locked(ctx, 0, "rmsnorm_customop_alloc cols=" + std::to_string(cols) +
                " rows=" + std::to_string(rows) + " bytes=" + std::to_string(total_bytes));
        }

        // C. Data preparation: F32→BF16 conversion and DMA write.
        const ggml_tensor * src0 = node->src[0];
        const size_t act_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        std::vector<uint16_t> in_bf16(act_elems);
        if (input_override != nullptr) {
            std::vector<float> d2h_buf(act_elems);
            input_override->read(reinterpret_cast<char *>(d2h_buf.data()), 0, act_elems * sizeof(float));
            for (size_t idx = 0; idx < act_elems; ++idx) {
                in_bf16[idx] = ggml_fmsh_f32_to_bf16(d2h_buf[idx]);
            }
        } else {
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < cols; ++i0) {
                            in_bf16[row_off + static_cast<size_t>(i0)] =
                                ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src0, i0, i1, i2, i3));
                        }
                    }
                }
            }
        }

        centry->mem_chunk.write(0, reinterpret_cast<char *>(in_bf16.data()), input_bytes);

        // Compute physical addresses.
        uint32_t mem_addr = centry->mem_chunk->begin.addr();
        uint32_t src_addr = mem_addr;
        uint32_t dst_addr = mem_addr + static_cast<uint32_t>(input_bytes);

        uint32_t reg_read_cfg  = (rsize_beats << 16) | (src_addr & 0xFFFFU);
        uint32_t reg_write_cfg = (wsize_beats << 16) | (dst_addr & 0xFFFFU);
        uint32_t reg_addr_hi   = ((dst_addr >> 16) << 16) | (src_addr >> 16);

        // Stale state detection and reset.
        auto & reg = ctx->zg_device.defaultRegRegion();

        // Re-assert hub select and unconditionally reset before every launch.
        reg.write(kCopBase + kCopRegHubSelW, kCopHubSelRmsnorm, false);
        reg.write(kCopBase + kCopRegReset, 1U, false);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        reg.write(kCopBase + kCopRegReset, 0U, false);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Write registers and start.
        float eps = 0.0f;
        std::memcpy(&eps, node->op_params, sizeof(float));

        reg.write(kCopBase + kCopRegAddrHi,   reg_addr_hi,   false);
        reg.write(kCopBase + kCopRegReadCfg,  reg_read_cfg,  false);
        reg.write(kCopBase + kCopRegWriteCfg, reg_write_cfg, false);
        reg.write(kCopBase + kCopRegCfg0,     2U,            false); // BF16 output + precise_rsqrt
        reg.write(kCopBase + kCopRegShape0,   static_cast<uint32_t>(rows), false);
        reg.write(kCopBase + kCopRegEps,      ggml_fmsh_float_to_u32(eps), false);
        reg.write(kCopBase + kCopRegStart,    1U,            false);

        // D. Poll for completion.
        auto t_start = std::chrono::steady_clock::now();
        for (;;) {
            uint32_t done   = static_cast<uint32_t>(reg.read(kCopBase + kCopRegDone, false));
            uint32_t status = static_cast<uint32_t>(reg.read(kCopBase + kCopRegStatus, false));
            if ((status & kCopErrorMask) != 0U) {
                reg.write(kCopBase + kCopRegReset, 1U, false);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                reg.write(kCopBase + kCopRegReset, 0U, false);
                if (err) *err = "rmsnorm_customop: hardware error status=0x" +
                    ([](uint32_t v) { char buf[16]; std::snprintf(buf, sizeof(buf), "%08x", v); return std::string(buf); })(status);
                return false;
            }
            if ((done & kCopDoneMask) != 0U) {
                break;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            if (elapsed > kCopTimeoutMs) {
                reg.write(kCopBase + kCopRegReset, 1U, false);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                reg.write(kCopBase + kCopRegReset, 0U, false);
                if (err) *err = "rmsnorm_customop: timeout after " + std::to_string(elapsed) + "ms";
                return false;
            }
        }

        // Read back BF16 output and convert to F32.
        std::vector<uint8_t> out_raw(output_bytes);
        centry->mem_chunk.read(reinterpret_cast<char *>(out_raw.data()), input_bytes, output_bytes);

        // Reset hardware after readback to clear internal state before next invocation.
        reg.write(kCopBase + kCopRegReset, 1U, false);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        reg.write(kCopBase + kCopRegReset, 0U, false);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        const uint16_t * out_bf16 = reinterpret_cast<const uint16_t *>(out_raw.data());
        const size_t out_elems = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        std::vector<float> out_f32(out_elems);
        for (size_t idx = 0; idx < out_elems; ++idx) {
            uint32_t tmp = static_cast<uint32_t>(out_bf16[idx]) << 16;
            std::memcpy(&out_f32[idx], &tmp, sizeof(float));
        }

        if (write_host_output) {
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < cols; ++i0) {
                            ggml_fmsh_write_f32_indexed(ctx, node, i0, i1, i2, i3,
                                out_f32[row_off + static_cast<size_t>(i0)]);
                        }
                    }
                }
            }
            std::string sync_err;
            if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "rmsnorm_customop_readback", &sync_err)) {
                if (err) *err = sync_err;
                return false;
            }
        }

        if (keep_device_output && chained_output != nullptr) {
            const size_t f32_bytes = out_elems * sizeof(float);
            if (!centry->output_chunk.defined() || centry->output_chunk->byte_size < f32_bytes) {
                centry->output_chunk = ctx->zg_device.defaultMemRegion().malloc(f32_bytes, true, 4096);
            }
            centry->output_chunk.write(0, reinterpret_cast<char *>(out_f32.data()), f32_bytes);
            icraft::xir::TensorType ttype(
                icraft::xir::FloatType::FP32(),
                icraft::xir::Array<int64_t>{static_cast<int64_t>(out_elems)},
                icraft::xir::Layout());
            *chained_output = Tensor(ttype, centry->output_chunk, 0);
        } else if (chained_output != nullptr) {
            *chained_output = Tensor();
        }

        ggml_fmsh_log_locked(ctx, 1, "rmsnorm_customop_done rows=" + std::to_string(rows) +
            " cols=" + std::to_string(cols) + " hits=" + std::to_string(centry->hit_count) +
            " keep_device=" + std::to_string(keep_device_output ? 1 : 0) +
            " write_host=" + std::to_string(write_host_output ? 1 : 0));
        return true;
    } catch (const std::exception & e) {
        if (err) *err = std::string("rmsnorm_customop: ") + e.what();
        return false;
    }
}

// Two-phase RMS_NORM execution: PL computes r² per row, host does scalar sqrt, PL applies X*r_inv.
// X is written once into a shared buffer used by both pre and post networks.
bool ggml_fmsh_execute_rmsnorm_split(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_rmsnorm_split_session_entry * entry,
    const Tensor * input_override,    // device-resident X from upstream (or nullptr)
    bool keep_device_output,          // if true, leave post output in ZG DDR
    bool write_host_output,
    Tensor * chained_output,          // out: device tensor if keep_device_output
    std::string * err) {
    if (!node || !entry || node->data == nullptr || node->src[0] == nullptr || node->src[0]->data == nullptr) {
        if (err) *err = "rmsnorm_split: tensor data is null";
        return false;
    }
    if (chained_output) *chained_output = Tensor();
    try {
        const int64_t exec_rows   = entry->compiled_rows;
        const int64_t actual_rows = ggml_fmsh_node_rows(node);
        const int64_t cols        = entry->cols;

        // Lazily allocate shared X tensor (used by both pre and post).
        if (!entry->x_input_tensor.defined()) {
            entry->x_input_tensor = Tensor(entry->x_input_type.clone());
            entry->x_input_tensor.mallocOn(HostDevice::MemRegion());
        }
        if (!entry->eps_input_tensor.defined()) {
            entry->eps_input_tensor = Tensor(entry->eps_input_type.clone());
            entry->eps_input_tensor.mallocOn(HostDevice::MemRegion());
        }
        if (!entry->r_inv_input_tensor.defined()) {
            entry->r_inv_input_tensor = Tensor(entry->r_inv_input_type.clone());
            entry->r_inv_input_tensor.mallocOn(HostDevice::MemRegion());
        }

        // Fill X: if upstream produced a device-resident tensor, D2H from it;
        // otherwise read from src[0] host memory.
        uint16_t * x_buf = reinterpret_cast<uint16_t *>(entry->x_input_tensor.data().cptr());
        if (input_override != nullptr) {
            // Upstream (e.g. ADD) left output in ZG DDR — read bf16 directly.
            const size_t act_bytes = static_cast<size_t>(actual_rows) * static_cast<size_t>(cols) * sizeof(float);
            std::vector<float> d2h(static_cast<size_t>(actual_rows) * static_cast<size_t>(cols));
            input_override->read(reinterpret_cast<char *>(d2h.data()), 0, act_bytes);
            for (size_t idx = 0; idx < d2h.size(); ++idx) {
                x_buf[idx] = ggml_fmsh_f32_to_bf16(d2h[idx]);
            }
        } else {
            const ggml_tensor * src0 = node->src[0];
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row    = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < cols; ++i0) {
                            x_buf[row_off + static_cast<size_t>(i0)] =
                                ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src0, i0, i1, i2, i3));
                        }
                    }
                }
            }
        }
        if (exec_rows > actual_rows) {
            const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
            for (int64_t r = actual_rows; r < exec_rows; ++r) {
                std::memcpy(x_buf + r * cols, x_buf + (actual_rows - 1) * cols, row_bytes);
            }
        }

        float eps = 0.0f;
        std::memcpy(&eps, node->op_params, sizeof(float));
        const uint16_t eps_bf16 = ggml_fmsh_f32_to_bf16(eps);
        std::memcpy(entry->eps_input_tensor.data().cptr(), &eps_bf16, sizeof(uint16_t));

        // Run pre: [X, eps] → r_sq[exec_rows, 1].
        std::vector<Tensor> pre_inputs = {entry->x_input_tensor, entry->eps_input_tensor};
        Tensor pre_output;
        if (!ggml_fmsh_forward_synced(
                ctx, entry->session_pre, pre_inputs, entry->zg_device,
                entry->layer_increment_pre, &pre_output, "rmsnorm_split_pre", err)) {
            return false;
        }

        // Read r_sq. Session cache is bucketed by compiled_rows, but the runtime output
        // tensor may be retyped to actual_rows on reuse; clamp to the tensor's live size.
        const int64_t pre_tensor_rows = std::max<int64_t>(1, pre_output.dtype().numElements());
        const int64_t pre_read_rows = std::min<int64_t>(exec_rows, pre_tensor_rows);
        std::vector<float> r_sq(static_cast<size_t>(pre_read_rows));
        pre_output.read(reinterpret_cast<char *>(r_sq.data()), 0,
                        static_cast<size_t>(pre_read_rows) * sizeof(float));

        // Compute r_inv on host.
        uint16_t * r_inv_buf = reinterpret_cast<uint16_t *>(entry->r_inv_input_tensor.data().cptr());
        for (int64_t r = 0; r < exec_rows; ++r) {
            const int64_t src_r = std::min<int64_t>(r, pre_read_rows - 1);
            const float rsq_val = r_sq[static_cast<size_t>(src_r)];
            const float r_inv = (rsq_val > 0.0f) ? (1.0f / std::sqrt(rsq_val)) : 0.0f;
            r_inv_buf[static_cast<size_t>(r)] = ggml_fmsh_f32_to_bf16(r_inv);
        }

        // Evict any tensor already occupying post_output_chunk.
        // (mirrors the elementwise output_chunk eviction pattern)
        // Run post: [X (shared, no re-upload needed), r_inv] → Y.
        std::vector<Tensor> post_inputs = {entry->x_input_tensor, entry->r_inv_input_tensor};
        Tensor post_output;
        if (!ggml_fmsh_forward_synced(
                ctx, entry->session_post, post_inputs, entry->zg_device,
                entry->layer_increment_post, &post_output, "rmsnorm_split_post", err)) {
            return false;
        }

        // Check output chunk consistency.
        if (entry->post_output_chunk.defined()) {
            if (post_output.chunk() != entry->post_output_chunk) {
                ggml_fmsh_log_locked(ctx, 2, "rmsnorm_split post output chunk mismatch");
            }
        }

        const bool should_readback = write_host_output;
        if (should_readback) {
            const int64_t post_tensor_elems = std::max<int64_t>(1, post_output.dtype().numElements());
            const int64_t post_tensor_rows = std::max<int64_t>(1, post_tensor_elems / std::max<int64_t>(1, cols));
            const int64_t post_read_rows = std::min<int64_t>(actual_rows, post_tensor_rows);
            const size_t total_elems = static_cast<size_t>(post_read_rows) * static_cast<size_t>(cols);
            std::vector<float> out_tmp(total_elems);
            post_output.read(reinterpret_cast<char *>(out_tmp.data()), 0, total_elems * sizeof(float));
            for (int64_t i3 = 0; i3 < node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < node->ne[1]; ++i1) {
                        const int64_t row    = ((i3 * node->ne[2]) + i2) * node->ne[1] + i1;
                        const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < cols; ++i0) {
                            ggml_fmsh_write_f32_indexed(ctx, node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                        }
                    }
                }
            }
            std::string sync_err;
            if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "rmsnorm_readback", &sync_err)) {
                if (err) *err = sync_err;
                return false;
            }
        } else if (chained_output != nullptr) {
            // Return device tensor for the next consumer to use directly.
            Tensor dev_out = post_output;
            if (entry->post_output_chunk.defined() && dev_out.chunk() == entry->post_output_chunk) {
                // Retype to actual rows if compiled_rows > actual_rows.
                if (exec_rows != actual_rows && entry->post_output_type.defined()) {
                    TensorType rtyped = entry->post_output_type.clone();
                    rtyped.setShape(icraft::xir::Array<int64_t>{actual_rows, cols});
                    dev_out = Tensor(rtyped, dev_out.chunk(), dev_out.offset());
                }
            }
            *chained_output = dev_out;
            ggml_fmsh_log_locked(ctx, 0, "rmsnorm_split keep_device_output=1");
        }
        ggml_fmsh_log_locked(ctx, 1, "rmsnorm_split_done rows=" + std::to_string(actual_rows) + " cols=" + std::to_string(cols));
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

// ROPE NeoX execution: compute theta table on host, upload, run PL network.
bool ggml_fmsh_execute_rope(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * node,
    ggml_fmsh_zg330_rope_session_entry * entry,
    std::string * err) {
    if (!node || !entry || node->data == nullptr || !node->src[0] || node->src[0]->data == nullptr) {
        if (err) *err = "rope_exec: null tensor";
        return false;
    }
    try {
        // Lazily allocate input tensors.
        for (size_t i = 0; i < entry->input_types.size(); ++i) {
            if (!entry->input_tensors[i].defined()) {
                entry->input_tensors[i] = Tensor(entry->input_types[i].clone());
                entry->input_tensors[i].mallocOn(HostDevice::MemRegion());
            }
        }

        const ggml_tensor * src0 = node->src[0]; // X[ne0=d, ne1=n_head, ne2=seq, ne3=batch]
        const ggml_tensor * src1 = node->src[1]; // positions [ne0=seq]

        const int32_t n_dims     = ((const int32_t *) node->op_params)[1];
        float freq_base  = 10000.0f;
        float freq_scale = 1.0f;
        std::memcpy(&freq_base,  (const float *) node->op_params + 5, sizeof(float));
        std::memcpy(&freq_scale, (const float *) node->op_params + 6, sizeof(float));

        const int64_t ne0 = node->ne[0]; // head_dim = cols
        const int64_t ne1 = node->ne[1]; // n_head
        const int64_t ne2 = node->ne[2]; // seq_len
        const int64_t ne3 = node->ne[3]; // batch
        const int64_t half = ne0 / 2;

        // Fill X input (bf16, row-major [rows, cols]).
        uint16_t * x_in = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const int64_t row = (i3 * ne2 + i2) * ne1 + i1;
                    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(ne0);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        x_in[row_off + static_cast<size_t>(i0)] =
                            ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src0, i0, i1, i2, i3));
                    }
                }
            }
        }

        // Fill THETA input (bf16, row-major [rows, half]).
        // Range-reduce to [-π, π] on host so the ZG330 Horner polynomial stays accurate.
        // cos/sin are evaluated entirely on ZG330 via the compiled Horner network.
        uint16_t * theta_in = reinterpret_cast<uint16_t *>(entry->input_tensors[1].data().cptr());
        const int32_t * pos_data = (src1 && src1->data)
            ? reinterpret_cast<const int32_t *>(ggml_fmsh_host_data_for_tensor(ctx, src1))
            : nullptr;
        constexpr float kTwoPi = 6.28318530717958647692f;
        constexpr float kPi    = 3.14159265358979323846f;
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const int32_t pos = pos_data
                    ? pos_data[i2 < src1->ne[0] ? i2 : src1->ne[0] - 1]
                    : static_cast<int32_t>(i2);
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const int64_t row = (i3 * ne2 + i2) * ne1 + i1;
                    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(half);
                    for (int64_t j = 0; j < half; ++j) {
                        float theta;
                        if (j < n_dims / 2) {
                            theta = static_cast<float>(pos) * freq_scale *
                                std::pow(freq_base, -2.0f * static_cast<float>(j) / static_cast<float>(n_dims));
                        } else {
                            theta = 0.0f;  // passthrough: cos=1, sin=0
                        }
                        // Reduce to [-π, π] so the degree-12 Horner stays within its accurate range.
                        theta = std::fmod(theta, kTwoPi);
                        if (theta >  kPi) theta -= kTwoPi;
                        if (theta < -kPi) theta += kTwoPi;
                        theta_in[row_off + static_cast<size_t>(j)] = ggml_fmsh_f32_to_bf16(theta);
                    }
                }
            }
        }

        // Run the PL ROPE network.
        Tensor output;
        if (!ggml_fmsh_forward_synced(
                ctx, entry->session, entry->input_tensors, entry->zg_device,
                entry->layer_increment, &output, "rope_exec", err)) {
            return false;
        }

        // Read output Y and write to host.
        const size_t total_elems = static_cast<size_t>(entry->rows) * static_cast<size_t>(ne0);
        const size_t total_bytes = total_elems * sizeof(float);
        std::vector<float> out_tmp(total_elems);
        output.read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);

        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    const int64_t row = (i3 * ne2 + i2) * ne1 + i1;
                    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(ne0);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ggml_fmsh_write_f32_indexed(
                            ctx, node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                    }
                }
            }
        }
        std::string sync_err;
        if (!ggml_fmsh_sync_zg_staging_to_device(ctx, node, "rope_readback", &sync_err)) {
            if (err) *err = sync_err;
            return false;
        }
        ggml_fmsh_log_locked(ctx, 1, "rope_exec_done rows=" + std::to_string(entry->rows) + " cols=" + std::to_string(ne0));
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}

bool ggml_fmsh_execute_fused_ew(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_fmsh_zg330_fused_ew_chain & chain,
    ggml_fmsh_zg330_fused_ew_entry * entry,
    const Tensor * input_override,
    bool keep_device_output_only,
    bool write_host_output,
    Tensor * chained_output,
    std::string * err) {
    ggml_tensor * out_node = chain.nodes.back();
    ggml_tensor * in_node  = chain.nodes[0];

    if (!out_node || !in_node || out_node->data == nullptr) {
        if (err) *err = "fused_ew: tensor data is null";
        return false;
    }
    if (input_override == nullptr && (in_node->src[0] == nullptr || in_node->src[0]->data == nullptr)) {
        if (err) *err = "fused_ew: primary src[0] is null";
        return false;
    }

    try {
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_begin net=" + entry->net_name);

        const int64_t exec_rows   = entry->compiled_rows;
        const int64_t actual_rows = chain.rows;
        const int64_t cols        = entry->cols;
        const size_t  total_elems = static_cast<size_t>(exec_rows) * static_cast<size_t>(cols);
        const size_t  total_bytes = total_elems * sizeof(float);

        // Lazily allocate host input tensors.
        for (size_t idx = 0; idx < entry->input_types.size(); ++idx) {
            if (!entry->input_tensors[idx].defined()) {
                entry->input_tensors[idx] = Tensor(entry->input_types[idx].clone());
                entry->input_tensors[idx].mallocOn(HostDevice::MemRegion());
            }
        }

        if (chained_output) *chained_output = Tensor();

        // ── Primary input (input_tensors[0]) ──────────────────────────────────
        uint16_t * in0 = reinterpret_cast<uint16_t *>(entry->input_tensors[0].data().cptr());
        if (input_override != nullptr) {
            // D2H: PLDDR F32 → BF16 host buffer. Read actual_rows only; the PLDDR tensor
            // from the previous op may have fewer rows than exec_rows (RMS_NORM pads 1→2).
            const size_t act_elems = static_cast<size_t>(actual_rows) * static_cast<size_t>(cols);
            std::vector<float> d2h_buf(act_elems);
            input_override->read(reinterpret_cast<char *>(d2h_buf.data()), 0, act_elems * sizeof(float));
            for (size_t idx = 0; idx < act_elems; ++idx) {
                in0[idx] = ggml_fmsh_f32_to_bf16(d2h_buf[idx]);
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(in0 + r * cols, in0 + (actual_rows - 1) * cols, row_bytes);
                }
            }
        } else {
            const ggml_tensor * src0 = in_node->src[0];
            for (int64_t i3 = 0; i3 < in_node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < in_node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < in_node->ne[1]; ++i1) {
                        const int64_t row     = ((i3 * in_node->ne[2]) + i2) * in_node->ne[1] + i1;
                        const size_t  row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < in_node->ne[0]; ++i0) {
                            in0[row_off + static_cast<size_t>(i0)] =
                                ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src0, i0, i1, i2, i3));
                        }
                    }
                }
            }
            if (exec_rows > actual_rows) {
                const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
                for (int64_t r = actual_rows; r < exec_rows; ++r) {
                    std::memcpy(
                        in0 + r * cols,
                        in0 + (actual_rows - 1) * cols,
                        row_bytes);
                }
            }
        }

        // ── Secondary inputs ──────────────────────────────────────────────────
        int secondary_idx = 1;
        for (size_t k = 0; k < chain.ops.size(); ++k) {
            const auto     op    = chain.ops[k];
            ggml_tensor * knode = chain.nodes[k];
            if (op == ggml::fmsh::netmake::ElementwiseZgOp::ADD ||
                op == ggml::fmsh::netmake::ElementwiseZgOp::MUL) {
                if (!knode->src[1] || !knode->src[1]->data) {
                    if (err) *err = "fused_ew: ADD/MUL src[1] null at op " + std::to_string(k);
                    return false;
                }
                uint16_t * sec = reinterpret_cast<uint16_t *>(
                    entry->input_tensors[secondary_idx].data().cptr());
                const ggml_tensor * src1  = knode->src[1];
                const int64_t       krows = knode->ne[1] * knode->ne[2] * knode->ne[3];
                for (int64_t i3 = 0; i3 < knode->ne[3]; ++i3) {
                    for (int64_t i2 = 0; i2 < knode->ne[2]; ++i2) {
                        for (int64_t i1 = 0; i1 < knode->ne[1]; ++i1) {
                            const int64_t row     = ((i3 * knode->ne[2]) + i2) * knode->ne[1] + i1;
                            const size_t  row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                            for (int64_t i0 = 0; i0 < knode->ne[0]; ++i0) {
                                sec[row_off + static_cast<size_t>(i0)] =
                                    ggml_fmsh_f32_to_bf16(ggml_fmsh_read_f32_broadcast(ctx, src1, i0, i1, i2, i3));
                            }
                        }
                    }
                }
                if (exec_rows > krows) {
                    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(uint16_t);
                    for (int64_t r = krows; r < exec_rows; ++r) {
                        std::memcpy(
                            sec + r * cols,
                            sec + (krows - 1) * cols,
                            row_bytes);
                    }
                }
                secondary_idx++;
            } else if (op == ggml::fmsh::netmake::ElementwiseZgOp::SCALE) {
                float scale = 1.0f;
                if (knode->src[1] && knode->src[1]->data && ggml_nelements(knode->src[1]) >= 1) {
                    scale = ggml_fmsh_read_f32_broadcast(ctx, knode->src[1], 0, 0, 0, 0);
                } else {
                    std::memcpy(&scale, knode->op_params, sizeof(float));
                }
                const uint16_t sv = ggml_fmsh_f32_to_bf16(scale);
                std::memcpy(entry->input_tensors[secondary_idx].data().cptr(), &sv, sizeof(uint16_t));
                secondary_idx++;
            } else if (op == ggml::fmsh::netmake::ElementwiseZgOp::RMS_NORM) {
                float eps = 0.0f;
                std::memcpy(&eps, knode->op_params, sizeof(float));
                const uint16_t ev = ggml_fmsh_f32_to_bf16(eps);
                std::memcpy(entry->input_tensors[secondary_idx].data().cptr(), &ev, sizeof(uint16_t));
                secondary_idx++;
            }
        }

        // ── Forward ───────────────────────────────────────────────────────────
        std::vector<Tensor> session_inputs = entry->input_tensors;
        Tensor output;
        if (!ggml_fmsh_forward_synced(
                ctx, entry->session, session_inputs, entry->zg_device,
                entry->layer_increment, &output, "fused_ew", err)) {
            return false;
        }

        if (chained_output) *chained_output = output;
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_forward_done net=" + entry->net_name);

        // Same rationale as execute_elementwise(): chained outputs must stay on
        // device even when debug-compare is enabled, because chained ops are
        // already excluded from comparison.
        if (write_host_output) {
            std::vector<float> out_tmp(total_elems);
            output.read(reinterpret_cast<char *>(out_tmp.data()), 0, total_bytes);
            for (int64_t i3 = 0; i3 < out_node->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < out_node->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < out_node->ne[1]; ++i1) {
                        const int64_t row     = ((i3 * out_node->ne[2]) + i2) * out_node->ne[1] + i1;
                        const size_t  row_off = static_cast<size_t>(row) * static_cast<size_t>(cols);
                        for (int64_t i0 = 0; i0 < out_node->ne[0]; ++i0) {
                            ggml_fmsh_write_f32_indexed(
                                ctx, out_node, i0, i1, i2, i3, out_tmp[row_off + static_cast<size_t>(i0)]);
                        }
                    }
                }
            }
            std::string sync_err;
            if (!ggml_fmsh_sync_zg_staging_to_device(ctx, out_node, "fused_ew_readback", &sync_err)) {
                if (err) *err = sync_err;
                return false;
            }
        } else {
            ggml_fmsh_log_locked(ctx, 0, "fused_ew_read_skip net=" + entry->net_name);
        }
        ggml_fmsh_log_locked(ctx, 1, "fused_ew_end net=" + entry->net_name);
        return true;
    } catch (const std::exception & e) {
        if (err) *err = e.what();
        return false;
    }
}


// ─── Helper function definitions ─────────────────────────────────────────

const char * ggml_fmsh_host_data_for_tensor(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * t) {
    if (!ctx || !t) {
        return t ? static_cast<const char *>(t->data) : nullptr;
    }
    const ggml_tensor * root = t;
    while (root->view_src) {
        root = root->view_src;
    }
    auto it = ctx->zg_staging_bufs.find(root);
    if (it == ctx->zg_staging_bufs.end() || it->second.empty()) {
        return static_cast<const char *>(t->data);
    }
    const uintptr_t root_zg = reinterpret_cast<uintptr_t>(root->data);
    const uintptr_t t_zg = reinterpret_cast<uintptr_t>(t->data);
    return it->second.data() + (t_zg - root_zg);
}

float ggml_fmsh_read_mask_f32(
    ggml_backend_fmsh_zg330_context * ctx,
    const ggml_tensor * mask,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
    const char * base = ggml_fmsh_host_data_for_tensor(ctx, mask);
    const char * p = base +
                     i0 * mask->nb[0] + i1 * mask->nb[1] +
                     i2 * mask->nb[2] + i3 * mask->nb[3];
    if (mask->type == GGML_TYPE_F16) {
        return GGML_FP16_TO_FP32(*reinterpret_cast<const ggml_fp16_t *>(p));
    }
    return *reinterpret_cast<const float *>(p);
}

bool ggml_fmsh_write_f32_flat_to_staging(
    ggml_backend_fmsh_zg330_context * ctx,
    ggml_tensor * t,
    const float * src,
    size_t n,
    std::string * err) {
    if (!ctx || !t || !src) {
        if (err) *err = "flat_write: null input";
        return false;
    }
    if (!ggml_is_contiguous(t)) {
        if (err) *err = "flat_write: tensor is not contiguous";
        return false;
    }
    if (ggml_nelements(t) != static_cast<int64_t>(n)) {
        if (err) *err = "flat_write: element count mismatch";
        return false;
    }
    if (!ggml_fmsh_is_zg_buffer_tensor(t, nullptr, nullptr)) {
        if (err) *err = "flat_write: tensor is not a ZG buffer";
        return false;
    }
    if (!ggml_fmsh_ensure_zg_staging(ctx, t, false, err)) {
        return false;
    }

    char * staging_base = const_cast<char *>(ggml_fmsh_host_data_for_tensor(ctx, t));
    switch (t->type) {
        case GGML_TYPE_F32:
            std::memcpy(staging_base, src, n * sizeof(float));
            return true;
        case GGML_TYPE_F16:
            ggml_fp32_to_fp16_row(src, reinterpret_cast<ggml_fp16_t *>(staging_base), static_cast<int64_t>(n));
            return true;
        case GGML_TYPE_BF16:
            ggml_fp32_to_bf16_row(src, reinterpret_cast<ggml_bf16_t *>(staging_base), static_cast<int64_t>(n));
            return true;
        default:
            if (err) *err = "flat_write: unsupported tensor type";
            return false;
    }
}

} // namespace ggml_fmsh_zg330_impl
