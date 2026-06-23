#include "ggml-fmsh-zg330-internal.h"

namespace ggml_fmsh_zg330_impl {

ggml_backend_i ggml_backend_fmsh_zg330_i = {
    /* .get_name                = */ ggml_backend_fmsh_zg330_get_name,
    /* .free                    = */ ggml_backend_fmsh_zg330_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
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

ggml_guid_t ggml_backend_fmsh_zg330_guid(void) {
    static ggml_guid guid = { 0x46, 0xac, 0x9f, 0x39, 0x21, 0x44, 0x59, 0x63, 0x91, 0x85, 0xe2, 0x66, 0x17, 0x8c, 0x04, 0x30 };
    return &guid;
}

// Forward declaration (defined after buffer callbacks).
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(
        ggml_fmsh_zg330_device_context * dev_ctx, ggml_backend_dev_t dev);

ggml_backend_t ggml_backend_fmsh_zg330_init_impl(ggml_fmsh_zg330_device_context * dev_ctx) {
    auto * ctx = new ggml_backend_fmsh_zg330_context;
    ctx->cpu_backend = ggml_backend_cpu_init();
    if (!ctx->cpu_backend) {
        delete ctx;
        return nullptr;
    }

    ctx->strict_mode = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_STRICT", false);
    ctx->enable_log = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_LOG", true);
    ctx->log_level = ggml_fmsh_get_log_level();
    ctx->offload_cpy_dup = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_CPY_DUP", true);
    ctx->offload_soft_max = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_SOFT_MAX", true);
    ctx->offload_rms_norm = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_RMS_NORM", true);
    ctx->rms_norm_native_sqrt = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_RMS_NORM_NATIVE_SQRT", false);
    ctx->rms_norm_custom_op = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_RMS_NORM_CUSTOM_OP", true);
    ctx->offload_rope = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_ROPE", true);
    ctx->offload_flash_attn_ext = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_FLASH_ATTN_EXT", true);
    ctx->disable_device_input_chain = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_DISABLE_DEVICE_INPUT_CHAIN", false);
    ctx->flash_softmax_cu = std::max<int64_t>(1, static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_FLASH_SOFTMAX_CU", 8)));
    ctx->flash_precompile_kv_depth = static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_FLASH_PRECOMPILE_KV_DEPTH", 1));
    ctx->flash_kv_bucket_max = std::max<int64_t>(1, static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_FLASH_KV_BUCKET_MAX", 8192)));
    ctx->offload_quant_weights = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_OFFLOAD_QUANT_WEIGHTS", true);
    ctx->mul_mat_m_bucket_max = std::max<int64_t>(1, static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_MUL_MAT_M_BUCKET_MAX", 1)));
    ctx->mul_mat_n_chunk = static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_MUL_MAT_N_CHUNK", 16384));
    ctx->mul_mat_precompile_m_depth = static_cast<int64_t>(ggml_fmsh_get_env_u64("GGML_FMSH_ZG330_MUL_MAT_PRECOMPILE_M_DEPTH", 0));
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
    ctx->debug_compare = ggml_fmsh_get_env_bool("GGML_FMSH_ZG330_DEBUG_COMPARE", true);
    ctx->debug_compare_atol = ggml_fmsh_get_env_double("GGML_FMSH_ZG330_DEBUG_COMPARE_ATOL", 1e-1);
    ctx->debug_compare_rtol = ggml_fmsh_get_env_double("GGML_FMSH_ZG330_DEBUG_COMPARE_RTOL", 1e-1);
#endif

    
    const char * cache_dir = std::getenv("GGML_FMSH_ZG330_CACHE_DIR");
    ctx->cache_dir = cache_dir && cache_dir[0] ? std::filesystem::path(cache_dir) : std::filesystem::path(".cache/ggml-fmsh_zg330");
    std::error_code ec;
    std::filesystem::create_directories(ctx->cache_dir, ec);
    ctx->log_file = ctx->cache_dir / "backend.log";



    // delete prev log file if exists, and create a new empty log
    if (std::filesystem::exists(ctx->log_file)) {
        std::filesystem::remove(ctx->log_file, ec);
        std::ofstream ofs(ctx->log_file, std::ios::out);

        // write current timestamp to log
        const auto now = std::chrono::system_clock::now();
        const std::time_t now_c = std::chrono::system_clock::to_time_t
            (now - std::chrono::hours(24)); // backdate 24h to ensure log is always newer than any cached session
        ofs << "Log start time: " << std::ctime(&now_c) << std::flush;

        // release ofs
        ofs.close();
    }

    
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_fmsh_zg330_guid(),
        /* .iface   = */ ggml_backend_fmsh_zg330_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_fmsh_zg330_reg(), 0),
        /* .context = */ ctx,
    };

    {
        std::lock_guard<std::mutex> lock(ctx->mu);
        ggml_fmsh_log_locked(
            ctx, 1,
            "backend init strict=" + std::to_string(ctx->strict_mode ? 1 : 0) +
            " log_level=" + std::to_string(ctx->log_level) +
            " offload_cpy_dup=" + std::to_string(ctx->offload_cpy_dup ? 1 : 0) +
            " offload_soft_max=" + std::to_string(ctx->offload_soft_max ? 1 : 0) +
            " offload_rms_norm=" + std::to_string(ctx->offload_rms_norm ? 1 : 0) +
            " rms_norm_native_sqrt=" + std::to_string(ctx->rms_norm_native_sqrt ? 1 : 0) +
            " rms_norm_custom_op=" + std::to_string(ctx->rms_norm_custom_op ? 1 : 0) +
            " offload_flash_attn_ext=" + std::to_string(ctx->offload_flash_attn_ext ? 1 : 0) +
            " flash_softmax_cu=" + std::to_string(ctx->flash_softmax_cu) +
            " flash_precompile_kv_depth=" + std::to_string(ctx->flash_precompile_kv_depth) +
            " flash_kv_bucket_max=" + std::to_string(ctx->flash_kv_bucket_max) +
            " offload_quant_weights=" + std::to_string(ctx->offload_quant_weights ? 1 : 0) +
            " mul_mat_m_bucket_max=" + std::to_string(ctx->mul_mat_m_bucket_max) +
            " mul_mat_n_chunk=" + std::to_string(ctx->mul_mat_n_chunk) +
            " mul_mat_precompile_m_depth=" + std::to_string(ctx->mul_mat_precompile_m_depth) +
#ifdef GGML_FMSH_ZG330_DEBUG_COMPARE
            " debug_compare=" + std::to_string(ctx->debug_compare ? 1 : 0) +
            " debug_compare_atol=" + std::to_string(ctx->debug_compare_atol) +
            " debug_compare_rtol=" + std::to_string(ctx->debug_compare_rtol) +
#endif
            " cache_dir=" + ctx->cache_dir.string() +
            " device_opened=" + std::to_string(ctx->device_opened ? 1 : 0));
    }

    // Bind this backend to the device's buft_ctx so alloc_buffer can use the device.
    if (dev_ctx) {
        std::lock_guard<std::mutex> lock(dev_ctx->mu);
        // Ensure buft_ctx exists (created lazily in buffer_type_impl).
        if (!dev_ctx->buft_ctx) {
            dev_ctx->buft_ctx = new ggml_fmsh_zg330_buft_ctx{};
            dev_ctx->buft_ctx->dev_ctx_ptr = dev_ctx;
        }
        dev_ctx->buft_ctx->backend_ctx = ctx;
    }

    return backend;
}

bool ggml_backend_is_fmsh_zg330_impl(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_fmsh_zg330_guid());
}

void ggml_backend_fmsh_zg330_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (!ggml_backend_is_fmsh_zg330_impl(backend)) {
        return;
    }
    auto * ctx = static_cast<ggml_backend_fmsh_zg330_context *>(backend->context);
    ggml_backend_cpu_set_n_threads(ctx->cpu_backend, n_threads);
}

// device interface
const char * ggml_backend_fmsh_zg330_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH_ZG330";
}

const char * ggml_backend_fmsh_zg330_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FMSH ZG330 Backend";
}

void ggml_backend_fmsh_zg330_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(dev ? dev->context : nullptr);
    if (dev_ctx) {
        std::lock_guard<std::mutex> lock(dev_ctx->mu);
        if (dev_ctx->memory_cache_valid) {
            *free = dev_ctx->memory_free;
            *total = dev_ctx->memory_total;
            return;
        }

        try {
            const std::string device_url = ggml_fmsh_get_device_url();
            Device memory_device;
            bool memory_device_opened = false;
            std::string err;
            if (!ggml_fmsh_acquire_shared_device(device_url, &memory_device, &err)) {
                GGML_LOG_WARN("%s: failed to acquire zg330 device for PL memory query: %s\n", __func__, err.c_str());
            } else {
                memory_device_opened = true;
                const bool ok = ggml_fmsh_query_pl_memory(memory_device, free, total, &err);
                ggml_fmsh_release_shared_device(device_url, &memory_device, &memory_device_opened);
                if (ok) {
                    dev_ctx->memory_free = *free;
                    dev_ctx->memory_total = *total;
                    dev_ctx->memory_cache_valid = true;
                    return;
                }
                GGML_LOG_WARN("%s: failed to query PL memory: %s\n", __func__, err.c_str());
            }
        } catch (const std::exception & e) {
            GGML_LOG_WARN("%s: failed to open zg330 device for PL memory query: %s\n", __func__, e.what());
        } catch (...) {
            GGML_LOG_WARN("%s: failed to open zg330 device for PL memory query\n", __func__);
        }
    }

    // Last-resort fallback so llama_params_fit does not silently reassign
    // layers to CPU when the device is unavailable during backend enumeration.
    const size_t fallback = size_t(1) << 50; // 1 PiB, well under INT64_MAX
    *free = fallback;
    *total = fallback;
}

enum ggml_backend_dev_type ggml_backend_fmsh_zg330_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void ggml_backend_fmsh_zg330_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_fmsh_zg330_device_get_name(dev);
    props->description = ggml_backend_fmsh_zg330_device_get_description(dev);
    props->type = ggml_backend_fmsh_zg330_device_get_type(dev);
    ggml_backend_fmsh_zg330_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

ggml_backend_t ggml_backend_fmsh_zg330_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return ggml_backend_fmsh_zg330_init_impl(static_cast<ggml_fmsh_zg330_device_context *>(dev ? dev->context : nullptr));
}

void ggml_fmsh_zg330_buffer_free(ggml_backend_buffer_t buffer) {
    delete static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
}

// get_base returns the ZG DDR physical address as a fake host pointer.
// This pointer is NEVER dereferenced on the host; it is only used as a unique
// identifier for tensor->data. The address is 4096-aligned (from ZG malloc),
// which satisfies the ggml allocator alignment requirement.
void * ggml_fmsh_zg330_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    const uint64_t addr = ctx->chunk->begin.addr();
    return reinterpret_cast<void *>(static_cast<uintptr_t>(addr));
}

enum ggml_status ggml_fmsh_zg330_buffer_init_tensor(
        ggml_backend_buffer_t /*buffer*/, struct ggml_tensor * /*tensor*/) {
    return GGML_STATUS_SUCCESS;
}

size_t ggml_fmsh_zg330_tensor_offset(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_fmsh_zg330_buffer_get_base(buffer));
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(tensor->data) - base);
}

void ggml_fmsh_zg330_buffer_set_tensor(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    const size_t toff = ggml_fmsh_zg330_tensor_offset(buffer, tensor);
    buf_ctx->chunk.write(toff + offset,
                         const_cast<char *>(static_cast<const char *>(data)),
                         size);
}

void ggml_fmsh_zg330_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    const size_t toff = ggml_fmsh_zg330_tensor_offset(buffer, tensor);
    buf_ctx->chunk.read(static_cast<char *>(data), toff + offset, size);
}

bool ggml_fmsh_zg330_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (!buffer || !src || !dst || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const size_t bytes = ggml_nbytes(src);
    if (bytes != ggml_nbytes(dst)) {
        return false;
    }

    std::vector<char> tmp(bytes);
    if (ggml_fmsh_is_zg_buffer_tensor(src, nullptr, nullptr)) {
        MemChunk src_chunk;
        size_t src_off = 0;
        if (!ggml_fmsh_is_zg_buffer_tensor(src, &src_chunk, &src_off)) {
            return false;
        }
        src_chunk.read(tmp.data(), src_off, bytes);
    } else if (src->data != nullptr) {
        std::memcpy(tmp.data(), src->data, bytes);
    } else {
        return false;
    }

    if (ggml_fmsh_is_zg_buffer_tensor(dst, nullptr, nullptr)) {
        MemChunk dst_chunk;
        size_t dst_off = 0;
        if (!ggml_fmsh_is_zg_buffer_tensor(dst, &dst_chunk, &dst_off)) {
            return false;
        }
        dst_chunk.write(dst_off, tmp.data(), bytes);
        return true;
    }

    GGML_UNUSED(buffer);
    return false;
}

void ggml_fmsh_zg330_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * buf_ctx = static_cast<ggml_fmsh_zg330_buffer_ctx *>(buffer->context);
    if (!buf_ctx || !buf_ctx->chunk.defined()) return;
    // Zero-fill the entire ZG DDR buffer so that staged D2H reads get deterministic data.
    // Done in 4 KB chunks to stay within socket RPC payload limits.
    constexpr size_t CHUNK = 4096;
    std::vector<char> zero(CHUNK, static_cast<char>(value));
    for (size_t done = 0; done < buf_ctx->size; done += CHUNK) {
        const size_t blk = std::min(CHUNK, buf_ctx->size - done);
        buf_ctx->chunk.write(done, zero.data(), blk);
    }
}

ggml_backend_buffer_t ggml_fmsh_zg330_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = static_cast<ggml_fmsh_zg330_buft_ctx *>(buft->context);
    if (!buft_ctx) {
        GGML_LOG_ERROR("%s: missing FMSH_ZG330 buffer type context\n", __func__);
        return nullptr;
    }

    auto * bctx = buft_ctx->backend_ctx;
    Device zg_dev;
    std::string err;
    if (bctx) {
        if (!ggml_fmsh_open_device_if_needed(bctx, &err)) {
            ggml_fmsh_log_locked(bctx, 3,
                "zg_alloc_buffer_open_fail size=" + std::to_string(size) + " " + err);
            return nullptr;
        }
        zg_dev = bctx->zg_device;
    } else {
        auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(buft_ctx->dev_ctx_ptr);
        if (!dev_ctx) {
            GGML_LOG_ERROR("%s: missing FMSH_ZG330 device context\n", __func__);
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(dev_ctx->mu);
        if (!ggml_fmsh_open_device_context_if_needed(dev_ctx, &zg_dev, &err)) {
            GGML_LOG_ERROR("%s: failed to open zg330 device for buffer allocation: %s\n",
                __func__, err.c_str());
            return nullptr;
        }
    }

    MemChunk chunk;
    try {
        chunk = zg_dev.defaultMemRegion().malloc(size, /*auto_free=*/true, /*alignment=*/4096);
    } catch (const std::exception & e) {
        ggml_fmsh_log_locked(bctx, 3,
            std::string("zg_alloc_buffer_fail size=") + std::to_string(size) + " " + e.what());
        if (!bctx) {
            GGML_LOG_ERROR("%s: zg_alloc_buffer_fail size=%zu %s\n", __func__, size, e.what());
        }
        return nullptr;
    }
    if (!chunk.defined()) {
        ggml_fmsh_log_locked(bctx, 3,
            "zg_alloc_buffer_fail size=" + std::to_string(size) + " undefined_chunk");
        if (!bctx) {
            GGML_LOG_ERROR("%s: zg_alloc_buffer_fail size=%zu undefined_chunk\n", __func__, size);
        }
        return nullptr;
    }
    ggml_fmsh_log_locked(bctx, 1, "zg_alloc_buffer size=" + std::to_string(size));
    auto * buf_ctx = new ggml_fmsh_zg330_buffer_ctx{std::move(chunk), size};
    static const struct ggml_backend_buffer_i zg_buffer_iface = {
        /* .free_buffer     = */ ggml_fmsh_zg330_buffer_free,
        /* .get_base        = */ ggml_fmsh_zg330_buffer_get_base,
        /* .init_tensor     = */ ggml_fmsh_zg330_buffer_init_tensor,
        /* .memset_tensor   = */ nullptr,
        /* .set_tensor      = */ ggml_fmsh_zg330_buffer_set_tensor,
        /* .get_tensor      = */ ggml_fmsh_zg330_buffer_get_tensor,
        /* .set_tensor_2d   = */ nullptr,
        /* .get_tensor_2d   = */ nullptr,
        /* .cpy_tensor      = */ ggml_fmsh_zg330_buffer_cpy_tensor,
        /* .clear           = */ ggml_fmsh_zg330_buffer_clear,
        /* .reset           = */ nullptr,
    };
    return ggml_backend_buffer_init(buft, zg_buffer_iface, buf_ctx, size);
}

// Returns the per-device ZG buffer type. Called from device_get_buffer_type;
// the buft object is cached on dev_ctx (created once, never freed).
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(
        ggml_fmsh_zg330_device_context * dev_ctx, ggml_backend_dev_t dev) {
    if (!dev_ctx->cached_buft) {
        if (!dev_ctx->buft_ctx) {
            dev_ctx->buft_ctx = new ggml_fmsh_zg330_buft_ctx{};
            dev_ctx->buft_ctx->dev_ctx_ptr = dev_ctx;
        }
        static struct ggml_backend_buffer_type_i iface = {
            /* .get_name      = */ [](ggml_backend_buffer_type_t) -> const char * { return "FMSH_ZG330"; },
            /* .alloc_buffer  = */ ggml_fmsh_zg330_buffer_type_alloc_buffer,
            /* .get_alignment = */ [](ggml_backend_buffer_type_t) -> size_t { return 4096; },
            /* .get_max_size  = */ nullptr,
            /* .get_alloc_size= */ nullptr,
            /* .is_host       = */ [](ggml_backend_buffer_type_t) -> bool { return false; },
        };
        dev_ctx->cached_buft = new ggml_backend_buffer_type{iface, /*device=*/dev, /*context=*/dev_ctx->buft_ctx};
    }
    return dev_ctx->cached_buft;
}

// Legacy zero-arg overload: fallback to CPU buffer type.
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type_impl(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_fmsh_zg330_reg(), 0);
    auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(dev ? dev->context : nullptr);
    if (!dev_ctx) {
        return nullptr;
    }
    return ggml_backend_fmsh_zg330_buffer_type_impl(dev_ctx, dev);
}

ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(dev ? dev->context : nullptr);
    if (!dev_ctx) {
        return nullptr;
    }
    return ggml_backend_fmsh_zg330_buffer_type_impl(dev_ctx, dev);
}

ggml_backend_buffer_t ggml_backend_fmsh_zg330_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

bool ggml_backend_fmsh_zg330_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return ggml_fmsh_is_supported_op(op);
}

bool ggml_backend_fmsh_zg330_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }
    // Accept ZG DDR buffer types created by ggml_fmsh_zg330_buffer_type_impl(bctx).
    if (buft && buft->iface.alloc_buffer == ggml_fmsh_zg330_buffer_type_alloc_buffer) {
        return true;
    }
    return false;
}

const ggml_backend_device_i ggml_backend_fmsh_zg330_device_i = {
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

// reg interface
const char * ggml_backend_fmsh_zg330_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FMSH_ZG330";
}

size_t ggml_backend_fmsh_zg330_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

ggml_backend_dev_t ggml_backend_fmsh_zg330_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static auto * dev_ctx = new ggml_fmsh_zg330_device_context;
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_fmsh_zg330_device_i,
        /* .reg     = */ reg,
        /* .context = */ dev_ctx,
    };
    return &dev;
}

// Returns the ZG DDR buffer type for KV cache tensors.
// Called via proc_address by llama-kv-cache.cpp when offloading KV cache.
ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_get_kv_buft(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_fmsh_zg330_device_context *>(dev->context);
    if (!dev_ctx) return nullptr;
    return ggml_backend_fmsh_zg330_buffer_type_impl(dev_ctx, dev);
}

void * ggml_backend_fmsh_zg330_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return reinterpret_cast<void *>(ggml_backend_fmsh_zg330_set_n_threads);
    }
    if (std::strcmp(name, "ggml_backend_fmsh_zg330_get_kv_buft") == 0) {
        return reinterpret_cast<void *>(ggml_backend_fmsh_zg330_get_kv_buft);
    }
    return nullptr;
}

const ggml_backend_reg_i ggml_backend_fmsh_zg330_reg_i = {
    /* .get_name         = */ ggml_backend_fmsh_zg330_reg_get_name,
    /* .get_device_count = */ ggml_backend_fmsh_zg330_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fmsh_zg330_reg_get_device,
    /* .get_proc_address = */ ggml_backend_fmsh_zg330_reg_get_proc_address,
};


} // namespace ggml_fmsh_zg330_impl
