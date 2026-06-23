#include "ggml-fmsh-zg330.h"
#include "ggml-fmsh-zg330-internal.h"

using namespace ggml_fmsh_zg330_impl;

ggml_backend_t ggml_backend_fmsh_zg330_init(void) {
    return ggml_backend_fmsh_zg330_init_impl();
}

bool ggml_backend_is_fmsh_zg330(ggml_backend_t backend) {
    return ggml_backend_is_fmsh_zg330_impl(backend);
}

ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type(void) {
    return ggml_backend_fmsh_zg330_buffer_type_impl();
}

ggml_backend_reg_t ggml_backend_fmsh_zg330_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fmsh_zg330_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}
