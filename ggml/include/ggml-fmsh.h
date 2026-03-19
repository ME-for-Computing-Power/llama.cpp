#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t     ggml_backend_fmsh_init(size_t device_index);
GGML_BACKEND_API bool               ggml_backend_is_fmsh(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_fmsh_reg(void);

#ifdef  __cplusplus
}
#endif
