#ifndef GGML_FMSH_ZG330_H
#define GGML_FMSH_ZG330_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t ggml_backend_fmsh_zg330_init(void);
GGML_BACKEND_API bool ggml_backend_is_fmsh_zg330(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_fmsh_zg330_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_fmsh_zg330_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_FMSH_ZG330_H
