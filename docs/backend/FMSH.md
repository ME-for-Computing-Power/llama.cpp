# FMSH backend (experimental)

This document describes the current `ggml-fmsh` backend for `llama.cpp`.

## Scope

`ggml-fmsh` is designed for FMSH ARM + Buyi NPU platforms. The current MVP provides:

- Backend registration (`FMSH`) and device discovery entry.
- Scheduler-side op capability surface aligned with common transformer workloads.
- Real NPU execution for selected ops (`MUL_MAT`, `ADD`, `ADD1`, `SUB`, `MUL`, `SQRT`, `SUM`, `MEAN`, `SOFT_MAX`, and selected `UNARY`) via iCraft runtime.
- Safe graph execution fallback to CPU for all other or non-lowerable nodes.

## Design notes

The backend design follows two references:

1. `ref_proj/include/icraft-xir/ops` (iCraft available ops), used to define a realistic operator envelope.
2. `ggml/src/ggml-opencl` partitioning style, used as a reference for backend/device separation and incremental operator support strategy.

The compute path is hybrid: backend tries NPU execution per node, and leaves residual nodes to CPU.

## Build

Enable FMSH backend:

```bash
cmake -B build -DGGML_FMSH=ON
cmake --build build --config Release
```

FMSH requires direct iCraft runtime link (fallback-only build is disabled):

```bash
cmake -B build \
  -DGGML_FMSH=ON \
  -DGGML_FMSH_ICRAFT=ON \
  -DGGML_FMSH_ICRAFT_ROOT=/path/to/icraft/sdk
cmake --build build --config Release
```

If iCraft headers/libraries are not found, build fails fast.

## Runtime environment variables

- `GGML_FMSH_ENABLE_NPU` (`1`/`0`): request NPU path (effective only when iCraft-linked build is available).
- `GGML_FMSH_FORCE_CPU` (`1`/`0`): force CPU execution path.
- `GGML_FMSH_VERBOSE` (`1`/`0`): print backend initialization details.
- `GGML_FMSH_DEVICE_URL`: override iCraft device URL (defaults to `axi://...` on Linux).

## Runtime behavior

- CPU fallback remains enabled for correctness.
- When fallback happens, backend prints explicit warning(s) with operator names.
- NPU execution currently implemented for:
  - `GGML_OP_MUL_MAT`
  - `GGML_OP_ADD` / `GGML_OP_ADD1` / `GGML_OP_SUB` / `GGML_OP_MUL` (including RHS broadcast when each dim is `1` or matches dst)
  - `GGML_OP_SQRT`
  - `GGML_OP_SUM` / `GGML_OP_MEAN`
  - `GGML_OP_SOFT_MAX`
  - `GGML_OP_UNARY` subset: `ABS`, `NEG`, `TANH`, `ELU`, `RELU`, `SIGMOID`, `GELU`, `SILU`, `HARDSWISH`, `HARDSIGMOID`

## Current limitations

- Host-buffer path only.
- Not all transformer-critical ops are lowered yet (`MUL_MAT`, `RMS_NORM`, etc. still use CPU fallback).

## Next steps

- Expand iCraft-lowered op coverage (`MUL_MAT`, normalization family, elementwise/unary set).
- Add NPU memory/buffer management path.
- Execute partitioned subgraphs on Buyi runtime with CPU fallback per-subgraph.
