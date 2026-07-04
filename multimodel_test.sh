#!/usr/bin/env bash
set -euo pipefail

docker run --network host -it --rm \
    -v "$(pwd):/workspace" \
    fpai-icraft:latest \
    bash -lc '
        cd /workspace/build-fmsh-zg330-x64/bin
        export LD_LIBRARY_PATH=/workspace/build-fmsh-zg330-x64/bin:/ModelzooDeps/x64/Dynamic/lib:${LD_LIBRARY_PATH:-}
        export GGML_FMSH_ZG330_LOG=1
        export GGML_FMSH_ZG330_CACHE_DIR=/workspace/.cache/deploy
        export MTMD_BACKEND_DEVICE=FMSH_ZG330
        ./llama-mtmd-cli \
            -m /workspace/Qwen3.5-0.8B-Q4_K_M.gguf \
            --mmproj /workspace/mmproj-Qwen3.5-0.8b-BF16.gguf \
            --device FMSH_ZG330 \
            --image /workspace/test.png \
            -p "Describe this image." \
            -n 16 \
            -c 2048 \
            --verbose \
            --no-warmup \
            --log-file /workspace/llama_zg330.log \
            --image-max-tokens 1024
    '
