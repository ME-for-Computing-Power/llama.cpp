# 项目目的

- 你现在位于 llama.cpp 的仓库中，这是一个大模型推理运行时。你需要为它增加一个名为 fmsh_zg330 的后端，其详细信息由 icraft-docs 技能提供

- 既然是要部署到zg330后端上，就不要尝试所谓的最小实现，最低要求是将所有主要算子都部署到 zg330 上，将 `mulmat` 类算子在 cpu 上执行是不可接受的

- fmsh_ops_testgroud 中的 matmul_test.cpp 是一个在zg330上运行matmul的最小样例。这是一个参考项目

# 目前进度

详细进度参考 PLAN.md. 你需要根据实际情况添加、删除或完成任务


# 精度问题

根据 [文档](.agents/skills/icraft-docs/docs/version/Icraft v3.33.0_FMQL30TAI.md)

FMQL30TAI (zg330)支持 int8 、 bf16 、 fp16 、 tf32 以满足不同用户的需求，下表罗列数据格式对比情况，请结合精度和性能选择合适的前向数据类型。

Icraft330支持的数格式对比

| 格式 | 总位数 | 符号位 (Sign) | 阶码 (Exponent) | 尾数 (Significand/Mantissa) |
| --- | --- | --- | --- | --- |
| TF32 | 32 | 1 | 8 | 23 (但实际有效位为10) |
| BF16 | 16 | 1 | 8 | 7 |
| FP16 | 16 | 1 | 5 | 10 |
| INT8 | 8 | 1 |  | 7 |

# 编译

要编译带有 fmsh_zg330 的后端，直接在宿主机上运行 

make fmsh-zg330-build-x64 FMSH_ZG330_TARGET=all


## Debug 可用功能

FMSH_ZG330_EXTRA_CMAKE_ARGS="-DGGML_FMSH_ZG330_DEBUG_COMPARE=ON"

使能对比，每次都会用 npu与 cpu 计算参考值对比

# 测试

要运行一次socket模式的推理，可以：

1. 进入docker  

docker run --network host  -it --rm -v $(pwd):/workspace fpai-icraft:latest

Icraft 相关的全部环境都在*容器内*，不要尝试在host上运行程序

2. 启动 llama-cli

cd /workspace/build-fmsh-zg330-x64/bin && export LD_LIBRARY_PATH=/workspace/build-fmsh-zg330-x64/bin:/ModelzooDeps/x64/Dynamic/lib:$LD_LIBRARY_PATH && export GGML_FMSH_ZG330_LOG=1 && export GGML_FMSH_ZG330_CACHE_DIR=/workspace/.cache/deploy && ./llama-cli -m /workspace/Qwen3.5-0.8B-Q4_K_M.gguf --device FMSH_ZG330 --reasoning-budget 0 -p 'Introduce yourself in 10 words' -n 40 -c 1024 --no-warmup --single-turn --seed 1024 --verbose --log-file /workspace/llama.log