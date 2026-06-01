# 项目目的

- 你现在位于 llama.cpp 的仓库中，这是一个大模型推理运行时。你需要为它增加一个名为 fmsh_zg330 的后端，其详细信息由 icraft-docs 技能提供

- 既然是要部署到zg330后端上，就不要尝试所谓的最小实现，最低要求是将所有主要算子都部署到 zg330 上，将 `mulmat` 等算子在 cpu 上执行是不可接受的


# 编译

要编译带有 fmsh_zg330 的后端，直接在宿主机上运行 

make fmsh-zg330-build-x64 


## Debug 可用功能

FMSH_ZG330_EXTRA_CMAKE_ARGS="-DGGML_FMSH_ZG330_DEBUG_COMPARE=ON"

使能对比，每次都会用 npu 与 cpu 计算参考值对比

## 常见错误

Socket 模式下，Device::Open 在已有连接时被调用（第二个临时 context 尝试连接），ARM 侧 icraft serve 会收到非法请求崩溃，后续会显示无法连接。避免这种写法。

# 测试

要运行一次socket模式的推理，可以：

1. 进入docker  

docker run --network host  -it --rm -v $(pwd):/workspace fpai-icraft:latest

Icraft 相关的全部环境都在*容器内*，不要尝试在host上运行程序

2. 启动 llama-cli

cd /workspace/build-fmsh-zg330-x64/bin && export LD_LIBRARY_PATH=/workspace/build-fmsh-zg330-x64/bin:/ModelzooDeps/x64/Dynamic/lib:$LD_LIBRARY_PATH && export GGML_FMSH_ZG330_LOG=1 && export GGML_FMSH_ZG330_CACHE_DIR=/workspace/.cache/deploy && ./llama-cli -m /workspace/Qwen3.5-0.8B-Q4_K_M.gguf --device FMSH_ZG330 --reasoning-budget 0 -p 'Hello there' -n 10 -c 1024 --no-warmup --single-turn --seed 1024 --verbose --log-file /workspace/llama.log

socket模式中一轮推理非常慢，因此运行时应当设置 timeout 时间，并打印完整日志；推理一次后，应在日志文件（cache 中的 backend.log 和 运行目录中的 llama.log ）中运行 grep 等工具，若无必要则不要反复执行推理浪费时间。
