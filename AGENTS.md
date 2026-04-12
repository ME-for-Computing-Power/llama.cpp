# 项目目的

- 你现在位于 llama.cpp 的仓库中，这是一个大模型推理运行时。你需要为它增加一个名为 fmsh_zg330 的后端，其详细信息由 icraft-docs 技能提供

- fmsh_ops_testgroud 中的matmul_test.cpp 是一个在zg330上运行matmul的最小样例


# 编译
要运行一次socket模式的推理，可以：

1. 进入docker  

docker run --network host  -it --rm -v $(pwd):/workspace fpai-icraft:latest

2. 启动 llama-cli

cd /workspace/build-fmsh-zg330-x64/bin && LD_LIBRARY_PATH=/workspace/build-fmsh-zg330-x64/bin:/ModelzooDeps/x64/Dynamic/lib:$LD_LIBRARY_PATH GGML_FMSH_ZG330_LOG=1 ./llama-cli -m /workspace/Qwen3.5-0.8B-Q4_K_M.gguf --device FMSH_ZG330 --reasoning-budget 0 -p 'Introduce yourself in 10 words' -n 8 -c 2048 --no-warmup