# 项目目的

- 你现在位于 llama.cpp 的仓库中，这是一个大模型推理运行时。你需要为它增加一个名为 fmsh_zg330 的后端，其详细信息由 icraft-docs 技能提供

- 既然是要部署到zg330后端上，就不要尝试所谓的最小实现，最低要求是将所有主要算子都部署到 zg330 上，将 `mulmat` 等算子在 cpu 上执行是不可接受的


# 编译

要编译带有 fmsh_zg330 的后端，直接在宿主机上运行 

make fmsh-zg330-build-arm64 # 生成 arm socket 模式的后端
make fmsh-zg330-build-x64 # 生成 x64 socket 模式的后端


## Debug 可用功能

FMSH_ZG330_EXTRA_CMAKE_ARGS="-DGGML_FMSH_ZG330_DEBUG_COMPARE=ON"

使能对比，每次都会用 npu 与 cpu 计算参考值对比

## 常见错误

Socket 模式下，Device::Open 在已有连接时被调用（第二个临时 context 尝试连接），ARM 侧 icraft serve 会收到非法请求崩溃，后续会显示无法连接。避免这种写法。

数值计算错误，考虑是否是滞后了
1. 产生原因与本质
- Session 内物理节点的复用：在 llama.cpp 图执行（graph_compute）中，许多相同形状（Shape）的节点会复用同一个底层 NPU 算子的 Session。在 Session 内部，物理设备端的 Output Tensor 是同一个，因此主机端在不同时刻读取到的也是相同的设备物理内存地址。
- NPU 与 CPU 的异步执行：NPU 执行网络前向计算（forward()）后，CPU 并非被硬件直接阻塞，而是继续向下执行，调用 waitForReady() 等待硬件计算完成并通知信号。
- Ready 标志的“永久置位”状态丢失：
- 在首次执行 waitForReady() 并等到结果后，该 Output Tensor 内部的 ready_ 标志会被置为 true。
- 在接下来的层或 Token 推理中，当 CPU 再次调用 forward() 后，如果直接调用 waitForReady()，由于该 Tensor 在底层的 ready_ 依然是旧的 true 状态，waitForReady() 就会立刻返回成功。
- 结果是，CPU 读出了上一轮/上一帧尚未被 NPU 覆盖的旧输出（stale output），导致了数据污染与时序滞后（Pipeline Lag）


# 测试

优先直接使用仓库中的测试脚本，不要手敲长命令。


## ARM socket 测试

在宿主机运行：

./arm_test.sh

该脚本会：

- 将本地 `./.cache` 上传到 `root@192.168.110.114:/root/llama`
- 将本地 `./build-fmsh-zg330-arm64/bin` 上传到 `root@192.168.110.114:/root/llama`
- 在远端 `192.168.110.114` 上运行同样的 `llama-cli` 测试，并额外加上 `-fa on`
- 将远端 `/root/llama/llama.log` 下载到本地 `./llama_arm.log`
- 将远端 `/root/llama/.cache/deploy/backend.log` 下载到本地 `./backend_arm.log`
- 在下载后执行 `grep -Ein 'error|warn|fail' llama_arm.log backend_arm.log`

## x64 socket 测试

在宿主机运行：

```bash
./socket_test.sh
```

该脚本会：

- 在 docker 容器内运行一次 socket 模式推理
- 默认使用 `timeout --signal=INT 900`
- 将实时输出写入 `socket_test.live.log`
- 生成 `/workspace/llama.log`
- 检查 `.cache/deploy/backend.log`
- 在推理结束后执行 `grep -Ein 'error|warn|fail' llama.log .cache/deploy/backend.log`

Icraft 相关环境都在容器内；不要尝试在 host 上直接运行 `llama-cli`。

## 注意事项

一轮推理很慢，需要多加等待，推理一次后优先检查日志而不是重复跑
