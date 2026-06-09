# 项目目的

- 你现在位于 llama.cpp 的仓库中，这是一个大模型推理运行时。你需要为它增加一个名为 fmsh_zg330 的后端，其详细信息由 icraft-docs 技能提供

- 既然是要部署到zg330后端上，就不要尝试所谓的最小实现，最低要求是将所有主要算子都部署到 zg330 上，将 `mulmat` 等算子在 cpu 上执行是不可接受的


# 编译

要编译带有 fmsh_zg330 的后端，直接在宿主机上运行 

make fmsh-zg330-build-x64 # 生成 x64 socket 模式的后端
make fmsh-zg330-build-arm64 # 生成 arm socket 模式的后端


## Debug 可用功能

FMSH_ZG330_EXTRA_CMAKE_ARGS="-DGGML_FMSH_ZG330_DEBUG_COMPARE=ON"

使能对比，每次都会用 npu 与 cpu 计算参考值对比

## 常见错误

Socket 模式下，Device::Open 在已有连接时被调用（第二个临时 context 尝试连接），ARM 侧 icraft serve 会收到非法请求崩溃，后续会显示无法连接。避免这种写法。

# 测试

优先直接使用仓库中的测试脚本，不要手敲长命令。

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

## 注意事项

一轮推理很慢，推理一次后优先检查日志而不是重复跑
