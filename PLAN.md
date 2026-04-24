## FMSH_ZG330 ggml Backend 落地计划（修订版）

### Summary
- 目标：为 `ggml` 新增 `FMSH_ZG330` 后端，参考 `ggml-opencl` 结构。
- 执行模型：将一次 OpenCL kernel 调度映射为一次 Icraft `Session` 执行，按 `op_signature` 做缓存复用。
- 拒绝任何“最小实现”，接入真实后端

### Implementation Changes
- [x] 构建与注册
- [x] 在 [ggml/CMakeLists.txt](/home/ray/git/llama.cpp/ggml/CMakeLists.txt) 增加 `GGML_FMSH_ZG330` 选项。
- [x] 在 [ggml/src/CMakeLists.txt](/home/ray/git/llama.cpp/ggml/src/CMakeLists.txt) 接入 `ggml_add_backend(FMSH_ZG330)`。
- [x] 新建 `ggml/src/ggml-fmsh_zg330/`，实现 backend/device/buffer/graph 核心接口。

- [ ] Session 缓存与执行
- [x] `op_signature` 定义为：`(op_type, shape, dtype)`。
- [x] 首次命中：构建网络并 `Session::Create<ZG330Backend, HostBackend>()` + `apply`。
- [x] 后续命中：复用 session，更新输入输出后直接 `forward`。
- [x] 权重与指令在 `apply` 后常驻，避免重复上传（指令常驻；权重按 `src0->data` 指针在 session 内缓存转置后输入张量并复用）。

- [x] 正确性优先修复（新增）
- [x] 修复错误：4D/batched `MUL_MAT` 被按 2D 执行导致输出异常（先加防护回退）
- [x] 实现 batched `MUL_MAT` 的 ZG330 正确执行路径（按 `ne[2]/ne[3]` 与广播语义逐 slice 执行）
- [x] 仅在满足语义与布局约束时走 ZG330，不满足则按节点 fallback（避免整图回退）
- [x] batched 路径通过 `llama-cli --device FMSH_ZG330` 与 CPU 对照验证输出可读性与稳定性（`runcheck6/runcheck6_b`：FMSH 两次输出稳定可读，CPU 对照同语义）

- [x] NPU 加速收敛（新增）
- [x] 将可支持算子形成连续 ZG 执行段，减少 Host↔ZG330 往返（增加 `MUL_MAT` 连续节点 device input chain 路径与命中统计）
- [x] 结合 `userReuseSegment/userConnectNetwork` 做跨 session/网络段复用，降低 ETM 与拷贝开销（`session_opt_config` 已记录复用段与连接 value id）
- [x] 在日志中量化 batched `MUL_MAT` 的 offload 命中率与 memcpy 占比（`offload_summary` 输出 `batched_hit_rate_pct/memcpy_ratio_pct`）

- [ ] 算子实现（合并后的第一阶段）
- [ ] 在zg 330 上覆盖最小可用 + llama 主链路关键算子：
  - [x] `MUL_MAT`
  - [ ] `MUL_MAT_ID`（含必要 MoE 子路径，当前通过 HostBackend 单节点 dispatch）
  - [x] `CPY/DUP`（已上板：`CPY/DUP` 通过 elementwise session 执行，`CPY` 采用 `Mul(X,1)` 网络规避 `Identity` session 崩溃）
  - [x] `ADD/MUL/SCALE`（已上板：支持 broadcast/stride 读写，避免错误 `memcpy` 假设）
  - [ ] `RMS_NORM`、`ROPE`、`SOFT_MAX`
  - [x] `RMS_NORM`（已上板：含 `rows=1` padding 到 `rows=2` 的稳定执行路径）
  - [ ] `ROPE`（当前 HostBackend dispatch）
  - [x] `SOFT_MAX`（已上板：支持 `src1` mask(F16/F32) + `scale/max_bias(ALiBi)` 预处理后 NPU softmax）
  - [ ] view/reshape 类元数据算子（不触发搬运）
- [x] 未支持算子走 Host fallback，保证功能可运行。

- [x] AXI 模式 MUL_MAT 上板修复（新增）
- [x] 根因定位：`ggml_backend_fmsh_zg330_device_get_memory` 返回 0/0 → `llama_get_device_memory_data` 回落到 CPU 物理内存（ARM 板 993 MiB） → `llama_params_fit_impl` 发现预算低于 1024 MiB margin → 重建图把 recurrent 层从 FMSH 迁到 CPU → `auto_fgdn` 从"mismatch→disable"翻为"match→enable" → SSM 整块融合进 `GGML_OP_GATED_DELTA_NET` 在 CPU 执行 → 所有 SSM MUL_MAT 不再进入 NPU。
- [x] 修复：`get_memory` 改为通过 Icraft XRT `MemRegion::memManager()` 查询 `plddr` PL 内存，使用 `getMemRegionInfo()["byte_size"]` 作为总量，并用 `getAllMemChunk()` 汇总已分配 chunk 后计算剩余量；仅在设备查询失败时保留 1 PiB 兜底，避免后端枚举阶段触发 CPU 层重分配。
- [x] 缓存对齐：ARM 无法 `icraft compile`，所有网络必须在 x86 socket 模式下预编译到 `.cache/deploy/`，`rsync -avh ./.cache root@<board>:/root/llama`，ARM 运行时 `GGML_FMSH_ZG330_CACHE_DIR=/root/llama/.cache/deploy`。不同 prompt/`-n`/`-c` 会触发不同 shape，需用与板端一致的参数在 docker 内预热一次。
- [x] 192.168.110.114 端到端验证：`Hello! How can I help`，`MUL_MAT 432/432 offloaded (100%)`，`FLASH_ATTN_EXT 54/60 offloaded (90%)`，`Prompt 1.2 t/s | Generation 0.7 t/s`。

- [x] Flash Attention 适配（新增）
- [x] 在 `ggml-fmsh-zg330.cpp` 接入 `GGML_OP_FLASH_ATTN_EXT`：`supports_op`/`validate`/`dispatch`/session 缓存全链路。
- [x] 在 `ggml-fmsh-zg330-netmake.h/.cpp` 增加 fused attention 网络生成器，按 `QK^T -> scale(+mask) -> softmax -> PV` 生成 ONNX 并复用现有 icraft compile 缓存流程。
- [x] 按 zg330 softmax 约束实现布局重排：softmax 维度放到倒数第二维，按 cu 对齐最后一维并在 softmax 后逆重排恢复。
- [x] session key 覆盖 `n_head/n_head_kv/head_dim/q_len/kv_len/causal/mask/logit_softcap` 及分桶维度；实现长度分桶与 kv 预编译，避免每次新长度现编译。
- [x] 新增独立随机激励对比程序：`ggml-fmsh-zg330-flash-attn-test`，同一组输入下对比 netmake(ZG330) 与 `ggml_cpu` 的 `FLASH_ATTN_EXT` 输出，并统计 `inf/nan/max_abs/nmse`。
- [x] 修复 AXI 同步 bug（两层）：
  - 第一次调用：`session.forward()` 内置的 `check_func_` 使用 `apply()` 时的绝对 layerCount 目标，在完整推理中该目标早已被超越，`waitForReady` 立即返回。改用稳态轮询（layerCount 稳定 5ms 无变化）正确学习 `layer_increment`（实测 21~30）。
  - 后续调用：`TensorNode::ready_` 默认 `true`，`waitForReady` 永久置 `true` 后不再等待。改为每次 `setReady(false)` + `setCheckFunc([zg_dev, layer_target]{ return layerCount >= layer_target; })` 强制重新同步。

- [ ] Fallback 与搬运控制
- [x] 连续可支持算子按节点 dispatch，并记录 `fallback_boundary`（方向/字节数）以量化 Host↔ZG330 往返。
- [ ] 权重、KV cache、中间张量优先常驻设备内存。
- [ ] 结合 `userReuseSegment/userConnectNetwork` 思路做跨执行段复用。
- [x] 提供 `STRICT` 模式（禁 fallback）用于缺算子排查。

### Logging & Observability
- [x] 增加后端日志（默认开启，支持级别控制）
- [x] session 生命周期：创建/命中缓存/释放
- [x] op dispatch：op 类型、shape、dtype、命中 backend（ZG330/Host）
- [x] fallback 边界：触发原因、tensor 字节数、方向
- [x] 性能统计：每 op `total_time`、`memcpy_time`、`hard_time` 汇总
- [x] 日志落盘到 `GGML_FMSH_ZG330_CACHE_DIR` 下，便于离线分析

### Test Plan
- [x] 基础测试：必须通过编译

- [x] 功能正确性
- [x] 用 `fmsh_ops_testgroud` matmul 基线验证 `MUL_MAT` 数值一致性
- [x] 覆盖关键链路算子的 CPU 对比测试（误差阈值固定）

- [ ] 集成验证
- [x] 通过 `Makefile` 现有 `fmsh-zg330-configure/build-x64` 完成构建
- [x] `llama-cli --device FMSH_ZG330` 做短 prompt smoke test（docker + socket，`--single-turn` 已验证）
- [x] `FLASH_ATTN_EXT` 端到端验证（AXI/ARM）：修复 AXI 同步 bug 后，ARM 设备端 flash attn 输出正常，推理文本可读。
- [x] `make fmsh-zg330-build-x64 FMSH_ZG330_TARGET=all` 可通过
- [x] 在可用 x64 icraft runtime 环境复测 `llama-cli --device FMSH_ZG330`，验证 flash offload 命中与日志指标

- [ ] 性能与搬运
- [x] 验证 session cache 命中率提升（同 shape 二次运行 `net_cache=HIT compile_now=NO`）
- [x] 对比 `memcpy_time/total_time`，确保合并阶段后搬运开销可量化下降（`op_summary` 已输出 `total/memcpy/hard`）

### Assumptions
- 按“从零创建 `ggml-fmsh_zg330`”实施。
- 先支持 x86_64 + socket，后续扩展 aarch64/axi。
- 第一阶段允许 fallback 保可用；后续持续收敛 fallback 覆盖面。
