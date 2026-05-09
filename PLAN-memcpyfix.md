# fmsh_zg330 memcpy / device chaining 修复计划

## 当前结论（按实际验证更新）

目标不是最小实现，也不是加开关绕过问题。目标是让主要 elementwise 链路在 ZG330 上正确运行，消除 `RMS_NORM -> MUL` 等链路中 HostBackend 读取 ADDR MemPtr 导致的 materialize/fallback，同时保持端到端推理正确。

### 已保留的正确基础

- [x] `MUL/ADD/SCALE/RMS_NORM` elementwise session 创建时，在 `apply()` 前用 `ZG330Backend::userConnectNetwork(output_chunk, output_value_id)` 绑定输出 PL DDR chunk。
- [x] elementwise entry 记录 `output_chunk/output_type/output_value_id/output_bytes`，并统计 `output_chunk_hits/output_chunk_misses`。
- [x] 对仍被 `device_tensor_map` 持有的同 entry output chunk 做冲突保护，避免复用 chunk 覆盖未消费结果。
- [x] `rows=1` 的 `RMS_NORM` 按 ZG330 网络约束用 `compiled_rows=2` 编译，写入 `device_tensor_map` 前按下游真实 rows 做 TensorType retype。
- [x] 真实上游 device Tensor 传给下游 session 时，端到端输出可恢复为 `I am Qwen3.`，证明参数和基础网络数值是正确的。

### 已证伪并禁止保留的路径

- [x] 禁止 dummy host tensor + input0 `userConnectNetwork`：该路径虽然去掉 ADDR 读取，但端到端输出连续 `?`，数值错误。
- [x] 禁止全局禁用 `RMS_NORM/SOFT_MAX` device chaining：会把输出打坏为连续 `?`，且不是结构修复。
- [x] 已移除 `RMS_NORM_MUL` fused 网络方案：`Session::Create<ZG330Backend, HostBackend>` 阶段崩溃，不能作为当前修复基础。
- [x] 禁止把 `RMS_NORM -> MUL` materialize 到 host 后继续 CPU/fallback：这正是要修的性能和结构问题。
- [x] 当前 `Backend::forwardOp(captured_output_op, mapped_network_inputs, explicit_output)` 方案已证伪：能跑但端到端输出乱码 `みました sensible!.housesamet还 forgiven`，不是正确计算。

## 当前故障

- `Session::forward(real_device_input0)`：端到端正确，但下游 `MUL` 仍触发 HostBackend 对 ADDR MemPtr 的读取，日志出现：
  - `The MemPtr is of ADDR type!`
  - `fallback op=MUL`
  - `materialize_to_host bytes=... op_src=RMS_NORM`
- `Backend::forwardOp` 直接调用：绕过部分 HostBackend 路径，但当前输入契约不对，端到端输出错误，而且仍有未编译 shape 的 fallback。

因此下一步不是 fallback，而是修正 ZG direct forward 的调用契约：必须使用 session apply 后的 ZG forward op，并传入与 XRT forward runner 一致的 Tensor 输入/输出排列。

## 下一步实现顺序

1. [x] 修正 direct ZG forward 输入契约。
   - 根因：`can_chain_input0` 路径将 `entry->input_tensors[1..N]`（HostDevice::MemRegion() 上的 ADDR 型 host tensor）直接传给 `zg_forward_func`（ZG330 FForwardOp），而 ZG330 backend 要求所有输入均为 PL DDR device tensor。
   - 修复：在 `ggml_fmsh_get_or_create_elementwise_session` 中（`enable_user_connect` 路径），为 inputs[1..N] 各预分配一个 PL DDR tensor（`mallocOn(zg_device.defaultMemRegion())`），存入 `entry->input_device_tensors`。在 `can_chain_input0` 块执行前，对每个 non-input0 input 调用 `dev_t.write(0, host_ptr, bytes)` 完成 H2D 拷贝，再传给 `zg_forward_func`。
   - direct 路径的所有失败已改为 `"fatal:"` 前缀，禁止 silent fallback。
   - 已编译通过（无 warning）。

2. [x] 消除 elementwise 编译产物发现竞态（非功能性 fallback）。
   - 已确认 `mul_bf16_32x256` 等目录里实际存在 `.cache/<net>_ZG/<net>_ZG.json/raw`。
   - `fallback op=MUL reason=cannot find generated *_ZG.json/raw` 的根因不是未编译 shape，而是编译后产物发现时序/并发竞态。
   - 已在 `ggml-fmsh-zg330-netmake.cpp` 落地修复：
     - `find_generated_zg_json_raw()` 增加短时重试（递归扫描 + 2s 等待窗口）。
     - `get_or_compile_*` 路径增加全局编译互斥 + 二次 cache 检查，避免同 net 并发编译/查找冲突。

3. [x] 端到端容器验证（2026-05-09 完成）。
   - 根因定位：ZG330 的 PL DDR output_chunk 以 **F32（4 bytes/elem）** 格式存储，而之前 D2H copy 用 `sizeof(uint16_t)` 读取，只读了前一半数据并误判为 BF16，导致乱码。
   - 修复：将 D2H 改为读 F32 后再用 `ggml_fmsh_f32_to_bf16()` 转换，与非链路路径的 `ggml_fmsh_f32_to_bf16(src0->data[i])` 逻辑一致。
   - ✅ 无 `The MemPtr is of ADDR type!` 错误
   - ✅ 无 `fallback op=MUL` 错误
   - ✅ `output_chunk_misses=0`（`output_chunk_hits=14110`）
   - ✅ axi_out 有限值（`[0.018 -0.166 ...]`，非 inf）
   - ✅ 端到端文本正确（"large language ... highly accurate and helpful responses"）

4. [x] 性能复测（2026-05-09 完成）。
   - 对比基线（deploy_retyped1）与当前链路，per-call 总耗时几乎持平：
     - MUL: 11.526 → 11.459 ms/call（-0.6%）
     - memcpy ratio: MUL 34.7% → 36.4%（D2H+F32→BF16 转换引入微小额外拷贝，在噪声范围内）
   - ADD / SCALE / RMS_NORM 的 per-call 总耗时均有小幅下降（7-11%），符合预期。
   - 结论：D2H → F32→BF16 转换路径不引入可观察的性能回归。

## 关键 API 依据

- `ZG330Backend::userConnectNetwork(memchunk, v_id)` 必须在 `Session::apply()` 前配置，用于多网络输入/输出 PL DDR chunk 连接。
- `Session::getForwards()` 返回 `(Operation, Backend, FForwardOp, input_value_ids, output_value_ids)`，direct path 必须遵守这个 forward runner 的输入输出契约。
- `Backend::forwardOp(op, inputs, outputs)` 只适合在输入排列和 backend forward function 语义明确时使用；当前压缩 value-id 输入排列已被端到端验证证伪。

## 禁止项

- 不加运行时开关掩盖问题。
- 不用 CPU fallback 或 host materialize 作为 `RMS_NORM -> MUL` 的解决方案。
- 不恢复已知输出 `?` 或乱码的路径。
- 不保留没有端到端推理验证的“看似优化”。
