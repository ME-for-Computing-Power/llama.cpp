# FMSH 后端适配计划（llama.cpp）

## 目标
- 参考 `ref_proj` 与 `docs`，为 `llama.cpp` 增加可编译、可加载、可配置的 `FMSH` 后端入口。
- 先完成 **MVP 集成**（编译系统 + 后端注册 + 运行时参数 + 文档 + ARM 构建脚本）。
- 在缺少板端专有 SDK/库 的情况下，优先保证主仓可维护、可渐进增强。

## 范围与策略
1. **后端形态**：先实现 `ggml-fmsh` 后端骨架，并通过环境变量/参数控制。
2. **运行路径**：优先支持“设备发现 + 后端注册 + 基本能力声明 + 可安全回退”。
3. **不做事项（MVP 阶段）**：
   - 不在本次提交中强行引入不可分发的闭源 SDK 源码。
   - 不实现全量算子 NPU 内核；先提供可扩展接口与占位实现。

## 执行步骤
- [x] 1) 盘点现有 ggml 后端最小实现约束（reg/device/backend/buffer 接口）
- [x] 2) 新增 `ggml/src/ggml-fmsh/` 目录与 `CMakeLists.txt`
- [x] 3) 在 `ggml/CMakeLists.txt` 与 `ggml/src/CMakeLists.txt` 接入 `GGML_FMSH` 开关
- [x] 4) 实现 `ggml-fmsh.cpp`：
  - [x] backend reg（名称、设备枚举、proc 导出）
  - [x] device 接口（props/caps/init/supports_op）
  - [x] backend 接口（buffer/type、graph_compute、sync）
  - [x] 安全回退策略（当前不支持 op 时回退到 CPU）
- [x] 5) 新增公开头文件（如 `ggml/include/ggml-fmsh.h`，按现有风格）
- [x] 6) 更新文档：
  - [x] `docs/backend/FMSH.md`（构建、环境变量、已知限制）
  - [x] `docs/build.md` 增加入口
  - [x] `README.md` supported backends 表格增加 FMSH
- [x] 7) 更新 `scripts/build_llama_arm.sh`，加入 `GGML_FMSH=ON` 可选构建
- [x] 8) 本地最小验证：CMake 配置、编译通过、后端可枚举
- [ ] 9) 输出后续路线（接入 BuyiBackend / XRT 真正 NPU 执行链）

## 验收标准（MVP）
- 能通过 CMake 打开 `GGML_FMSH=ON` 编译。
- 运行时可看到 `FMSH` 后端注册与设备条目。
- 不支持路径下不崩溃，可回退至 CPU 完成执行。
- 文档完整给出板端 SDK 接入点与下一阶段工作项。

## 风险
- 缺少板端运行库/头文件时，无法在本机完成真实 NPU 调度验证。
- 若强耦合闭源依赖，可能影响主仓可移植性；因此采用分层/可选编译策略。
