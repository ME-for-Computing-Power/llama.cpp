调试总结
问题现象
AXI 模式（ARM 直接内存映射）下，每次 forward() 读到的是上一帧的 NPU 输出（1帧延迟），而 socket 模式正常。

根本原因（两层）
第一层：check_func_ 使用绝对 layerCount 目标值

ZG330Backend 为输出 Tensor 安装的 check_func_ 捕获了一个绝对 layerCount 目标（例如 23）。第一次 forward 后 layerCount=23，第二次 forward 时 check_func_ 的目标仍是 23，而寄存器已经是 23，所以立即返回 true。

第二层：Session 复用同一个 Tensor 对象（决定性原因）

SessionNode 内部有 tmap_: unordered_map<int64_t, Tensor>，跨 forward 调用复用同一个 TensorNode。waitForReady() 成功后会将 TensorNode::ready_ 置为 true，此后所有调用都立即返回，完全跳过 check_func_ 的轮询。

这就是为什么 socket 模式正常：网络延迟足够长，NPU 早已完成；AXI 模式寄存器读写是纳秒级，waitForReady 立即返回时 NPU 还在跑。

wTileCount/rTileCount 始终为 0：这两个寄存器在当前固件/配置下不工作，不能用于同步。

layerCount 计数的是层开始执行，不是完成：所以仅等待 layerCount > layer_before 不够，NPU 可能还没写回 ETM。

修复方案

// 第一次 forward：让 waitForReady 正常工作（ready_=false，check_func_ 有效）
// 同时学习每次 forward 的 layerCount 增量（= 23）

// 后续 forward：
outputs[0].setReady(false);   // 重置 ready_ 标志
outputs[0].setCheckFunc([zg_dev, layer_target](const Device &) -> bool {
    return zg_dev.layerCount() >= layer_target;  // layer_target = layer_before + 23
});
outputs[0].waitForReady(30s); // 真正等待 NPU 完成
核心是：重置 ready_ 并安装以相对增量为目标的新 check_func_，而不是依赖 icraft 内部那个固定绝对值的 check_func_。