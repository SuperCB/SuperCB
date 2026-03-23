# PyTorch 2.11 中 torch.compile 与 torch.inductor 相关 PR 总结

> 版本信息：PyTorch 2.11 发布日期约 2026 年 3 月 18 日（release/2.11 分支于 2026-02-16 切出）  
> 本文整理了从 2.10.0 发布（2026-01-21）至 2.11 分支切出（2026-02-16）期间，主分支中与 `torch.compile` / `torch.inductor` 相关的主要 PR。

---

## 一、torch.inductor 核心改动

### 1. Combo Kernels（组合内核）优化

#### 扁平化 Grid 分发 [PR #172527]
- **问题**：原有 Combo Kernel 使用各子内核 Y 维块数的最大值作为公共 Y grid 大小，导致大量无效线程块。
- **改动**：将 2D/3D grid 改为 1D 扁平化 grid（`sum of x*y blocks per sub-kernel, 1, 1`），每个子内核只分配恰好需要的块数。
- **效果**：对 8 个子内核的典型场景，总块数从 **809,984** 减少到 **3,795**（节省 **213×**）。

#### 每子内核独立 Block 维度 [PR #171671]
- 为 Combo Kernel 中每个子内核设置独立的 block 维度（`YBLOCK_0, XBLOCK_0, YBLOCK_1, XBLOCK_1 ...`），允许不同子内核采用不同的 tile 大小。

#### PDL（Programmatic Dependent Launch）支持 [PR #174232]
- 在 Combo Kernels 中加入 PDL 支持，允许依赖图中的 kernel 以 PDL 方式异步调度，进一步减少 kernel launch overhead。

#### 仅 Pointwise 子内核的配置开关 [PR #174894]
- 新增 `combo_kernels_pointwise_only` 配置项，控制 Combo Kernel 是否仅融合 pointwise 类型的子内核。

---

### 2. NVGEMM（NVIDIA 专用矩阵乘加速）

#### Scaled MM NVGEMM 支持 [PR #172525]
- 为 NVGEMM 后端增加 scaled mm（FP8 等精度的量化矩阵乘）支持，利用 CUTLASS 的 scaled matmul 接口生成高效内核。

#### Group GEMM 支持 [PR #172417]
- NVGEMM 后端新增 GroupGEMM 支持，允许批处理多个不同尺寸的矩阵乘。

#### Scaled MM 启发式规则 [PR #174827]
- 为 scaled mm 开启 NVGEMM heuristics，自动选择最优 CUTLASS 配置。

#### 代码重构系列 [PR #172607, #172582, #172402, #172391]
- 重构 NVGEMM render 代码、全局 kernel 缓存初始化逻辑、config 变量等，为后续功能扩展打好基础。

---

### 3. TMA（Tensor Memory Accelerator）模板改进

#### TlxGemmConfig 自动调优支持 [PR #172016]
- 新增 `TlxGemmConfig`，为 TLX（Tensor Layout eXtension）模板增加自动调优配置支持。

#### TMA 兼容性要求更新 [PR #174857]
- 更新 TMA persistent mm 模板的设备兼容性检测。

#### 禁用 B200+ 上的 mm_persistent_tma 模板 [PR #174745]
- 针对 Blackwell B200 及更高规格 GPU，暂时禁用 `mm_persistent_tma` 模板，避免已知问题。

#### TMA 工作区指针复用边界情况修复 [PR #172650]
- 修复了 TMA persistent 模板中工作区指针复用时的边界情况。

---

### 4. 矩阵乘法模板与调优

#### Triton MM 模板改为 2D 循环 [PR #172520]
- 将 Triton mm 模板从 1D 循环改为 2D 循环（`(k, m*n)` → `(m*n, k)`），提升了编译器生成代码的循环结构质量。

#### Better Batch Matmul Codegen [PR #172678]
- 改进批量矩阵乘（bmm）代码生成，更高效地处理 batch 维度。

#### Add BLOCK_K=64 and 128 for conv [PR #174752]
- 为 conv（卷积）的矩阵化 kernel 新增 `BLOCK_K=64` 和 `128` 的自动调优配置，扩大搜索空间。

---

### 5. 异步流水线自动调优（Async Pipelined Autotuning）

#### 优化：Epilogue 融合与流水线 Autotuning 并发 [PR #171011]
- 实现了 epilogue 融合（如 bias 加法、激活函数）与异步流水线自动调优的最优并发执行。

#### 进程池空闲自动关闭 [PR #174684]
- 当自动调优进程池长时间空闲时自动关闭，避免资源泄漏。

#### 自动调优继承 TF32 设置 [PR #174742]
- 修复异步自动调优子进程未继承父进程 TF32 配置的问题。

#### 防止竞态条件（Race Condition）[PR #174834]
- 用手动保存/恢复替代 `mock.patch.dict`，避免并发自动调优中的竞态条件。

---

### 6. Overlap Scheduling（计算通信重叠调度）

#### 预分桶路径外调度 [PR #170578]
- 实现"调度预分桶到路径外"的优化策略，减少计算与通信的串行等待。

#### 桶优先级调度 [PR #170575]
- 调度时优先处理已分桶的内核，提升流水线利用率。

#### 支持 non-flat 参数 [PR #173083]
- 在控制依赖（control_deps）中支持 non-flat 类型的参数。

---

### 7. 布局（Layout）优化

#### 延迟布局约束求解 [PR #172649]
- 记录布局约束但推迟到模板选择阶段才求解，允许 Inductor 根据所选模板选择最优内存布局。

---

### 8. Inductor XPU GEMM 重构（Step 1–7）

| Step | PR | 内容 |
|------|-----|------|
| 1/N | #160174 | 重构 CUTLASS 配置（CutlassConfig）|
| 2/N | #160685 | 将 cutlass 文件移出 `torch/_inductor/codegen/cuda` |
| 3/N | #160686 | 将 `CUDATemplate` 重构为 `CUTLASSTemplate` |
| 4/N | #160687 | 将 `CUDAKernel` 重构为 `CUTLASSKernel` |
| 5/N | #160688 | 重构 `CUDACombinedScheduling` 和 `CUDACppScheduling` |
| 6/N | #160706 | 重构 `CUDACodeCache` |
| 7/N | #160729 | 重构 `CUDABenchmarkRequest` |

这一系列重构将原本 CUDA 独有的 GEMM 框架推广到 XPU（Intel GPU），以实现代码复用。

---

### 9. 性能与正确性修复

| PR | 修复内容 |
|----|---------|
| #174947 | Mix-order Reduction 避免不必要的重编译 |
| #174408 | 移除若干昂贵的 `dynamo_timed` 调用，减少编译开销 |
| #173703 | 修复 WeakDeps 在 clone op 中的逻辑 |
| #174929 | 修复 persistent reduction 的代码生成错误 |
| #174354 | 修复 combo kernel y grid 溢出 bug |
| #174729 | 修复 cudagraphs 分区中的符号传播遗漏 |
| #173043 | 修复 narrow_copy 分解的步长（strides）计算 |
| #174583 | 修复自动调优时 unspec tensor 的解包逻辑 |
| #170794 | 修复 `aten.uniform` 的 strides 计算 |
| #173426 | 修复动态 shape 的无穷大判断 bug |
| #173412 | 修复 Inductor scheduler 融合中的 DDE（死数据消除）bug |

---

### 10. 其他 Inductor 新功能

#### Helion + torch.compile 集成 [PR #174148]
- 添加了模板 codegen 和 emit 的 override 钩子（hook），允许外部模板（如 Helion）接入 Inductor 的代码生成流程。

#### 支持 pin_memory 的 empty tensor [PR #172578]
- Inductor 现在能够正确处理 `torch.empty(..., pin_memory=True)` 的内核生成。

#### 支持 `torchcomms` lowering [PR #171634]
- 在 Inductor IR 中新增对 torchcomms（通信算子）的 lowering 支持，为计算通信融合打基础。

#### 使用自定义 Triton 内核子类 [PR #167456]
- 当有自定义 Triton 内核子类可用时，Inductor 优先使用，提升灵活性。

#### XPU Pointwise 启发式规则 [PR #167712]
- 为 XPU 新增 pointwise kernel 的启发式规则，改善 Intel GPU 上的性能。

#### 静态 Triton Kernel Launcher（XPU）[PR #169938]
- 为 XPU 实现静态 Triton kernel launcher，减少 kernel launch 的动态开销。

---

## 二、torch._dynamo（torch.compile 前端）核心改动

### 1. 编译时间（Compile Time）性能优化

#### bind_args 快速路径 [PR #174438]
- 为简单函数（无 `*args`/`**kwargs` 的函数）提供 `bind_args` 的快速路径，减少参数绑定开销。

#### 缓存 `inspect.signature` 结果 [PR #174437]
- 对 `inspect.signature` 的调用结果进行缓存，避免重复解析函数签名。

#### 缓存 attr source 构建 [PR #174020]
- 缓存属性源（attr source）的构建结果。

#### 缓存 InlineInstructionTranslator 常量 attr [PR #174141]
- 缓存 inline 翻译器中的常量属性访问，减少重复计算。

#### 引入 CONSTANT_VARIABLE_NONE 单例 [PR #174728]
- 用单例模式替代每次 `ConstantVariable(None)` 的新对象创建，减少内存分配。

#### 简化 VT 缓存，支持 LazyVT [PR #174242]
- 简化 VariableTracker 的缓存逻辑，并扩展到 LazyVariableTracker。

#### 加速 DictItemsVariable 的迭代 [PR #173645]
- 使用更高效的数据结构加速字典视图（items/keys/values）的迭代。

#### 加速 GET_ITER 对 tuple [PR #173582]
- 为 tuple 的 `GET_ITER` 操作提供快速路径。

#### 加速 index 方法 [PR #173612]
- 为常量数据结构的 `index()` 方法提供快速路径。

---

### 2. 新功能

#### 自引用列表/字典支持 [PR #173672]
- Dynamo 现在能够正确追踪和重建自引用（self-referential）的列表和字典对象。

#### 支持 sourceless MappingProxyObjects [PR #173749]
- 支持对 `MappingProxyType` 对象的无源（sourceless）追踪。

#### 新的 UserDefinedEnumVariable [PR #173223]
- 新增专用的枚举变量追踪器，支持用户定义 Enum 的 `__contains__` 等操作。

#### 自定义 DTensor grad_placements [PR #173787]
- 支持在 DTensor 中自定义梯度的 placements 策略。

#### 可自动微分的叶子模块 [PR #170471]
- 为支持 `torch.compile` 的叶子模块（leaf module）增加自动微分支持的初步实现。

#### DDP Optimizer 支持 composable replicate [PR #174307]
- `DDPOptimizer` 现在可与可组合的 `replicate` 一起在 `torch.compile` 下使用。

#### Handle List/Dict 推导式图断裂（Python 3.12+）[PR #173558]
- 正确处理 Python 3.12+ 中列表/字典推导式可能引发的图断裂（graph break）。

#### 嵌套推导式图断裂 [PR #174413]
- 修复嵌套推导式时的图断裂处理。

---

### 3. 调试与可观测性

#### Dynamo Profiler [PR #173942]
- 新增 `dynamo_profiler`，可生成详细的编译时间分析报告，帮助定位编译耗时热点。

#### 编译事件在 Profiler 中可见 [PR #174191]
- 使 torch.compile 的编译事件出现在 PyTorch Profiler 的时间线中。

#### 添加变量构建时间到 tlparse [PR #174908]
- 在 tlparse 工具中加入变量构建（variable builder）阶段的时间统计。

#### 每图 Inductor Config 覆盖 [PR #174228]
- 新增 per-graph 的 Inductor config override 机制，方便 debug/bisect 特定图的编译行为。

#### GraphBackendRouter / GraphConfigRouter 重构 [PR #174229]
- 将 `GraphBackendRouter` 和 `GraphConfigRouter` 的公共逻辑提取为共用代码。

---

### 4. 行为变更

#### FSDP2 移除 dynamo tracing 支持 [PR #174863]
- 从 `fully_shard`（FSDP2）中移除了 dynamo tracing 支持，未来 FSDP2 将通过不同机制与 torch.compile 集成。

#### Make DTensor 在 compile 中有一致的 cache key [PR #173526]
- 确保 DTensor 在编译缓存中有一致的键，避免不必要的重编译。

---

## 三、AOTI（Ahead-of-Time Inductor）核心改动

### 1. XPU 支持扩展

#### XPU 独立编译 API [PR #171450]
- 在 `_Exporter` 中支持 XPU 的 AOTI 独立编译（standalone compile）API。

#### XPU 多架构 kernel 发射 [PR #171432]
- 支持 `aot_inductor.emit_multi_arch_kernel`，允许同一模型生成支持多个 XPU 架构的 AOTI 产物。

#### XPU Lazy 依赖 Intel Level Zero [PR #173497]
- XPU 构建改为懒加载 Intel Level Zero，减少不必要的依赖。

### 2. 序列化与运行时

#### Triton 内核 side table 序列化 [PR #173556]
- 为捆绑式 AOT artifacts（precompile 产物）序列化 Triton 内核的 side table，支持跨进程/机器部署。

#### 修复 ScalarType/MemoryFormat/Layout 反序列化 [PR #173562]
- 修复 proxy executor 在 AOTI 场景中对 `ScalarType`、`MemoryFormat`、`Layout` 枚举值的反序列化 bug。

#### 添加 `torch_from_blob` 的导出宏 [PR #174270]
- 为 `torch::from_blob` 实现添加 `AOTI_TORCH_EXPORT` 宏，确保正确的符号可见性。

#### nonzero_static 的 C-shim 支持 [PR #173229]
- 在 AOTI C-shim 中为 `nonzero_static` 算子生成对应的 shim，完善算子覆盖。

---

## 四、FlexAttention 相关改进

| PR | 内容 |
|----|------|
| #174610 | 支持 blockmask 接受任意可调用对象（callable），扩展灵活性 |
| #164931 | 完整修复 FlexAttention decode 对非 2 的幂次方 head 维度的支持 |
| #174251 | 修复 score_mod 中捕获梯度的数据类型（dtype）问题 |
| #166927 | XPU 上的 FlexAttention backward 启用 tensor descriptor |

---

## 五、ROCm 相关（Inductor + torch.compile）

| PR | 内容 |
|----|------|
| #166492 | 为 ROCm 启用 StaticCudaLauncher |
| #173166 | 在 ROCm 上启用并修复若干 AOT inductor 测试 |
| #173209 | ROCm CI 切换至 MI350 runner |
| #172780 | 修复 ROCm 上多个 unit test 的失败（Navi/MI200/MI300/MI350）|

---

## 六、torch.compile + Deterministic（确定性计算）专题

> 该方向的核心目标是：当用户设置 `torch.use_deterministic_algorithms(True)` 时，`torch.compile` 编译产物的行为应与 eager 完全一致。

### 背景：2.10 建立的基础

| PR | 版本 | 内容 |
|----|------|------|
| [#163589](https://github.com/pytorch/pytorch/pull/163589) | 2.10 | **[Inductor] 确定性模式**：新增 `torch._inductor.config.deterministic`，在开启时跳过所有会影响数值结果的 on-device benchmark，包括：pad-mm、dynamic rblock scaling、template autotuning、coordinate descent tuning for reduction、reduction config autotuning（RBLOCK/num_warps 影响数值，XBLOCK 不影响）、计算通信重排序 benchmark |
| [#165950](https://github.com/pytorch/pytorch/pull/165950) | 2.10 | **自动联动 `use_deterministic_algorithms`**：当用户调用 `torch.use_deterministic_algorithms(True)` 时，自动激活 Inductor 的 deterministic mode（`config.deterministic = True`） |
| [#164532](https://github.com/pytorch/pytorch/pull/164532) | 2.10 | 配套的测试与 config 传播修复 |

这三个 PR 是 2.10 的核心，也是 2.11 的起点。

---

### 2.11 开发窗口（2026-01-22 至 2026-02-16）中的相关 PR

#### ❌ PR #174718 —— `empty`/`empty_like` 的确定性填充（已合并后被 Revert）

- **Issue [#174386](https://github.com/pytorch/pytorch/issues/174386)**：用户发现在 `torch.use_deterministic_algorithms(True)` 下，`torch.compile` 中 `empty_like` 返回的是**未初始化的随机内存**，而非 eager 下规范的 NaN 填充。
- **修复（PR #174718）**：在 Inductor 的 allocation 路径中加入确定性 guard，检测到确定性模式时改用 `torch.empty_strided(...)` 等走 eager 语义的分配方式。
- **状态**：PR 于 2026-02-12/13 合并，但随后因某些 CI 问题被 **Revert**，最终未进入 2.11 release。该 fix 在 2.11 分支切出后重新尝试推进（后续有 PR #178119 以略不同的方式在 Windows 上再次修复）。

#### 🟡 PR #174813 —— FlexAttention Backward 的确定性实现（未及时合并）

- **背景**：FlexAttention backward 中使用了 atomicAdd，在 `use_deterministic_algorithms(True)` 时会 throw error。
- **修复**：通过计算 `dq_write_order`（exclusive prefix sum 确定写入顺序），实现确定性的 backward pass。
- **性能**：在 S≥8192 时开销 <0.3%，基本无负担。
- **状态**：PR 于 2026-02-11 创建，在 2.11 分支切出（2026-02-16）时尚未合并。

---

### 2.11 分支切出后继续推进的相关修复

#### ✅ PR #177166 —— `nn.functional.pad` + deterministic + compile 崩溃修复（2026-03-18 合并）

- **Issue [#170079](https://github.com/pytorch/pytorch/issues/170079)**（发现于 2.9.1）：`torch.compile(ReplicationPad1d(...), fullgraph=True)` 在 `use_deterministic_algorithms(True)` 时 crash，错误为 `Unsupported: Attempted to call function marked as skipped`。
- **根因**：`replication_pad1d_backward` 的 CUDA 实现使用了 atomicAdd（非确定性），PyTorch 走一条通过 `importlib.import_module` 的 Python decomposition 路径来绕过它。但 Dynamo 无法 trace `importlib.import_module`，导致 `fullgraph=True` 时报错。
- **修复**：用 `@nonstrict_trace` 装饰该 decomposition 函数，让 Dynamo 跳过其内部追踪、由 AOTAutograd 负责展开——这样 Dynamo 不需要进入，AOTAutograd 能正常处理 backward 的确定性分解。
- **来源**：PT2 Bug Bash 活动（专项 bug 修复冲刺）的成果。

#### ❌ PR #167318 —— `max_pool2d_with_indices_backward` 确定性分解（2026-03-19 合并后被 Revert）

- **背景**：`max_pool2d_with_indices_backward` 的 CUDA 实现使用了 atomicAdd，导致 `use_deterministic_algorithms(True)` 时会抛出错误。
- **修复**：新增一个纯 tensor op 的 backward 分解（通过 gather + scatter_add 实现），不依赖任何 atomicAdd，从而可在确定性模式下正常运行。
- **关键注意**：该分解优先级低于 Inductor 自身的 max_pool2d 优化路径，**不影响 Inductor 编译的代码路径**；仅影响非 Inductor backend 和 `torch.export`。
- **状态**：合并后因其他问题被 Revert，未最终落地。

#### 🟡 PR #176842 —— `reorder_for_locality` 导致 RNG 顺序变化（仍在 review）

- **Issue [#175156](https://github.com/pytorch/pytorch/issues/175156)**（发现于 2.11.0.dev）：在 `torch.compile(backend='inductor')` 下，多个 `randint` 调用的结果与 eager 不一致，即使手动 `manual_seed` 也无法复现 eager 结果。
- **根因**：`reorder_for_locality` 调度优化 pass 会对 RNG 操作（如 `aten.randint`）进行重排序，而 RNG 操作消耗全局 RNG state，重排序会改变随机数序列。
- **修复**：在 `reorder_for_locality` 中将 RNG op 标记为不可重排，保持与 eager 相同的 RNG 消耗顺序。
- **状态**：PR 于 2026-03-08 提交，截至 2026-03-23 仍在 review 阶段。

#### 🟡 PR #178119 —— Windows 上 Inductor CPU 确定性 empty 路径修复（2026-03-23 提交，仍在 review）

- 专门修复 Windows 平台上 Inductor CPU 的 empty 分配路径未正确遵守确定性语义的问题（是 PR #174718 的后续尝试）。

---

### 小结：2.11 中 deterministic 方向的整体情况

| 类别 | 结论 |
|------|------|
| **2.10 已解决** | Inductor 确定性模式（`config.deterministic`）和与 `use_deterministic_algorithms` 的联动 |
| **2.11 窗口内** | 主要修复（`empty_like` 确定性）被合并又 Revert，未能进入 release |
| **2.11 分支切出后** | `nn.functional.pad` + 确定性 crash 修复（#177166）合并进 2.11；`max_pool2d` 确定性分解（#167318）Revert |
| **进行中** | RNG 重排序问题（#176842）、Windows empty 路径（#178119）仍在推进 |

**核心结论**：2.11 对 deterministic 方向**没有重大新功能落地**。2.10 建立了框架（`config.deterministic` + `use_deterministic_algorithms` 联动），2.11 主要是**针对具体 op 的 corner case 修复**（replication pad crash、empty_like 语义差异、RNG 顺序不一致），且部分修复因各种原因被 Revert，属于**持续修缮阶段**而非架构性突破。

---

## 七、总结

PyTorch 2.11 的 `torch.compile` / `torch.inductor` 方向延续了 2.10 的优化路线，主要改进集中在：

1. **Combo Kernels 全面优化**：扁平化 Grid、PDL 支持、独立子内核块维度，大幅提升 kernel 合并后的 GPU 利用率。
2. **NVGEMM 扩展**：FP8 scaled mm 和 GroupGEMM 支持，配合 heuristics 自动选择最优配置。
3. **Inductor XPU GEMM 重构**：7 步系列 PR 完成了 CUTLASS GEMM 框架从 CUDA 到 XPU 的泛化。
4. **编译时间持续降低**：Dynamo 前端对 `bind_args`、`inspect.signature`、attr 缓存等热点路径做了专项优化。
5. **可观测性提升**：Dynamo Profiler、per-graph config override、编译事件在 Profiler 中可见，让调试更友好。
6. **AOTI XPU 完善**：独立编译 API、多架构 kernel 等特性进一步扩展了 AOTI 对 Intel GPU 的支持。
7. **Deterministic 持续修缮**：针对 `empty_like`、replication pad、RNG 顺序等 corner case 的修复正在推进，但无架构性突破。

---

*参考：[pytorch/pytorch releases](https://github.com/pytorch/pytorch/releases) | [release/2.11 branch](https://github.com/pytorch/pytorch/tree/release/2.11)*
