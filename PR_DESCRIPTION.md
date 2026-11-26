# [2026秋季][赛题1-2] lexing-2026

> 分支：`2026-autumn-lexing-2026-1-2`（提交至 ntops.lab；配套编译器改动提交至
> `ninetoothed` 同名分支）

## 1. 测试环境说明

- **适配算子**：以 ntops.lab 中 `reduction/log_softmax` 为切入点，基线覆盖 13 个算子
  × 3 档形状（`add` / `mul` / `gelu` / `silu` / `functional_gelu` / `softmax` /
  `log_softmax` / `sum` / `amax` / `cross_entropy` / `mm` / `addmm` / `bmm`）；
  通用性验证另覆盖 12 个归约类算子。
- **目标硬件平台**：本地实际测试平台为 **NVIDIA GeForce RTX 3090（24GB）**。
  赛题要求的昇腾、天数智芯、沐曦、摩尔线程、海光平台本地无硬件，故未声称实测数据；
  通用性通过 SSA pass 的后端无关性、各后端 `default_spec()` 注册检查与生成代码检查验证。
- **软件环境**：Python 3.13.13、PyTorch 2.13.0+cu130、Triton 3.7.1、
  ninetoothed `4295e9c`、ntops.lab `17b1dbd`（均为 editable 安装）。
- **计时方法**：`ninetoothed.compiler.benchmark.benchmark_kernel`（CUDA graph → device
  event → Python 回退），最小采样 20ms，每点 3 轮取最小值。优化前后使用同一脚本、
  同一输入规模与同一评测条件。

## 2. 正确性验证

1. 直接调用算子 spec 的 `check()`：`add` / `softmax` / `mm` 全部通过
   （max_abs_error 分别为 0 / 1.79e-7 / 0）。
2. 基线脚本中 13 个算子 × 3 档形状共 39 组，**全部 `correct=True`**；
   优化后重跑同样 39 组，仍全部 `correct=True`，误差量级不变。
3. `log_softmax` 优化前后 max_abs_error 均为 `9.54e-07`，数值语义未改变。
4. 新增 10 个 CSE pass 单元测试（`ninetoothed/tests/test_cse.py`）：覆盖重复纯操作消除、
   attrs/结果类型区分、输出与操作数重命名、副作用操作保留、默认流水线注册、
   全部 5 个后端支持，全部通过。
5. 运行 `ninetoothed` 全量 pytest 测试套件（结果见 PR 评论/附件）。

## 3. 性能测试结果

### 3.1 log_softmax（同一 application，仅切换 CSE pass）

| rows | dim | 优化前(ms) | 优化后(ms) | 加速比 | 误差 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 32 | 0.1586 | 0.0129 | **12.30x** | 9.54e-07 |
| 32768 | 32 | 0.9206 | 0.0725 | **12.70x** | 9.54e-07 |
| 131072 | 32 | 3.5697 | 0.2693 | **13.25x** | 9.54e-07 |
| 4096 | 64 | 0.8893 | 0.0330 | **26.94x** | 9.54e-07 |
| 32768 | 64 | 6.4154 | 0.2343 | **27.38x** | 9.54e-07 |
| 131072 | 64 | 25.9097 | 0.9581 | **27.04x** | 9.54e-07 |

复现：`python benchmarks/cse_measure.py`（ninetoothed 仓库），原始数据见
`benchmarks/cse_results.json`。

### 3.2 全量基线回归（13 算子 × 3 形状）

优化前后逐点对比：`log_softmax` 三档分别获得 **12.25x / 12.98x / 12.75x**；
其余 36 组变化均在 ±2% 以内（噪声范围），无性能回退。

| 算子 | 优化前(ms) | 优化后(ms) | 加速比 |
|---|---:|---:|---:|
| log_softmax s0 | 0.1407 | 0.0115 | **12.25x** |
| log_softmax s1 | 0.8509 | 0.0655 | **12.98x** |
| log_softmax s2 | 3.3531 | 0.2629 | **12.75x** |
| softmax s1 | 0.0645 | 0.0641 | 1.01x |
| add s2 | 0.0158 | 0.0157 | 1.00x |
| mm s2 | 0.0406 | 0.0406 | 1.00x |
| （其余 30 组） | — | — | 0.96–1.02x |

### 3.3 相对 PyTorch

`log_softmax` 优化前相对 PyTorch 仅 0.01x（慢约 80–100 倍），优化后提升至
0.15–0.18x（慢约 6–7 倍）。绝对延迟从 0.85ms 降至 0.065ms（rows=32768, dim=32）。

## 4. 优化方案说明

### 4.1 瓶颈定位

查看生成的 Triton kernel（`~/.ninetoothed/artifacts/.../ntops_lab_log_softmax.triton.py`）
发现存在**嵌套 O(dim²) 循环**：

```python
for v2_elem_0_i in range(0, size_1):        # reduce.max
    ...
for v8_elem_1_i in range(0, size_1):        # reduce.sum
    for v2_v8_elem_1_body_elem_5_i in range(0, size_1):  # max 被重复计算
        ...
```

SSA 层面，`log_softmax` 的 application 为：

```python
m = max(value, axis=1)
e = exp(value - m[:, None])
out = value - m[:, None] - log(sum(e, axis=1))[:, None]
```

前端产出两条重复计算链：

```
%3 = tensor.view(%2)        # m[:, None]
%4 = arith.sub(%1, %3)      # value - m
%5 = math.exp(%4)           # e
%6 = tensor.view(%2)        # 重复的 m[:, None]
%7 = arith.sub(%1, %6)      # 重复的 value - m  <-- 在 reduce.sum 之后消费
%8 = reduce.sum(%5)
%11 = arith.sub(%7, %10)    # 最终输出使用重复计算的 %7
```

`%6` / `%7` 是 `%3` / `%4` 的**重复纯计算**，且 `%7` 在 `reduce.sum` 之后才被消费。
后端 emitter 为保证 `%7` 可用，会在 `reduce.sum` 循环体内重新发射其完整依赖链，
导致 `reduce.max` 被重复执行一次，形成 O(dim²) 嵌套循环。`softmax` 没有重复计算，
因此只有两个顺序循环，无此问题。

### 4.2 编译器优化：通用 CSE pass

新增硬件无关的 SSA pass `ssa.eliminate_common_subexpressions`，注册在默认流水线中
`ssa.canonicalize` 之后、`ssa.analyze_effects` 之前，使所有后端受益。

关键实现（`src/ninetoothed/compiler/cse.py`）：

1. 对每个 SSA 块做前向扫描，为纯操作构造结构键
   `(opcode, 规范化操作数, attrs, 结果类型)`。
2. 命中重复键时，删除后出现的重复操作，并把其所有使用点（操作数、attrs 中的
   SSA 名、Program outputs）重定向到首次定义。
3. 候选仅限确定性的纯操作：`arith.*`、`cmp.*`、`math.*`（排除 `math.rand`）、
   `reduce.*`、`tensor.view` / `tensor.cast` / `tensor.extract` 等；排除 `mem.*`、
   `call.*`、带 region 的操作与随机值。
4. 递归处理 region，保持嵌套块的替换语义；SSA verify 在流水线每个 pass 后执行。

优化后 `log_softmax` 的 SSA 变为（与手写改写完全一致）：

```
%3 = view(%2)
%4 = sub(%1, %3)
%5 = exp(%4)
%8 = sum(%5)
%11 = sub(%4, %10)          # 复用 %4，不再重复消费 max
store %11
```

生成代码只剩**两个顺序循环**，延迟大幅下降。该 pass 不改变任何归约本身的实现或
浮点归约顺序，因此数值误差保持不变。

### 4.3 关键改动文件（ninetoothed）

- `src/ninetoothed/compiler/cse.py`（新增）：CSE 实现。
- `src/ninetoothed/compiler/passes.py`：注册 `EliminateCommonSubexpressions` 到默认流水线。
- `tests/test_cse.py`（新增）：10 个单元测试。
- `benchmarks/cse_measure.py` / `benchmarks/cse_results.json`（新增）：可复现基准。

## 5. 通用性验证

1. **后端无关**：pass 的 `category = HARDWARE_INDEPENDENT`，不引用任何后端 API；
   单元测试验证 `triton` / `cuda` / `tilelang` / `ascendc` / `bangc` 五个后端的
   `default_spec()` 均包含该 pass，且位于效果分析之前。
2. **跨算子无回退**：13 个算子 × 3 形状的回归测试中，除目标 `log_softmax` 外
   其余 36 组变化均在 ±2% 内，全部 `correct=True`。
3. **跨算子正确性**：另测 12 个归约类算子
   （`softmax` / `softmax_entropy_width64` / `logsumexp_width32` / `var` /
   `center_width64` / `normalize_rms_width64` / `l1_l2_ratio_width8` /
   `cosine_embedding_loss` / `pairwise_distance` / `triplet_margin_loss` /
   `cross_entropy` 等），优化前后均 `correct=True`，误差在同等量级。
4. **更 wide 形状收益更大**：因原瓶颈是 O(dim²) 嵌套循环，dim=64 时加速比达到
   约 **27x**，体现 pass 对重复归约链的消除效果随问题规模增长。
5. **跨平台说明**：本地仅有 NVIDIA GPU；对昇腾/天数智芯/沐曦/摩尔线程/海光，
   由于 pass 在 SSA 层、位于所有后端特定 pass 之前，且不依赖任何 Triton 专有结构，
   其优化效果来源于消除 SSA 层重复计算，不依赖具体硬件。已通过后端注册检查与
   SSA/生成代码检查验证，未声称无硬件平台的实测性能数据。

## 6. 诚信与引用声明

已提交本人署名的 `HONOR_CODE.md`，并在 `REFERENCE.md` 中逐条列明引用的开源仓库、
文档与工具。所有性能数据可复现，未伪造测试结果。
