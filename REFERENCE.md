# REFERENCES / 引用清单

本提交参考并使用的公开资源如下。所有第三方代码均按其原许可证使用，未复制其代码到
本提交的新增源文件中；新增实现为独立编写。

## 代码仓库

1. **InfiniTensor/ninetoothed** — <https://github.com/InfiniTensor/ninetoothed>
   - 提交基线：`4295e9c`
   - 用途：赛题目标编译器。本次改动在其 SSA pass 框架
     (`src/ninetoothed/compiler/passes.py`) 中新增通用 CSE pass；参考了
     `compiler/canonicalize.py` 的 SSA 重写风格、`backends/triton_reductions.py` 的
     pass 注册方式，以及 `compiler/benchmark.py` 的计时工具。
   - 许可证：Apache-2.0（随原仓库保留）。

2. **InfiniTensor/ntops.lab** — <https://github.com/InfiniTensor/ntops.lab>
   - 提交基线：`17b1dbd`
   - 用途：赛题指定算子集合。使用 `ntops_lab.list_operators()`、`get_operator()`，
     以及 `testing/specs/` 下的算子 spec 做正确性与性能测试；分析
     `kernels/reduction/log_softmax.py` 与 `softmax.py` 定位瓶颈。
   - 许可证：Apache-2.0（随原仓库保留）。

## 公开文档

3. NineToothed README / CONTRIBUTING — <https://github.com/InfiniTensor/ninetoothed>
   - 用途：编译流程、pass 注册约定、分支与提交规范。

4. ntops.lab README / CONTRIBUTING — <https://github.com/InfiniTensor/ntops.lab>
   - 用途：算子清单、spec 接口、正确性检查流程。

5. PyTorch 文档 — <https://pytorch.org/docs/stable/>
   - 用途：参考实现 `torch.log_softmax` / `torch.softmax` / `torch.allclose` 的语义。

6. Triton 文档 — <https://triton-lang.org/>
   - 用途：理解生成代码中的 `tl.sum` / `tl.max` / `@triton.jit` 归约循环语义。

## 赛事材料

7. 2026 秋季人工智能大赛推理赛题 1-2 题目描述（本地存档：
   `九齿优化赛/2026秋季推理赛题-九齿优化.md`）
   - 用途：赛题要求、分支命名、PR 描述要求、目标平台与评优规则。

## 自有产物（本次提交新增）

8. `HONOR_CODE.md` — 本人署名声明。
9. `benchmarks/cse_measure.py` 与 `benchmarks/cse_results.json`（ninetoothed 仓库）—
   本人编写的可复现基准脚本与原始数据。
10. `bench/baseline.py`、`bench/generality_probe.py`、`bench/FINDINGS.md`
    （本地工作区，未随 PR 提交）— 本人编写的诊断、基线与通用性验证脚本与记录。

## 未使用的资源

本人未参考任何付费、私有或他人未公开的 compiler 优化实现；未使用任何外部 CSE 算法的
具体实现代码，仅依据标准编译原理（common subexpression elimination）自行设计并实现
针对本 SSA IR 的简化版本。
