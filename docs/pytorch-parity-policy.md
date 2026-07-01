# PyTorch 顶层对标策略

ntops.lab 的 manifest 只记录可运行、可验证的算子实现。PyTorch 顶层命名空间里还包含别名、in-place 变体、后端内部入口、序列化/状态/编译 API 和控制流变换。如果把这些名字全部直接注册成 runnable operator，会导致重复语义和不可验证条目。

因此当前对标策略分为：

- `alias_covered`：由已有 manifest 算子覆盖，不重复注册。
- `mutation_variant_pending`：in-place 语义需要单独测试 mutation 行为，不能用 functional 算子冒充。
- `deprecated_replacement`：PyTorch 已弃用或建议迁移到新 API 的入口，记录替代目标。
- `excluded_backend_internal`：cuDNN、MIOpen、MKLDNN、FBGEMM 等后端内部入口默认不进入普通算子 manifest；其中能用当前环境真实运行且能稳定验证的 helper 会作为显式覆盖项进入 manifest。
- `excluded_non_operator`：序列化、编译、线程、RNG 状态、JIT parser、控制流变换等非数值 kernel 默认不作为数值算子；其中可稳定调用并可恢复状态的 runtime/API callable 会进入 `runtime` 分类覆盖。
- `metadata-only creation`：`empty*` 这类未初始化分配进入 manifest，但验证只比较 shape、dtype、device、stride 和量化参数，不比较未定义数据内容。
- `runtime_blocked`：当前 PyTorch/CUDA 环境自身无法运行或后端为 NYI，需要独立策略。
- `pending_real_operator`：仍应继续实现或验证的真实算子。

PyTorch 对标策略通过 manifest、文档和 pytest 测试维护；过程性审计脚本和报告不随提交保存。
