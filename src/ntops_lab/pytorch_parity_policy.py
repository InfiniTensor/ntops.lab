"""PyTorch top-level parity policy for ntops.lab.

The manifest contains runnable numeric/operator entries.  Some public
``torch`` names are aliases, in-place variants, backend internals, or global
runtime utilities; registering all of those as runnable operators would create
duplicate semantics.  This module centralizes the classification used by audit
scripts and tests.
"""


ALIAS_COVERED = {
    "arccos_": "acos_",
    "arccosh_": "acosh_",
    "arcsin_": "asin_",
    "arctan_": "atan_",
    "arctanh": "atanh",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arctan": "atan",
    "concat": "cat",
    "concatenate": "cat",
    "multiply": "mul",
    "row_stack": "vstack",
    "unsafe_chunk": "chunk",
    "unsafe_split": "split",
    "unsafe_split_with_sizes": "split_with_sizes",
}


MUTATION_VARIANTS = {
    "acos_": "acos",
    "acosh_": "acosh",
    "addmv_": "addmv",
    "alpha_dropout_": "alpha_dropout",
    "arccos_": "acos",
    "arccosh_": "acosh",
    "arcsin_": "asin",
    "arctan_": "atan",
    "arctanh": "atanh",
    "asin_": "asin",
    "as_strided_": "as_strided",
    "atan_": "atan",
    "atanh_": "atanh",
    "conj_physical_": "conj_physical",
    "deg2rad_": "deg2rad",
    "detach_": "detach",
    "dropout_": "dropout",
    "embedding_renorm_": "embedding_renorm",
    "erfc_": "erfc",
    "feature_alpha_dropout_": "feature_alpha_dropout",
    "feature_dropout_": "feature_dropout",
    "fill_": "fill",
    "fix_": "fix",
    "frac_": "frac",
    "gcd_": "gcd",
    "i0_": "i0",
    "index_put_": "index_put",
    "lcm_": "lcm",
    "ldexp_": "ldexp",
    "log2_": "log2",
    "log_": "log",
    "nan_to_num_": "nan_to_num",
    "negative_": "negative",
    "rad2deg_": "rad2deg",
    "resize_as_": "resize_as",
    "resize_as_sparse_": "resize_as_sparse",
    "round_": "round",
    "sinc_": "sinc",
    "trunc_": "trunc",
    "xlogy_": "xlogy",
}


DEPRECATED_REPLACEMENTS = {
    "eig": "linalg.eig",
    "lstsq": "linalg.lstsq",
    "matrix_rank": "linalg.matrix_rank",
    "range": "arange",
    "solve": "linalg.solve",
    "symeig": "linalg.eigh",
}


BACKEND_INTERNAL_PREFIXES = {
    "batch_norm_backward_": "autograd/backend internal batch-norm helper",
    "batch_norm_gather_": "backend internal batch-norm helper",
    "batch_norm_update_": "backend internal batch-norm helper",
    "cudnn_": "cuDNN backend entry point",
    "fbgemm_": "FBGEMM backend entry point",
    "miopen_": "MIOpen backend entry point",
    "mkldnn_": "MKLDNN backend entry point",
    "quantized_rnn_": "quantized backend RNN helper",
    "sym_": "symbolic-shape helper",
}


EXCLUDED_EXACT = {
    "are_deterministic_algorithms_enabled": "global deterministic runtime state",
    "get_autocast_cpu_dtype": "autocast runtime state",
    "get_autocast_dtype": "autocast runtime state",
    "get_autocast_gpu_dtype": "autocast runtime state",
    "get_autocast_ipu_dtype": "autocast runtime state",
    "get_autocast_xla_dtype": "autocast runtime state",
    "get_default_device": "global default device state",
    "get_default_dtype": "global default dtype state",
    "get_deterministic_debug_mode": "global deterministic runtime state",
    "get_file_path": "build/runtime utility",
    "get_float32_matmul_precision": "global matmul precision state",
    "get_num_interop_threads": "thread runtime state",
    "get_num_threads": "thread runtime state",
    "is_anomaly_check_nan_enabled": "autograd debug runtime state",
    "is_anomaly_enabled": "autograd debug runtime state",
    "is_autocast_cache_enabled": "autocast runtime state",
    "is_autocast_cpu_enabled": "autocast runtime state",
    "is_autocast_enabled": "autocast runtime state",
    "is_autocast_ipu_enabled": "autocast runtime state",
    "is_autocast_xla_enabled": "autocast runtime state",
    "is_deterministic_algorithms_warn_only_enabled": "global deterministic runtime state",
    "is_distributed": "distributed build/runtime query",
    "is_grad_enabled": "autograd runtime state",
    "is_inference_mode_enabled": "autograd/inference runtime state",
    "is_vulkan_available": "backend availability query",
    "is_warn_always_enabled": "warning runtime state",
    "set_anomaly_enabled": "autograd debug runtime state",
    "set_autocast_cache_enabled": "autocast runtime state",
    "set_autocast_cpu_dtype": "autocast runtime state",
    "set_autocast_cpu_enabled": "autocast runtime state",
    "set_autocast_dtype": "autocast runtime state",
    "set_autocast_enabled": "autocast runtime state",
    "set_autocast_gpu_dtype": "autocast runtime state",
    "set_autocast_ipu_dtype": "autocast runtime state",
    "set_autocast_ipu_enabled": "autocast runtime state",
    "set_autocast_xla_dtype": "autocast runtime state",
    "set_autocast_xla_enabled": "autocast runtime state",
    "set_default_device": "global default device state",
    "set_default_dtype": "global default dtype state",
    "set_default_tensor_type": "global default tensor type state",
    "set_deterministic_debug_mode": "global deterministic runtime state",
    "set_float32_matmul_precision": "global matmul precision state",
    "set_flush_denormal": "global floating-point runtime state",
    "set_num_interop_threads": "thread runtime state",
    "set_num_threads": "thread runtime state",
    "set_printoptions": "formatting/runtime display state",
    "set_rng_state": "global RNG state API",
    "set_warn_always": "warning runtime state",
    "use_deterministic_algorithms": "global deterministic runtime state",
    "align_tensors": "named-tensor alignment utility, not a runnable numeric kernel",
    "autocast_decrement_nesting": "autocast runtime state",
    "autocast_increment_nesting": "autocast runtime state",
    "batch_norm_elemt": "backend internal batch-norm helper",
    "batch_norm_stats": "backend internal batch-norm helper",
    "choose_qparams_optimized": "quantization calibration helper",
    "classproperty": "Python utility",
    "clear_autocast_cache": "autocast runtime state",
    "compile": "compiler frontend, not an operator kernel",
    "compiled_with_cxx11_abi": "build metadata query",
    "cond": "control-flow transform",
    "fork": "JIT future/runtime utility",
    "from_file": "file-backed storage construction",
    "get_rng_state": "global RNG state API",
    "import_ir_module": "JIT serialization API",
    "import_ir_module_from_buffer": "JIT serialization API",
    "init_num_threads": "thread runtime state",
    "initial_seed": "global RNG state API",
    "load": "serialization API",
    "manual_seed": "global RNG state API",
    "merge_type_from_type_comment": "JIT parser utility",
    "parse_ir": "JIT parser utility",
    "parse_schema": "JIT parser utility",
    "parse_type_comment": "JIT parser utility",
    "prepare_multiprocessing_environment": "runtime setup utility",
    "profiler_allow_cudagraph_cupti_lazy_reinit_cuda12": "profiler/runtime utility",
    "save": "serialization API",
    "seed": "global RNG state API",
    "thread_safe_generator": "RNG helper",
    "to_dlpack": "DLPack export utility; covered by interop policy separately",
    "typename": "debug/type-name utility",
    "unify_type_list": "JIT type utility",
    "vmap": "higher-order transform",
    "wait": "JIT future/runtime utility",
    "while_loop": "control-flow transform",
}


RUNTIME_BLOCKED = {
    "quantized_gru_cell": "requires FBGEMM PackBMatrix packed weight wrappers that ntops.lab cannot construct yet",
    "quantized_lstm_cell": "requires FBGEMM PackBMatrix packed weight wrappers that ntops.lab cannot construct yet",
    "indices_copy": "PyTorch reports SparseCPU/SparseCUDA backend unavailable in the current environment",
}


TENSOR_METADATA_PREDICATES = {
    "get_device": "tensor device metadata query",
    "is_complex": "tensor dtype predicate",
    "is_conj": "tensor view metadata predicate",
    "is_floating_point": "tensor dtype predicate",
    "is_inference": "tensor autograd/inference metadata predicate",
    "is_neg": "tensor view metadata predicate",
    "is_nonzero": "scalar tensor truth-value predicate",
    "is_same_size": "tensor shape metadata predicate",
    "is_signed": "tensor dtype predicate",
    "is_storage": "Python object/storage predicate",
    "is_tensor": "Python object/tensor predicate",
}


PENDING_REAL_OPS = {
    "fused_moving_avg_obs_fake_quant": "quantization observer/fake-quant fused kernel",
    "lobpcg": "iterative sparse/dense eigen solver",
    "quantized_batch_norm": "quantized normalization API",
    "quantized_gru": "quantized recurrent API",
    "quantized_lstm": "quantized recurrent API",
    "quantized_max_pool1d": "quantized pooling API",
    "quantized_max_pool2d": "quantized pooling API",
    "quantized_max_pool3d": "quantized pooling API",
    "slice_inverse": "layout inverse for slice/scatter-style transformations",
}


def classify_name(name: str, manifest_ops: set[str]) -> tuple[str, str]:
    """Return ``(classification, detail)`` for a missing public torch name."""

    if name in ALIAS_COVERED:
        target = ALIAS_COVERED[name]
        status = "alias_covered" if target in manifest_ops else "alias_target_missing"
        return status, target
    if name in MUTATION_VARIANTS:
        return "mutation_variant_pending", MUTATION_VARIANTS[name]
    if name in DEPRECATED_REPLACEMENTS:
        return "deprecated_replacement", DEPRECATED_REPLACEMENTS[name]
    if name in RUNTIME_BLOCKED:
        return "runtime_blocked", RUNTIME_BLOCKED[name]
    if name in TENSOR_METADATA_PREDICATES:
        return "metadata_predicate", TENSOR_METADATA_PREDICATES[name]
    if name in PENDING_REAL_OPS:
        return "pending_real_operator", PENDING_REAL_OPS[name]
    if name in EXCLUDED_EXACT:
        return "excluded_non_operator", EXCLUDED_EXACT[name]
    for prefix, reason in BACKEND_INTERNAL_PREFIXES.items():
        if name.startswith(prefix):
            return "excluded_backend_internal", reason
    return "unclassified_missing", ""
