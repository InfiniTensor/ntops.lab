# Operator Coverage

This repository intentionally includes only runnable NineToothed operator implementations.
Unsupported/todo scaffold source files are excluded from the commit-ready tree.

- Total operators in this repo: 246
- Runnable operators: 246
- Unsupported source files included: 0

## Categories

| Category | Runnable operators |
| --- | ---: |
| `creation` | 9 |
| `fused/fla` | 1 |
| `fused/general` | 9 |
| `layout` | 21 |
| `linear` | 18 |
| `normalization` | 2 |
| `pointwise` | 162 |
| `reduction` | 24 |

## Operators

### creation

| Operator | Kernel file |
| --- | --- |
| `eye` | `kernels/creation/eye.py` |
| `eye_m` | `kernels/creation/eye_m.py` |
| `full` | `kernels/creation/full.py` |
| `full_like` | `kernels/creation/full_like.py` |
| `new_full` | `kernels/creation/new_full.py` |
| `ones` | `kernels/creation/ones.py` |
| `ones_like` | `kernels/creation/ones_like.py` |
| `zeros` | `kernels/creation/zeros.py` |
| `zeros_like` | `kernels/creation/zeros_like.py` |

### fused/fla

| Operator | Kernel file |
| --- | --- |
| `cumsum` | `kernels/fused/fla/cumsum.py` |

### fused/general

| Operator | Kernel file |
| --- | --- |
| `add_rms_norm` | `kernels/fused/general/add_rms_norm.py` |
| `fused_add_rms_norm` | `kernels/fused/general/fused_add_rms_norm.py` |
| `geglu` | `kernels/fused/general/geglu.py` |
| `gelu_and_mul` | `kernels/fused/general/gelu_and_mul.py` |
| `reglu` | `kernels/fused/general/reglu.py` |
| `rotary_embedding` | `kernels/fused/general/rotary_embedding.py` |
| `silu_and_mul` | `kernels/fused/general/silu_and_mul.py` |
| `skip_layernorm` | `kernels/fused/general/skip_layernorm.py` |
| `swiglu` | `kernels/fused/general/swiglu.py` |

### layout

| Operator | Kernel file |
| --- | --- |
| `_upsample_nearest_exact1d` | `kernels/layout/upsample_nearest_exact1d.py` |
| `alias_copy` | `kernels/layout/alias_copy.py` |
| `avg_pool2d` | `kernels/layout/avg_pool2d.py` |
| `avg_pool3d` | `kernels/layout/avg_pool3d.py` |
| `contiguous` | `kernels/layout/contiguous.py` |
| `conv1d` | `kernels/layout/conv1d.py` |
| `conv2d` | `kernels/layout/conv2d.py` |
| `conv3d` | `kernels/layout/conv3d.py` |
| `conv_depthwise2d` | `kernels/layout/conv_depthwise2d.py` |
| `conv_transpose1d` | `kernels/layout/conv_transpose1d.py` |
| `conv_transpose2d` | `kernels/layout/conv_transpose2d.py` |
| `copy` | `kernels/layout/copy.py` |
| `cudnn_convolution` | `kernels/layout/cudnn_convolution.py` |
| `max_pool2d_with_indices` | `kernels/layout/max_pool2d_with_indices.py` |
| `max_pool3d_with_indices` | `kernels/layout/max_pool3d_with_indices.py` |
| `resolve_conj` | `kernels/layout/resolve_conj.py` |
| `to` | `kernels/layout/to.py` |
| `upsample_linear1d` | `kernels/layout/upsample_linear1d.py` |
| `upsample_nearest1d` | `kernels/layout/upsample_nearest1d.py` |
| `upsample_nearest2d` | `kernels/layout/upsample_nearest2d.py` |
| `upsample_nearest3d` | `kernels/layout/upsample_nearest3d.py` |

### linear

| Operator | Kernel file |
| --- | --- |
| `addbmm` | `kernels/linear/addbmm.py` |
| `addmm` | `kernels/linear/addmm.py` |
| `addmv` | `kernels/linear/addmv.py` |
| `addr` | `kernels/linear/addr.py` |
| `baddbmm` | `kernels/linear/baddbmm.py` |
| `bmm` | `kernels/linear/bmm.py` |
| `bmm_out` | `kernels/linear/bmm_out.py` |
| `dot` | `kernels/linear/dot.py` |
| `gemm_bias` | `kernels/linear/gemm_bias.py` |
| `gemm_bias_gelu` | `kernels/linear/gemm_bias_gelu.py` |
| `gemm_bias_relu` | `kernels/linear/gemm_bias_relu.py` |
| `grouped_mm` | `kernels/linear/grouped_mm.py` |
| `linear` | `kernels/linear/linear.py` |
| `mm` | `kernels/linear/mm.py` |
| `mm_out` | `kernels/linear/mm_out.py` |
| `mv` | `kernels/linear/mv.py` |
| `outer` | `kernels/linear/outer.py` |
| `vdot` | `kernels/linear/vdot.py` |

### normalization

| Operator | Kernel file |
| --- | --- |
| `layernorm` | `kernels/normalization/layernorm.py` |
| `rms_norm` | `kernels/normalization/rms_norm.py` |

### pointwise

| Operator | Kernel file |
| --- | --- |
| `abs` | `kernels/pointwise/abs.py` |
| `abs_` | `kernels/pointwise/abs_inplace.py` |
| `absolute` | `kernels/pointwise/absolute.py` |
| `acos` | `kernels/pointwise/acos.py` |
| `add` | `kernels/pointwise/add.py` |
| `add_` | `kernels/pointwise/add_inplace.py` |
| `addcdiv` | `kernels/pointwise/addcdiv.py` |
| `addcmul` | `kernels/pointwise/addcmul.py` |
| `angle` | `kernels/pointwise/angle.py` |
| `arcsinh` | `kernels/pointwise/arcsinh.py` |
| `arcsinh_` | `kernels/pointwise/arcsinh_inplace.py` |
| `arctanh_` | `kernels/pointwise/arctanh_inplace.py` |
| `asinh` | `kernels/pointwise/asinh.py` |
| `asinh_` | `kernels/pointwise/asinh_inplace.py` |
| `atan` | `kernels/pointwise/atan.py` |
| `atan2` | `kernels/pointwise/atan2.py` |
| `bitwise_and` | `kernels/pointwise/bitwise_and.py` |
| `bitwise_and_tensor` | `kernels/pointwise/bitwise_and_tensor.py` |
| `bitwise_and_tensor_` | `kernels/pointwise/bitwise_and_tensor_inplace.py` |
| `bitwise_left_shift` | `kernels/pointwise/bitwise_left_shift.py` |
| `bitwise_not` | `kernels/pointwise/bitwise_not.py` |
| `bitwise_not_` | `kernels/pointwise/bitwise_not_inplace.py` |
| `bitwise_or` | `kernels/pointwise/bitwise_or.py` |
| `bitwise_or_tensor` | `kernels/pointwise/bitwise_or_tensor.py` |
| `bitwise_or_tensor_` | `kernels/pointwise/bitwise_or_tensor_inplace.py` |
| `bitwise_right_shift` | `kernels/pointwise/bitwise_right_shift.py` |
| `ceil` | `kernels/pointwise/ceil.py` |
| `ceil_` | `kernels/pointwise/ceil_inplace.py` |
| `ceil_out` | `kernels/pointwise/ceil_out.py` |
| `celu` | `kernels/pointwise/celu.py` |
| `celu_` | `kernels/pointwise/celu_inplace.py` |
| `clamp` | `kernels/pointwise/clamp.py` |
| `clamp_` | `kernels/pointwise/clamp_inplace.py` |
| `clamp_max` | `kernels/pointwise/clamp_max.py` |
| `clamp_max_` | `kernels/pointwise/clamp_max_inplace.py` |
| `clamp_min` | `kernels/pointwise/clamp_min.py` |
| `clamp_min_` | `kernels/pointwise/clamp_min_inplace.py` |
| `clip` | `kernels/pointwise/clip.py` |
| `clip_` | `kernels/pointwise/clip_inplace.py` |
| `copysign` | `kernels/pointwise/copysign.py` |
| `cos` | `kernels/pointwise/cos.py` |
| `cos_` | `kernels/pointwise/cos_inplace.py` |
| `cosh` | `kernels/pointwise/cosh.py` |
| `cosh_` | `kernels/pointwise/cosh_inplace.py` |
| `cosh_out` | `kernels/pointwise/cosh_out.py` |
| `div` | `kernels/pointwise/div.py` |
| `div_mode` | `kernels/pointwise/div_mode.py` |
| `div_mode_` | `kernels/pointwise/div_mode_inplace.py` |
| `elu` | `kernels/pointwise/elu.py` |
| `elu_` | `kernels/pointwise/elu_inplace.py` |
| `eq` | `kernels/pointwise/eq.py` |
| `erf` | `kernels/pointwise/erf.py` |
| `erf_` | `kernels/pointwise/erf_inplace.py` |
| `exp` | `kernels/pointwise/exp.py` |
| `exp2` | `kernels/pointwise/exp2.py` |
| `exp2_` | `kernels/pointwise/exp2_inplace.py` |
| `exp_` | `kernels/pointwise/exp_inplace.py` |
| `exp_out` | `kernels/pointwise/exp_out.py` |
| `expm1` | `kernels/pointwise/expm1.py` |
| `expm1_` | `kernels/pointwise/expm1_inplace.py` |
| `expm1_out` | `kernels/pointwise/expm1_out.py` |
| `fill` | `kernels/pointwise/fill.py` |
| `floor` | `kernels/pointwise/floor.py` |
| `floor_` | `kernels/pointwise/floor_inplace.py` |
| `floor_divide` | `kernels/pointwise/floor_divide.py` |
| `floor_divide_` | `kernels/pointwise/floor_divide_inplace.py` |
| `floor_out` | `kernels/pointwise/floor_out.py` |
| `fmin` | `kernels/pointwise/fmin.py` |
| `fmod` | `kernels/pointwise/fmod.py` |
| `ge` | `kernels/pointwise/ge.py` |
| `gelu` | `kernels/pointwise/gelu.py` |
| `gelu_` | `kernels/pointwise/gelu_inplace.py` |
| `greater` | `kernels/pointwise/greater.py` |
| `gt` | `kernels/pointwise/gt.py` |
| `hardsigmoid` | `kernels/pointwise/hardsigmoid.py` |
| `hardswish_` | `kernels/pointwise/hardswish_inplace.py` |
| `hypot` | `kernels/pointwise/hypot.py` |
| `isclose` | `kernels/pointwise/isclose.py` |
| `isfinite` | `kernels/pointwise/isfinite.py` |
| `isinf` | `kernels/pointwise/isinf.py` |
| `isnan` | `kernels/pointwise/isnan.py` |
| `isneginf` | `kernels/pointwise/isneginf.py` |
| `le` | `kernels/pointwise/le.py` |
| `leaky_relu` | `kernels/pointwise/leaky_relu.py` |
| `leaky_relu_` | `kernels/pointwise/leaky_relu_inplace.py` |
| `leaky_relu_out` | `kernels/pointwise/leaky_relu_out.py` |
| `lerp` | `kernels/pointwise/lerp.py` |
| `log` | `kernels/pointwise/log.py` |
| `log10` | `kernels/pointwise/log10.py` |
| `log10_` | `kernels/pointwise/log10_inplace.py` |
| `log10_out` | `kernels/pointwise/log10_out.py` |
| `log1p` | `kernels/pointwise/log1p.py` |
| `log1p_` | `kernels/pointwise/log1p_inplace.py` |
| `log_sigmoid` | `kernels/pointwise/log_sigmoid.py` |
| `logaddexp` | `kernels/pointwise/logaddexp.py` |
| `logical_and` | `kernels/pointwise/logical_and.py` |
| `logical_and_` | `kernels/pointwise/logical_and_inplace.py` |
| `logical_not` | `kernels/pointwise/logical_not.py` |
| `logical_or` | `kernels/pointwise/logical_or.py` |
| `logical_or_` | `kernels/pointwise/logical_or_inplace.py` |
| `logical_xor` | `kernels/pointwise/logical_xor.py` |
| `logit` | `kernels/pointwise/logit.py` |
| `logit_` | `kernels/pointwise/logit_inplace.py` |
| `lt` | `kernels/pointwise/lt.py` |
| `maximum` | `kernels/pointwise/maximum.py` |
| `minimum` | `kernels/pointwise/minimum.py` |
| `mul` | `kernels/pointwise/mul.py` |
| `mul_` | `kernels/pointwise/mul_inplace.py` |
| `nan_to_num` | `kernels/pointwise/nan_to_num.py` |
| `ne` | `kernels/pointwise/ne.py` |
| `neg` | `kernels/pointwise/neg.py` |
| `neg_` | `kernels/pointwise/neg_inplace.py` |
| `pow` | `kernels/pointwise/pow.py` |
| `prelu` | `kernels/pointwise/prelu.py` |
| `rad2deg` | `kernels/pointwise/rad2deg.py` |
| `reciprocal` | `kernels/pointwise/reciprocal.py` |
| `reciprocal_` | `kernels/pointwise/reciprocal_inplace.py` |
| `relu` | `kernels/pointwise/relu.py` |
| `relu6` | `kernels/pointwise/relu6.py` |
| `relu_` | `kernels/pointwise/relu_inplace.py` |
| `remainder` | `kernels/pointwise/remainder.py` |
| `remainder_` | `kernels/pointwise/remainder_inplace.py` |
| `resolve_neg` | `kernels/pointwise/resolve_neg.py` |
| `round` | `kernels/pointwise/round.py` |
| `rsqrt` | `kernels/pointwise/rsqrt.py` |
| `rsqrt_` | `kernels/pointwise/rsqrt_inplace.py` |
| `rsub` | `kernels/pointwise/rsub.py` |
| `selu` | `kernels/pointwise/selu.py` |
| `selu_` | `kernels/pointwise/selu_inplace.py` |
| `sgn_` | `kernels/pointwise/sgn_inplace.py` |
| `sigmoid` | `kernels/pointwise/sigmoid.py` |
| `sigmoid_` | `kernels/pointwise/sigmoid_inplace.py` |
| `signbit` | `kernels/pointwise/signbit.py` |
| `silu` | `kernels/pointwise/silu.py` |
| `silu_` | `kernels/pointwise/silu_inplace.py` |
| `sin` | `kernels/pointwise/sin.py` |
| `sin_` | `kernels/pointwise/sin_inplace.py` |
| `sinh_` | `kernels/pointwise/sinh_inplace.py` |
| `softplus` | `kernels/pointwise/softplus.py` |
| `softshrink` | `kernels/pointwise/softshrink.py` |
| `sqrt` | `kernels/pointwise/sqrt.py` |
| `sqrt_` | `kernels/pointwise/sqrt_inplace.py` |
| `square` | `kernels/pointwise/square.py` |
| `square_` | `kernels/pointwise/square_inplace.py` |
| `sub` | `kernels/pointwise/sub.py` |
| `sub_` | `kernels/pointwise/sub_inplace.py` |
| `tan` | `kernels/pointwise/tan.py` |
| `tan_` | `kernels/pointwise/tan_inplace.py` |
| `tanh` | `kernels/pointwise/tanh.py` |
| `tanh_` | `kernels/pointwise/tanh_inplace.py` |
| `threshold` | `kernels/pointwise/threshold.py` |
| `true_divide` | `kernels/pointwise/true_divide.py` |
| `true_divide_` | `kernels/pointwise/true_divide_inplace.py` |
| `true_divide_out` | `kernels/pointwise/true_divide_out.py` |
| `trunc_divide` | `kernels/pointwise/trunc_divide.py` |
| `trunc_divide_` | `kernels/pointwise/trunc_divide_inplace.py` |
| `where_scalar_other` | `kernels/pointwise/where_scalar_other.py` |
| `where_scalar_self` | `kernels/pointwise/where_scalar_self.py` |
| `where_self` | `kernels/pointwise/where_self.py` |
| `where_self_out` | `kernels/pointwise/where_self_out.py` |
| `zero` | `kernels/pointwise/zero.py` |
| `zero_` | `kernels/pointwise/zero_inplace.py` |

### reduction

| Operator | Kernel file |
| --- | --- |
| `_safe_softmax` | `kernels/reduction/safe_softmax.py` |
| `all_dim` | `kernels/reduction/all_dim.py` |
| `all_dims` | `kernels/reduction/all_dims.py` |
| `amax` | `kernels/reduction/amax.py` |
| `aminmax` | `kernels/reduction/aminmax.py` |
| `any` | `kernels/reduction/any.py` |
| `any_dim` | `kernels/reduction/any_dim.py` |
| `any_dims` | `kernels/reduction/any_dims.py` |
| `log_softmax` | `kernels/reduction/log_softmax.py` |
| `logsumexp` | `kernels/reduction/logsumexp.py` |
| `max` | `kernels/reduction/max.py` |
| `max_dim` | `kernels/reduction/max_dim.py` |
| `mean` | `kernels/reduction/mean.py` |
| `mean_dim` | `kernels/reduction/mean_dim.py` |
| `mean_dim_comm` | `kernels/reduction/mean_dim_comm.py` |
| `min` | `kernels/reduction/min.py` |
| `min_dim` | `kernels/reduction/min_dim.py` |
| `scaled_softmax` | `kernels/reduction/scaled_softmax.py` |
| `softmax` | `kernels/reduction/softmax.py` |
| `std` | `kernels/reduction/std.py` |
| `sum` | `kernels/reduction/sum.py` |
| `var` | `kernels/reduction/var.py` |
| `var_mean` | `kernels/reduction/var_mean.py` |
| `vector_norm` | `kernels/reduction/vector_norm.py` |
