# Unrunnable Operator Support Analysis

- total unrunnable operators: 97

## manual_layout_indexing

- count: 20
- needed support: Can be attempted with current NineToothed, but needs custom offsets, masks, boundary handling, and stride/window mapping.

- `affine_grid_generator` | `layout` | `operators/layout/affine_grid_generator.py`
- `as_strided_copy` | `layout` | `operators/layout/as_strided_copy.py`
- `col2im` | `layout` | `operators/layout/col2im.py`
- `diagonal` | `layout` | `operators/layout/diagonal.py`
- `flip` | `layout` | `operators/layout/flip.py`
- `grid_sample` | `layout` | `operators/layout/grid_sample.py`
- `pad` | `layout` | `operators/layout/pad.py`
- `pixel_shuffle` | `layout` | `operators/layout/pixel_shuffle.py`
- `pixel_unshuffle` | `layout` | `operators/layout/pixel_unshuffle.py`
- `reflection_pad1d` | `layout` | `operators/layout/reflection_pad1d.py`
- `reflection_pad1d_backward` | `layout` | `operators/layout/reflection_pad1d_backward.py`
- `reflection_pad2d` | `layout` | `operators/layout/reflection_pad2d.py`
- `replication_pad1d` | `layout` | `operators/layout/replication_pad1d.py`
- `replication_pad3d` | `layout` | `operators/layout/replication_pad3d.py`
- `roll` | `layout` | `operators/layout/roll.py`
- `t_copy` | `layout` | `operators/layout/t_copy.py`
- `trace` | `layout` | `operators/layout/trace.py`
- `upsample_bicubic2d` | `layout` | `operators/layout/upsample_bicubic2d.py`
- `upsample_bicubic2d_aa` | `layout` | `operators/layout/upsample_bicubic2d_aa.py`
- `upsample_bicubic2d_aa_backward` | `layout` | `operators/layout/upsample_bicubic2d_aa_backward.py`

## manual_fused_analysis

- count: 16
- needed support: Needs per-kernel source analysis and a custom lowering plan.

- `triton_ops_helper` | `fused/fla` | `operators/fused/fla/triton_ops_helper.py`
- `wy_fast` | `fused/fla` | `operators/fused/fla/wy_fast.py`
- `apply_repetition_penalties` | `fused/general` | `operators/fused/general/apply_repetition_penalties.py`
- `cross_entropy_loss` | `fused/general` | `operators/fused/general/cross_entropy_loss.py`
- `cutlass_scaled_mm` | `fused/general` | `operators/fused/general/cutlass_scaled_mm.py`
- `fused_inv_rope_fp8_quant` | `fused/general` | `operators/fused/general/fused_inv_rope_fp8_quant.py`
- `outer` | `fused/general` | `operators/fused/general/outer.py`
- `pack_seq` | `fused/general` | `operators/fused/general/pack_seq.py`
- `rwkv_ka_fusion` | `fused/general` | `operators/fused/general/rwkv_ka_fusion.py`
- `rwkv_mm_sparsity` | `fused/general` | `operators/fused/general/rwkv_mm_sparsity.py`
- `top_k_per_row_prefill` | `fused/general` | `operators/fused/general/top_k_per_row_prefill.py`
- `unpack_seq` | `fused/general` | `operators/fused/general/unpack_seq.py`
- `hc_head_fused_kernel` | `fused/mhc` | `operators/fused/mhc/hc_head_fused_kernel.py`
- `hc_split_sinkhorn` | `fused/mhc` | `operators/fused/mhc/hc_split_sinkhorn.py`
- `mhc_bwd` | `fused/mhc` | `operators/fused/mhc/mhc_bwd.py`
- `mhc_post` | `fused/mhc` | `operators/fused/mhc/mhc_post.py`

## dynamic_selection_atomic

- count: 13
- needed support: Needs top-k/sort/radix selection, histogram/bincount, atomics, or dynamic output compaction.

- `bin_topk` | `fused/dsa` | `operators/fused/dsa/bin_topk.py`
- `sparse_mla` | `fused/dsa` | `operators/fused/dsa/sparse_mla.py`
- `bincount` | `fused/general` | `operators/fused/general/bincount.py`
- `deepseek_v4_attention_combine_topk_swa_indices` | `fused/general` | `operators/fused/general/deepseek_v4_attention_combine_topk_swa_indices.py`
- `deepseek_v4_attention_compute_global_topk_indices_and_lens` | `fused/general` | `operators/fused/general/deepseek_v4_attention_compute_global_topk_indices_and_lens.py`
- `deepseek_v4_attention_fused_q_kv_rmsnorm` | `fused/general` | `operators/fused/general/deepseek_v4_attention_fused_q_kv_rmsnorm.py`
- `flash_mla` | `fused/general` | `operators/fused/general/flash_mla.py`
- `flashmla_sparse` | `fused/general` | `operators/fused/general/flashmla_sparse.py`
- `grouped_topk` | `fused/general` | `operators/fused/general/grouped_topk.py`
- `sparse_attention` | `fused/general` | `operators/fused/general/sparse_attention.py`
- `top_k_per_row_decode` | `fused/general` | `operators/fused/general/top_k_per_row_decode.py`
- `topk_softmax` | `fused/general` | `operators/fused/general/topk_softmax.py`
- `topk_softplus_sqrt` | `fused/general` | `operators/fused/general/topk_softplus_sqrt.py`

## dynamic_index_cache

- count: 13
- needed support: Needs dynamic gather/scatter, indirect load/store, masked updates, KV-cache writes, or quantization metadata access.

- `indexer_k_tiled` | `fused/dsa` | `operators/fused/dsa/indexer_k_tiled.py`
- `utils` | `fused/fla` | `operators/fused/fla/utils.py`
- `concat_and_cache_mla` | `fused/general` | `operators/fused/general/concat_and_cache_mla.py`
- `cp_gather_indexer_k_quant_cache` | `fused/general` | `operators/fused/general/cp_gather_indexer_k_quant_cache.py`
- `deepseek_v4_attention_dequantize_and_gather_k_cache` | `fused/general` | `operators/fused/general/deepseek_v4_attention_dequantize_and_gather_k_cache.py`
- `flash_mla_with_kvcache` | `fused/general` | `operators/fused/general/flash_mla_with_kvcache.py`
- `fused_marlin_moe` | `fused/general` | `operators/fused/general/fused_marlin_moe.py`
- `fused_moe` | `fused/general` | `operators/fused/general/fused_moe.py`
- `indexer_k_quant_and_cache` | `fused/general` | `operators/fused/general/indexer_k_quant_and_cache.py`
- `moe_align_block_size` | `fused/general` | `operators/fused/general/moe_align_block_size.py`
- `reshape_and_cache` | `fused/general` | `operators/fused/general/reshape_and_cache.py`
- `reshape_and_cache_flash` | `fused/general` | `operators/fused/general/reshape_and_cache_flash.py`
- `mhc_pre` | `fused/mhc` | `operators/fused/mhc/mhc_pre.py`

## scan_recurrent_triangular

- count: 11
- needed support: Needs cross-tile scan, recurrent state carry, triangular solve, or block-to-block state transfer.

- `chunk` | `fused/fla` | `operators/fused/fla/chunk.py`
- `chunk_delta_h` | `fused/fla` | `operators/fused/fla/chunk_delta_h.py`
- `chunk_fused_tail_vblock` | `fused/fla` | `operators/fused/fla/chunk_fused_tail_vblock.py`
- `chunk_gated_delta_direct` | `fused/fla` | `operators/fused/fla/chunk_gated_delta_direct.py`
- `chunk_o` | `fused/fla` | `operators/fused/fla/chunk_o.py`
- `chunk_scaled_dot_kkt` | `fused/fla` | `operators/fused/fla/chunk_scaled_dot_kkt.py`
- `fused_cumsum_kkt_solve_tril` | `fused/fla` | `operators/fused/fla/fused_cumsum_kkt_solve_tril.py`
- `fused_recurrent` | `fused/fla` | `operators/fused/fla/fused_recurrent.py`
- `index` | `fused/fla` | `operators/fused/fla/index.py`
- `solve_tril` | `fused/fla` | `operators/fused/fla/solve_tril.py`
- `chunk_gated_delta_rule` | `fused/general` | `operators/fused/general/chunk_gated_delta_rule.py`

## rng_distribution

- count: 9
- needed support: Needs a NineToothed RNG/state API and distribution primitives.

- `act_quant` | `pointwise` | `operators/pointwise/act_quant.py`
- `cauchy` | `pointwise` | `operators/pointwise/cauchy.py`
- `digamma_` | `pointwise` | `operators/pointwise/digamma_inplace.py`
- `exponential_` | `pointwise` | `operators/pointwise/exponential_inplace.py`
- `gcd` | `pointwise` | `operators/pointwise/gcd.py`
- `i0` | `pointwise` | `operators/pointwise/i0.py`
- `i0_` | `pointwise` | `operators/pointwise/i0_inplace.py`
- `special_i0e` | `pointwise` | `operators/pointwise/special_i0e.py`
- `special_i1` | `pointwise` | `operators/pointwise/special_i1.py`

## normalization_state_backward

- count: 6
- needed support: Needs dedicated normalization templates for running stats, backward, multi-output reductions, or affine parameters.

- `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | `fused/general` | `operators/fused/general/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py`
- `instance_norm` | `fused/general` | `operators/fused/general/instance_norm.py`
- `weight_norm` | `fused/general` | `operators/fused/general/weight_norm.py`
- `batch_norm` | `normalization` | `operators/normalization/batch_norm.py`
- `groupnorm` | `normalization` | `operators/normalization/groupnorm.py`
- `weightnorm` | `normalization` | `operators/normalization/weightnorm.py`

## manual_reduction_loss

- count: 5
- needed support: Needs operator-specific reduction/loss templates such as target-index loads, ignore_index, product, median, or multi-output reductions.

- `diff` | `reduction` | `operators/reduction/diff.py`
- `hadamard_transform` | `reduction` | `operators/reduction/hadamard_transform.py`
- `nll_loss_nd` | `reduction` | `operators/reduction/nll_loss_nd.py`
- `nllloss` | `reduction` | `operators/reduction/nllloss.py`
- `prod` | `reduction` | `operators/reduction/prod.py`

## linear_broadcast_quant

- count: 2
- needed support: Needs outer/broadcast 2D layout, scaled/quantized matmul, or more flexible matrix lowering.

- `addr` | `linear` | `operators/linear/addr.py`
- `scaled_mm` | `linear` | `operators/linear/scaled_mm.py`

## special_math_or_pointwise_api

- count: 2
- needed support: Needs special math approximations/intrinsics or dynamic pointwise API support.

- `isin` | `pointwise` | `operators/pointwise/isin.py`
- `polar` | `pointwise` | `operators/pointwise/polar.py`
