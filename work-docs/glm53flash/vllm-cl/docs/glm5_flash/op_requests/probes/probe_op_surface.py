import torch
dev='dlc'
# --- existence probes (operator surface) ---
names = ['dlc_vllm_zero_block_ids_','dlc_mhc_pre','dlc_mhc_post','_causal_conv1d','_causal_conv1d_update',
         'chunk_gated_delta_rule','fused_recurrent_gated_delta_rule','fused_gdn_gating',
         'dlc_vllm_recurrent_gated_delta_rule_packed_decode','dlc_multi_query_attention',
         'dlc_sparse_attention_kv_cache','dlc_sparse_attention','dlc_vllm_paged_mqa',
         'dlc_fused_rms_norm_f32_bf16','dlc_fused_rms_norm_qkv_rotary_emb','dlc_fused_post_conv_prep_bf16',
         'dlc_vllm_concat_and_cache_mla_','dlc_indexer_k_quant_and_cache','dlc_indexer_k_cache_update_',
         'dlc_vllm_cp_gather_indexer_','dlc_vllm_expand_kernel_','dlc_vllm_topk_per_row_decode_',
         'silu_and_mul','silu_and_mul_with_clamp','dlc_silu_and_mul','dlc_kpool_compress','dlc_mhc_fused_post_pre',
         'dlc_kda_chunk_prefill','dlc_gather_initial_states','dlc_scatter_states']
missing, present = [], []
for n in names:
    (present if hasattr(torch, n) else missing).append(n)
print("PRESENT:", present)
print("MISSING:", missing)

# --- generic index_select / index_copy_ on a mamba-shaped state (covers #05 semantics) ---
P,H,V,K = 8,4,16,16
state = torch.randn(P,H,V,K, dtype=torch.float32, device=dev)
idx = torch.tensor([3,1,3], dtype=torch.int64, device=dev)
mask = torch.tensor([True,False,True], device=dev)
g = state.index_select(0, idx)
g = torch.where(mask.view(-1,1,1,1), g, torch.zeros_like(g))
src = torch.randn(2,H,V,K, dtype=torch.float32, device=dev)
state.index_copy_(0, torch.tensor([2,5], dtype=torch.int64, device=dev), src)
torch.dlc.synchronize()
ok_g = torch.equal(g[2].cpu(), torch.zeros(H,V,K))
ok_s = torch.equal(state[5].cpu(), src[1].cpu())
print("index_select/index_copy_ on mamba state:", "gather-mask-ok" if ok_g else "gather-FAIL", "scatter-ok" if ok_s else "scatter-FAIL")
