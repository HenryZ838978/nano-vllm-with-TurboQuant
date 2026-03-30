import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def gather_from_paged_cache(cache: torch.Tensor, block_table: torch.Tensor,
                            context_lens: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Gather contiguous KV sequences from paged block cache.

    Args:
        cache:        (num_blocks, block_size, num_kv_heads, head_dim)
        block_table:  (batch, max_num_blocks) — physical block ids per sequence
        context_lens: (batch,) — actual token count per sequence
        block_size:   tokens per block

    Returns:
        (batch, max_seq_len, num_kv_heads, head_dim) padded tensor
    """
    B = block_table.shape[0]
    Hkv, D = cache.shape[2], cache.shape[3]
    max_seq = context_lens.max().item()

    out = torch.zeros(B, max_seq, Hkv, D, dtype=cache.dtype, device=cache.device)
    for b in range(B):
        L = context_lens[b].item()
        nblocks = (L + block_size - 1) // block_size
        block_ids = block_table[b, :nblocks]
        gathered = cache[block_ids].reshape(-1, Hkv, D)[:L]
        out[b, :L] = gathered
    return out


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.tq_engine = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if self.tq_engine is not None:
                o = self._tq_decode(q, context)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o

    def _tq_decode(self, q: torch.Tensor, context) -> torch.Tensor:
        """
        Decode attention using TurboQuant asymmetric estimator.
        Gathers K/V from paged cache, compresses K, computes attention.
        """
        block_size = self.k_cache.shape[1]
        keys = gather_from_paged_cache(
            self.k_cache, context.block_tables, context.context_lens, block_size)
        vals = gather_from_paged_cache(
            self.v_cache, context.block_tables, context.context_lens, block_size)

        B, S, Hkv, D = keys.shape
        keys_flat = keys.reshape(B * S * Hkv, D)
        compressed_k = self.tq_engine.compress_keys(keys_flat)

        compressed_k = {
            "k_mse": compressed_k["k_mse"].reshape(B, S, Hkv, D).transpose(1, 2),
            "qjl_signs": compressed_k["qjl_signs"].reshape(B, S, Hkv, D).transpose(1, 2),
            "residual_norm": compressed_k["residual_norm"].reshape(B, S, Hkv).transpose(1, 2),
        }

        q_4d = q.unsqueeze(1).reshape(B, self.num_heads, 1, D)
        vals_4d = vals.transpose(1, 2)  # (B, Hkv, S, D)

        output = self.tq_engine.asymmetric_attention(q_4d, compressed_k, vals_4d, self.scale)
        return output.reshape(B, 1, self.num_heads, D)
