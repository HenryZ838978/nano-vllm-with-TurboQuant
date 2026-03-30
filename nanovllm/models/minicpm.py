"""
MiniCPM4.1-8B model implementation for nano-vllm.

Key differences from Qwen3:
  - Separate q/k/v and gate/up projections (not fused)
  - scale_emb on input embeddings
  - scale_depth / sqrt(num_layers) residual scaling
  - μP logit scaling: hidden / (hidden_size / dim_model_base)
  - LongRoPE positional encoding
"""

import math
import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.rotary_embedding import apply_rotary_emb


class LongRoPEEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        max_position_embeddings: int,
        base: float,
        short_factor: list[float],
        long_factor: list[float],
        original_max_position_embeddings: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.original_max_position_embeddings = original_max_position_embeddings

        scale = max_position_embeddings / original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
        short_ext = torch.tensor(short_factor, dtype=torch.float)
        long_ext = torch.tensor(long_factor, dtype=torch.float)

        t = torch.arange(max_position_embeddings, dtype=torch.float)

        freqs_short = torch.outer(t, inv_freq / short_ext)
        cos_short = freqs_short.cos() * self.scaling_factor
        sin_short = freqs_short.sin() * self.scaling_factor
        cache_short = torch.cat((cos_short, sin_short), dim=-1).unsqueeze_(1)

        freqs_long = torch.outer(t, inv_freq / long_ext)
        cos_long = freqs_long.cos() * self.scaling_factor
        sin_long = freqs_long.sin() * self.scaling_factor
        cache_long = torch.cat((cos_long, sin_long), dim=-1).unsqueeze_(1)

        self.register_buffer("cos_sin_cache_short", cache_short, persistent=False)
        self.register_buffer("cos_sin_cache_long", cache_long, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = positions.max().item() + 1
        if seq_len > self.original_max_position_embeddings:
            cos_sin = self.cos_sin_cache_long[positions]
        else:
            cos_sin = self.cos_sin_cache_short[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


class MiniCPMAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_emb: nn.Module,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ColumnParallelLinear(hidden_size, self.total_num_heads * self.head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = rotary_emb
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class MiniCPMMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)

    @torch.compile
    def _gate_up_fwd(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._gate_up_fwd(x))


class MiniCPMDecoderLayer(nn.Module):

    def __init__(self, config, rotary_emb: nn.Module) -> None:
        super().__init__()
        self.self_attn = MiniCPMAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_emb=rotary_emb,
        )
        self.mlp = MiniCPMMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale_factor = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states * self.scale_factor

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.scale_factor
        return hidden_states


class MiniCPMModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.scale_emb = config.scale_emb
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        head_dim = config.hidden_size // config.num_attention_heads
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and rope_scaling.get("rope_type") == "longrope":
            rotary_emb = LongRoPEEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                short_factor=rope_scaling["short_factor"],
                long_factor=rope_scaling["long_factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            )
        else:
            from nanovllm.layers.rotary_embedding import get_rope
            rotary_emb = get_rope(head_dim, head_dim, config.max_position_embeddings, config.rope_theta)

        self.layers = nn.ModuleList([MiniCPMDecoderLayer(config, rotary_emb) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) * self.scale_emb
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MiniCPMForCausalLM(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = MiniCPMModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logit_scale = config.hidden_size / getattr(config, "dim_model_base", config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states / self.logit_scale)
