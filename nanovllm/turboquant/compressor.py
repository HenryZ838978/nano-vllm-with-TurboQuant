"""
TurboQuant engine for nano-vllm KV cache compression.

Implements the two-stage quantizer from TurboQuant (ICLR 2026):
  Stage 1: Random rotation + Lloyd-Max per-coordinate quantization (MSE-optimal)
  Stage 2: QJL 1-bit sign correction on residuals (unbiased inner products)

The asymmetric estimator computes attention scores directly from compressed K
without full decompression:
  <q, k> ≈ <q, k_mse> + ||r_k|| * sqrt(π/2) / m * <S@q, sign(S@r_k)>

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
"""

import torch
import math
from .lloyd_max import solve_lloyd_max


class TurboQuantEngine:
    """
    Stateless TQ compressor/estimator. One instance per (head_dim, bits) pair.
    Thread-safe: all matrices are read-only after init.
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cuda"):
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen, device="cpu", dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)
        self.PiT = self.Pi.T.contiguous()

        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(seed + 10000)
        self.S = torch.randn(head_dim, head_dim, generator=gen2, device="cpu", dtype=torch.float32).to(device)

        self.centroids = solve_lloyd_max(head_dim, self.mse_bits).to(device)

    @torch.no_grad()
    def compress_keys(self, keys: torch.Tensor) -> dict:
        """
        Compress key vectors for asymmetric attention.

        Args:
            keys: (N, head_dim) — flat key vectors, already post-RoPE

        Returns:
            dict with k_mse (fp16), qjl_signs (int8), residual_norm (fp16)
        """
        flat = keys.float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normed = flat / (norms + 1e-8)

        rotated = normed @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        reconstructed = self.centroids[indices] @ self.Pi
        k_mse = reconstructed * norms

        residual = flat - k_mse
        r_norm = torch.norm(residual, dim=-1)
        projected = residual @ self.S.T
        signs = (projected >= 0).to(torch.int8) * 2 - 1

        return {
            "k_mse": k_mse.half(),
            "qjl_signs": signs,
            "residual_norm": r_norm.half(),
        }

    @torch.no_grad()
    def compress_values(self, values: torch.Tensor) -> dict:
        """MSE-only compression for values (no QJL needed)."""
        flat = values.float()
        norms = torch.norm(flat, dim=-1, keepdim=True)
        normed = flat / (norms + 1e-8)
        rotated = normed @ self.Pi.T
        diffs = rotated.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1)
        reconstructed = self.centroids[indices] @ self.Pi
        return {"v_mse": (reconstructed * norms).half()}

    @torch.no_grad()
    def asymmetric_attention(
        self,
        queries: torch.Tensor,
        compressed_k: dict,
        values: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        Compute full attention output using TQ asymmetric estimator on keys
        and standard weighted sum on values.

        Args:
            queries:     (batch, num_q_heads, 1, head_dim) — decode queries
            compressed_k: dict from compress_keys, reshaped to (batch, num_kv_heads, seq_len, head_dim)
            values:      (batch, num_kv_heads, seq_len, head_dim) — FP16 values from cache
            scale:       attention softmax scale (1/sqrt(head_dim))

        Returns:
            output: (batch, num_q_heads, 1, head_dim)
        """
        k_mse = compressed_k["k_mse"].float()
        signs = compressed_k["qjl_signs"].float()
        r_norm = compressed_k["residual_norm"].float()
        q = queries.float()

        B, Hq, _, D = q.shape
        Hkv = k_mse.shape[1]
        num_groups = Hq // Hkv

        if num_groups > 1:
            k_mse = k_mse.repeat_interleave(num_groups, dim=1)
            signs = signs.repeat_interleave(num_groups, dim=1)
            r_norm = r_norm.repeat_interleave(num_groups, dim=1)
            values = values.repeat_interleave(num_groups, dim=1)

        # Term 1: Q @ K_mse^T
        term1 = torch.matmul(q, k_mse.transpose(-2, -1))

        # Term 2: QJL correction
        q_proj = torch.matmul(q, self.S.T)
        qjl_ip = torch.matmul(q_proj, signs.transpose(-2, -1))
        m = self.head_dim
        correction = math.sqrt(math.pi / 2) / m
        term2 = correction * qjl_ip * r_norm.unsqueeze(-2)

        scores = (term1 + term2) * scale

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, values.float())
        return output.to(queries.dtype)
