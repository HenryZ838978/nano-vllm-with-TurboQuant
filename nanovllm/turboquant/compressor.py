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
        GQA-aware asymmetric attention — avoids repeat_interleave expansion.

        Args:
            queries:     (B, Hq, 1, D) — decode queries
            compressed_k: dict with shapes (B, Hkv, S, D) / (B, Hkv, S)
            values:      (B, Hkv, S, D)
            scale:       1/sqrt(D)

        Returns:
            (B, Hq, 1, D)
        """
        k_mse = compressed_k["k_mse"]
        signs = compressed_k["qjl_signs"]
        r_norm = compressed_k["residual_norm"]
        q = queries

        B, Hq, _, D = q.shape
        Hkv = k_mse.shape[1]
        G = Hq // Hkv
        S = k_mse.shape[2]

        # Reshape Q to group dim: (B, Hkv, G, 1, D)
        q = q.reshape(B, Hkv, G, 1, D)

        # K/V stay at (B, Hkv, S, D) → add dim for broadcast: (B, Hkv, 1, S, D)
        k_mse_5 = k_mse.unsqueeze(2)          # (B, Hkv, 1, S, D)
        signs_5 = signs.unsqueeze(2).float()   # (B, Hkv, 1, S, D)
        vals_5 = values.unsqueeze(2)           # (B, Hkv, 1, S, D)

        # Term 1: Q @ K_mse^T  → (B, Hkv, G, 1, S)
        term1 = torch.matmul(q.float(), k_mse_5.float().transpose(-2, -1))

        # Term 2: QJL correction
        q_proj = torch.matmul(q.float(), self.S.T)   # (B, Hkv, G, 1, D)
        qjl_ip = torch.matmul(q_proj, signs_5.transpose(-2, -1))  # (B, Hkv, G, 1, S)
        correction = math.sqrt(math.pi / 2) / D
        r_5 = r_norm.float().unsqueeze(2).unsqueeze(-2)  # (B, Hkv, 1, 1, S)
        term2 = correction * qjl_ip * r_5

        scores = (term1 + term2) * scale
        attn_w = torch.softmax(scores, dim=-1)         # (B, Hkv, G, 1, S)

        out = torch.matmul(attn_w, vals_5.float())     # (B, Hkv, G, 1, D)
        out = out.reshape(B, Hq, 1, D)
        return out.to(queries.dtype)
