"""Multi-Head Latent Attention and rotary-embedding utilities."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig

class RMSNorm(nn.Module):
    """Root-mean-square normalization."""
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        norm = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * norm).to(dtype=x.dtype) * self.weight

def precompute_rope(
    dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine/sine rotary tables."""
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even.")
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,) -> torch.Tensor:
    """Apply RoPE to x with position-aligned cosine and sine tables.
    x is expected to have shape (B, H, T, D) and cos/sin shape (T, D).
    """
    return x * cos[None, None, :, :] + rotate_half(x) * sin[None, None, :, :]

PastKV = Tuple[torch.Tensor, torch.Tensor]

class MultiHeadLatentAttention(nn.Module):
    """Educational Multi-Head Latent Attention implementation.
    Data flow per token/layer:
        c_KV = W_DKV h
        k_R = RoPE(W_KR h)
        c_Q = RMSNorm(W_DQ h)
        q = W_UQ c_Q
        k_nope^h = W_UK_h c_KV
        v^h = W_UV_h c_KV
    The attention score combines the content and RoPE components.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size,
                config.num_heads * self.q_head_dim,
                bias=False,)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                config.num_heads * self.q_head_dim,
                bias=False,)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,)
        self.o_proj = nn.Linear(
            config.num_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,)
        self.dropout_p = config.attention_dropout
        self.causal = config.causal_attention

        cos, sin = precompute_rope(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope_tables(
        self,
        start: int,
        length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,) -> tuple[torch.Tensor, torch.Tensor]:
        end = start + length
        if end > self.rope_cos.size(0):
            raise ValueError(
                f"Sequence position {end} exceeds max_position_embeddings="
                f"{self.rope_cos.size(0)}.")
        return (
            self.rope_cos[start:end].to(device=device, dtype=dtype),
            self.rope_sin[start:end].to(device=device, dtype=dtype),)

    def _project_queries(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(batch, tokens, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,)

        q_pe = apply_rope(
            q_pe.transpose(1, 2),
            cos,
            sin,).transpose(1, 2)
        return q_nope, q_pe

    def _compress_kv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        ckv, k_pe = self.kv_a_proj_with_mqa(x).split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,)
        ckv = self.kv_a_layernorm(ckv)
        k_pe = apply_rope(
            k_pe.unsqueeze(1),
            cos,
            sin,).squeeze(1)
        return ckv, k_pe

    def _apply_mask(
        self,
        scores: torch.Tensor,
        *,
        query_length: int,
        key_length: int,
        past_length: int,
        attention_mask: Optional[torch.Tensor],) -> torch.Tensor:
        if self.causal:
            causal = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=scores.device,).tril(diagonal=past_length)
            scores = scores.masked_fill(~causal, float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask.to(dtype=scores.dtype, device=scores.device)
        return scores

    def _dropout(self, probs: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_p > 0.0:
            return F.dropout(probs, p=self.dropout_p)
        return probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[PastKV] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,) -> tuple[torch.Tensor, Optional[PastKV]]:
        """Standard MLA path.
        Args:
            hidden_states: Tensor of shape (B, T, hidden_size).
            past_kv: Optional tuple `(c_kv, k_pe)` from prior tokens.
            use_cache: Return the updated latent cache when True.
            attention_mask: Optional additive mask broadcastable to (B, H, T, T_k).
        """
        batch, query_length, _ = hidden_states.shape
        past_length = 0 if past_kv is None else past_kv[0].shape[1]

        cos_q, sin_q = self._rope_tables(
            past_length,
            query_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_nope, q_pe = self._project_queries(hidden_states, cos_q, sin_q)
        ckv_new, k_pe_new = self._compress_kv(hidden_states, cos_q, sin_q)

        if past_kv is None:
            ckv = ckv_new
            k_pe = k_pe_new
        else:
            ckv = torch.cat([past_kv[0], ckv_new], dim=1)
            k_pe = torch.cat([past_kv[1], k_pe_new], dim=1)

        key_length = ckv.shape[1]

        kv = self.kv_b_proj(ckv).view(
            batch,
            key_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,)

        k = torch.cat(
            [
                k_nope,
                k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1),
            ],
            dim=-1,
        )

        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) * self.scale
        scores = self._apply_mask(
            scores,
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            attention_mask=attention_mask,
        )

        probs = F.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        probs = self._dropout(probs)

        out = (probs @ v).transpose(1, 2).reshape(
            batch,
            query_length,
            self.num_heads * self.v_head_dim,
        )
        return self.o_proj(out), (ckv, k_pe) if use_cache else None

    def forward_absorbed(
        self,
        hidden_states: torch.Tensor,
        past_kv: Optional[PastKV] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[PastKV]]:
        """Weight-absorbed MLA path.

        This path avoids materializing full per-head K/V tensors. The content
        part of K is absorbed into Q through W_UK, attention mixes directly in
        the compressed KV latent, and W_UV is applied after the mixture.

        It is mathematically equivalent to :meth:`forward` up to floating-point
        round-off.
        """
        batch, query_length, _ = hidden_states.shape
        past_length = 0 if past_kv is None else past_kv[0].shape[1]

        cos_q, sin_q = self._rope_tables(
            past_length,
            query_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_nope, q_pe = self._project_queries(hidden_states, cos_q, sin_q)
        ckv_new, k_pe_new = self._compress_kv(hidden_states, cos_q, sin_q)

        if past_kv is None:
            ckv = ckv_new
            k_pe = k_pe_new
        else:
            ckv = torch.cat([past_kv[0], ckv_new], dim=1)
            k_pe = torch.cat([past_kv[1], k_pe_new], dim=1)

        key_length = ckv.shape[1]

        # W_UK and W_UV are stored together as the output rows of kv_b_proj.
        # Shape: (H, D_nope + D_v, R_kv)
        w = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk = w[:, : self.qk_nope_head_dim, :]
        w_uv = w[:, self.qk_nope_head_dim :, :]

        # q_nope · (W_UK c_kv) = (W_UK^T q_nope) · c_kv
        q_latent = torch.einsum(
            "bthd,hdr->bthr",
            q_nope,
            w_uk,
        )

        content_scores = torch.einsum(
            "bthr,bsr->bhts",
            q_latent,
            ckv,
        )
        rope_scores = torch.einsum(
            "bthr,bsr->bhts",
            q_pe,
            k_pe,
        )
        scores = (content_scores + rope_scores) * self.scale
        scores = self._apply_mask(
            scores,
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            attention_mask=attention_mask,
        )

        probs = F.softmax(scores.float(), dim=-1).to(dtype=ckv.dtype)
        probs = self._dropout(probs)

        latent_mix = torch.einsum(
            "bhts,bsr->bthr",
            probs,
            ckv,)
        out = torch.einsum(
            "bthr,hvr->bthv",
            latent_mix,
            w_uv,)
        out = out.reshape(
            batch,
            query_length,
            self.num_heads * self.v_head_dim,)

        return self.o_proj(out), (ckv, k_pe) if use_cache else None
