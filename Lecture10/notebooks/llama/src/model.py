"""
model.py — LLaMA-style Transformer Decoder (single layer, educational version).

Key components:
  - RMSNorm          : simpler normalization (no mean centering)
  - RoPE             : Rotary Position Embedding on Q and K
  - Grouped Query Attention (GQA) : fewer K/V heads than Q heads
  - SwiGLU           : gated activation in the FFN
  - Causal mask      : autoregressive (each token only sees past tokens)

All shapes are annotated: B=batch, T=seq_len, D=d_model, H=n_heads, h=head_dim
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int   = 512    # will be overwritten after tokenizer training
    d_model: int      = 128    # embedding / hidden dimension
    n_heads: int      = 4      # number of query heads
    n_kv_heads: int   = 2      # number of key/value heads  (GQA: n_kv_heads < n_heads)
    d_ff: int         = 256    # feed-forward inner dimension (before gate)
    max_seq_len: int  = 128    # maximum context length
    dropout: float    = 0.0    # dropout probability


# ---------------------------------------------------------------------------
# 1. RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Simpler than LayerNorm: no mean subtraction, just scale by RMS.

    y = x / RMS(x) * gamma
    where RMS(x) = sqrt( mean(x^2) + eps )
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma


# ---------------------------------------------------------------------------
# 2. RoPE — Rotary Position Embedding
# ---------------------------------------------------------------------------
def precompute_rope_freqs(head_dim: int, max_seq_len: int,
                          base: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex rotation frequencies for each position.
    Returns: (max_seq_len, head_dim//2) complex tensor
    """
    # theta_i = 1 / base^(2i / head_dim)   for i in [0, head_dim/2)
    i = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta = 1.0 / (base ** (i / head_dim))               # (head_dim//2,)
    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (T,)
    # outer product: each position gets its angle per dimension
    freqs = torch.outer(positions, theta)                 # (T, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)     # complex: (T, head_dim//2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.
    x    : (B, T, H, head_dim)   — real
    freqs: (T, head_dim//2)      — complex
    """
    B, T, H, D = x.shape
    # view as complex: pair up (cos,sin) dimensions
    x_complex = torch.view_as_complex(x.float().reshape(B, T, H, D // 2, 2))
    # broadcast freqs over batch and heads: (1, T, 1, D//2)
    freqs = freqs[:T].unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs
    # back to real
    return torch.view_as_real(x_rotated).reshape(B, T, H, D).to(x.dtype)


# ---------------------------------------------------------------------------
# 3. Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------
class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention where K and V have fewer heads than Q.

    n_heads    : Q heads  (e.g. 4)
    n_kv_heads : K/V heads (e.g. 2)
    Each K/V head is shared by (n_heads // n_kv_heads) Q heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_heads % config.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads  # repetitions per KV head
        self.head_dim   = config.d_model // config.n_heads

        D  = config.d_model
        Dh = self.head_dim

        self.Wq = nn.Linear(D, config.n_heads * Dh,    bias=False)
        self.Wk = nn.Linear(D, config.n_kv_heads * Dh, bias=False)
        self.Wv = nn.Linear(D, config.n_kv_heads * Dh, bias=False)
        self.Wo = nn.Linear(config.n_heads * Dh, D,    bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor,
                rope_freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        Dh = self.head_dim

        # Project to Q, K, V
        q = self.Wq(x).reshape(B, T, self.n_heads,    Dh)   # (B,T,Hq,Dh)
        k = self.Wk(x).reshape(B, T, self.n_kv_heads, Dh)   # (B,T,Hkv,Dh)
        v = self.Wv(x).reshape(B, T, self.n_kv_heads, Dh)   # (B,T,Hkv,Dh)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Expand K, V to match Q heads: repeat each KV head n_rep times
        # (B, T, Hkv, Dh) → (B, T, Hq, Dh)
        k = k.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, Dh) \
             .reshape(B, T, self.n_heads, Dh)
        v = v.unsqueeze(3).expand(B, T, self.n_kv_heads, self.n_rep, Dh) \
             .reshape(B, T, self.n_heads, Dh)

        # Transpose for attention: (B, H, T, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale  = math.sqrt(Dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # (B,H,T,T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out    = torch.matmul(attn, v)                           # (B,H,T,Dh)

        # Merge heads and project
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * Dh)
        return self.Wo(out)                                      # (B,T,D)


# ---------------------------------------------------------------------------
# 4. SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
    where Swish(x) = x * sigmoid(x)   (a.k.a. SiLU)

    Two parallel linear projections → element-wise gate → project down.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        D, Dff = config.d_model, config.d_ff
        self.gate_proj = nn.Linear(D, Dff, bias=False)   # the "gating" branch
        self.up_proj   = nn.Linear(D, Dff, bias=False)   # the "value" branch
        self.down_proj = nn.Linear(Dff, D, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))   # (B,T,Dff)
        up   = self.up_proj(x)             # (B,T,Dff)
        return self.down_proj(self.dropout(gate * up))   # (B,T,D)


# ---------------------------------------------------------------------------
# 5. Single Transformer Decoder Layer
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    One LLaMA-style decoder block:
      x → RMSNorm → GQA → residual
        → RMSNorm → SwiGLU FFN → residual
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor,
                rope_freqs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + residual  (LLaMA style: norm BEFORE attention)
        x = x + self.attn(self.norm1(x), rope_freqs, mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 6. Full LLaMA-style Model
# ---------------------------------------------------------------------------
class MiniLLaMA(nn.Module):
    """
    A single-layer LLaMA-style language model for educational purposes.

    Forward pass:
      token_ids → embedding → decoder layer → RMSNorm → logits
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed    = nn.Embedding(config.vocab_size, config.d_model)
        self.layer    = DecoderLayer(config)
        self.norm_out = RMSNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embedding and lm_head weights (common in LLaMA)
        self.lm_head.weight = self.embed.weight

        # Precompute RoPE frequencies (not a parameter, just a buffer)
        rope_freqs = precompute_rope_freqs(
            config.d_model // config.n_heads, config.max_seq_len
        )
        self.register_buffer("rope_freqs", rope_freqs)

        # Causal mask: lower-triangular matrix of 1s
        causal = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", causal)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        """
        input_ids : (B, T)  — token indices
        targets   : (B, T)  — shifted token indices for next-token prediction

        Returns
        -------
        logits : (B, T, vocab_size)
        loss   : scalar cross-entropy loss (only when targets provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Token embeddings: (B, T, D)
        x = self.embed(input_ids)

        # Causal mask for this sequence length
        mask = self.causal_mask[:T, :T]   # (T, T)

        # Single decoder layer
        x = self.layer(x, self.rope_freqs, mask)

        # Output normalization + projection to vocabulary
        x      = self.norm_out(x)          # (B, T, D)
        logits = self.lm_head(x)           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=512, d_model=128, n_heads=4, n_kv_heads=2,
                      d_ff=256, max_seq_len=64)
    model = MiniLLaMA(cfg)
    print(f"MiniLLaMA | parameters: {model.num_parameters():,}")

    # dummy forward pass
    B, T = 2, 32
    ids     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(ids, targets)
    print(f"logits shape : {logits.shape}")   # (2, 32, 512)
    print(f"loss         : {loss.item():.4f}")
    print("Sanity check passed ✓")
