"""Modules reused from BrainOmni: https://github.com/OpenTSLab/BrainOmni"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args


class RMSNorm(torch.nn.Module):
    def __init__(self, n_dim, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim)) if elementwise_affine else 1.0
        self.eps = eps

    def forward(self, x: torch.Tensor):
        weight = self.weight
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, n_dim, init_seq_len, base=10000):
        super().__init__()
        self.register_buffer(
            "freqs",
            1.0 / (base ** (torch.arange(0, n_dim, 2)[: (n_dim // 2)].float() / n_dim)),
        )
        self._set_rotate_cache(init_seq_len)

    def _set_rotate_cache(self, seq_len):
        self.max_seq_len_cache = seq_len
        t = torch.arange(seq_len, device=self.freqs.device).type_as(self.freqs)
        rotate = torch.outer(t, self.freqs).float()
        self.register_buffer("rotate", torch.polar(torch.ones_like(rotate), rotate))

    def reshape_for_broadcast(self, x: torch.Tensor):
        """
        x      Batch seq n_head d_head
        rotate seq dim
        """
        B, T, H, D = x.shape
        if T > self.max_seq_len_cache:
            self._set_rotate_cache(T)
        rotate = self.rotate[:T, :]
        assert H * D == rotate.shape[1]
        return rearrange(rotate, "T (H D)-> T H D", H=H).unsqueeze(0)

    def forward(self, q, k):
        assert len(q.shape) == len(k.shape) == 4
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        rotate = self.reshape_for_broadcast(q_)
        q_out = torch.view_as_real(q_ * rotate).flatten(3)
        k_out = torch.view_as_real(k_ * rotate).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


class SpatialTemporalAttentionBlock(nn.Module):
    def __init__(self, n_dim, n_head, dropout, causal):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.time_attn = SelfAttention(
            n_dim // 2, n_head // 2, dropout, causal=causal, rope=True
        )
        self.spatial_attn = SelfAttention(
            n_dim // 2, n_head // 2, dropout, causal=False, rope=False
        )
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)

    def forward(self, x, mask=None):
        """
        True element in mask will take part in attention
        """
        x = x + self._attn_operator(self.pre_attn_norm(x))
        x = x + self.ff(self.pre_ff_norm(x))
        return x

    def _attn_operator(self, x):
        B, C, W, D = x.shape
        xs = rearrange(x[:, :, :, D // 2 :], "B C W D -> (B W) C D")
        xt = rearrange(x[:, :, :, : D // 2], "B C W D->(B C) W D")
        xs = self.spatial_attn(xs, None)
        xt = self.time_attn(xt, None)
        xs = rearrange(xs, "(B W) C D -> B C W D", B=B)
        xt = rearrange(xt, "(B C) W D->B C W D", B=B)
        return torch.cat([xs, xt], dim=-1)


class SpatialTemporalEncoder(nn.Module):
    """
    Stack of SpatialTemporalAttentionBlocks with interface similar to x_transformers.Encoder.

    Args:
        dim: Feature dimension (will be split D//2 for spatial and temporal attention)
        depth: Number of SpatialTemporalAttentionBlock layers to stack
        heads: Number of attention heads (will be split heads//2 for each attention type)
        dropout: Dropout rate for attention and feedforward layers
        causal: Whether temporal attention should be causal

    Input:
        x: [B, C, W, D] tensor (Batch, Channels/Spatial, Time, Features)
        mask: Optional mask tensor (True elements take part in attention)

    Output:
        [B, C, W, D] tensor (same shape as input)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [
                SpatialTemporalAttentionBlock(
                    n_dim=dim,
                    n_head=heads,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(depth)
            ]
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for this encoder."""
        self.gradient_checkpointing = True

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, C, W, D] tensor
            mask: Optional mask tensor

        Returns:
            [B, C, W, D] tensor
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask=mask)
        return x


# Attention
class SelfAttnBlock(nn.Module):
    def __init__(self, n_dim, n_head, dropout, causal, rope):
        super().__init__()
        self.pre_attn_norm = RMSNorm(n_dim)
        self.attn = SelfAttention(n_dim, n_head, dropout, causal=causal, rope=rope)
        self.pre_ff_norm = RMSNorm(n_dim)
        self.ff = FeedForward(n_dim, dropout)

    def forward(self, x, mask=None):
        """
        True element in mask will take part in attention
        """
        x = x + self.attn(self.pre_attn_norm(x), mask)
        x = x + self.ff(self.pre_ff_norm(x))
        return x


class SelfAttention(nn.Module):
    def __init__(
        self, n_dim, n_head, dropout, causal: bool = False, rope: bool = False
    ):
        super().__init__()
        assert n_dim % n_head == 0
        self.dropout = dropout
        self.n_dim = n_dim
        self.n_head = n_head
        self.causal = causal
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.proj = nn.Linear(n_dim, n_dim)
        self.rope = rope
        self.rope_embedding_layer = (
            RotaryEmbedding(n_dim=n_dim, init_seq_len=240) if self.rope else Identity()
        )

    def forward(self, x: torch.Tensor, mask=None):
        """
        True element in mask will take part in attention
        """
        B, T, C = x.shape
        x = self.qkv(x)
        q, k, v = torch.split(x, split_size_or_sections=self.n_dim, dim=-1)

        # 有无rope对形状变换有影响，需要判断
        if self.rope:
            q = q.view(B, T, self.n_head, -1)
            k = k.view(B, T, self.n_head, -1)
            q, k = self.rope_embedding_layer(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        else:
            q = rearrange(q, "B T (H D) -> B H T D", H=self.n_head)
            k = rearrange(k, "B T (H D) -> B H T D", H=self.n_head)

        v = rearrange(v, "B T (H D) -> B H T D", H=self.n_head)

        # add head_dim
        if mask != None:
            mask = mask.unsqueeze(1)

        output = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                dropout_p=self.dropout,
                is_causal=self.causal,
            )
            .transpose(1, 2)
            .contiguous()
        )
        output = output.view(B, T, -1)
        return self.proj(output)


class FeedForward(nn.Module):
    def __init__(self, n_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_dim, int(4 * n_dim)),
            nn.SELU(),
            nn.Linear(int(4 * n_dim), n_dim),
            nn.Dropout(dropout) if dropout != 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.layer(x)


if __name__ == "__main__":
    print("Testing SpatialTemporalEncoder...")

    # Test parameters
    batch_size = 2
    channels = 64      # Spatial dimension
    time_steps = 100   # Temporal dimension
    feature_dim = 128  # Must be even (splits D//2 for spatial and temporal)

    # Model parameters
    depth = 4
    heads = 8
    dropout = 0.1

    # Create encoder
    encoder = SpatialTemporalEncoder(
        dim=feature_dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
        causal=False
    )

    # Create random input [B, C, W, D]
    x = torch.randn(batch_size, channels, time_steps, feature_dim)
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = encoder(x)
    print(f"Output shape: {output.shape}")

    # Test with mask
    mask = torch.ones(batch_size, channels, time_steps, dtype=torch.bool)
    output_masked = encoder(x, mask=mask)
    print(f"Output with mask shape: {output_masked.shape}")

    # Verify shape preservation
    assert output.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {output.shape}"
    print("✓ Shape preservation test passed")

    # Test with different sequence length (>240 to test RoPE expansion)
    x_long = torch.randn(batch_size, channels, 300, feature_dim)
    output_long = encoder(x_long)
    assert output_long.shape == x_long.shape
    print(f"✓ Long sequence test passed (seq_len=300, RoPE auto-expanded)")

    # Test causal encoder
    encoder_causal = SpatialTemporalEncoder(
        dim=feature_dim,
        depth=depth,
        heads=heads,
        dropout=dropout,
        causal=True
    )
    output_causal = encoder_causal(x)
    assert output_causal.shape == x.shape
    print("✓ Causal encoder test passed")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nModel statistics:")
    print(f"  Depth: {depth} layers")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n✓ All tests passed!")
