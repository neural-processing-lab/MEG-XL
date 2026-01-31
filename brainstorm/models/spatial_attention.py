import torch.nn as nn
import math
import torch

import torch
import torch.nn as nn
import numpy as np

class GaussianFourierEmb3D(nn.Module):
    def __init__(self, embed_dim: int = 256, scale: float = 1.0):
        """
        Gaussian Fourier Embeddings for 3D coordinates (e.g., MEG sensors).
        
        Args:
            embed_dim: Total output dimension (must be even).
            scale: Standard deviation of the Gaussian distribution.
                   - Low scale (e.g., 1.0): Focuses on low-freq (global shape).
                   - High scale (e.g., 10.0): Focuses on high-freq (fine grain differences).
                   For MEG sensors normalized to [-1, 1], a scale of ~1.0 to 5.0 is usually good.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even."
        
        # We need embed_dim // 2 frequencies because each generates a Sin and Cos
        num_freqs = embed_dim // 2
        
        # Random Gaussian Matrix: [3, num_freqs]
        # 3 is the input dimension (x, y, z)
        # We sample from Normal(0, scale^2)
        self.B = nn.Parameter(torch.randn(3, num_freqs) * scale, requires_grad=False)

    def forward(self, positions):
        """
        Args:
            positions: [Batch, ..., 3] coordinates. 
                       CRITICAL: Must be normalized (roughly -1 to 1).
        """
        # 1. Project positions: [..., 3] @ [3, F] -> [..., F]
        # 2 * pi ensures the spectrum covers the period correctly
        x_proj = 2 * np.pi * (positions @ self.B)
        
        # 2. Concatenate Sin and Cos: [..., F] -> [..., 2*F] = [..., embed_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierEmb3D(nn.Module):
    """
    Extended from Défossez et al. 2023 to handle 3D positions.
    """
    def __init__(self, dimension: int = 250, margin: float = 0.2):
        super().__init__()
        # For 3D: dimension = 2 * n_freqs^3 (sin and cos for each freq combination)
        n_freqs = round((dimension / 2) ** (1/3))
        actual_dim = 2 * (n_freqs ** 3)
        if actual_dim != dimension:
            raise ValueError(f"dimension must be 2 * n^3 for some integer n. "
                           f"Got {dimension}, closest valid is {actual_dim}")
        self.dimension = dimension
        self.margin = margin
        self.n_freqs = n_freqs

    def forward(self, positions):
        # positions: [..., 3] for (x, y, z)
        *O, D = positions.shape
        assert D == 3, f"Expected 3D positions, got {D}D"
        
        n = self.n_freqs
        device = positions.device
        
        # Create 3D frequency grid
        freqs = torch.arange(n, device=device)
        freqs_z = freqs.view(n, 1, 1)
        freqs_y = freqs.view(1, n, 1)
        freqs_x = freqs.view(1, 1, n)
        
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        
        # Scale frequencies
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        
        # Broadcast positions
        positions = positions[..., None, None, None, :]  # [..., 1, 1, 1, 3]
        
        # Combine: x*px + y*py + z*pz
        loc = (positions[..., 0] * p_x + 
               positions[..., 1] * p_y + 
               positions[..., 2] * p_z).view(*O, -1)
        
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

class Batched3DSpatialAttention(nn.Module):
    """
    Inspired by Défossez et al. 2023's spatial attention. Adapted to handle 3D coordinate spaces and batched inputs with samples from heterogeneous sensor layouts.

    When used, no need to include channel embeddings as sensor position will be naturally encoded via the attention mechanism.
    """

    def __init__(self, chout: int, pos_dim: int = 250, dropout: float = 0):
        super().__init__()
        self.heads = nn.Parameter(torch.randn(chout, pos_dim))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb3D(pos_dim)

    def forward(self, data, sensor_xyz, mask=None):
        # data: [B, channels, timepoints]
        # sensor_xyz: [B, channels, 3]
        # mask: [B, channels] - 1 for real channels, 0 for padding
        B, C, T = data.shape
        
        positions = sensor_xyz.float()  # [B, channels, 3]
        
        # Vectorized normalization per sample
        if mask is not None:
            mask = mask.bool()
            mask_expanded = mask.unsqueeze(-1)  # [B, channels, 1]
            
            # Min/max only over valid channels
            masked_for_min = torch.where(mask_expanded, positions, torch.tensor(float('inf'), device=positions.device))
            masked_for_max = torch.where(mask_expanded, positions, torch.tensor(float('-inf'), device=positions.device))
            
            pos_min = masked_for_min.min(dim=1, keepdim=True)[0]  # [B, 1, 3]
            pos_max = masked_for_max.max(dim=1, keepdim=True)[0]  # [B, 1, 3]
            
            # Normalize and zero out padding
            positions = (positions - pos_min) / (pos_max - pos_min + 1e-8)
            positions = torch.where(mask_expanded, positions, 0.)
        else:
            pos_min = positions.min(dim=1, keepdim=True)[0]
            pos_max = positions.max(dim=1, keepdim=True)[0]
            positions = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        
        # Embed positions
        embedding = self.embedding(positions)  # [B, channels, pos_dim]
        
        # Mask padded channels
        score_offset = torch.zeros(B, C, device=data.device)
        if mask is not None:
            score_offset[~mask.bool()] = float('-inf')
        
        # Spatial dropout during training
        if self.training and self.dropout:
            center = torch.rand(3, device=data.device)
            banned = (positions - center).norm(dim=-1) <= self.dropout
            score_offset[banned] = float('-inf')
        
        # Attention: heads attend to position embeddings
        heads = self.heads[None].expand(B, -1, -1)  # [B, chout, pos_dim]
        scores = torch.einsum("bcp,bop->boc", embedding, heads)  # [B, chout, channels]
        scores = scores + score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        
        # Mix channels based on attention weights
        out = torch.einsum("bct,boc->bot", data, weights)  # [B, chout, timepoints]
        return out

if __name__ == "__main__":
    # Test the Batched3DSpatialAttention
    
    print("Testing Batched3DSpatialAttention with 3D positions\n")
    
    # Create model
    chout = 64
    pos_dim = 250  # Valid for 3D (2 * 5^3)
    model = Batched3DSpatialAttention(chout=chout, pos_dim=pos_dim, dropout=0.1)
    
    # Test 1: Single sample, no padding
    print("Test 1: Single sample, no padding")
    B, C, T = 1, 32, 1000
    data = torch.randn(B, C, T)
    sensor_xyz = torch.randn(B, C, 3)  # Random 3D positions
    
    out = model(data, sensor_xyz, mask=None)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: [1, {chout}, {T}]")
    assert out.shape == (B, chout, T), "Shape mismatch!"
    print("✓ Passed\n")
    
    # Test 2: Batch with different channel counts (padded)
    print("Test 2: Batch with heterogeneous channel counts")
    max_channels = 100
    
    # Sample 1: 64 channels
    # Sample 2: 32 channels (will be padded)
    # Sample 3: 100 channels (no padding needed)
    real_channels = [64, 32, 100]
    B = len(real_channels)
    T = 500
    
    data_list = []
    pos_list = []
    mask_list = []
    
    for n_chan in real_channels:
        # Create real data
        d = torch.randn(n_chan, T)
        p = torch.randn(n_chan, 3)
        
        # Pad to max_channels
        d_padded = torch.nn.functional.pad(d, (0, 0, 0, max_channels - n_chan))
        p_padded = torch.nn.functional.pad(p, (0, 0, 0, max_channels - n_chan))
        
        # Create mask
        m = torch.cat([torch.ones(n_chan), torch.zeros(max_channels - n_chan)])
        
        data_list.append(d_padded)
        pos_list.append(p_padded)
        mask_list.append(m)
    
    data_batch = torch.stack(data_list)
    pos_batch = torch.stack(pos_list)
    mask_batch = torch.stack(mask_list)
    
    print(f"Batch size: {B}")
    print(f"Real channels per sample: {real_channels}")
    print(f"Padded channel dim: {max_channels}")
    print(f"Data shape: {data_batch.shape}")
    print(f"Positions shape: {pos_batch.shape}")
    print(f"Mask shape: {mask_batch.shape}")
    
    out = model(data_batch, pos_batch, mask_batch)
    print(f"Output shape: {out.shape}")
    print(f"Expected: [{B}, {chout}, {T}]")
    assert out.shape == (B, chout, T), "Shape mismatch!"
    
    # Check that padded channels don't contribute (attention weights should be 0)
    print("\n✓ Passed - heterogeneous batching works!")
    
    # Test 3: Check attention weights sum to 1 (excluding padded)
    print("\nTest 3: Checking attention weight properties")
    model.eval()
    with torch.no_grad():
        # Intercept attention weights
        positions = pos_batch.float()
        mask_batch = mask_batch.bool()
        mask_expanded = mask_batch.unsqueeze(-1)
        
        masked_for_min = torch.where(mask_expanded, positions, torch.tensor(float('inf')))
        masked_for_max = torch.where(mask_expanded, positions, torch.tensor(float('-inf')))
        pos_min = masked_for_min.min(dim=1, keepdim=True)[0]
        pos_max = masked_for_max.max(dim=1, keepdim=True)[0]
        positions = (positions - pos_min) / (pos_max - pos_min + 1e-8)
        positions = torch.where(mask_expanded, positions, 0.)
        
        embedding = model.embedding(positions)
        score_offset = torch.zeros(B, max_channels, device=data_batch.device)
        score_offset[~mask_batch.bool()] = float('-inf')
        
        heads = model.heads[None].expand(B, -1, -1)
        scores = torch.einsum("bcp,bop->boc", embedding, heads)
        scores = scores + score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        
        # Check weights sum to 1 for each output channel
        weight_sums = weights.sum(dim=2)
        print(f"Attention weight sums (should all be ~1.0):")
        print(f"  Min: {weight_sums.min().item():.6f}")
        print(f"  Max: {weight_sums.max().item():.6f}")
        print(f"  Mean: {weight_sums.mean().item():.6f}")

        # Check that padded channels have zero weight
        for b_idx in range(B):
            n_real = real_channels[b_idx]
            if n_real < max_channels:  # Only check if there's actual padding
                padded_weights = weights[b_idx, :, n_real:]  # Weights on padded channels
                print(f"  Sample {b_idx} ({n_real} real channels): max weight on padding = {padded_weights.max().item():.10f}")
                assert padded_weights.max() < 1e-6, f"Padded channels have non-zero weights!"
            else:
                print(f"  Sample {b_idx} ({n_real} real channels): no padding")
    
    print("\n✓ All tests passed!")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Heads: {model.heads.numel():,}")