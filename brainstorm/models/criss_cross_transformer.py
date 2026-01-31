import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import math
import time
from typing import Optional, Tuple, Dict, Any

from brainstorm.models.spatial_attention import GaussianFourierEmb3D
from brainstorm.models.attentional.attn import SpatialTemporalEncoder


class CrissCrossTransformerModule(pl.LightningModule):
    """
    Criss-Cross Transformer Lightning Module for multi-channel biosignal processing.

    This module uses a frozen tokenizer (BioCodec) to encode multi-channel MEG signals
    into discrete RVQ codes, then trains a criss-cross transformer with temporal block masking.

    Architecture:
    1. BioCodec encoding: [B, C, T] → [B, C, Q, T', D] quantized embeddings
    2. RVQ projection: Combine Q levels to get [B, C, T', D_model]
    3. Add RoPE embeddings in time dimension
    4. Add sensor embeddings (position xyz + orientation abc + type GRAD/MAG)
    5. Temporal block masking: Randomly select time position, mask 3s block across all channels
    6. Criss-cross transformer: Alternate attention between time and channel dimensions
    7. Predict masked tokens across all RVQ levels

    Args:
        tokenizer: Frozen BioCodecModule instance for encoding signals
        latent_dim: Transformer hidden dimension (default: 512)
        num_layers: Number of transformer layers (default: 8)
        num_heads: Number of attention heads (default: 8)
        vocab_size: Codebook vocabulary size (default: 256)
        learning_rate: Learning rate for AdamW optimizer (default: 1e-4)
        warmup_steps: Number of warmup steps for scheduler (default: 1000)
        training_steps: Total number of training steps for scheduler (default: 10000)
        max_seq_len: Maximum sequence length for positional embeddings (default: 2048)
        mask_duration: Duration of temporal mask in seconds (default: 3.0)
        num_subsegments_to_mask: Number of subsegments to mask simultaneously (default: 1)
        sampling_rate: MEG sampling rate in Hz (default: 250)
        fourier_pos_dim: Fourier embedding dimension for sensor positions (default: 250)
    """

    # Allow non-strict checkpoint loading (for RoPE buffer size mismatches)
    # RoPE buffers are deterministic and will be recomputed on first forward pass
    strict_loading = False

    def __init__(
        self,
        tokenizer: nn.Module,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        vocab_size: int = 256,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        training_steps: int = 10000,
        mask_duration: float = 3.0,
        num_subsegments_to_mask: int = 1,
        sampling_rate: int = 250,
        fourier_pos_dim: int = 250,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])

        # Validate num_subsegments_to_mask
        if num_subsegments_to_mask < 0:
            raise ValueError(
                f"num_subsegments_to_mask must be non-negative, got {num_subsegments_to_mask}"
            )

        # Freeze tokenizer
        self.tokenizer = tokenizer
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self.tokenizer.eval()

        # Get number of RVQ levels from tokenizer
        self.n_q = tokenizer.quantizer.n_q
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.mask_duration = mask_duration
        self.num_subsegments_to_mask = num_subsegments_to_mask
        self.sampling_rate = sampling_rate

        # Calculate mask length in encoded timesteps
        # BioCodec downsampling ratio is 12 (ratios=[3, 2, 2])
        self.biocodec_downsample_ratio = 12
        mask_samples = round(mask_duration * sampling_rate)
        self.mask_length = mask_samples // self.biocodec_downsample_ratio

        # Get BioCodec embeddings from frozen quantizer
        # Each layer has: tokenizer.quantizer.vq.layers[q]._codebook.embed
        # Shape: [vocab_size, codebook_dim] where codebook_dim = dimension // 8
        biocodec_embeddings = []
        for q in range(self.n_q):
            codebook_embed = tokenizer.quantizer.vq.layers[q]._codebook.embed
            # Register as buffer (non-trainable but tracked by the model)
            self.register_buffer(f'biocodec_embedding_{q}', codebook_embed)
            biocodec_embeddings.append(codebook_embed)

        # Get codebook dimension from first layer
        self.codebook_dim = biocodec_embeddings[0].shape[1]

        # RVQ projector: Concatenate all Q levels and project to latent_dim
        # Input: Q * codebook_dim -> Output: latent_dim
        self.rvq_projector = nn.Linear(self.n_q * self.codebook_dim, latent_dim)

        # Sensor position embeddings: Fourier embedding for xyz coordinates
        self.position_fourier_emb = GaussianFourierEmb3D(embed_dim=fourier_pos_dim, scale=1.8)
        self.position_projector = nn.Linear(fourier_pos_dim, latent_dim)

        # Sensor orientation embeddings (dir) - needs data extraction in dataloader
        self.orientation_fourier_emb = GaussianFourierEmb3D(embed_dim=fourier_pos_dim, scale=1.0)
        self.orientation_projector = nn.Linear(fourier_pos_dim, latent_dim)

        # Sensor type embedding: binary GRAD (0) or MAG (1)
        self.sensor_type_layer = nn.Embedding(2, latent_dim)

        # Single learnable mask token for all levels
        self.mask_token = nn.Parameter(torch.randn(latent_dim))

        # Criss-cross attention transformer
        # This should alternate between:
        # - Row attention (along time axis for each channel)
        # - Column attention (along channel axis for each timestep)
        # Automatically handles RoPE in time dimension
        self.criss_cross_transformer = SpatialTemporalEncoder(
            dim=latent_dim,
            depth=num_layers,
            heads=num_heads,
            dropout=0.1,
            causal=False
        )

        # Output head: single linear layer predicting all Q levels
        # Output shape: [B, C, T', Q * vocab_size]
        self.output_head = nn.Linear(latent_dim, self.n_q * vocab_size)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing to reduce memory usage.

        This trades compute for memory by not storing intermediate activations
        during the forward pass. Instead, activations are recomputed during
        the backward pass. Particularly useful for large models with FSDP.
        """
        self.gradient_checkpointing = True
        # Enable checkpointing on transformer encoder
        if hasattr(self.criss_cross_transformer, 'gradient_checkpointing_enable'):
            self.criss_cross_transformer.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    def train(self, mode: bool = True):
        """Override train() to keep tokenizer in eval mode."""
        super().train(mode)
        self.tokenizer.eval()  # Always keep tokenizer frozen
        return self

    def _tokenize_multichannel(self, raw_meg: torch.Tensor) -> torch.Tensor:
        """
        Tokenize multi-channel MEG signals using the frozen BioCodec.

        Args:
            raw_meg: [B, C, T] raw MEG signals (T = segment_duration * sampling_rate)

        Returns:
            codes: [B, C, Q, T'] discrete RVQ codes where T' = T/12 (downsample ratio)
        """
        B, C, T = raw_meg.shape

        # Reshape to batch all channels: [B*C, 1, T]
        raw_batched = raw_meg.reshape(B * C, 1, T)

        # Encode through frozen tokenizer
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(raw_batched)
            # Extract codes from encoded frames: List[(codes, scale)]
            # codes shape: [B*C, Q, T']
            codes = torch.stack([frame[0] for frame in encoded_frames], dim=0)

        codes = codes[0]

        # Reshape back to [B, C, Q, T']
        _, Q, T_prime = codes.shape
        codes = codes.reshape(B, C, Q, T_prime)

        return codes

    def _construct_embeddings(
        self,
        codes: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct embeddings with RVQ projection, sensor info, and temporal RoPE.

        Args:
            codes: [B, C, Q, T'] discrete RVQ codes
            sensor_xyz: [B, C, 3] sensor XYZ coordinates (normalized, batched)
            sensor_abc: [B, C, 3] sensor orientation vectors (batched)
            sensor_type: [B, C] sensor types, 0=GRAD, 1=MAG (batched)

        Returns:
            embeddings: [B, C, T', latent_dim] embeddings
            codes_flat: [B, C, T', Q] codes reshaped for loss computation
        """
        B, C, Q, T_prime = codes.shape

        # Reorder to [B, C, T', Q] for easier processing
        codes = codes.permute(0, 1, 3, 2)  # [B, C, T', Q]

        # Step 1: Embed each RVQ level using BioCodec embeddings
        # [B, C, T', Q, codebook_dim]
        embedded_levels = []
        for q in range(Q):
            # Get codes for this level: [B, C, T']
            codes_q = codes[..., q]
            # Look up BioCodec embeddings: [B, C, T', codebook_dim]
            biocodec_emb_q = getattr(self, f'biocodec_embedding_{q}')
            biocodec_emb = F.embedding(codes_q.long(), biocodec_emb_q)
            embedded_levels.append(biocodec_emb)
        embedded_levels = torch.stack(embedded_levels, dim=3)  # [B, C, T', Q, codebook_dim]

        # Step 2: Concatenate all Q levels and project to latent_dim
        # [B, C, T', Q * codebook_dim]
        embedded_concat = embedded_levels.reshape(B, C, T_prime, Q * self.codebook_dim)
        # Project to latent_dim: [B, C, T', latent_dim]
        embeddings = self.rvq_projector(embedded_concat)

        # Step 3: Add sensor position embeddings (Fourier embedding of xyz)
        # Process batched sensor data: [B, C, 3] -> [B*C, 3]
        pos_fourier = self.position_fourier_emb(sensor_xyz.reshape(B * C, 3))  # [B*C, fourier_pos_dim]
        pos_fourier = pos_fourier.reshape(B, C, -1)  # [B, C, fourier_pos_dim]
        pos_emb = self.position_projector(pos_fourier)  # [B, C, latent_dim]
        embeddings = embeddings + pos_emb.unsqueeze(2)  # [B, C, T', latent_dim]

        # Add sensor orientation embeddings (Fourier embedding of abc)
        ori_fourier = self.orientation_fourier_emb(sensor_abc.reshape(B * C, 3))  # [B*C, fourier_pos_dim]
        ori_fourier = ori_fourier.reshape(B, C, -1)  # [B, C, fourier_pos_dim]
        ori_emb = self.orientation_projector(ori_fourier)  # [B, C, latent_dim]
        embeddings = embeddings + ori_emb.unsqueeze(2)  # [B, C, T', latent_dim]

        # Add sensor type embeddings
        type_emb = self.sensor_type_layer(sensor_type.long())  # [B, C, latent_dim]
        embeddings = embeddings + type_emb.unsqueeze(2)  # [B, C, T', latent_dim]

        return embeddings, codes

    def _generate_temporal_block_mask(
        self,
        B: int,
        n_channels: int,
        n_timesteps: int,
        sensor_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate temporal block mask using subsegment-based strategy:
        1. Divide the segment into subsegments of 3s each (word segment length)
        2. Randomly select num_subsegments_to_mask subsegments per batch sample
        3. Randomly place the mask within each selected subsegment
        4. Combine all masks via logical OR
        Vectorized implementation with different subsegment selection per batch sample.
        """
        # 1. Calculate number of subsegments based on segment length / 3s (word segment length)
        # Segment length in seconds = n_timesteps * downsample_ratio / sampling_rate
        segment_length_seconds = (n_timesteps * self.biocodec_downsample_ratio) / self.sampling_rate
        num_subsegments = int(segment_length_seconds / 3.0)
        subseg_length = n_timesteps // num_subsegments  # Integer division for clean boundaries

        # 2. Handle edge cases
        if self.num_subsegments_to_mask <= 0:
            # No masking - return empty mask
            mask = torch.zeros(B, n_channels, n_timesteps, dtype=torch.bool, device=device)
            return mask, 0.0

        actual_n_to_mask = min(self.num_subsegments_to_mask, num_subsegments)

        # Log warning if clamping (only once)
        if not hasattr(self, '_warned_subseg_clamp'):
            if self.num_subsegments_to_mask > num_subsegments:
                print(f"Warning: num_subsegments_to_mask ({self.num_subsegments_to_mask}) "
                      f"exceeds available subsegments ({num_subsegments}). "
                      f"Masking all {num_subsegments} subsegments instead.")
                self._warned_subseg_clamp = True

        # 3. Sample unique subsegment indices (without replacement)
        # Use multinomial for efficient vectorized sampling
        # Shape: [B, actual_n_to_mask]
        probs = torch.ones(B, num_subsegments, device=device) / num_subsegments
        subseg_indices = torch.multinomial(
            probs,
            num_samples=actual_n_to_mask,
            replacement=False
        ).unsqueeze(-1)  # [B, actual_n_to_mask, 1]

        # 4. Calculate subsegment start positions
        subseg_starts = subseg_indices * subseg_length  # [B, actual_n_to_mask, 1]

        # 5. Calculate valid range for random placement
        max_offset_within_subseg = max(0, subseg_length - self.mask_length)

        # 6. Generate random offset within each subsegment
        random_offsets = torch.randint(
            0, max_offset_within_subseg + 1,
            (B, actual_n_to_mask, 1),
            device=device
        )  # [B, actual_n_to_mask, 1]

        # 7. Calculate final mask start positions
        start_indices = subseg_starts + random_offsets  # [B, actual_n_to_mask, 1]

        # 8. Create time index grid
        time_indices = torch.arange(n_timesteps, device=device).view(1, 1, -1)  # [1, 1, T]

        # 9. Create masks for all subsegments simultaneously via broadcasting
        # [B, actual_n_to_mask, 1] broadcast with [1, 1, T] -> [B, actual_n_to_mask, T]
        mask_per_subseg = (time_indices >= start_indices) & \
                          (time_indices < start_indices + self.mask_length)

        # 10. Combine masks across subsegments (logical OR)
        # Any timestep masked in ANY subsegment is masked
        mask = mask_per_subseg.any(dim=1)  # [B, T]

        # 11. Expand to channels
        mask = mask.unsqueeze(1).expand(B, n_channels, n_timesteps)  # [B, C, T]

        # 12. Apply sensor mask if provided
        if sensor_mask is not None:
            # Convert to boolean if needed (sensor_mask might be float)
            sensor_mask_bool = sensor_mask.bool()
            sensor_mask_expanded = sensor_mask_bool.unsqueeze(-1)  # [B, C, 1]
            mask = mask & sensor_mask_expanded  # Only mask valid sensors

        # 13. Calculate actual mask ratio
        if sensor_mask is not None:
            total_valid_tokens = sensor_mask.sum() * n_timesteps
            # Avoid division by zero if total_valid_tokens is 0
            if total_valid_tokens > 0:
                actual_mask_ratio = (mask.sum().float() / total_valid_tokens.float()).item()
            else:
                actual_mask_ratio = 0.0
        else:
            actual_mask_ratio = mask.float().mean().item()

        return mask, actual_mask_ratio

    def _apply_mask(
        self,
        embeddings: torch.Tensor,
        codes: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mask tokens to embeddings at masked positions.

        Replace the original concatenated BioCodec projection with the mask token.

        Args:
            embeddings: [B, C, T', latent_dim] input embeddings
            codes: [B, C, T', Q] discrete codes
            mask: [B, C, T'] boolean mask (True = masked)

        Returns:
            masked_embeddings: [B, C, T', latent_dim] with masked positions replaced
        """
        B, C, T_prime, Q = codes.shape

        # Embed all RVQ levels using BioCodec embeddings
        embedded_levels = []
        for q in range(Q):
            codes_q = codes[..., q]
            biocodec_emb_q = getattr(self, f'biocodec_embedding_{q}')
            biocodec_emb = F.embedding(codes_q.long(), biocodec_emb_q)
            embedded_levels.append(biocodec_emb)
        embedded_levels = torch.stack(embedded_levels, dim=3)  # [B, C, T', Q, codebook_dim]

        # Concatenate and project to get original embeddings
        embedded_concat = embedded_levels.reshape(B, C, T_prime, Q * self.codebook_dim)
        orig_emb = self.rvq_projector(embedded_concat)  # [B, C, T', latent_dim]

        # Replace with mask token at masked positions
        mask_expanded = mask.unsqueeze(-1)  # [B, C, T', 1]
        masked_embeddings = torch.where(
            mask_expanded,
            embeddings - orig_emb + self.mask_token,
            embeddings
        )

        return masked_embeddings

    def forward(
        self,
        raw_meg: torch.Tensor,
        sensor_xyz: torch.Tensor,
        sensor_abc: torch.Tensor,
        sensor_type: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        apply_mask: bool = True,
        collect_timing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Criss-Cross Transformer.

        Args:
            raw_meg: [B, C, T] raw MEG signals (T = segment_duration * sampling_rate)
            sensor_xyz: [B, C, 3] sensor XYZ coordinates (normalized, batched)
            sensor_abc: [B, C, 3] sensor orientation vectors (batched)
            sensor_type: [B, C] sensor types, 0=GRAD, 1=MAG (batched)
            sensor_mask: [B, C] boolean mask for valid sensors (optional)
            apply_mask: whether to apply masking (True for training)
            collect_timing: whether to collect timing information (for profiling)

        Returns:
            Dictionary containing:
                - logits: [B, C, T', Q, vocab_size] output logits
                - features: [B, C, T', latent_dim] transformer features before output head
                - mask: [B, C, T'] boolean mask (if apply_mask=True)
                - mask_ratio: float (if apply_mask=True)
                - codes: [B, C, T', Q] ground truth codes
                - timing: dict of timing info in ms (if collect_timing=True)
        """
        timing = {}
        if collect_timing:
            t0 = time.perf_counter()

        # Step 1: Tokenize multi-channel input
        codes = self._tokenize_multichannel(raw_meg)  # [B, C, Q, T']
        B, C, _, T_prime = codes.shape

        if collect_timing:
            t1 = time.perf_counter()
            timing['tokenize_ms'] = (t1 - t0) * 1000

        # Step 2: Construct embeddings
        embeddings, codes_reordered = self._construct_embeddings(
            codes, sensor_xyz, sensor_abc, sensor_type
        )  # [B, C, T', latent_dim], [B, C, T', Q]

        if collect_timing:
            t2 = time.perf_counter()
            timing['embeddings_ms'] = (t2 - t1) * 1000

        # Step 3: Apply masking (if training)
        mask_info = {}
        if apply_mask:
            mask, mask_ratio = self._generate_temporal_block_mask(
                B, C, T_prime, sensor_mask, embeddings.device
            )
            embeddings = self._apply_mask(embeddings, codes_reordered, mask)
            mask_info['mask'] = mask
            mask_info['mask_ratio'] = mask_ratio

        if collect_timing:
            t3 = time.perf_counter()
            timing['masking_ms'] = (t3 - t2) * 1000

        # Step 4: Pass through criss-cross transformer
        # Shape: [B, C, T', latent_dim] matches [B, C, W, D] expected by SpatialTemporalEncoder
        # The encoder alternates between:
        # - Temporal attention along T' (with RoPE) for each channel
        # - Spatial attention along C for each timestep
        transformer_out = self.criss_cross_transformer(embeddings)

        if collect_timing:
            t4 = time.perf_counter()
            timing['transformer_ms'] = (t4 - t3) * 1000

        # Step 5: Project to output logits
        logits = self.output_head(transformer_out)  # [B, C, T', Q * vocab_size]
        logits = logits.reshape(B, C, T_prime, self.n_q, self.vocab_size)  # [B, C, T', Q, vocab_size]

        if collect_timing:
            t5 = time.perf_counter()
            timing['output_proj_ms'] = (t5 - t4) * 1000
            timing['total_ms'] = (t5 - t0) * 1000

        result = {
            'logits': logits,
            'codes': codes_reordered,
            'features': transformer_out,  # [B, C, T', latent_dim] - transformer features before output head
            **mask_info
        }

        if collect_timing:
            result['timing'] = timing

        return result
    
    def _compute_metrics(
        self,
        logits: torch.Tensor,
        codes: torch.Tensor,
        mask: torch.Tensor,
        sensor_mask: Optional[torch.Tensor],
        raw_meg: Optional[torch.Tensor] = None,
        sensor_xyz: Optional[torch.Tensor] = None,
        sensor_abc: Optional[torch.Tensor] = None,
        sensor_types: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Shared logic to compute loss and accuracy for a batch (or subset).
        """
        B, C, T_prime, Q, V = logits.shape

        # 1. Apply Sensor Masking (if provided)
        # Exclude padded sensors from loss computation
        if sensor_mask is not None:
            # Expand sensor_mask: [B, C] -> [B, C, T']
            sensor_mask_bool = sensor_mask.bool()
            sensor_mask_expanded = sensor_mask_bool.unsqueeze(-1).expand(-1, -1, T_prime)
            valid_mask = mask & sensor_mask_expanded
        else:
            valid_mask = mask

        # 2. Flatten Tensors
        logits_flat = logits.reshape(-1, self.n_q, V)     # [N_total, Q, V]
        codes_flat = codes.reshape(-1, self.n_q)          # [N_total, Q]
        valid_mask_flat = valid_mask.reshape(-1)          # [N_total]

        # 3. Select Masked Positions Only
        logits_masked = logits_flat[valid_mask_flat]      # [N_masked, Q, V]
        codes_masked = codes_flat[valid_mask_flat]        # [N_masked, Q]

        metrics = {}

        if logits_masked.numel() > 0:
            loss = 0.0
            accuracies = []
            
            # Compute cross-entropy and accuracy for each Quantizer level
            for q in range(self.n_q):
                logits_q = logits_masked[:, q, :]
                targets_q = codes_masked[:, q]
                
                # Loss
                loss_q = F.cross_entropy(logits_q, targets_q)
                loss += loss_q

                # If there is a NaN in loss or logits, print debug info
                if torch.isnan(loss_q) or torch.isnan(logits_q).any():
                    print(f"\n{'='*60}")
                    print(f"NaN DETECTED at quantizer level q={q}")
                    print(f"{'='*60}")

                    # MEG Input Stats
                    if raw_meg is not None:
                        print(f"\nMEG Input stats (raw_meg shape: {raw_meg.shape}):")
                        print(f"  - Contains NaN: {torch.isnan(raw_meg).any().item()}")
                        print(f"  - Contains Inf: {torch.isinf(raw_meg).any().item()}")
                        print(f"  - Min: {raw_meg.min().item():.6e}")
                        print(f"  - Max: {raw_meg.max().item():.6e}")
                        print(f"  - Mean: {raw_meg.mean().item():.6e}")
                        print(f"  - Std: {raw_meg.std().item():.6e}")
                        print(f"  - NaN count: {torch.isnan(raw_meg).sum().item()}")
                        print(f"  - Inf count: {torch.isinf(raw_meg).sum().item()}")

                    # Sensor Stats
                    if sensor_xyz is not None:
                        print(f"\nSensor XYZ stats (sensor_xyz shape: {sensor_xyz.shape}):")
                        print(f"  - Contains NaN: {torch.isnan(sensor_xyz).any().item()}")
                        print(f"  - Contains Inf: {torch.isinf(sensor_xyz).any().item()}")
                        print(f"  - Min: {sensor_xyz.min().item():.6f}")
                        print(f"  - Max: {sensor_xyz.max().item():.6f}")
                        print(f"  - Mean: {sensor_xyz.mean().item():.6f}")
                        print(f"  - Std: {sensor_xyz.std().item():.6f}")

                    if sensor_abc is not None:
                        print(f"\nSensor ABC (orientation) stats (sensor_abc shape: {sensor_abc.shape}):")
                        print(f"  - Contains NaN: {torch.isnan(sensor_abc).any().item()}")
                        print(f"  - Contains Inf: {torch.isinf(sensor_abc).any().item()}")
                        print(f"  - Min: {sensor_abc.min().item():.6f}")
                        print(f"  - Max: {sensor_abc.max().item():.6f}")
                        print(f"  - Mean: {sensor_abc.mean().item():.6f}")

                    if sensor_types is not None:
                        print(f"\nSensor Types stats (sensor_types shape: {sensor_types.shape}):")
                        print(f"  - Unique types: {sensor_types.unique().tolist()}")
                        print(f"  - Type counts: {[(t.item(), (sensor_types == t).sum().item()) for t in sensor_types.unique()]}")

                    if dataset_ids is not None:
                        print(f"\nDataset IDs:")
                        print(f"  - Unique datasets: {dataset_ids.unique().tolist()}")
                        print(f"  - Dataset distribution: {[(d.item(), (dataset_ids == d).sum().item()) for d in dataset_ids.unique()]}")

                    print(f"\nLogits stats (logits_q shape: {logits_q.shape}):")
                    print(f"  - Contains NaN: {torch.isnan(logits_q).any().item()}")
                    print(f"  - Contains Inf: {torch.isinf(logits_q).any().item()}")
                    print(f"  - Min: {logits_q.min().item():.6f}")
                    print(f"  - Max: {logits_q.max().item():.6f}")
                    print(f"  - Mean: {logits_q.mean().item():.6f}")
                    print(f"  - Std: {logits_q.std().item():.6f}")
                    print(f"  - NaN count: {torch.isnan(logits_q).sum().item()}")
                    print(f"  - Inf count: {torch.isinf(logits_q).sum().item()}")

                    print(f"\nTargets stats (targets_q shape: {targets_q.shape}):")
                    print(f"  - Contains NaN: {torch.isnan(targets_q.float()).any().item()}")
                    print(f"  - Min: {targets_q.min().item()}")
                    print(f"  - Max: {targets_q.max().item()}")
                    print(f"  - Unique values: {targets_q.unique().numel()}")
                    print(f"  - Out of range (<0 or >={logits_q.shape[-1]}): {((targets_q < 0) | (targets_q >= logits_q.shape[-1])).sum().item()}")

                    print(f"\nLoss stats:")
                    print(f"  - loss_q: {loss_q.item() if not torch.isnan(loss_q) else 'NaN'}")
                    print(f"  - loss (accumulated): {loss if isinstance(loss, float) else loss.item() if not torch.isnan(loss) else 'NaN'}")

                    print(f"\nMasked data stats:")
                    print(f"  - N_masked samples: {logits_masked.shape[0]}")
                    print(f"  - N_total samples: {logits_flat.shape[0]}")
                    print(f"  - Mask ratio: {valid_mask_flat.float().mean().item():.4f}")

                    print(f"\nOriginal input stats (logits shape: {logits.shape}):")
                    print(f"  - Contains NaN: {torch.isnan(logits).any().item()}")
                    print(f"  - Contains Inf: {torch.isinf(logits).any().item()}")
                    print(f"{'='*60}\n")

                    # exit(0)

                # Accuracy
                preds_q = logits_q.argmax(dim=-1)
                acc_q = (preds_q == targets_q).float().mean()
                accuracies.append(acc_q)
                
                # Store individual Q accuracy
                metrics[f'accuracy_q{q}'] = acc_q

            # Average metrics
            metrics['loss'] = loss / self.n_q
            metrics['accuracy'] = torch.stack(accuracies).mean()
        else:
            # Fallback for empty masks (e.g. sanity checks or extreme edge cases)
            metrics['loss'] = torch.tensor(0.0, device=logits.device, requires_grad=True)
            metrics['accuracy'] = torch.tensor(0.0, device=logits.device)
            for q in range(self.n_q):
                metrics[f'accuracy_q{q}'] = torch.tensor(0.0, device=logits.device)

        return metrics

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # Unpack batch (handles optional dataset_ids)
        dataset_ids = None
        if len(batch) == 5:
            raw_meg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids = batch
        else:
            raw_meg, sensor_xyzdir, sensor_types, sensor_mask = batch

        sensor_xyz = sensor_xyzdir[..., :3]
        sensor_abc = sensor_xyzdir[..., 3:]

        # --- Forward Pass ---
        t0 = time.perf_counter()

        output = self.forward(
            raw_meg, sensor_xyz, sensor_abc, sensor_types,
            sensor_mask=sensor_mask,
            apply_mask=True,
            collect_timing=True
        )

        t1 = time.perf_counter()

        # Unpack output
        logits = output['logits']
        codes = output['codes']
        mask = output['mask']
        mask_ratio = output['mask_ratio']
        timing = output['timing']

        # --- Compute Metrics using Helper ---
        metrics = self._compute_metrics(
            logits, codes, mask, sensor_mask,
            raw_meg=raw_meg,
            sensor_xyz=sensor_xyz,
            sensor_abc=sensor_abc,
            sensor_types=sensor_types,
            dataset_ids=dataset_ids
        )
        
        t2 = time.perf_counter()

        # --- Logging ---
        # 1. Log Loss & Accuracy
        self.log('train/loss', metrics['loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/accuracy', metrics['accuracy'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # 2. Log per-level accuracy (only on epoch to reduce clutter)
        for q in range(self.n_q):
             self.log(f'train/accuracy_q{q}', metrics[f'accuracy_q{q}'], on_step=True, on_epoch=True, sync_dist=True)

        # 3. Log Timing
        for key, value in timing.items():
            self.log(f'timing/{key}', value, on_step=True, on_epoch=True, sync_dist=True)
        self.log('timing/loss_calc_ms', (t2 - t1) * 1000, on_step=True, on_epoch=True, sync_dist=True)
        self.log('timing/total_step_ms', (t2 - t0) * 1000, on_step=True, on_epoch=True, sync_dist=True)

        # 4. Log Mask Ratio
        self.log('train/mask_ratio', mask_ratio, on_step=True, on_epoch=True, sync_dist=True)

        return metrics['loss']

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        # Unpack batch
        dataset_ids = None
        if len(batch) == 5:
            raw_meg, sensor_xyzdir, sensor_types, sensor_mask, dataset_ids = batch
        else:
            raw_meg, sensor_xyzdir, sensor_types, sensor_mask = batch

        sensor_xyz = sensor_xyzdir[..., :3]
        sensor_abc = sensor_xyzdir[..., 3:]

        # --- Forward Pass ---
        output = self.forward(
            raw_meg, sensor_xyz, sensor_abc, sensor_types, 
            sensor_mask=sensor_mask, apply_mask=True
        )
        
        logits = output['logits'].contiguous()
        codes = output['codes'].contiguous()
        mask = output['mask'].contiguous()

        # --- 1. Global Metrics ---
        global_metrics = self._compute_metrics(
            logits, codes, mask, sensor_mask,
            raw_meg=raw_meg,
            sensor_xyz=sensor_xyz,
            sensor_abc=sensor_abc,
            sensor_types=sensor_types,
            dataset_ids=dataset_ids
        )
        
        self.log('val/loss', global_metrics['loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', global_metrics['accuracy'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log per-level accuracy for global
        for q in range(self.n_q):
            self.log(f'val/accuracy_q{q}', global_metrics[f'accuracy_q{q}'], on_step=False, on_epoch=True, sync_dist=True)

        # --- 2. Dataset-Specific Metrics ---
        if dataset_ids is not None:
            unique_ids = torch.unique(dataset_ids)

            # Get dataset name mapping from datamodule
            dataset_name_mapping = {}
            if self.trainer is not None and hasattr(self.trainer, 'datamodule'):
                dataset_name_mapping = self.trainer.datamodule.get_dataset_name_mapping()

            for d_id in unique_ids:
                # Get integer indices for this dataset (safer for CUDA than boolean indexing)
                batch_indices = torch.where(dataset_ids == d_id)[0]

                if batch_indices.numel() == 0:  # Skip if no samples from this dataset
                    continue

                # If only one sample is present duplicate
                if batch_indices.numel() == 1:
                    batch_indices = batch_indices.repeat(2)

                # Slice the batch using integer indices
                sub_logits = torch.index_select(logits, 0, batch_indices)
                sub_codes = torch.index_select(codes, 0, batch_indices)
                sub_mask = torch.index_select(mask, 0, batch_indices)
                sub_sensor_mask = torch.index_select(sensor_mask, 0, batch_indices) if sensor_mask is not None else None

                # Slice input tensors for debug info
                sub_raw_meg = torch.index_select(raw_meg, 0, batch_indices)
                sub_sensor_xyz = torch.index_select(sensor_xyz, 0, batch_indices)
                sub_sensor_abc = torch.index_select(sensor_abc, 0, batch_indices)
                sub_sensor_types = torch.index_select(sensor_types, 0, batch_indices)
                sub_dataset_ids = torch.index_select(dataset_ids, 0, batch_indices)

                # Compute subset metrics
                sub_metrics = self._compute_metrics(
                    sub_logits, sub_codes, sub_mask, sub_sensor_mask,
                    raw_meg=sub_raw_meg,
                    sensor_xyz=sub_sensor_xyz,
                    sensor_abc=sub_sensor_abc,
                    sensor_types=sub_sensor_types,
                    dataset_ids=sub_dataset_ids
                )

                # Log with dataset name (fallback to ID if name not available)
                d_id_int = int(d_id.item())
                dataset_name = dataset_name_mapping.get(d_id_int, f'dataset_{d_id_int}')
                self.log(f'val/{dataset_name}_loss', sub_metrics['loss'], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'val/{dataset_name}_acc', sub_metrics['accuracy'], on_step=False, on_epoch=True, sync_dist=True)

        return global_metrics['loss']

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook called before loading checkpoint state dict.

        Filters out RoPE rotation buffers that may have size mismatches due to
        different sequence lengths between checkpoint and current model initialization.
        RoPE rotation matrices are deterministically computed from position indices
        and frequencies, so they will be automatically recomputed on first forward pass.

        Args:
            checkpoint: The checkpoint dictionary being loaded
        """
        if 'state_dict' not in checkpoint:
            return

        state_dict = checkpoint['state_dict']
        skipped_rope_keys = []
        filtered_state_dict = {}

        for key, value in state_dict.items():
            if 'rope_embedding_layer.rotate' in key:
                skipped_rope_keys.append(key)
            else:
                filtered_state_dict[key] = value

        if skipped_rope_keys:
            print(f"\n{'='*60}")
            print(f"Checkpoint Loading: Skipping RoPE rotation buffers")
            print(f"{'='*60}")
            print(f"  Skipped {len(skipped_rope_keys)} RoPE keys (will auto-expand on first forward pass)")
            print(f"  Example keys: {skipped_rope_keys[:2]}")
            print(f"{'='*60}\n")

        # Replace state dict with filtered version
        checkpoint['state_dict'] = filtered_state_dict

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure AdamW optimizer with cosine annealing.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = AdamW(
            list(filter(lambda p: p.requires_grad, self.parameters())),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.training_steps,
            eta_min=self.learning_rate * 0.01
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


if __name__ == "__main__":
    """Test the Criss-Cross Transformer forward pass."""
    print("Testing Criss-Cross Transformer Module...")

    # Import BioCodec tokenizer
    import sys
    sys.path.append('/path/to/BrainStorm')
    from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel

    # Create a test BioCodec tokenizer
    print("\n1. Creating BioCodec tokenizer...")
    tokenizer = BioCodecModel._get_optimized_model()
    checkpoint = torch.load("./brainstorm/neuro_tokenizers/biocodec_ckpt.pt", map_location="cpu")
    # Rename keys to remove _orig_mod prefix
    new_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod.") :]
        else:
            new_key = key
        new_state_dict[new_key] = value
    tokenizer.load_state_dict(new_state_dict)
    tokenizer.eval()
    print(f"   ✓ Tokenizer created with {tokenizer.quantizer.n_q} RVQ levels")
    print(f"   ✓ Codebook size: {tokenizer.quantizer.bins}")

    # Create Criss-Cross Transformer module
    print("\n2. Creating Criss-Cross Transformer...")
    model = CrissCrossTransformerModule(
        tokenizer=tokenizer,
        latent_dim=512,
        num_layers=4,
        num_heads=8,
        vocab_size=256,
        learning_rate=1e-4,
        training_steps=10000,
        mask_duration=3.0,
        sampling_rate=250,
        fourier_pos_dim=250,
    )
    print(f"   ✓ Model created with {model.latent_dim}D latent space")
    print(f"   ✓ Codebook dimension: {model.codebook_dim}")
    print(f"   ✓ Mask duration: {model.mask_duration}s = {model.mask_length} encoded timesteps")

    # Create dummy data
    print("\n3. Creating test data...")
    batch_size = 2
    num_channels = 300
    seq_len = 7500  # Example: 30 seconds at 250 Hz (segment duration can vary)

    # Random MEG signals
    raw_meg = torch.randn(batch_size, num_channels, seq_len)

    # Random sensor positions and orientations (batched, normalized to unit sphere)
    sensor_xyz = torch.randn(batch_size, num_channels, 3)
    sensor_xyz = F.normalize(sensor_xyz, p=2, dim=-1)

    sensor_abc = torch.randn(batch_size, num_channels, 3)
    sensor_abc = F.normalize(sensor_abc, p=2, dim=-1)

    # Random sensor types (0=GRAD, 1=MAG)
    sensor_types = torch.randint(0, 2, (batch_size, num_channels))

    # Sensor mask (all valid for now)
    sensor_mask = torch.ones(batch_size, num_channels, dtype=torch.bool)

    print(f"   ✓ Input shape: {raw_meg.shape}")
    print(f"   ✓ Sensor positions: {sensor_xyz.shape}")
    print(f"   ✓ Sensor orientations: {sensor_abc.shape}")
    print(f"   ✓ Sensor types: {sensor_types.shape}")
    print(f"   ✓ Input duration: {seq_len / 250}s at 250Hz")

    # Test forward pass without masking
    print("\n4. Testing forward pass (no masking)...")
    with torch.no_grad():
        output = model(raw_meg, sensor_xyz, sensor_abc, sensor_types, apply_mask=False)

    logits = output['logits']
    codes = output['codes']
    print(f"   ✓ Output logits shape: {logits.shape}")
    print(f"   ✓ Codes shape: {codes.shape}")
    print(f"   ✓ Expected logits shape: [B={batch_size}, C={num_channels}, T', Q={model.n_q}, vocab={model.vocab_size}]")

    # Test forward pass with masking
    print("\n5. Testing forward pass (with temporal block masking)...")
    for i in range(3):
        with torch.no_grad():
            output = model(raw_meg, sensor_xyz, sensor_abc, sensor_types, apply_mask=True)

        logits = output['logits']
        codes = output['codes']
        mask = output['mask']
        mask_ratio = output['mask_ratio']

        print(f"\n   Run {i+1}:")
        print(f"   ✓ Mask ratio: {mask_ratio:.3f}")
        print(f"   ✓ Number of masked positions: {mask.sum().item()}/{mask.numel()}")
        if i == 0:
            print(f"   ✓ Output logits shape: {logits.shape}")
            print(f"   ✓ Mask shape: {mask.shape}")

    # Test with sensor masking (multi-dataset scenario)
    print("\n6. Testing with sensor masking (padded sensors)...")
    # Create sensor mask where only first 200 channels are valid
    sensor_mask_partial = torch.zeros(batch_size, num_channels, dtype=torch.bool)
    sensor_mask_partial[:, :200] = True

    with torch.no_grad():
        output = model(raw_meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask=sensor_mask_partial, apply_mask=True)

    mask = output['mask']
    mask_ratio = output['mask_ratio']

    # Check that padded sensors are not masked
    masked_padded = mask[:, 200:].sum()
    print(f"   ✓ Valid sensors per sample: 200/{num_channels}")
    print(f"   ✓ Mask ratio (over valid sensors): {mask_ratio:.3f}")
    print(f"   ✓ Masked tokens in padded region: {masked_padded} (should be 0)")
    assert masked_padded == 0, "Padded sensors should not be masked!"

    # Test training step simulation
    print("\n7. Simulating training step...")
    model.train()
    # Create batch in format expected by training_step: (raw_meg, sensor_xyzdir, sensor_types, sensor_mask)
    sensor_xyzdir = torch.cat([sensor_xyz, sensor_abc], dim=-1)  # [B, C, 6]
    batch = (raw_meg, sensor_xyzdir, sensor_types, sensor_mask_partial)

    loss = model.training_step(batch, 0)
    print(f"   ✓ Loss: {loss.item():.4f}")

    # Count parameters
    print("\n8. Model statistics...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Frozen parameters (tokenizer): {frozen_params:,}")

    print("\n✅ All tests passed!")
    print("\n✅ Implementation complete:")
    print("  ✓ Criss-cross transformer (alternating time/channel attention)")
    print("  ✓ RoPE (Rotary Position Embeddings) for time dimension")
