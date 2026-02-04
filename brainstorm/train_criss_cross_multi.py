"""Training script for Criss-Cross Transformer on multiple MEG datasets."""

import os
import sys
import warnings
from pathlib import Path

import torch
import pytorch_lightning as pl

# Filter out Pydantic warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
from brainstorm.data.multi_datamodule import MultiMEGDataModule
from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel


class SamplerVerificationCallback(Callback):
    """
    Verify that Lightning properly wrapped our RecordingShuffleSampler with DistributedSampler.

    This callback checks on the first training epoch that:
    1. The sampler is wrapped with DistributedSamplerWrapper (when using DDP/FSDP)
    2. The underlying sampler is our RecordingShuffleSampler
    """

    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        # Only verify on first epoch
        if trainer.current_epoch > 0:
            return

        sampler = trainer.train_dataloader.sampler

        # Import here to avoid circular dependencies
        from brainstorm.data.samplers import RecordingShuffleSampler

        # Check if we're using distributed training
        if trainer.world_size > 1:
            # Lightning wraps samplers - check the wrapper and underlying sampler
            # The wrapper class name depends on Lightning version, so check by attribute
            if hasattr(sampler, 'sampler'):
                # This is a wrapper - get the underlying sampler
                underlying_sampler = sampler.sampler
                if isinstance(underlying_sampler, RecordingShuffleSampler):
                    print(f"✓ Rank {trainer.global_rank}/{trainer.world_size}: "
                          f"RecordingShuffleSampler properly wrapped with {type(sampler).__name__}")
                else:
                    print(f"⚠ Rank {trainer.global_rank}: Expected RecordingShuffleSampler, "
                          f"got {type(underlying_sampler).__name__}")
            else:
                print(f"⚠ Rank {trainer.global_rank}: Expected wrapped sampler for multi-GPU, "
                      f"got {type(sampler).__name__}")
        else:
            # Single GPU - should NOT be wrapped
            if isinstance(sampler, RecordingShuffleSampler):
                print(f"✓ Single GPU: Using RecordingShuffleSampler (no distributed wrapper needed)")
            else:
                print(f"⚠ Expected RecordingShuffleSampler, got {type(sampler).__name__}")


def load_tokenizer(ckpt_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load frozen BioCodec tokenizer from checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to BioCodec checkpoint file
    device : str
        Device to load model on ("cpu" or "cuda")

    Returns
    -------
    tokenizer : torch.nn.Module
        Frozen BioCodec tokenizer
    """
    print(f"\n=== Loading BioCodec Tokenizer ===")
    print(f"Checkpoint: {ckpt_path}")

    # Create model
    tokenizer = BioCodecModel._get_optimized_model()

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Remove _orig_mod prefix from state dict keys
    new_state_dict = {}
    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    tokenizer.load_state_dict(new_state_dict)
    tokenizer.eval()

    print(f"✓ Tokenizer loaded successfully")
    print(f"  RVQ levels: {tokenizer.quantizer.n_q}")
    print(f"  Codebook size: {tokenizer.quantizer.bins}")

    return tokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="train_criss_cross_multi")
def main(cfg: DictConfig):
    """
    Main training function for multi-dataset pre-training.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    print("\n" + "=" * 80)
    print("MULTI-DATASET CRISS-CROSS TRANSFORMER TRAINING")
    print("=" * 80)
    print("\n=== Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    # Validate config consistency
    assert cfg.data.target_sfreq == cfg.model.sampling_rate, (
        f"Config mismatch: data.target_sfreq ({cfg.data.target_sfreq}) must match "
        f"model.sampling_rate ({cfg.model.sampling_rate}). The model uses sampling_rate "
        f"to calculate segment subdivisions, so it must match the actual sampling rate "
        f"of the preprocessed data (target_sfreq)."
    )
    print("\n✓ Config validation passed: data.target_sfreq matches model.sampling_rate")

    # Configure TF32 precision for Tensor Cores (must be set before CUDA operations)
    torch.set_float32_matmul_precision('high')
    print("✓ Set float32 matmul precision to 'high' for Tensor Cores")

    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        pl.seed_everything(cfg.seed, workers=True)
        print(f"\n✓ Random seed set to {cfg.seed}")

    # -------------------------------------------------------------------------
    # 1. Setup DataModule
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SETTING UP MULTI-DATASET DATA")
    print("=" * 80)

    datamodule = MultiMEGDataModule(
        datasets_config=OmegaConf.to_container(cfg.datasets_config, resolve=True),
        segment_length=cfg.data.segment_length,
        cache_dir=cfg.data.cache_dir,
        l_freq=cfg.data.l_freq,
        h_freq=cfg.data.h_freq,
        target_sfreq=cfg.data.target_sfreq,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        persistent_workers=cfg.training.persistent_workers,
        use_recording_sampler=cfg.training.use_recording_sampler,
        sampler_seed=cfg.training.sampler_seed,
        debug_mode=cfg.data.get("debug_mode", False),
        shuffle_segments=cfg.data.get("shuffle_segments", False),
        shuffle_segment_duration=cfg.data.get("shuffle_segment_duration", 3.0),
        recording_subsample_prop=cfg.data.get("recording_subsample_prop", None),
    )

    # Setup datasets (triggers preprocessing and caching)
    datamodule.setup("fit")

    # Validate training configuration (either num_epochs or max_steps, not both)
    num_epochs = cfg.training.get("num_epochs", None)
    max_steps = cfg.training.get("max_steps", None)

    if num_epochs is not None and max_steps is not None:
        raise ValueError(
            "Cannot specify both num_epochs and max_steps. "
            "Please set one to null in the config."
        )
    if num_epochs is None and max_steps is None:
        raise ValueError(
            "Must specify either num_epochs or max_steps in the config."
        )

    # Calculate training steps for scheduler
    train_loader = datamodule.train_dataloader()
    steps_per_epoch = len(train_loader)

    if num_epochs is not None:
        training_steps = num_epochs * steps_per_epoch
        print(f"\n=== Training Configuration (Epoch-based) ===")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Total training steps: {training_steps}")
    else:
        training_steps = max_steps
        estimated_epochs = max_steps / steps_per_epoch
        print(f"\n=== Training Configuration (Step-based) ===")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Maximum training steps: {max_steps}")
        print(f"Estimated epochs: {estimated_epochs:.2f}")

    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Max channel dim: 306 (multi-dataset)")

    # -------------------------------------------------------------------------
    # 2. Load Tokenizer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    tokenizer_path = Path(cfg.model.tokenizer_ckpt)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer checkpoint not found: {tokenizer_path}\n"
            f"Please ensure the BioCodec checkpoint is at this location."
        )

    tokenizer = load_tokenizer(str(tokenizer_path), device="cpu")

    # -------------------------------------------------------------------------
    # 3. Optional: Setup Checkpoint Resumption
    # -------------------------------------------------------------------------
    ckpt_path = None
    if cfg.checkpoint.get("resume", False):
        resume_path = cfg.checkpoint.get("resume_path", None)
        if resume_path:
            ckpt_path = Path(resume_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {ckpt_path}\n"
                    f"Please ensure the checkpoint file exists."
                )
            print(f"\n=== Resuming Training ===\n")
            print(f"Checkpoint: {ckpt_path}")
            print(f"✓ Will resume training from checkpoint")
        else:
            raise ValueError(
                "checkpoint.resume is True but checkpoint.resume_path is not set. "
                "Please provide a valid checkpoint path in the config."
            )

    # -------------------------------------------------------------------------
    # 4. Create Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CREATING MODEL")
    print("=" * 80)

    model = CrissCrossTransformerModule(
        tokenizer=tokenizer,
        latent_dim=cfg.model.latent_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        vocab_size=cfg.model.vocab_size,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        training_steps=training_steps,
        mask_duration=cfg.model.mask_duration,
        num_subsegments_to_mask=cfg.model.num_subsegments_to_mask,
        sampling_rate=cfg.model.sampling_rate,
        fourier_pos_dim=cfg.model.fourier_pos_dim,
    )

    # Enable gradient checkpointing if configured
    if cfg.model.get("use_gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled for memory efficiency")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters (tokenizer): {frozen_params:,}")
    print(f"Latent dimension: {cfg.model.latent_dim}")
    print(f"Number of layers: {cfg.model.num_layers}")
    print(f"Number of heads: {cfg.model.num_heads}")
    print(f"\n=== Temporal Block Masking Configuration ===")
    print(f"Mask duration: {cfg.model.mask_duration}s")
    print(f"Number of subsegments to mask: {cfg.model.num_subsegments_to_mask}")
    print(f"Sampling rate: {cfg.model.sampling_rate}Hz")
    # Calculate mask length in encoded timesteps
    biocodec_downsample_ratio = 12
    mask_samples = round(cfg.model.mask_duration * cfg.model.sampling_rate)
    mask_length = mask_samples // biocodec_downsample_ratio
    print(f"Mask length per subsegment (encoded timesteps): {mask_length}")
    print(f"Loss computed on ALL RVQ levels (averaged)")

    # -------------------------------------------------------------------------
    # 5. Setup Logger (must be before callbacks to get run ID)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SETTING UP LOGGER")
    print("=" * 80)

    # Weights & Biases logger
    wandb_logger = WandbLogger(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"✓ WandB logger: project={cfg.logging.wandb_project}")

    # Get WandB run ID for checkpoint directory structure
    wandb_run_id = wandb_logger.experiment.id
    print(f"✓ WandB run ID: {wandb_run_id}")

    # Also log to CSV for local backup
    csv_logger = CSVLogger(save_dir="./logs", name=cfg.logging.experiment_name)
    loggers = [wandb_logger, csv_logger]
    print(f"✓ CSV logger: ./logs/{cfg.logging.experiment_name}")

    # -------------------------------------------------------------------------
    # 6. Setup Callbacks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SETTING UP CALLBACKS")
    print("=" * 80)

    callbacks = []

    # Create checkpoint directory: ./checkpoints/<experiment_name>/<wandb_run_id>/
    checkpoint_dir = Path(cfg.checkpoint.save_dir) / cfg.logging.experiment_name / wandb_run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint directory: {checkpoint_dir}")

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="criss-cross-transformer-{epoch:02d}-{step:06d}",
        every_n_train_steps=cfg.checkpoint.every_n_train_steps,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    print(f"✓ Model checkpoint: every_n_train_steps={cfg.checkpoint.every_n_train_steps}, save_top_k={cfg.checkpoint.save_top_k}")

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    print(f"✓ Learning rate monitor enabled")

    # Sampler verification callback (runs once on first epoch)
    sampler_verification = SamplerVerificationCallback()
    callbacks.append(sampler_verification)
    print(f"✓ Sampler verification callback enabled")

    # Note: Lightning automatically calls set_epoch() on samplers with this method
    # when using distributed training, so no custom callback is needed.

    # Optional: Early stopping
    # early_stop = EarlyStopping(
    #     monitor="val/loss",
    #     patience=10,
    #     mode="min",
    #     verbose=True,
    # )
    # callbacks.append(early_stop)

    # -------------------------------------------------------------------------
    # 7. Setup Trainer
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SETTING UP TRAINER")
    print("=" * 80)

    # Configure trainer with either max_epochs or max_steps
    trainer_kwargs = {
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "callbacks": callbacks,
        "logger": loggers,
        "gradient_clip_val": cfg.training.gradient_clip_val,
        "log_every_n_steps": cfg.logging.log_every_n_steps,
        "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
        "val_check_interval": cfg.trainer.val_check_interval,
        "deterministic": "warn" if hasattr(cfg, "seed") else False,
    }

    if num_epochs is not None:
        trainer_kwargs["max_epochs"] = num_epochs
    else:
        trainer_kwargs["max_steps"] = max_steps

    trainer = pl.Trainer(**trainer_kwargs)

    print(f"✓ Trainer configured")
    print(f"  Accelerator: {cfg.trainer.accelerator}")
    print(f"  Devices: {cfg.trainer.devices}")
    print(f"  Precision: {cfg.trainer.precision}")
    print(f"  Gradient clipping: {cfg.training.gradient_clip_val}")
    if num_epochs is not None:
        print(f"  Max epochs: {num_epochs}")
    else:
        print(f"  Max steps: {max_steps}")

    # -------------------------------------------------------------------------
    # 8. Start Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STARTING MULTI-DATASET TRAINING")
    print("=" * 80)

    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\nBest model checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("TRAINING INTERRUPTED")
        print("=" * 80)
        print("\nTraining was interrupted by user. Saving checkpoint...")

    finally:
        # Cleanup
        print("\n=== Cleanup ===")
        datamodule.teardown("fit")
        print("✓ Closed HDF5 file handles")

        # Finish WandB run
        if wandb_logger is not None:
            wandb_logger.experiment.finish()
            print("✓ Finished WandB logging")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
