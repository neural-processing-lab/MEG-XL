"""BioCodec MEG Trainer - Train from scratch on Armeni MEG data."""

import os, wandb, logging, warnings
from collections import defaultdict
from tqdm import tqdm

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import hydra, torch
import torch.optim as optim
import torch.distributed as dist
import torch._functorch.config
from torch.amp import GradScaler, autocast

from .datasets import MEGDATASET
from .objective import total_loss
from .scheduler import WarmupCosineLrScheduler
from .model import BioCodecModel
from .utils import (
    count_parameters,
    save_master_checkpoint,
    set_seed,
    start_distributed_train,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
torch.backends.nnpack.enabled = False
torch._functorch.config.donated_buffer = False


def train_one_epoch(
    epoch,
    optimizer,
    model,
    trainloader,
    config,
    scheduler,
    scaler=None,
):
    """
    Train for one epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number
    optimizer : torch.optim.Optimizer
        Optimizer
    model : BioCodecModel
        Model to train
    trainloader : DataLoader
        Training data loader
    config : OmegaConf
        Configuration object
    scheduler : WarmupCosineLrScheduler
        Learning rate scheduler
    scaler : GradScaler, optional
        Gradient scaler for mixed precision
    """
    model.train()
    data_length = len(trainloader)

    # Initialize variables to accumulate losses
    accumulated_losses = defaultdict(float)
    accumulated_loss_g = 0.0
    accumulated_loss_w = 0.0
    log_count = 0

    for idx, (signal_in, sr) in enumerate(trainloader):
        # signal_in shape: (B, C, T) where C=269 channels
        signal_in = signal_in.cuda(non_blocking=True)

        # Flatten channels into batch dimension: (B, C, T) -> (B*C, 1, T)
        B, C, T = signal_in.shape
        signal_in = signal_in.reshape(B * C, 1, T)  # (B*C, 1, T)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=config.common.amp):
            # Model processes each channel independently
            signal_out, loss_w, _ = model(signal_in)  # output: (B*C, 1, T) | loss_w: [1]

            # Compute reconstruction loss
            losses = total_loss(signal_in, signal_out, sr=sr)

        # Multiple backward calls --> retain_graph = True
        if config.common.amp:
            loss_g = losses["l_t"] * 0.1 + losses["l_f"] * 1.0 + loss_w
            scaler.scale(loss_g).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_g = torch.tensor([0.0], device="cuda", requires_grad=True)
            loss_g = losses["l_t"] * config.optimization.weights.l_t + \
                     losses["l_f"] * config.optimization.weights.l_f + loss_w
            loss_g.backward(retain_graph=True)

        loss_w.backward()
        optimizer.step()

        # Accumulate losses
        log_count += 1
        accumulated_loss_g += loss_g.item()
        accumulated_loss_w += loss_w.item()
        for k, l in losses.items():
            accumulated_losses[k] += l.item()

        scheduler.step()

        # Step logger
        if idx % config.common.log_interval == 0 or idx == data_length - 1:
            if config.distributed.data_parallel and dist.get_rank() != 0:
                continue  # Only log from the main process

            zidx = str(idx + 1).zfill(len(str(data_length)))
            global_step = (epoch - 1) * data_length + idx
            log_loss_g = accumulated_loss_g / log_count
            log_loss_w = accumulated_loss_w / log_count

            wandb.log(
                {
                    "global_step": global_step,
                    "Train/Loss": log_loss_g,
                    "Train/Loss_W": log_loss_w,
                }
            )
            for k, this_loss in accumulated_losses.items():
                wandb.log({f"Train/{k}": this_loss / log_count})
            logger.info(
                f"Epoch {epoch} {zidx}/{data_length}\tAvg loss: {log_loss_g:.4f}\tAvg loss_W: {log_loss_w:.6f}"
            )

            # Log codebook usage statistics
            quantizer = (
                model.module.quantizer
                if config.distributed.data_parallel
                else model.quantizer
            )
            for i, vq_layer in enumerate(quantizer.vq.layers):
                cluster_size = vq_layer._codebook.cluster_size
                dead_codes = (cluster_size < quantizer.threshold_ema_dead_code).sum()
                wandb.log(
                    {
                        f"RVQ/DeadCodes_layer{i}": dead_codes.item(),
                        "global_step": global_step,
                    }
                )

            # Reset accumulated losses
            accumulated_losses = defaultdict(float)
            accumulated_loss_g = 0.0
            accumulated_loss_w = 0.0
            log_count = 0


@torch.no_grad()
def test_one_epoch(epoch, model, testloader, config):
    """
    Test for one epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number
    model : BioCodecModel
        Model to evaluate
    testloader : DataLoader
        Test data loader
    config : OmegaConf
        Configuration object
    """
    model.eval()
    losses = defaultdict(float)
    n_samples = 0

    progress_bar = (
        tqdm(testloader)
        if not config.distributed.data_parallel or dist.get_rank() == 0
        else testloader
    )

    for test_sig, sr in progress_bar:
        # test_sig shape: (B, C, T) where C=269
        signal_in = test_sig.cuda(non_blocking=True)

        # Flatten channels into batch dimension: (B, C, T) -> (B*C, 1, T)
        B, C, T = signal_in.shape
        signal_in = signal_in.reshape(B * C, 1, T)

        # Forward pass (eval mode returns only reconstructed signal)
        signal_out = model(signal_in)  # (B*C, 1, T)

        # Compute loss (squeeze to match expected input shape [B*C, T])
        batch_losses = total_loss(signal_in.squeeze(1), signal_out.squeeze(1), sr=sr)

        # Accumulate losses
        for k, v in batch_losses.items():
            losses[k] += v.item() * B  # Weight by original batch size

        n_samples += B

    if config.distributed.data_parallel and dist.get_rank() != 0:
        return  # Only log from the main process

    # Average losses over all samples
    for k in losses.keys():
        losses[k] /= n_samples

    log_msg = f"| TEST | epoch: {epoch} | Loss: {sum(losses.values()):.4f}"
    for k, l in losses.items():
        wandb.log({f"Test/{k}": l, "epoch": epoch})
        log_msg += f" | {k}: {l:.4f}"

    logger.info(log_msg)


def perform_training(rank, world_size, config):
    """
    Main training function.

    Parameters
    ----------
    rank : int
        Process rank (for distributed training)
    world_size : int
        Total number of processes
    config : OmegaConf
        Configuration object
    """
    # Set up logger
    logger.handlers.clear()
    file_handler = logging.FileHandler(
        f"{config.checkpoint.save_folder}/biocodec_meg_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log"
    )
    formatter = logging.Formatter(
        "%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # Initialize datasets
    logger.info("Initializing MEG datasets...")
    trainset = MEGDATASET(config=config, mode="train")
    testset = MEGDATASET(config=config, mode="test")

    logger.info(f"Trainset size: {len(trainset)} samples")
    logger.info(f"Testset size: {len(testset)} samples")

    # Initialize model from scratch
    logger.info("Initializing BioCodec model from scratch...")
    model = BioCodecModel._get_optimized_model(
        sample_rate=config.model.sample_rate,
        causal=config.model.causal,
        model_norm=config.model.norm,
        signal_normalize=config.model.normalize,
        segment=eval(config.model.segment) if config.model.segment != "None" else None,
        name=config.model.name,
        n_q=config.model.n_q,
        q_bins=config.model.q_bins,
    )
    model = torch.compile(model)

    logger.info(model)
    logger.info(config)
    logger.info(f"\nModel Parameters: {count_parameters(model)}")
    logger.info(
        f"Model training: {model.training} | RVQ training: {model.quantizer.training}"
    )

    # Resume training (if specified)
    resume_epoch = 0
    if config.checkpoint.resume:
        assert config.checkpoint.checkpoint_path != "", "resume path is empty"
        model_checkpoint = torch.load(
            config.checkpoint.checkpoint_path, map_location="cpu"
        )
        model.load_state_dict(model_checkpoint["model_state_dict"])
        resume_epoch = model_checkpoint["epoch"]
        logger.info(f"Load model checkpoint, resume from {resume_epoch}")
        if resume_epoch >= config.common.max_epoch:
            raise ValueError(
                f"Resume epoch {resume_epoch} is larger than total epochs {config.common.max_epoch}"
            )

    # Set up distributed training
    train_sampler, test_sampler = None, None
    if config.distributed.data_parallel:
        if config.distributed.init_method == "tcp":
            distributed_init_method = "tcp://%s:%s" % (
                os.environ["MASTER_ADDR"],
                os.environ["MASTER_PORT"],
            )
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=distributed_init_method,
                rank=rank,
                world_size=world_size,
            )

        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        # Set distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    model = model.cuda()
    torch.autograd.set_detect_anomaly(True)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        sampler=train_sampler,
        drop_last=True,
        shuffle=(train_sampler is None),
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory,
        prefetch_factor=4,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        sampler=test_sampler,
        drop_last=True,
        shuffle=False,
        num_workers=config.datasets.num_workers // 2,
        pin_memory=config.datasets.pin_memory,
        prefetch_factor=4,
    )

    # Set up optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        [{"params": params, "lr": config.optimization.lr}], betas=(0.5, 0.9)
    )
    scheduler = WarmupCosineLrScheduler(
        optimizer,
        max_iter=config.common.max_epoch * len(trainloader),
        warmup_iter=config.optimization.warmup * len(trainloader),
        warmup_ratio=5e-4,
        warmup="linear",
    )
    scaler = GradScaler("cuda") if config.common.amp else None

    if config.checkpoint.resume and "scheduler_state_dict" in model_checkpoint.keys():
        optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(model_checkpoint["scheduler_state_dict"])
        logger.info(f"Load optimizer state_dict from epoch {resume_epoch}")

    # Wrap model for distributed training
    if config.distributed.data_parallel:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            find_unused_parameters=config.distributed.find_unused_parameters,
        )

    # Initialize wandb
    if not config.distributed.data_parallel or dist.get_rank() == 0:
        wandb.init(
            project="biocodec_meg",
            name=f"biocodec_meg_bs{config.datasets.batch_size}_lr{config.optimization.lr}",
        )
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        wandb.define_metric("Train/*", step_metric="global_step")
        wandb.define_metric("Test/*", step_metric="epoch")

    # Training loop
    start_epoch = max(1, resume_epoch + 1)
    logger.info(f"Starting training from epoch {start_epoch}...")

    # Initial test
    test_one_epoch(0, model, testloader, config)

    for epoch in range(start_epoch, config.common.max_epoch + 1):
        train_one_epoch(
            epoch,
            optimizer,
            model,
            trainloader,
            config,
            scheduler,
            scaler,
        )
        if epoch % config.common.test_interval == 0:
            test_one_epoch(epoch, model, testloader, config)

        # Save checkpoint
        if epoch % config.common.save_interval == 0:
            model_to_save = model.module if config.distributed.data_parallel else model
            if not config.distributed.data_parallel or dist.get_rank() == 0:
                save_master_checkpoint(
                    epoch,
                    model_to_save,
                    optimizer,
                    scheduler,
                    f"{config.checkpoint.save_location}epoch{epoch}.pt",
                )

    # Clean up
    if config.distributed.data_parallel:
        dist.destroy_process_group()
    if wandb.run is not None:
        wandb.finish()

    logger.info("Training complete!")


@hydra.main(config_path="configs", config_name="meg_config", version_base=None)
def main(config):
    """Main entry point."""
    os.makedirs(config.checkpoint.save_folder, exist_ok=True)
    torch.backends.cudnn.enabled = False

    os.environ["CUDA_VISIBLE_DEVICES"] = config.common.gpus
    if config.distributed.data_parallel:
        start_distributed_train(
            perform_training,
            config.distributed.world_size,
            config,
            dist_init_method=config.distributed.init_method,
        )
    else:
        perform_training(0, 1, config)  # Single GPU train


if __name__ == "__main__":
    main()
