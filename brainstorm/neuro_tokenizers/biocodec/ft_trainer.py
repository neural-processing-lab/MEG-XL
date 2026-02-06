import os, wandb, numpy as np
import logging, warnings
from tqdm import tqdm

import hydra, torch
import torch.nn as nn
from torch.utils import data
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    average_precision_score,
    balanced_accuracy_score,
)

from biocodec.datasets import (
    TUABDataset,
    EEGMMIDataset,
    BCI2aDataset,
    KaggleERN,
    SleepEDFDataset,
    NinaproDataset,
    MCSDataset,
)
from biocodec.ft_model import BioCodecFT
from biocodec.ft_base import BaselineCNN
from biocodec.scheduler import WarmupCosineLrScheduler
from biocodec.ft_eval import evaluate
from biocodec.utils import *

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_one_epoch(
    epoch,
    optimizer,
    model,
    criterion,
    trainloader,
    config,
    scheduler,
    scaler=None,
):
    """
    epoch (int): current epoch
    optimizer (_type_) : generator optimizer
    model (_type_): generator model
    trainloader (_type_): train dataloader
    config (_type_): hydra config file
    scheduler (_type_): adjust generate model learning rate
    """
    model.train()
    data_length = len(trainloader)

    accumulated_loss, log_count = 0.0, 0
    for idx, (signal_in, y_true, _) in enumerate(trainloader):
        signal_in = signal_in.cuda(non_blocking=True)  # [B, C, T, n_books]
        y_true = y_true.cuda(non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=config.common.amp):
            logits = model(signal_in)  # [B, n_classes]
            this_loss = criterion(logits, y_true)

        if config.common.amp:
            scaler.scale(this_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            this_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        accumulated_loss += this_loss.item()
        log_count += 1

        if idx % config.common.log_interval == 0 or idx == data_length - 1:
            zidx = str(idx + 1).zfill(len(str(data_length)))
            global_step = (epoch - 1) * data_length + idx
            avg_loss = accumulated_loss / log_count
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "global_step": global_step,
                    "Train/Loss": avg_loss,
                    "Train/lr": current_lr,
                }
            )
            logger.info(f"Epoch {epoch} {zidx}/{data_length}\tCE loss: {avg_loss:.4f}")
            accumulated_loss, log_count = 0.0, 0


@torch.no_grad()
def test_one_epoch(epoch, model, criterion, testloader, is_test=False):
    model.eval()
    all_losses = 0.0
    all_preds = []
    all_logits = []
    all_labels = []
    all_names = []

    for test_sig, test_y, test_id in tqdm(testloader):
        signal_in = test_sig.cuda(non_blocking=True)
        test_y = test_y.cuda(non_blocking=True)
        logits = model(signal_in)  # [B, n_classes]
        this_loss = criterion(logits, test_y).item()

        all_losses += this_loss * len(test_sig)
        all_preds.append(torch.argmax(logits, dim=1).cpu())
        all_labels.append(test_y.cpu())
        all_names.extend(test_id)

        if logits.size(1) == 2:
            all_logits.append(logits[:, 1].cpu())
        else:
            logits = torch.softmax(logits, dim=1)
            all_logits.append(logits.cpu())

    # Concatenate all results
    all_logits = torch.cat(all_logits)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Metrics over the whole dataset
    n_classes = logits.size(1)
    metrics = {
        "roc_auc": roc_auc_score(all_labels, all_logits, multi_class="ovr"),
        "pr_auc": average_precision_score(all_labels, all_logits, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    } if n_classes == 2 else {
        "cohen's_kappa": cohen_kappa_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    }

    # Logging
    name = "Test" if is_test else "Valid"
    log_msg = f"| {name} | epoch: {epoch}"

    avg = all_losses / len(testloader.dataset)
    wandb.log({"epoch": epoch, f"{name}/Loss": avg})
    log_msg += f" | Loss: {avg:.4f}"

    for k, v in metrics.items():
        wandb.log({"epoch": epoch, f"{name}/{k}": v})
        log_msg += f" | {k}: {v:.4f}"

    logger.info(log_msg)
    return balanced_accuracy_score(all_labels, all_preds)


def perform_training(config):
    # set logger
    logger.handlers.clear()
    file_handler = logging.FileHandler(
        f"{config.checkpoint.save_folder}/ft_biocodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log"
    )
    formatter = logging.Formatter(
        "%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s"
    )
    file_handler.setFormatter(formatter)

    # print on screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # set datasets
    if config.datasets.name == "TUAB":
        trainset = data.ConcatDataset(
            [
                TUABDataset(
                    config=config,
                    is_test=False,
                    split_num=i,
                    continuous=config.datasets.continuous,
                )
                for i in config.datasets.train_folds
            ]
        )
        validset = TUABDataset(
            config=config,
            is_test=False,
            split_num=config.datasets.valid_folds[0],
            continuous=config.datasets.continuous,
        )
        testset = TUABDataset(
            config=config, is_test=True, continuous=config.datasets.continuous
        )

    elif config.datasets.name == "BCIIV2A":
        trainset = BCI2aDataset(config=config, subjects=[1, 2, 3, 4, 5])
        validset = BCI2aDataset(config=config, subjects=[6, 7])
        testset = BCI2aDataset(config=config, subjects=[8, 9])

    elif config.datasets.name == "Ninapro":
        db = "DB2"
        trainset = NinaproDataset(
            config=config, task_type=db, folds=config.datasets.train_folds
        )
        validset = NinaproDataset(
            config=config, task_type=db, folds=config.datasets.valid_folds
        )
        testset = NinaproDataset(
            config=config, task_type=db, folds=config.datasets.test_folds
        )

    elif config.datasets.name == "MCS":
        trainset = MCSDataset(config=config, folds=config.datasets.train_folds)
        validset = MCSDataset(config=config, folds=config.datasets.valid_folds)
        testset = MCSDataset(config=config, folds=config.datasets.test_folds)

    elif config.datasets.name == "KaggleERN":
        trainset = KaggleERN(
            config=config,
            mode="train",
            splits=config.datasets.train_folds,
            sr=config.model.sample_rate,
        )
        validset = KaggleERN(
            config=config,
            mode="train",
            splits=config.datasets.valid_folds,
            sr=config.model.sample_rate,
        )
        testset = KaggleERN(
            config=config,
            mode="test",
            splits=[0],
            sr=config.model.sample_rate,
        )

    elif config.datasets.name == "SleepEDF":
        trainset = SleepEDFDataset(
            config=config,
            splits=config.datasets.train_folds,
            sr=config.model.sample_rate,
        )
        validset = SleepEDFDataset(
            config=config,
            splits=config.datasets.valid_folds,
            sr=config.model.sample_rate,
        )
        testset = SleepEDFDataset(
            config=config,
            splits=config.datasets.test_folds,
            sr=config.model.sample_rate,
        )
        
    elif config.datasets.name == "MMI":
        this_task = "eyes_open_closed"
        trainset = EEGMMIDataset(
            config=config,
            task_type=this_task,
            folds=config.datasets.train_folds,
        )
        validset = EEGMMIDataset(
            config=config,
            task_type=this_task,
            folds=config.datasets.valid_folds,
        )
        testset = EEGMMIDataset(
            config=config,
            task_type=this_task,
            folds=config.datasets.test_folds,
        )

    elif config.datasets.name == "TUEV":
        trainset = TUEVDataset(
            config,
            mode="train",
            splits=config.datasets.train_folds,
        )
        validset = TUEVDataset(
            config,
            mode="train",
            splits=config.datasets.valid_folds,
        )
        testset = TUEVDataset(
            config,
            mode="test",
            splits=config.datasets.test_folds,
        )

    else:
        raise NotImplementedError(f"Dataset {config.datasets.name} not implemented")

    # initialize downstream model
    model = BioCodecFT(
        config=config,
        C=config.model.n_channels,
        T=config.model.n_timesteps,
        num_classes=config.model.n_classes,
        d_model=config.model.d_model,
        n_heads=config.model.num_heads,
        n_layers=config.model.num_layers,
        n_books=config.pretrained.n_q,
        n_used=config.model.n_used,
        n_bins=config.pretrained.q_bins,
        continuous=config.datasets.continuous,
        is_emg=config.common.is_emg,
    )
    torch.compile(model)

    logger.info(model)
    logger.info(config)
    logger.info(f"Model Parameters: {count_parameters(model)}")

    # resume training
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
                f"Resume epoch {resume_epoch} is larger than total epochs {config.common.epochs}"
            )

    model.cuda()
    model = torch.compile(model)
    torch.autograd.set_detect_anomaly(True)

    trainloader = data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        pin_memory=config.datasets.pin_memory,
        num_workers=config.datasets.num_workers,
        drop_last=True,
        shuffle=True,
    )
    validloader = data.DataLoader(
        validset,
        batch_size=config.datasets.batch_size,
        pin_memory=config.datasets.pin_memory,
        num_workers=config.datasets.num_workers,
        shuffle=False,
    )
    testloader = data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        pin_memory=config.datasets.pin_memory,
        shuffle=False,
    )
    logger.info(f"There are {len(trainset)} training samples")
    logger.info(f"There are {len(validset)} validation samples")
    logger.info(f"There are {len(testset)} testing samples")

    # set optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.5, 0.9),
        lr=config.optimization.lr,
        weight_decay=config.optimization.decay,
    )
    scheduler = WarmupCosineLrScheduler(
        optimizer,
        max_iter=config.common.max_epoch * len(trainloader),
        warmup_iter=config.common.max_epoch * len(trainloader) / 5,
        warmup_ratio=1e-4,
        warmup="linear",
    )
    scaler = GradScaler() if config.common.amp else None

    annots = (
        np.concatenate([trainset.datasets[i].labels for i in range(1)])
        if isinstance(trainset, data.ConcatDataset)
        else list(trainset.labels)
    )
    cweights = compute_class_weights(annots, config.model.n_classes)
    criterion = nn.CrossEntropyLoss(
        reduction="mean",
        weight=cweights,
        label_smoothing=config.optimization.label_smoothing,
    )
    logger.info("Class weights: %s", cweights)

    if config.checkpoint.resume and "scheduler_state_dict" in model_checkpoint.keys():
        optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(model_checkpoint["scheduler_state_dict"])
        logger.info(f"Load optimizer and disc_optimizer state_dict from {resume_epoch}")

    wandb.init(
        project="ft_biocodec",
        name=f"ft_5M_bs{config.datasets.batch_size}_lr{config.optimization.lr}",
    )
    wandb.define_metric("epoch")
    wandb.define_metric("global_step")
    wandb.define_metric("Train/*", step_metric="global_step")
    wandb.define_metric("Valid/*", step_metric="epoch")
    wandb.define_metric("Test/*", step_metric="epoch")

    best_val_bac = 0.0
    best_ckpt_path = None
    start_epoch = max(1, resume_epoch + 1)

    test_one_epoch(0, model, criterion, validloader)
    for epoch in range(start_epoch, config.common.max_epoch + 1):
        train_one_epoch(
            epoch,
            optimizer,
            model,
            criterion,
            trainloader,
            config,
            scheduler,
            scaler,
        )
        if epoch % config.common.test_interval == 0:
            val_bac = test_one_epoch(epoch, model, criterion, validloader)
            if val_bac > best_val_bac:
                best_val_bac = val_bac
                best_ckpt_path = (
                    f"{config.checkpoint.save_location}best_epoch{epoch}.pt"
                )
                save_master_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_ckpt_path,
                )

        # save checkpoint and epoch
        if epoch % config.common.save_interval == 0:
            save_master_checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                f"{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt",
            )

    # load best checkpoint
    if best_ckpt_path is not None:
        logger.info(f"Loading best model from {best_ckpt_path}")
        best = torch.load(best_ckpt_path, map_location="cuda")
        model.load_state_dict(best["model_state_dict"])
    else:
        logger.warning("No best‐model checkpoint found; using last epoch instead")

    # test the model
    logger.info("Testing the model")
    # test_one_epoch(epoch, model, criterion, testloader, is_test=True)
    evaluate(config, model, testloader)
    if wandb.run is not None:
        wandb.finish()


@hydra.main(config_path="configs", config_name="ft_config")
def main(config):
    os.makedirs(config.checkpoint.save_folder, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.common.gpus
    torch.backends.cudnn.enabled = False
    perform_training(config)


if __name__ == "__main__":
    main()
