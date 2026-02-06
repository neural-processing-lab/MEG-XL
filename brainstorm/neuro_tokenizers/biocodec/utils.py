import typing as tp, random
import os, torch, numpy as np
import torch.multiprocessing as mp


def compute_class_weights(labels, num_classes):
    annots = torch.from_numpy(np.array(labels))
    counts = torch.bincount(annots, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts + 1e-8)
    return weights.cuda()


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset : offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset : offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_master_checkpoint(epoch, model, optimizer, scheduler, ckpt_name):
    """save master checkpoint

    Args:
        epoch (int): epoch number
        model (nn.Module): model
        optimizer (optimizer): optimizer
        scheduler (_type_): _description_
        ckpt_name (str): checkpoint name
    """
    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state_dict, ckpt_name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def start_distributed_train(train_fn, world_size, config, dist_init_method="tcp"):
    """
    Starts distributed training using torch.multiprocessing.spawn

    Args:
        train_fn: Training function, must accept (rank, world_size, config, dist_init_method)
        world_size: Total number of processes (1 per available GPU)
        config: Hydra or argparse config
        dist_init_method: e.g., tcp://IP:PORT or file://<path>
    """
    # Ensure safe CUDA context creation
    torch.multiprocessing.set_start_method("spawn", force=True)

    # For multi-node or multi-GPU, be explicit about master envs if using TCP
    if dist_init_method.startswith("tcp"):
        os.environ["MASTER_ADDR"] = config.distributed.master_addr  # e.g., 127.0.0.1
        os.environ["MASTER_PORT"] = str(config.distributed.master_port)  # e.g., 23456

    # Let TORCH_DISTRIBUTED_DEBUG flow through for troubleshooting
    if config.distributed.torch_distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    mp.spawn(
        fn=train_fn,
        args=(world_size, config, dist_init_method),
        nprocs=world_size,
        join=True,
    )
