import hydra, os, numpy as np
import torch, logging, warnings
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    average_precision_score,
    balanced_accuracy_score,
)

from biocodec.datasets import TUABDataset, SleepEDFDataset  
from biocodec.ft_model import BioCodecFT
from biocodec.utils import count_parameters, set_seed

warnings.filterwarnings("ignore")
logger = logging.getLogger("ft_eval")
logger.setLevel(logging.INFO)


def aggregate_fixed_length(
    names, logits, probas, labels, win_len_sec=5, agg_len_sec=10, hop_sec=5
):
    """
    Aggregate contiguous windows within each recording
    into fixed-length segments (e.g., 10, 12, or 60 s).
    Returns arrays of aggregated probas and labels.
    """
    assert (
        agg_len_sec % win_len_sec == 0
    ), "agg_len_sec must be multiple of window length"
    k = agg_len_sec // win_len_sec
    out_l, out_p, out_y, out_id = [], [], [], []

    for rid in np.unique(names):
        # contiguous segments with same id
        idx = np.where(names == rid)[0]
        i = 0
        N = len(idx)
        step = max(1, hop_sec // win_len_sec)
        while i + k - 1 < N:
            span = idx[i : i + k]
            out_id.append(rid)
            out_l.append(logits[span].mean(0))
            out_p.append(probas[span].mean())
            out_y.append(int(labels[span].mean()))
            i += step
    return {
        "ids": np.asarray(out_id),
        "logits": np.asarray(out_l),
        "probas": np.asarray(out_p),
        "labels": np.asarray(out_y, int),
    }


def aggregate_session(names, logits, probas, labels):
    out_l, out_p, out_y, out_id = [], [], [], []
    for rid in np.unique(names):
        idx = np.where(names == rid)[0]
        out_id.append(rid)
        out_l.append(logits[idx].mean(0))
        out_p.append(float(np.mean(probas[idx])))
        out_y.append(int(labels[idx].mean()))
    return {
        "ids": np.asarray(out_id),
        "logits": np.asarray(out_l),
        "probas": np.asarray(out_p),
        "labels": np.asarray(out_y, int),
    }


@torch.no_grad()
def evaluate(config, model, testloader):
    model.eval()

    agg = config.eval.aggregate
    hop = config.eval.hop_sec

    all_logits, all_preds = [], []
    all_labels, all_names = [], []
    for x, y, rid in tqdm(testloader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = model(x)  # [B, n_classes]
        if logits.shape[1] == 2:
            # binary: keep positive-class logit
            all_logits.append(logits[:, 1].detach().cpu())
        else:
            # multi-class: keep full logits
            all_logits.append(logits.detach().cpu())

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(y.detach().cpu())
        all_names.extend(rid)

    if agg == "session":
        res_dict = aggregate_session(
            np.array(all_names),
            torch.cat(all_logits).numpy(),
            (
                torch.cat(all_logits).numpy()
                if all_logits[0].ndim == 1
                else torch.softmax(torch.cat(all_logits), dim=1)[:, 1].numpy()
            ),
            torch.cat(all_labels).numpy(),
        )
    elif isinstance(agg, int) and agg > 0:
        res_dict = aggregate_fixed_length(
            np.array(all_names),
            torch.cat(all_logits).numpy(),
            (
                torch.sigmoid(torch.cat(all_logits)).numpy()
                if all_logits[0].ndim == 1
                else torch.softmax(torch.cat(all_logits), dim=1)[:, 1].numpy()
            ),
            torch.cat(all_labels).numpy(),
            win_len_sec=5,
            agg_len_sec=agg,
            hop_sec=hop,
        )
    else:
        res_dict = {
            "ids": np.array(all_names),
            "logits": torch.cat(all_logits).numpy(),
            "probas": (
                torch.sigmoid(torch.cat(all_logits)).numpy()
                if all_logits[0].ndim == 1
                else torch.softmax(torch.cat(all_logits), dim=1)[:, 1].numpy()
            ),
            "labels": torch.cat(all_labels).numpy(),
        }

    if all_logits[0].ndim == 1:
        # binary
        probas = res_dict["probas"]
        y_true = res_dict["labels"]
        y_pred = (res_dict["probas"] >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_true, probas),
            "pr_auc": average_precision_score(y_true, probas),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
    else:
        # multi-class
        probas = torch.softmax(torch.tensor(res_dict["logits"]), dim=1).numpy()
        y_true = res_dict["labels"]
        y_pred = np.argmax(res_dict["logits"], axis=1)
        metrics = {
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "cohen's_kappa": cohen_kappa_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }

    logger.info(" ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    return {**metrics}


