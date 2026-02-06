import torch, h5py, os
import numpy as np, hydra
from torch.utils.data import Dataset
from tqdm import tqdm

from ..model import BioCodecModel


class TUABDataset(Dataset):
    def __init__(
        self,
        config,
        is_test=False,
        split_num=0,
        transform=None,
    ):
        self.config = config
        self.mode = "test" if is_test else "train"
        self.split_num = split_num
        self.transform = transform
        self.sample_rate = self.config.model.sample_rate
        np.random.seed(self.config.common.seed)

        self.proc_path = os.path.join(
            self.config.datasets.proc_path,
            f"tuab-processed-{self.mode}-{self.split_num}.h5",
        )
        self.codes_path = os.path.join(
            self.config.datasets.codes_path,
            f"tuab-codes-{self.mode}-{self.split_num}.h5",
        )

        if not os.path.exists(self.codes_path):
            os.makedirs(self.config.datasets.codes_path, exist_ok=True)
            self.preprocess(
                proc_path=self.proc_path,
                codes_path=self.codes_path,
                codec_path=self.config.common.codec_path,
            )
            
        self.h5_file = h5py.File(self.codes_path, "r")
        try:
            self.data = self.h5_file["eeg_input"]
        except KeyError:
            self.data = self.h5_file["eeg_data"]
        self.labels = self.h5_file["eeg_label"]
        self.names = self.h5_file["eeg_name"]
        self.names = np.array([n.decode("utf-8") for n in self.names])

    def preprocess(self, proc_path, codes_path, codec_path, bs=128):
        # initialize the codec
        model = BioCodecModel._get_optimized_model(
            sample_rate=self.sample_rate,
            causal=self.config.pretrained.causal,
            model_norm=self.config.pretrained.norm,
            signal_normalize=self.config.pretrained.normalize,
            segment=eval(self.config.pretrained.segment),
            name=self.config.pretrained.name,
            n_q=self.config.pretrained.n_q,
            q_bins=self.config.pretrained.q_bins,
        )
        model = torch.compile(model, dynamic=True)
        # load state dict from codec path
        checkpoint = torch.load(codec_path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.cuda()
        model.eval()

        # load segmented EEG data
        proc_h5_file = h5py.File(proc_path, "r")
        proc_data = proc_h5_file["eeg_input"]
        proc_labels = proc_h5_file["eeg_label"]
        proc_names = proc_h5_file["eeg_name"]

        codes_data = []
        n_q = self.config.pretrained.n_q
        for i in tqdm(range(0, len(proc_data), bs)):
            # isolate a batch of size B
            seg = torch.from_numpy(proc_data[i : i + bs])
            B = seg.size(0)
            # pin memory and create the channel dimension
            seg = seg.pin_memory().unsqueeze(1).float()
            # collapse batch and channel for the encoder
            seg = seg.view(B * 21, 1, -1).cuda(non_blocking=True)
            # encode to discrete codes
            with torch.no_grad():
                enc_frames = model.encode(seg)
            # extract codes in shape [B, C=21, n_q, T]
            codes = torch.cat([c.view(B, 21, n_q, -1) for c, _ in enc_frames], dim=-1)
            # permute to [B, C=21, T, n_q] for saving
            codes_data.append(codes.permute(0, 1, 3, 2).cpu())

        codes_data = torch.cat(codes_data, dim=0)
        codes_data = codes_data.cpu().numpy()

        # save the codes to h5 in codes_path
        if os.path.exists(codes_path):
            os.remove(codes_path)
        with h5py.File(codes_path, "w") as h5_file:
            h5_file.create_dataset("eeg_input", data=codes_data)
            h5_file.create_dataset("eeg_label", data=proc_labels)
            h5_file.create_dataset("eeg_name", data=proc_names)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate

        eeg = torch.from_numpy(self.data[idx])
        if self.transform:
            eeg = self.transform(eeg)

        return eeg, self.labels[idx], self.names[idx]

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False
    dataset = TUABDataset(
        config,
        is_test=False,
        split_num=0,
        transform=None,
    )
    print("Dataset length:", len(dataset))
    eeg_in, label, name = dataset[0]
    print(eeg_in.shape, label, name)
    print(eeg_in[17, ::4])  # print the 18th channel


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
