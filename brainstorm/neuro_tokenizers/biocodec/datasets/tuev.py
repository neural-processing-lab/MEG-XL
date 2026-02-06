import torch, h5py, os
import numpy as np, hydra
from torch.utils.data import Dataset
from tqdm import tqdm

from ..model import BioCodecModel


class TUEVDataset(Dataset):
    def __init__(
        self,
        config,
        mode="train",
        splits=[0, 1, 2, 3, 4],
        transform=None,
    ):
        self.config = config
        self.mode = mode
        self.splits = splits
        self.transform = transform
        self.sample_rate = self.config.model.sample_rate
        np.random.seed(self.config.common.seed)

        self.proc_path = os.path.join(
            self.config.datasets.proc_path,
            f"{self.mode}.h5",
        )
        self.codes_path = os.path.join(
            self.config.datasets.codes_path,
            f"tuev-{self.mode}.h5",
        )
        if not os.path.exists(self.codes_path):
            os.makedirs(self.config.datasets.codes_path, exist_ok=True)
            self.preprocess(
                proc_path=self.proc_path,
                codes_path=self.codes_path,
                codec_path=self.config.common.codec_path,
            )
        with h5py.File(self.codes_path, "r") as h5_file:
            all_data = h5_file["eeg_input"][:]
            all_labels = h5_file["eeg_label"][:]

        # split the dataset
        if self.mode == "train":
            total_len, split_size = len(all_data), len(all_data) // 5
            self.data, self.labels = [], []
            for split in self.splits:
                start = split * split_size
                end = start + split_size if split < 4 else total_len
                self.data.append(all_data[start:end])
                self.labels.append(all_labels[start:end])

            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0).astype(np.int64)
        else:
            self.data = all_data
            self.labels = all_labels.astype(np.int64)

    def preprocess(self, proc_path, codes_path, codec_path, bs=256):
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
        with h5py.File(proc_path, "r") as proc_h5_file:
            proc_data = proc_h5_file["eeg_input"][:]
            proc_labels = proc_h5_file["eeg_label"][:]

        for i, signal in tqdm(
            enumerate(proc_data), total=len(proc_data), desc="Preprocessing: "
        ):
            # bandpass filter between 0.5-40Hz
            from scipy.signal import butter, filtfilt

            def butter_bandpass(lowcut, highcut, fs, order=5):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype="band")
                return b, a

            def bandpass_filter(data, lowcut, highcut, fs, order=5):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = filtfilt(b, a, data)
                return y

            for ch in range(signal.shape[0]):
                signal[ch] = bandpass_filter(
                    signal[ch], 0.5, 40, self.sample_rate, order=5
                )
            
            # normalize each channel
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-8)
            proc_data[i] = signal

        codes_data = []
        n_q = self.config.pretrained.n_q
        for i in tqdm(range(0, len(proc_data), bs)):
            # isolate a batch of size B
            seg = torch.from_numpy(proc_data[i : i + bs])
            B = seg.size(0)
            # pin memory and create the channel dimension
            seg = seg.pin_memory().unsqueeze(2).float()
            # collapse batch and channel for the encoder
            seg = seg.view(B * 16, 1, -1).cuda(non_blocking=True)
            # encode to discrete codes
            with torch.no_grad():
                enc_frames = model.encode(seg)
            # extract codes in shape [B, C=16, n_q, T]
            codes = torch.cat([c.view(B, 16, n_q, -1) for c, _ in enc_frames], dim=-1)
            # permute to [B, C=16, T, n_q] for saving
            codes_data.append(codes.permute(0, 1, 3, 2).cpu())

        codes_data = torch.cat(codes_data, dim=0)
        codes_data = codes_data.cpu().numpy()

        # save the codes to h5 in codes_path
        if os.path.exists(codes_path):
            os.remove(codes_path)
        with h5py.File(codes_path, "w") as h5_file:
            h5_file.create_dataset("eeg_input", data=codes_data)
            h5_file.create_dataset("eeg_label", data=proc_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate

        eeg = torch.from_numpy(self.data[idx])
        if self.transform:
            eeg = self.transform(eeg)

        return eeg, int(self.labels[idx]), "None"


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False
    dataset = TUEVDataset(
        config,
        mode="test",
        splits=[0, 1, 2, 3, 4],
        transform=None,
    )
    print("Dataset length:", len(dataset))
    
    eeg_in, label, name = dataset[10]
    print(eeg_in.shape, label, name)
    print(eeg_in[7, ::5])

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

