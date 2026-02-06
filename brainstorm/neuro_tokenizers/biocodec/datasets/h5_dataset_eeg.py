import torch, h5py
from torch.utils.data import Dataset
from torchaudio.functional import resample


class HDF5EEG(Dataset):
    def __init__(self, config, transform=None, mode="train"):
        self.h5_files = None  # Open per worker
        self.fixed_len = config.datasets.fixed_length

        if mode == "test":
            self.fixed_len = self.fixed_len // 5
            self.n_splits = int(max(1, self.fixed_len // 1e6))
            self.datapaths = [
                config.datasets.test_path + f"split_{19-i}.h5"
                for i in range(self.n_splits)
            ]
        elif mode == "train":
            self.n_splits = int(max(1, self.fixed_len // 1e6))
            self.datapaths = [
                config.datasets.train_path + f"split_{8+i}.h5"
                for i in range(self.n_splits)
            ]
        else:
            raise ValueError("Invalid mode. Must be either 'train' or 'test'.")

        self.sample_rate = config.model.sample_rate
        self.transform = transform

    def __len__(self):
        return self.fixed_len

    def __getitem__(self, idx, sample_rate=None):
        if self.h5_files is None:
            self.h5_files = [h5py.File(p, "r") for p in self.datapaths]
            self.h5_files = [
                h5py.File(
                    p,
                    "r",
                    libver="latest",
                    rdcc_nbytes=128 * 1024**2,
                    rdcc_w0=0.75,
                    rdcc_nslots=1 << 17,
                )
                for p in self.datapaths
            ]
            self.data = [self.h5_files[i]["eeg_input"] for i in range(self.n_splits)]

        if sample_rate is None:
            sample_rate = self.sample_rate

        split_idx = idx // (self.fixed_len // self.n_splits)
        idx = idx % (self.fixed_len // self.n_splits)
        waveform = self.data[split_idx][idx]
        waveform = torch.from_numpy(waveform)

        # clip values for smoothing outlier effects
        waveform = torch.clamp(waveform, max=10, min=-10)

        if sample_rate != self.sample_rate:
            waveform = resample(waveform, sample_rate, self.sample_rate)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate

    def __del__(self):
        # Close the H5 when finished
        if self.h5_files is not None:
            for h5_file in self.h5_files:
                h5_file.close()
