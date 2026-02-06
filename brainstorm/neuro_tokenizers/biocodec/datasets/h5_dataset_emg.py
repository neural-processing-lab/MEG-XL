import torch, h5py, os
from torch.utils.data import Dataset
from torchaudio.functional import resample


class HDF5EMG(Dataset):
    def __init__(self, config, transform=None, mode="dev"):
        self.h5_files = None  # Open per worker
        self.h5_size = 2000
        self.sample_rate = config.model.sample_rate
        self.transform = transform
        self.mode = mode
        self.fpaths = [
            config.datasets.path + s
            for s in os.listdir(config.datasets.path)
            if mode in s
        ]

    def __len__(self):
        return len(self.fpaths) * self.h5_size

    def __getitem__(self, idx, sample_rate=None):
        if self.h5_files is None:
            self.h5_files = [h5py.File(p, "r") for p in self.fpaths]
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
            self.data = [
                self.h5_files[i]["emg_input"] for i in range(len(self.h5_files))
            ]

        if sample_rate is None:
            sample_rate = self.sample_rate

        split_idx = idx // self.h5_size
        idx = idx % self.h5_size
        waveform = self.data[split_idx][idx]
        waveform = torch.from_numpy(waveform)

        # clip values for smoothing outlier effects
        waveform = torch.clamp(waveform, max=10, min=-10)
        ch_idx = torch.randint(0, waveform.shape[0], (1,)).item()
        waveform = waveform[ch_idx]

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


