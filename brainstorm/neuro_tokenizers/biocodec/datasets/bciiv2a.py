import numpy as np, torch
import hydra, mne, os, h5py
from scipy.io import loadmat
from torch.utils.data import Dataset

from ..model import BioCodecModel


class BCI2aDataset(Dataset):
    """
    BCI Competition IV-2a motor imagery dataset.
    """

    MAPPING = {
        "276": 10,  # idle EEG (eyes open)
        "277": 11,  # idle EEG (eyes closed)
        "768": 0,  # start of trial
        "769": 1,
        "770": 2,
        "771": 3,
        "772": 4,
        "783": -1,  # unknown onset
        "1023": -100,  # rejected trial
        "1072": 100,  # eye movements
        "32766": 1000,  # start of new run
    }

    def __init__(
        self,
        root: str = "/PATH/TO/BCICIV_2a/",
        config: dict = None,
        sr: int = 250,
        mode: str = "T",  # T or E
        pre_sec: float = 1.0,
        post_sec: float = 4.0,
        normalize: bool = True,
        subjects: list = list(range(1, 10)),  # all subjects
        transform=None,
    ):
        super().__init__()
        assert mode in ("T", "E")
        self.sr = sr
        self.config = config
        self.root = root
        self.mode = mode
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.normalize = normalize
        self.transform = transform

        self.codes_path = self.config.datasets.codes_path
        if not os.path.exists(self.codes_path + f"BCI2a_{self.mode}.h5"):
            self.preprocess(
                codes_path=self.codes_path + f"BCI2a_{self.mode}.h5",
                codec_path=self.config.common.codec_path,
            )

        self.h5_file = h5py.File(self.codes_path + f"BCI2a_T.h5", "r")
        self.data = list(self.h5_file["eeg_input"])
        self.labels = list(self.h5_file["eeg_label"])
        self.names = self.h5_file["eeg_name"]
        self.names = np.array([n.decode("utf-8") for n in self.names])

        self.h5_file.close()
        self.h5_file = h5py.File(self.codes_path + f"BCI2a_E.h5", "r")
        self.data = np.concatenate((self.data, self.h5_file["eeg_input"]), axis=0)
        self.labels = np.concatenate((self.labels, self.h5_file["eeg_label"]), axis=0)
        self.names = np.concatenate(
            (
                self.names,
                np.array([n.decode("utf-8") for n in self.h5_file["eeg_name"]]),
            ),
            axis=0,
        )
        self.h5_file.close()

        # select only the specified subject
        subject_str = [f"A{str(s).zfill(2)}" for s in subjects]
        mask = np.isin(self.names, subject_str)
        self.data = self.data[mask]
        self.labels = self.labels[mask]
        self.names = self.names[mask]

    def preprocess(self, codes_path, codec_path):
        # initialize the codec
        model = BioCodecModel._get_optimized_model(
            sample_rate=self.config.model.sample_rate,
            causal=self.config.pretrained.causal,
            model_norm=self.config.pretrained.norm,
            signal_normalize=self.config.pretrained.normalize,
            segment=eval(self.config.pretrained.segment),
            name=self.config.pretrained.name,
            n_q=self.config.pretrained.n_q,
            q_bins=self.config.pretrained.q_bins,
        )
        # load state dict from codec path
        checkpoint = torch.load(codec_path, map_location="cuda")
        checkpoint["model_state_dict"] = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.cuda()
        model.eval()

        # load EEG data
        gdf_files = sorted(
            [
                f
                for f in os.listdir(self.root)
                if f.endswith(".gdf") and f[3] == self.mode
            ]
        )

        codes_data = []
        proc_labels = []
        proc_names = []

        for fname in gdf_files:
            rec_id = fname[:3]
            fpath = os.path.join(self.root, fname)

            # Load raw recording
            raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)
            # Pick only 22 EEG channels
            non_eog = [ch for ch in raw.ch_names if "EOG" not in ch.upper()]
            picks = mne.pick_channels(raw.ch_names, include=non_eog)
            raw.pick(picks, verbose=False)

            # Extract events and epochs
            events = mne.events_from_annotations(
                raw, event_id=self.MAPPING, verbose=False
            )[0]
            cue_mask = np.isin(events[:, 2], [1, 2, 3, 4, -1])  # cue onsets
            cue_events = events[cue_mask]
            if cue_events.size == 0:
                continue

            X = raw.get_data()  # shape: [22, T]
            pre_samp = int(round(self.pre_sec * self.sr))
            post_samp = int(round(self.post_sec * self.sr))

            # For evaluation
            eval_labels = None
            if self.mode == "E":
                # Provided labels are 1..4
                mat_path = os.path.join(self.root, f"{rec_id}E.mat")
                md = loadmat(mat_path)["classlabel"]
                eval_labels = md.squeeze().astype(int) - 1

            # Build epochs
            trial_idx = 0
            for e in cue_events:
                onset = e[0]  # sample index at cue onset

                start = onset - pre_samp
                end = onset + post_samp
                seg = torch.tensor(X[:, start:end])

                if self.normalize:
                    mu = seg[:, :pre_samp].mean(dim=1, keepdim=True)
                    sd = seg[:, :pre_samp].std(dim=1, keepdim=True)
                    seg = (seg - mu) / (sd + 1e-8)

                with torch.no_grad():
                    inp = seg.unsqueeze(1).pin_memory()
                    inp = inp.float().cuda(non_blocking=True)
                    enc_frames = model.encode(inp)
                    # extract codes in shape [C=22, n_q, T]
                    codes = torch.cat([c for c, _ in enc_frames], dim=-1)
                    # permute to [C=22, T, n_q] for saving
                    codes_data.append(codes.permute(0, 2, 1).cpu())

                if self.mode == "T":
                    proc_labels.append(int(e[2]) - 1)
                else:
                    # use provided eval label order
                    if eval_labels is not None and trial_idx < len(eval_labels):
                        proc_labels.append(int(eval_labels[trial_idx]))
                    else:
                        proc_labels.append(-1)

                proc_names.append(rec_id)
                trial_idx += 1

        codes_data = torch.stack(codes_data, dim=0)
        codes_data = codes_data.cpu().numpy()
        proc_labels = np.array(proc_labels)
        proc_names = np.array(proc_names, dtype="S3")

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
            sample_rate = self.sr

        eeg = torch.from_numpy(self.data[idx])
        if self.transform:
            eeg = self.transform(eeg)

        return eeg, self.labels[idx], self.names[idx]


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False
    dataset = BCI2aDataset(config=config, mode="T")
    print("Dataset length:", len(dataset))
    
    eeg_in, label, name = dataset[0]
    print(eeg_in.shape, label, name)
    print(eeg_in[17])  # print the 18th channel


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
