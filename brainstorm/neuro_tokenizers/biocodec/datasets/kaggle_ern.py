import hydra, os, h5py, mne, glob
import numpy as np, torch, pandas as pd
from torch.utils.data import Dataset
from scipy import signal
from tqdm import tqdm

from ..model import BioCodecModel


class KaggleERN(Dataset):
    """
    Kaggle Error-Related Negativity (ERN) Dataset - P300 Speller BCI Paradigm.
    Binary classification: 0 (incorrect feedback), 1 (correct feedback)
    """
    
    @staticmethod
    def get_subject_splits(mode="train"):
        """
        Returns 4 subject lists for cross-validation.
        """
        if mode == "train":
            all_subjects = ['S02', 'S06', 'S07', 'S11', 'S12', 'S13', 'S14', 'S16', 'S17', 'S18', 'S20', 'S21', 'S22', 'S23', 'S24', 'S26']
        elif mode == "test":
            all_subjects = ['S01', 'S03', 'S04', 'S05', 'S08', 'S09', 'S10', 'S15', 'S19', 'S25']
            return [all_subjects]  # Single split for test set

        # Shuffle subjects for random splits
        np.random.seed(42)
        shuffled_subjects = np.random.permutation(all_subjects).tolist()
        
        # Create 4 validation splits
        n_subjects = len(shuffled_subjects)
        val_size = n_subjects // 4
        
        val_splits = []
        for i in range(4):
            val_start = i * val_size
            val_splits.append(
                shuffled_subjects[val_start:]
                if i == 3
                else shuffled_subjects[val_start:val_start + val_size]
            )
        
        return val_splits

    def __init__(
        self,
        root: str = "/PATH/TO/kaggle-ern/",
        config: dict = None,
        sr: int = 250,
        pre_sec: float = 0.2,
        post_sec: float = 1.0,
        normalize: bool = True,
        mode: str = "train",
        splits: list = None,
        transform=None,
    ):
        super().__init__()
        self.sr = sr
        self.config = config
        self.root = root
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.normalize = normalize
        self.mode = mode
        self.transform = transform
        self.splits = [0] if mode == "test" else splits
        
        data_dir = os.path.join(self.root, self.mode)
        all_splits = self.get_subject_splits(self.mode)

        self.data = []
        self.labels = []
        self.names = []

        self.codes_path = self.config.datasets.codes_path
        for split in self.splits:
            codes_file = f"KaggleERN_{self.mode}_split{split}.h5"
            if not os.path.exists(self.codes_path + codes_file):
                self.preprocess(
                    codes_path=self.codes_path + codes_file,
                    codec_path=self.config.common.codec_path,
                    subjects=all_splits[split],
                )

            # Load preprocessed data
            self.h5_file = h5py.File(self.codes_path + codes_file, "r")
            self.data.extend(self.h5_file["eeg_input"])
            self.labels.extend(self.h5_file["eeg_label"])
            self.names.extend(self.h5_file["eeg_name"])
            self.h5_file.close()

        self.data = np.array(self.data)
        self.labels = np.array([s - 1 for s in self.labels])  # Convert to 0/1
        self.names = np.array([s.decode("utf-8") for s in self.names])

    def load_data(self, data_path):
        file_list = glob.glob(os.path.join(data_path, "*.csv"))
        return {
            os.path.basename(file): pd.read_csv(file)
            for file in tqdm(file_list)
        }

    def create_raw_objects(self, data, ch_locations, sr, mode_labels):
        ch_names = list(ch_locations['Labels']) + ['EOG']
        ch_types = ['eeg'] * 56 + ['eog']

        # Create montage
        montage = mne.channels.make_dig_montage(
            ch_pos=dict(
                zip(
                    ch_locations['Labels'],
                    zip(
                        ch_locations['Radius'] * np.cos(ch_locations['Phi']),
                        ch_locations['Radius'] * np.sin(ch_locations['Phi']),
                        np.zeros(len(ch_locations['Labels']))
                    )
                )
            ),
            coord_frame='head'
        )
        
        raw_objects = []
        y_split = [x[:-6] for x in mode_labels["IdFeedBack"]]
        for file_name, df in tqdm(data.items()):
            eeg_data = df[ch_names[:-1]].to_numpy().T
            eog_data = df['EOG'].to_numpy()[np.newaxis, :]
            all_data = np.vstack([eeg_data, eog_data])
            
            # Create info object
            info = mne.create_info(ch_names=ch_names, sfreq=sr, ch_types=ch_types)
            # Create raw object
            raw = mne.io.RawArray(all_data, info)
            raw.set_montage(montage)
            # Add annotations
            feedback_indices = df[df['FeedBackEvent'] == 1].index
            feedback_times = df.loc[feedback_indices, 'Time'].to_numpy()

            y_idx = [index for index, x in enumerate(y_split) if x == file_name[5:-4]]
            descriptions = [mode_labels.loc[idx, "Prediction"] for idx in y_idx]
            annotations = mne.Annotations(
                onset=feedback_times,
                duration=[0] * len(feedback_times),
                description=descriptions
            )
            raw.set_annotations(annotations)
            raw_objects.append(raw)
        
        return raw_objects

    def preprocess_raw(self, raw, visualize=False):
        raw_processed = raw.copy()
        raw_processed.filter(l_freq=1, h_freq=40, picks=['eeg', 'eog'])
        raw_processed.notch_filter(freqs=50, picks=['eeg'])

        ica = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=1000)
        ica.fit(raw_processed, picks='eeg')
        eog_indices, eog_scores = ica.find_bads_eog(raw_processed, ch_name='EOG')
        ica.exclude = eog_indices
        ica.apply(raw_processed)
        
        return raw_processed

    def extract_epochs(self, raw, tmin=-0.2, tmax=1.0):
        events = mne.events_from_annotations(raw)
        return mne.Epochs(
            raw, events[0],
            event_id=events[1],
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            preload=True
        )

    def preprocess(self, codes_path, codec_path, subjects):
        # Initialize codec model
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

        # Load codec state dict
        checkpoint = torch.load(codec_path, map_location="cuda")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Handle _orig_mod prefix from torch.compile()
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {
                key.replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }

        model.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        
        data = self.load_data(os.path.join(self.root, self.mode))
        data = {k: v for k, v in data.items() if k.split("_")[1] in subjects}
        ch_locations = pd.read_csv(self.root + "ChannelsLocation.csv")
        mode_labels = pd.read_csv(self.root + f"{self.mode.capitalize()}Labels.csv")
        raw_objects = self.create_raw_objects(
            data, ch_locations, sr=200, mode_labels=mode_labels
        )

        proc_data = []
        proc_labels = []
        proc_names = []
        for i, obj in enumerate(raw_objects):
            proc = self.preprocess_raw(obj)

            epochs = self.extract_epochs(proc)
            proc_data.append(epochs.get_data())

            labels = epochs.events[:, -1]
            proc_labels.append(labels)

            name = list(data.keys())[i].split("_")[1]
            proc_names.append([name] * len(epochs))

        proc_data = np.vstack(proc_data)
        proc_labels = np.concatenate(proc_labels)
        proc_names = np.concatenate(proc_names).astype("S10")

        codes_data = []
        for seg in tqdm(proc_data):
            # Resample to 250 Hz
            seg = signal.resample_poly(seg, up=5, down=4, axis=1)
            
            # Convert to tensor
            seg = torch.tensor(seg, dtype=torch.float32)
            if self.normalize:
                seg -= seg.mean(dim=0, keepdim=True)
                mu = seg.mean(dim=1, keepdim=True)
                sd = seg.std(dim=1, keepdim=True)
                seg = (seg - mu) / (sd + 1e-8)
                
            # Extract codes
            with torch.no_grad():
                inp = seg.unsqueeze(1).cuda()  # Add batch dim
                enc_frames = model.encode(inp)
                # Extract codes in shape [C, n_q, T]
                codes = torch.cat([c for c, _ in enc_frames], dim=-1)
                # Permute to [C, T, n_q] for consistency
                codes_data.append(codes.permute(0, 2, 1).cpu())
            
        # Stack and convert to numpy
        codes_data = torch.stack(codes_data, dim=0).cpu().numpy()
        print(f"Processed {len(codes_data)} trials from {len(set(proc_names))} subjects")

        # Save to HDF5
        if os.path.exists(codes_path):
            os.remove(codes_path)
        with h5py.File(codes_path, "w") as h5_file:
            h5_file.create_dataset("eeg_input", data=codes_data)
            h5_file.create_dataset("eeg_label", data=proc_labels)
            h5_file.create_dataset("eeg_name", data=proc_names)
        print(f"Saved preprocessed data to {codes_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(self.data[idx]).float()
        label = int(self.labels[idx])
        name = self.names[idx]
        if self.transform:
            eeg = self.transform(eeg)

        return eeg[:56].long(), label, name


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False

    dataset = KaggleERN(
        root="/PATH/TO/kaggle-ern/",
        config=config,
        mode="test",
        splits=[0, 1, 2, 3],
        sr=250,  # Target sampling rate
    )
    print("Dataset length:", len(dataset))
    
    eeg_in, label, subject = dataset[0]
    print(f"Sample shape: {eeg_in.shape}")
    print(f"Label: {label}, Subject: {subject}")
    print(eeg_in[17, :10])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
