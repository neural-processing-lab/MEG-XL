import numpy as np, torch, pandas as pd
import hydra, os, h5py, mne, glob
from torch.utils.data import Dataset
from scipy import signal
from tqdm import tqdm
from typing import List, Tuple, Optional

from ..model import BioCodecModel
mne.set_log_level("ERROR")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SleepEDFDataset(Dataset):
    """
    Sleep-EDF dataset for sleep stage classification.
    30-second sleep stage annotations (W, N1, N2, N3, REM)
    """
    SLEEP_STAGES = {
        'Sleep stage W': 0,    # Wake
        'Sleep stage 1': 1,    # N1
        'Sleep stage 2': 2,    # N2  
        'Sleep stage 3': 3,    # N3 (deep sleep)
        'Sleep stage 4': 3,    # N3 (treat as same as stage 3)
        'Sleep stage R': 4,    # REM
        'Sleep stage ?': -1,   # Unknown/artifact
        'Movement time': -1,   # Movement artifact
    }
    
    @staticmethod
    def get_subject_splits(root):
        """
        Returns 5 subject lists for cross-validation.
        """
        np.random.seed(42)
        
        # Get cassette subjects
        cassette_dir = os.path.join(root, "sleep-cassette")
        files = [f for f in os.listdir(cassette_dir) if f.endswith("-PSG.edf")]
        files = sorted(files)
        cassette_subjects = [f.replace("-PSG.edf", "")[:-3] for f in files]
        cassette_sessions = [f.replace("-PSG.edf", "") for f in files]
            
        # Get telemetry subjects
        telemetry_dir = os.path.join(root, "sleep-telemetry")
        files = [f for f in os.listdir(telemetry_dir) if f.endswith("-PSG.edf")]
        files = sorted(files)
        telemetry_subjects = [f.replace("-PSG.edf", "")[:-3] for f in files]
        telemetry_sessions = [f.replace("-PSG.edf", "") for f in files]
        
        all_subjects = list(set(cassette_subjects + telemetry_subjects))
        all_subjects = np.random.permutation(sorted(all_subjects)).tolist()
        val_size = len(all_subjects) // 5

        splits = []
        for i in range(5):
            if i == 4:
                fold_subjects = all_subjects[i*val_size:]
            else:
                fold_subjects = all_subjects[i*val_size:(i+1)*val_size]

            # find all the files for these subjects
            fold_sessions = [
                s for s in cassette_sessions + telemetry_sessions
                if any(s.startswith(sub) for sub in fold_subjects)
            ]
            splits.append(fold_sessions)

        return splits

    def __init__(
        self,
        root: str = "/PATH/TO/sleep-edf/",
        config: dict = None,
        sr: int = 250,
        window_size: int = 30,  # seconds
        channels: List[str] = None,
        normalize: bool = True,
        splits: list = None,
        transform=None,
    ):
        super().__init__()
        self.sr = sr
        self.config = config
        self.root = root
        self.window_size = window_size
        self.normalize = normalize
        self.transform = transform
        self.splits = splits
        self.channels = ['EEG Fpz-Cz', 'EEG Pz-Oz'] if channels is None else channels

        all_splits = self.get_subject_splits(self.root)

        self.data = []
        self.labels = []
        self.names = []

        self.codes_path = self.config.datasets.codes_path
        for split in self.splits:
            codes_file = f"SleepEDF_split{split}_new.h5"
            if not os.path.exists(self.codes_path + codes_file):
                self.preprocess(
                    codes_path=self.codes_path + codes_file,
                    codec_path=self.config.common.codec_path,
                    subjects=all_splits[split],
                )

            # Load pre-processed data
            self.h5_file = h5py.File(self.codes_path + codes_file, "r")
            self.data.extend(self.h5_file["eeg_input"])
            self.labels.extend(self.h5_file["eeg_label"])
            self.names.extend(self.h5_file["eeg_name"])
            self.h5_file.close()

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        # repeat 6 times
        self.labels = np.repeat(self.labels, 6)
        self.names = np.array([s.decode("utf-8") for s in self.names])
        self.names = np.repeat(self.names, 6)

    def _find_files(self, subjects):
        """Find PSG and hypnogram files for specified subjects"""
        cassette_dir = os.path.join(self.root, "sleep-cassette")
        telemetry_dir = os.path.join(self.root, "sleep-telemetry")
        
        edf_files = []
        annotation_files = []
        
        for base_dir in [cassette_dir, telemetry_dir]:
            # Find all PSG files first
            psg_pattern = os.path.join(base_dir, "*0-PSG.edf")            
            for psg_file in glob.glob(psg_pattern):
                # Extract base name
                psg_basename = os.path.basename(psg_file)
                base_name = psg_basename.replace("0-PSG.edf", "")
                
                # Check if this file belongs to any of our target subjects
                subject_match = False
                for subject in subjects:
                    if psg_basename.startswith(subject):
                        subject_match = True
                        break
                
                if subject_match:
                    # Find corresponding hypnogram file with any suffix
                    hypno_pattern = os.path.join(base_dir, f"{base_name}*-Hypnogram.edf")
                    hypno_files = glob.glob(hypno_pattern)
                    if hypno_files and os.path.exists(psg_file) and os.path.exists(hypno_files[0]):
                        edf_files.append(psg_file)
                        annotation_files.append(hypno_files[0])
            
        return edf_files, annotation_files

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
        
        # Find files for target subjects
        edf_files, annotation_files = self._find_files(subjects)
        
        proc_data = []
        proc_labels = []
        proc_names = []

        for edf_file, ann_file in tqdm(
            zip(edf_files, annotation_files),
            desc="Processing subjects",
            total=len(edf_files),
        ):
            # Preprocess PSG data
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            raw.pick([ch for ch in self.channels if ch in raw.ch_names])
            raw.filter(l_freq=0.5, h_freq=40, picks='eeg')

            # resample if needed
            if raw.info['sfreq'] != self.sr:
                raw.resample(self.sr, npad="auto", method="polyphase")
            
            # Load annotations
            ann = mne.read_annotations(ann_file)
            raw.set_annotations(ann)
            onsets = (ann.onset * self.sr).astype(int)
            durations = (ann.duration * self.sr).astype(int)
            descriptions = ann.description
            
            # extract 30s epochs from annotations
            for onset, dur, desc in zip(onsets, durations, descriptions):
                if dur >= 3e6:  # very long annotations
                    # keep last 30 minutes
                    dur = 30 * 60 * self.sr
                    onset = onset + (dur - 30 * 60 * self.sr)

                n_epochs = int(dur / (30 * self.sr))
                if desc in self.SLEEP_STAGES and self.SLEEP_STAGES[desc] != -1:
                    epochs_data = [
                        raw.get_data(start=onset + i * 30 * self.sr, stop=onset + (i + 1) * 30 * self.sr)
                        for i in range(n_epochs)
                    ]
                    if epochs_data[-1].shape[1] < 30 * self.sr:
                        # skip incomplete last epoch
                        epochs_data = epochs_data[:-1]
                        n_epochs -= 1
                    proc_data.extend(np.array(epochs_data))

                    epochs_label = self.SLEEP_STAGES[desc]
                    proc_labels.extend([epochs_label] * n_epochs)

                    subject_id = os.path.basename(edf_file)
                    subject_id = subject_id.split("E0")[0].split("J0")[0]
                    proc_names.extend([subject_id] * n_epochs)
            
        proc_data = np.array(proc_data)
        proc_labels = np.array(proc_labels)
        proc_names = np.array(proc_names).astype("S10")

        # proc_data is shape [N, C, T]  (N = number of 30s epochs)
        proc_tensor = torch.tensor(proc_data, dtype=torch.float32)   # (N, C, T)
        if self.normalize:
            mu = proc_tensor.mean(dim=2, keepdim=True)
            sd = proc_tensor.std(dim=2, keepdim=True)
            proc_tensor = (proc_tensor - mu) / (sd + 1e-8)

        chunks = proc_tensor.split(self.sr * 5, dim=2)
        chunks = torch.stack(chunks, dim=1)       # (N, 6, C, 1250)
        chunks = chunks.reshape(-1, self.sr * 5)  # (N*6, 1250)

        codes_data = []
        for i in range(0, chunks.shape[0], 1024):
            with torch.no_grad():
                # Model expects (B, 1, T) → add channel dim
                inp = chunks[i:i+1024].unsqueeze(1)   # (B, 1, 1024)
                enc_frames = model.encode(inp.cuda())

            # Concatenate codebook outputs
            codes = torch.cat([c for c, _ in enc_frames], dim=-1)
            # Reshape to [B, C, n_q, T']
            codes = codes.view(-1, 2, codes.shape[1], codes.shape[2])
            # Permute to (B, C, T', n_q)
            codes_data.append(codes.permute(0, 1, 3, 2).cpu())
        
        # Stack and convert to numpy
        codes_data = torch.cat(codes_data, dim=0).cpu().numpy()

        # Save to HDF5
        if os.path.exists(codes_path):
            os.remove(codes_path)
        with h5py.File(codes_path, "w") as h5_file:
            h5_file.create_dataset("eeg_input", data=codes_data)
            h5_file.create_dataset("eeg_label", data=proc_labels)
            h5_file.create_dataset("eeg_name", data=proc_names)
        print(f"Saved preprocessed data to {codes_path}")
        return

        codes_data = []
        for seg in tqdm(proc_data, desc="Extracting codes"):
            # Convert to tensor (C, T)
            seg = torch.tensor(seg, dtype=torch.float32)
            if self.normalize:
                mu = seg.mean(dim=1, keepdim=True)
                sd = seg.std(dim=1, keepdim=True)
                seg = (seg - mu) / (sd + 1e-8)

            # Extract codes
            chunks = seg.split(self.sr * 5, dim=1)  # [6, C, 5*sr]
            # convert to [6*C, 5*sr]
            chunks = [c.reshape(-1, c.shape[-1]) for c in chunks]
            with torch.no_grad():
                inp = chunks.unsqueeze(1).cuda()  # Add channel dim
                enc_frames = model.encode(inp)
                # Extract codes in shape [6*C, n_q, T]
                codes = torch.cat([c for c, _ in enc_frames], dim=-1)
                # Reshape to [5, C, n_q, T]
                codes = codes.view(5, -1, codes.shape[1], codes.shape[2])
                print(codes[0, 0])
                # Permute to [5, C, T, n_q] for consistency
                codes_data.append(codes.permute(0, 1, 3, 2))

        # Stack and convert to numpy
        codes_data = torch.cat(codes_data, dim=0).cpu().numpy()
        print(f"Processed {len(codes_data)} epochs from {len(set(proc_names))} subjects")

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
        eeg = torch.from_numpy(self.data[idx])
        label = int(self.labels[idx])
        if self.transform:
            eeg = self.transform(eeg)

        return eeg.long(), label, self.names[idx]


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False

    dataset = SleepEDFDataset(
        config=config,
        splits=[0, 1, 2, 3, 4],
        sr=250,
    )
    print("Dataset length:", len(dataset))

    eeg_in, label, subject = dataset[0]
    print(f"Sample shape: {eeg_in.shape}")
    print(f"Label: {label}, Subject: {subject}")
    print(eeg_in[0, ::4])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
