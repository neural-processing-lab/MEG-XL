import numpy as np, pandas as pd
import hydra, mne, os, h5py, json, torch
from tqdm import tqdm
from scipy.io import loadmat
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ..model import BioCodecModel


class N400Dataset(Dataset):
    """
    N400 EEG Dataset for semantic processing research.
    Responses to congruent and incongruent word pairs.
    """
    
    # Channel selection based on FESDE
    EEG_CHANNELS = (
        [f'A{i}' for i in range(1, 33)] + \
        [f'B{i}' for i in range(1, 33)] + \
        [f'C{i}' for i in range(1, 33)] + \
        [f'D{i}' for i in range(1, 33)]
    )

    def __init__(
        self,
        root: str = "/PATH/TO/n400/",
        config: dict = None,
        sr: int = 512,  # Original sampling rate
        normalize: bool = True,
        subjects: List[int] = None,
        channels: Optional[List[str]] = None,
        folds: List[int] = [0, 1, 2, 3],  # Use all folds by default
        transform=None,
    ):
        super().__init__()
        self.root = root
        self.config = config
        self.normalize = normalize
        self.transform = transform
        self.sr = sr
        
        self.subjects = subjects or list(range(1, 25))
        self.channels = channels or self.EEG_CHANNELS
        self.folds = None if subjects else folds
        
        # Setup codes path and preprocess if needed
        self.codes_path = self.config.datasets.codes_path
        os.makedirs(self.codes_path, exist_ok=True)

        if not folds:
            # create and load everyone
            codes_file = "N400_24.h5"
            codes_filepath = os.path.join(self.codes_path, codes_file)
            if not os.path.exists(codes_filepath) or True:
                self.preprocess(
                    codes_path=codes_filepath,
                    codec_path=self.config.common.codec_path if config else None,
                    subjects=list(range(1, 25))
                )
            # select the subjects to keep
            data = self._load_preprocessed(codes_filepath, subjects=self.subjects)
            self.eeg, self.aud, self.names = data

        else:
            self.eeg, self.aud, self.names = [], [], []
            for fold in folds:
                codes_file = f"N400_fold{fold}.h5"
                codes_filepath = os.path.join(self.codes_path, codes_file)
                these_subjects = [s for s in list(range(1, 25)) if (s - 1) % 4 == fold]
                if not os.path.exists(codes_filepath):
                    self.preprocess(
                        codes_path=codes_filepath,
                        codec_path=self.config.common.codec_path if config else None,
                        subjects=these_subjects
                    )
                data = self._load_preprocessed(codes_filepath, subjects=these_subjects)
                self.eeg.append(data[0])
                self.aud.append(data[1])
                self.names.append(data[2])
                
            self.eeg = np.concatenate(self.eeg, axis=0)
            self.aud = np.concatenate(self.aud, axis=0)
            self.names = np.concatenate(self.names, axis=0)
        
        print(f"Loaded {len(self.eeg)} samples from {len(set(self.names))} subjects.")

    def preprocess(self, codes_path: str, codec_path: Optional[str], subjects: List[int]):
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
            
        # Handle _orig_mod prefix
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {
                key.replace("_orig_mod.", ""): value
                for key, value in state_dict.items()
            }
            
        model.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()

        all_codes = []
        all_stims = []
        all_names = []
        
        # Process each subject
        for sub_num in tqdm(subjects, desc="Processing subjects"):
            sub_id = f"sub-{sub_num:02d}"
            codes, stimuli, names = self._process_raw_data(sub_id, model)

            all_codes.extend(codes)
            all_stims.extend(stimuli)
            all_names.extend(names)

        # Pad to the same T length (dim 1)
        max_len = max(code.shape[1] for code in all_codes)
        padded_codes = []
        for code in all_codes:
            pad_width = max_len - code.shape[1]
            pad = - np.ones((code.shape[0], pad_width, code.shape[2]))
            padded_codes.append(np.concatenate([code, pad], axis=1))

        all_codes = np.stack(padded_codes, axis=0)
        all_stims = np.array(all_stims, dtype='S60')
        all_names = np.array(all_names, dtype='S10')
        
        # Save to HDF5
        with h5py.File(codes_path, 'w') as h5_file:
            h5_file.create_dataset('eeg_input', data=all_codes)
            h5_file.create_dataset('eeg_stimuli', data=all_stims)
            h5_file.create_dataset('eeg_names', data=all_names)

        print(f"Data shape: {all_codes.shape}, {all_stims.shape}")
        print(f"Saved preprocessed data to {codes_path}")

    def _epoch_data(self, eeg_data, sub_id, these_events):
        samples, stimuli = [], []
        for index, row in these_events.iterrows():
            # Get the start sample
            start_time = row["stim_onset(s)"]
            start_sample = int(start_time * self.sr)

            # Get the end time
            end_time = start_time + row["stim_dur(s)"]
            end_sample = int(end_time * self.sr)

            this_sample = eeg_data[:, start_sample:end_sample]
            if self.normalize:
                mu = this_sample.mean(axis=1, keepdims=True)
                std = this_sample.std(axis=1, keepdims=True)
                this_sample = (this_sample - mu) / (std + 1e-8)

            samples.append(this_sample)
            stimuli.append(row["stim_file"])
        return samples, stimuli

    def _process_raw_data(self, sub_id: str, model):
        """Process raw BDF data"""
        bdf_file = os.path.join(self.root, sub_id, "eeg", f"{sub_id}_task-N400Stimset_eeg.bdf")
        if not os.path.exists(bdf_file):
            print(f"BDF file not found: {bdf_file}")
            return False
            
        # Load and filter raw EEG data
        raw = mne.io.read_raw_bdf(bdf_file, preload=True, verbose=False)
        eeg_channels = [ch for ch in self.channels if ch in raw.ch_names]
        raw.pick_channels(eeg_channels, verbose=False)
        raw.filter(l_freq=0.1, h_freq=45, picks="eeg", fir_design='firwin', verbose=False)

        # Eye Blink Correction
        ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
        ica.fit(raw)
        eog_indices, _ = ica.find_bads_eog(raw, ch_name="C17")
        ica.exclude = eog_indices
        raw = ica.apply(raw.copy())

        # DC offset correction
        correction = lambda x: x - np.mean(x, axis=1).reshape(-1, 1)
        raw = raw.apply_function(correction, channel_wise=False, picks="eeg")

        # Resample if needed
        if raw.info['sfreq'] != self.sr:
            raw.resample(self.sr, verbose=False)

        # Finally get the data
        eeg_data = raw.get_data()

        # Load events file
        events_file = os.path.join(self.root, sub_id, "eeg", f"{sub_id}_task-N400Stimset_events.tsv")
        events_data = pd.read_csv(events_file, sep='\t')

        # Avoid NaN onsets/durations
        events_data = events_data[
            events_data["onset"].notna() & events_data["duration"].notna()
        ]
        # Define Congruent Events
        congruent_events = events_data[events_data["trial_type"] == "NPC"]
        incongruent_events = events_data[events_data["trial_type"] == "NPI"]

        # Epoching the EEG data using the congruent events onset time
        co_samples, co_stims = self._epoch_data(eeg_data, sub_id, congruent_events)
        ic_samples, ic_stims = self._epoch_data(eeg_data, sub_id, incongruent_events)

        codes_data, names, stimuli  = [], [], []
        for epoch, stim_key in zip(co_samples + ic_samples, co_stims + ic_stims):
            # Extract codes
            with torch.no_grad():
                inp = torch.tensor(epoch, dtype=torch.float32)
                enc_frames = model.encode(inp.unsqueeze(1).cuda())
                codes = torch.cat([c for c, _ in enc_frames], dim=-1)
                codes = codes.permute(0, 2, 1).squeeze()  # (C, T, n_q)
                codes_data.append(codes.cpu().numpy())

            stimuli.append(stim_key)
            names.append(sub_id)

        return codes_data, stimuli, names

    def _load_preprocessed(
        self,
        codes_filepath: str,
        subjects: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from HDF5 file
        """
        with h5py.File(codes_filepath, 'r') as h5_file:
            this_eeg = h5_file['eeg_input'][:]
            this_aud = h5_file['eeg_stimuli'][:]
            this_names = h5_file['eeg_names'][:]
        # Decode string arrays
        this_names = np.array([s.decode('utf-8') for s in this_names])

        # keep only the selected subjects
        mask = np.isin(this_names, [f"sub-{sub:02d}" for sub in subjects])
        return this_eeg[mask], this_aud[mask], this_names[mask]

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        eeg = torch.from_numpy(self.eeg[idx])
        if self.transform:
            eeg = self.transform(eeg)

        aud = self.aud[idx].decode('utf-8')
        return eeg.long(), aud, self.names[idx]


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False
    dataset = N400Dataset(
        config=config,
        subjects=list(range(1, 25)),
        folds=None,
        sr=250,
    )
    print("Dataset length:", len(dataset))

    eeg_data, stim_key, subject = dataset[0]
    print(f"Sample shape: {eeg_data.shape}")
    print(f"Stimulus: {stim_key}, Subject: {subject}")
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
