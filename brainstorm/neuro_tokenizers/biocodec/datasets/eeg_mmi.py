import numpy as np, torch
import hydra, mne, os, h5py
from torch.utils.data import Dataset

from ..model import BioCodecModel


class EEGMMIDataset(Dataset):
    """
    PhysioNet EEG Motor Movement/Imagery Database.

    Dataset contains 109 subjects performing various motor/imagery tasks:
    - R01: Baseline, eyes open
    - R02: Baseline, eyes closed
    - R03: Task 1 (open/close left or right fist)
    - R04: Task 2 (imagine opening/closing left or right fist)
    - R05: Task 3 (open/close both fists or both feet)
    - R06: Task 4 (imagine opening/closing both fists or both feet)
    - R07: Task 1 (repeat of R03)
    - R08: Task 2 (repeat of R04)
    - R09: Task 3 (repeat of R05)
    - R10: Task 4 (repeat of R06)
    - R11: Task 1 (repeat of R03)
    - R12: Task 2 (repeat of R04)
    - R13: Task 3 (repeat of R05)
    - R14: Task 4 (repeat of R06)
    """

    def __init__(
        self,
        root: str = "/PATH/TO/eegmmidb/",
        config: dict = None,
        sr: int = 250,
        task_type: str = "left_right_fist",
        pre_sec: float = 0.0,
        post_sec: float = 5.0,
        normalize: bool = True,
        folds: list = [0, 1, 2, 3, 4],  # all folds
        transform=None,
    ):
        super().__init__()
        self.sr = sr
        self.config = config
        self.root = root
        self.folds = folds
        self.task_type = task_type
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.normalize = normalize
        self.transform = transform

        self.subjects = np.array([f"S{str(s).zfill(3)}" for s in range(1, 110)])
        np.random.seed(42)
        np.random.shuffle(self.subjects)

        # Select runs based on task type
        if task_type == "left_right_fist":
            self.runs = ["R04", "R08", "R12"]  # Left/right fist imagery
            self.num_classes = 2  # left fist, right fist
        elif task_type == "fists_feet":
            self.runs = ["R06", "R10", "R14"]  # Fists/feet imagery
            self.num_classes = 2  # both fists, both feet
        elif task_type == "four-class":
            self.runs = ["R04", "R06", "R08", "R10", "R12", "R14"]
            self.num_classes = 4  # left fist, right fist, both fists, both feet
        elif task_type == "eyes_open_closed":
            self.runs = ["R01", "R02"]
            self.num_classes = 2  # eyes open, eyes closed
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        self.codes_path = self.config.datasets.codes_path

        all_data = []
        all_labels = []
        all_names = []
        for fold in self.folds:
            if self.task_type == "eyes_open_closed":
                codes_file = f"MMI_eyes_fold{fold}.h5"
            else:
                codes_file = f"MMI_{self.task_type}_fold{fold}_both.h5"
            if not os.path.exists(self.codes_path + codes_file):
                self.preprocess(
                    codes_path=self.codes_path + codes_file,
                    codec_path=self.config.common.codec_path,
                    subjects=self.subjects[fold::5],  # 5-fold
                )
            with h5py.File(self.codes_path + codes_file, "r") as h5_file:
                all_data.append(h5_file["eeg_input"][:])
                all_labels.append(h5_file["eeg_label"][:])
                all_names.append(h5_file["eeg_subject"][:])

        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.names = np.concatenate(all_names, axis=0)
        self.names = np.array([s.decode("utf-8") for s in self.names])

    def preprocess(self, codes_path, codec_path, subjects):
        print(f"Preprocessing MMI dataset for task: {self.task_type}")
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

        codes_data = []
        proc_labels = []
        proc_names = []

        # Process each subject
        for subj_num in subjects:
            subj_id = f"{str(subj_num).zfill(3)}"
            subj_dir = os.path.join(self.root, subj_id)
            print(f"Processing {subj_id}...")

            # Process each run for this subject
            for run_id in self.runs:
                edf_file = os.path.join(subj_dir, f"{subj_id}{run_id}.edf")

                # Load EEG data and extract events using MNE
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                # Pick only EEG channels (64 channels, exclude reference)
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
                raw.pick(eeg_picks, verbose=False)
                # Filter for motor imagery events
                raw.filter(l_freq=0.5, h_freq=40, method="fir", verbose=False)

                # Resample to target sampling rate if needed
                if raw.info["sfreq"] != self.sr:
                    raw.resample(self.sr, verbose=False)

                # Extract events from annotations
                events, event_id = mne.events_from_annotations(raw, verbose=False)

                # Filter for task-relevant events (T1, T2)
                task_events = []
                for event in events:
                    event_code = event[2]
                    # Find corresponding annotation
                    event_name = None
                    for ann_key, ann_code in event_id.items():
                        if ann_code == event_code:
                            event_name = ann_key
                            break

                    if self.task_type == "eyes_open_closed":
                        event_name = "T1" if run_id == "R01" else "T2"
                    
                    if event_name in ["T1", "T2"]:
                        task_events.append([event[0], 0, event_name])

                X = raw.get_data()  # shape: [n_channels, n_samples]
                n_channels = X.shape[0]
                pre_samp = int(round(self.pre_sec * self.sr))
                post_samp = int(round(self.post_sec * self.sr))

                # Extract epochs around events
                for event in task_events:
                    onset = int(event[0])
                    event_type = event[2]

                    start = onset - pre_samp
                    end = onset + post_samp
                    seg = torch.tensor(X[:, start:end], dtype=torch.float32)

                    # pad if segment is shorter than expected
                    expected_len = pre_samp + post_samp
                    if seg.shape[1] < expected_len:
                        pad_len = expected_len - seg.shape[1]
                        seg = torch.nn.functional.pad(seg, (0, pad_len), "reflect")

                    if self.normalize:
                        # Z-score normalization using pre-stimulus
                        mu = seg[:, :self.sr // 2].mean(dim=1, keepdim=True)
                        sd = seg[:, :self.sr // 2].std(dim=1, keepdim=True)
                        seg = (seg - mu) / (sd + 1e-8)

                    # Extract codes using codec model
                    with torch.no_grad():
                        inp = seg.unsqueeze(1).cuda()  # Add batch dimension
                        enc_frames = model.encode(inp)
                        # Extract codes in shape [C, n_q, T]
                        codes = torch.cat([c for c, _ in enc_frames], dim=-1)
                        # Permute to [C, T, n_q] for consistency
                        codes_data.append(codes.permute(0, 2, 1).cpu())
                    
                    # Assign labels based on task type and event
                    if self.task_type == "left_right_fist":
                        label = 0 if event_type == "T1" else 1  # T1=left, T2=right
                    elif self.task_type == "fists_feet":
                        label = 0 if event_type == "T1" else 1  # T1=fists, T2=feet
                    elif self.task_type == "eyes_open_closed":
                        label = 0 if event_type == "T1" else 1  # T1=eyes open, T2=eyes closed
                    elif self.task_type == "four-class":
                        label = (
                            0
                            if event_type == "T1" and run_id in ["R04", "R08", "R12"]
                            else 1
                            if event_type == "T2" and run_id in ["R04", "R08", "R12"]
                            else 2
                            if event_type == "T1" and run_id in ["R06", "R10", "R14"]
                            else 3
                        )  # T1=left, T2=right, T3=fists, T4=feet

                    proc_labels.append(label)
                    proc_names.append(subj_id)

        # Stack and convert to numpy
        codes_data = torch.stack(codes_data, dim=0).cpu().numpy()
        proc_labels = np.array(proc_labels)
        proc_names = np.array(proc_names, dtype="S10")

        print(f"Processed{len(set(proc_names))} subjects")
        print(f"Data shape: {codes_data.shape}")
        print(f"Labels distribution: {np.bincount(proc_labels)}")

        if os.path.exists(codes_path):
            os.remove(codes_path)
        with h5py.File(codes_path, "w") as h5_file:
            h5_file.create_dataset("eeg_input", data=codes_data)
            h5_file.create_dataset("eeg_label", data=proc_labels)
            h5_file.create_dataset("eeg_subject", data=proc_names)
            
        print(f"Saved preprocessed data to {codes_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, sample_rate=None):
        eeg = torch.from_numpy(self.data[idx])
        if self.transform:
            eeg = self.transform(eeg)

        return eeg, self.labels[idx], self.names[idx]


@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False

    dataset = EEGMMIDataset(
        config=config,
        task_type="four-class",
        folds=[0, 1, 2, 3, 4],  # all folds
        sr=250,
    )
    print("Dataset length:", len(dataset))
    
    eeg_in, label, subject = dataset[0]
    print(f"Sample shape: {eeg_in.shape}, Label: {label}, Subject: {subject}")
    print(f"Channel 17 data: {eeg_in[17][:10]}")  # First 10 samples of ch 17
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

