import numpy as np, torch
import hydra, mne, os, h5py, pickle
from torch.utils.data import Dataset
from tqdm import tqdm

from ..model import BioCodecModel


class MCSDataset(Dataset):
    """
    MCS EMG Dataset from EMGBench
    """
    FOLD_MAPPING = {
        0: "[1 2 3 4 5 6 7]",
        1: "[8 9 10 11 12 13 14]",
        2: "[15 16 17 18 19 20]",
        3: "[21 22 23 24 25 26]",
        4: "[27 28 29 30 31 32]",
    }

    def __init__(
        self,
        root: str = "/PATH/TO/MCS_EMG/",
        config: dict = None,
        sr: int = 1000,
        normalize: bool = True,
        folds: list = [0, 1, 2, 3, 4],
        transform=None,
    ):
        super().__init__()
        self.normalize = normalize
        self.transform = transform
        self.config = config
        self.root = root
        self.sr = sr

        self.codes_path = self.config.datasets.codes_path

        all_data, all_labels = [], []
        for this_fold in folds:
            codes_file = f"MCS_{this_fold}.h5"
            if not os.path.exists(self.codes_path + codes_file):
                self.preprocess(
                    codes_path=self.codes_path + codes_file,
                    codec_path=self.config.common.codec_path,
                    testfold=this_fold,
                )

            # Load pre-processed data
            self.h5_file = h5py.File(self.codes_path + codes_file, "r")
            all_data.extend(self.h5_file["emg_input"][:])
            all_labels.extend(self.h5_file["emg_label"][:])
            self.h5_file.close()

        self.data = np.stack(all_data, axis=0)
        self.labels = np.array(all_labels)
        
        # discard all samples with label > 6
        valid_indices = np.where(self.labels < 7)[0]
        self.data = self.data[valid_indices]
        self.labels = self.labels[valid_indices]

    def preprocess(self, codes_path, codec_path, testfold):
        print(f"Preprocessing fold {testfold}")
        actual = self.FOLD_MAPPING[testfold]
        proc_path = os.path.join(self.root, f"testfold_{actual}.pkl")

        # Initialize codec model
        model = BioCodecModel._get_emg_model(
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

        codes_data, proc_labels = [], []
        with open(proc_path, "rb") as f:
            fold_data = pickle.load(f)

        # Extract epochs around events
        for segment in tqdm(fold_data):
            this_emg, this_lab = segment

            if self.normalize:
                mu = this_emg.mean(axis=1, keepdims=True)
                sd = this_emg.std(axis=1, keepdims=True)
                this_emg = (this_emg - mu) / (sd + 1e-8)

            with torch.no_grad():
                this_emg = torch.from_numpy(this_emg)
                inp = this_emg.unsqueeze(1).cuda()
                enc_frames = model.encode(inp.float())
                # Extract codes in shape [C, n_q, T]
                codes = torch.cat([c for c, _ in enc_frames], dim=-1)
                # Permute to [C, T, n_q] for consistency
                codes_data.append(codes.permute(0, 2, 1).cpu())

            proc_labels.append(this_lab)

        # Stack and convert to numpy
        codes_data = torch.stack(codes_data, dim=0).cpu().numpy()
        proc_labels = np.array(proc_labels)

        print(f"Processed {len(codes_data)} trials")
        print(f"Data shape: {codes_data.shape}")
        print(f"Labels distribution: {np.bincount(proc_labels)}")

        if os.path.exists(codes_path):
            os.remove(codes_path)
        with h5py.File(codes_path, "w") as h5_file:
            h5_file.create_dataset("emg_input", data=codes_data)
            h5_file.create_dataset("emg_label", data=proc_labels)

        print(f"Saved preprocessed data to {codes_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emg = torch.from_numpy(self.data[idx])
        if self.transform:
            emg = self.transform(emg)

        return emg, self.labels[idx], "None"

@hydra.main(config_path="../configs", config_name="ft_config", version_base=None)
def main(config):
    torch.backends.cudnn.enabled = False

    dataset = MCSDataset(
        config=config,
        folds=[0, 1, 2, 3, 4],
        normalize=True,
    )
    print("Dataset length:", len(dataset))

    emg, label, _ = dataset[0]
    print(f"Sample shape: {emg.shape}, Label: {label}")
    print(f"Label distribution: {np.bincount(dataset.labels)}")
    print(emg[2, ::5])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
