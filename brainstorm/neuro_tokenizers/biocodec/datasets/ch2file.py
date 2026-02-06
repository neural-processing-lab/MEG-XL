import numpy as np
import os, mat73, h5py
from tqdm import tqdm
from scipy.stats import iqr

orig_path = "/PATH/TO/TUHEEG/"
proc_path = "/PATH/TO/tueg-processed-h5-splits-norm/"
names_path = "/PATH/TO/tueg_excluded.txt"
os.makedirs(proc_path, exist_ok=True)
    
if __name__ == "__main__":
    # create an empty array to store
    counter = 0
    eeg_repo = np.empty((0, 1250), dtype=np.float32)

    # load names to exclude from txt file
    with open(names_path, "r") as f:
        names = f.readlines()
        names = [name.strip() for name in names]

    # iterate over the TUEG files
    for num, matfile in tqdm(enumerate(os.listdir(orig_path)), total=len(os.listdir(orig_path))):
        if (not matfile.endswith(".mat")) or (matfile in names):
            continue
        
        fpath = os.path.join(orig_path, matfile)
        data = mat73.loadmat(fpath)["Result"]
        data = [d for d in data["data"] if d is not None]
        
        # sub-sample the channels for variability
        data = np.array(data, dtype=np.float32)
        data = data[np.random.choice(len(data), 5, replace=False)]

        # normalize the data across time
        data -= np.median(data, axis=1, keepdims=True)
        data /= (iqr(data, axis=1, keepdims=True) + 1e-6)

        # chunk it to 5-sec segments
        seq_len = 5 * 250
        num_seg = len(data[0]) // seq_len
        if num_seg == 0:
            print(f"Skipping {matfile}, too short!")
            continue
        segments = data[:, :num_seg * seq_len]
        segments = segments.reshape(-1, seq_len)

        # store it until we can save
        eeg_repo = np.vstack((eeg_repo, segments))
        if len(eeg_repo) > 1000000:
            # shuffle the rows
            np.random.shuffle(eeg_repo)
            # save the data
            name = os.path.join(proc_path, f"split_{counter}.h5")
            with h5py.File(name, 'w') as h5f:
                h5f.create_dataset(
                'eeg_input',
                data=eeg_repo.astype(np.float32),
                compression='gzip',
                chunks=(1, eeg_repo.shape[1])  # row-wise chunking
            )

            counter += 1
            eeg_repo = np.empty((0, 1250), dtype=np.float32)
            print(f"Saved {name}")

    print("Done")
