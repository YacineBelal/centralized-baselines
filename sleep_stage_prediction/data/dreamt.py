from pathlib import Path

import numpy as np
import pandas as pd

from .utils import cache_exists, federate_data, multimodal_cache_exists, patient_leave_out_split

COLS_TO_DROP = [
    "TIMESTAMP",
    "IBI",
    "Obstructive_Apnea",
    "Central_Apnea",
    "Hypopnea",
    "Multiple_Events",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# TODO: remove workflow if not used
def load_dreamt(nb_patients, workflow, frequency=64, seed=42):
    # TODO: a dataclass to encapsulate rng etc
    # TODO: seed should be saved here too
    path = PROJECT_ROOT / "data" / "processed" / "dreamt"
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    if cache_exists(path, nb_patients):
        for idx in range(nb_patients):
            client_path = path / f"client_{idx}"
            X_train.append(np.load(client_path / "train_data.npy"))
            X_test.append(np.load(client_path / "test_data.npy"))
            y_train.append(np.load(client_path / "train_target.npy"))
            y_test.append(np.load(client_path / "test_target.npy"))

    else:
        rng = np.random.default_rng(seed=seed)
        signals, labels = _load_dreamt(
            nb_patients,
            rng,
            frequency,
        )
        signals_preprocessed, labels_preprocessed = _preprocess_dreamt(signals, labels)
        split_data = federate_data(signals_preprocessed, labels_preprocessed, "dreamt", rng)

        for client_data in split_data:
            X_train.append(client_data[0])
            X_test.append(client_data[1])
            y_train.append(client_data[2])
            y_test.append(client_data[3])

    X_train = np.concat(X_train, axis=0)
    y_train = np.concat(y_train, axis=0)

    return X_train, X_test, y_train, y_test


def _load_dreamt(
    nb_patients,
    rng,
    frequency=64,
):
    # TODO: download dataset if not present
    folder = PROJECT_ROOT / "data" / "raw" / "dreamt" / f"data_{frequency}Hz"
    if not folder.exists():
        raise FileNotFoundError("Data Folder does not exist")

    patient_file_list = [f for f in folder.iterdir() if f.is_file()]
    rng.shuffle(patient_file_list)
    labels = []
    signals = []
    nb_patients = min(100, nb_patients)
    for _ in range(nb_patients):
        patient_file = patient_file_list.pop()
        df = pd.read_csv(patient_file)
        df = df.drop(columns=COLS_TO_DROP)
        df = df[~df["Sleep_Stage"].isin(["P", "Missing"])]
        labels.append(df["Sleep_Stage"].to_numpy())
        signals.append(df.drop(columns=["Sleep_Stage"]).to_numpy())

    return signals, labels


def load_dreamt_multimodal(nb_patients, frequency=64, seed=42):
    """Load DREAMT with a patient leave-out split and sensor-specific resolutions.

    20% of patients are held out as a shared test set; the remaining 80% form
    per-client train sets.  Each modality is downsampled to its natural rate:

    - BVP:      (N, 1, 1920)  64 Hz, 1 channel
    - ACC:      (N, 3, 960)   32 Hz, 3 channels
    - EDA+Temp: (N, 2, 120)    4 Hz, 2 channels
    - HR:       (N, 30)        1 Hz, 1 channel

    Returns:
        Tuple of 10 arrays (train then test):
        X_bvp_train, X_acc_train, X_eda_temp_train, X_hr_train, y_train,
        X_bvp_test,  X_acc_test,  X_eda_temp_test,  X_hr_test,  y_test

        Train arrays are concatenated across all train patients.
    """
    path = PROJECT_ROOT / "data" / "processed" / "dreamt_multimodal"
    rng = np.random.default_rng(seed=seed)

    signals, labels = _load_dreamt(nb_patients, rng, frequency)
    bvps, accs, eda_temps, hrs, ys = _preprocess_dreamt_multimodal(signals, labels)

    train_idx, test_idx = patient_leave_out_split(nb_patients, test_size=0.2, rng=rng)
    nb_train = len(train_idx)

    if multimodal_cache_exists(path, nb_train):
        X_bvp_train = [np.load(path / f"client_{i}" / "train_bvp.npy") for i in range(nb_train)]
        X_acc_train = [np.load(path / f"client_{i}" / "train_acc.npy") for i in range(nb_train)]
        X_eda_temp_train = [np.load(path / f"client_{i}" / "train_eda_temp.npy") for i in range(nb_train)]
        X_hr_train = [np.load(path / f"client_{i}" / "train_hr.npy") for i in range(nb_train)]
        y_train = [np.load(path / f"client_{i}" / "train_target.npy") for i in range(nb_train)]
        X_bvp_test = np.load(path / "test" / "bvp.npy")
        X_acc_test = np.load(path / "test" / "acc.npy")
        X_eda_temp_test = np.load(path / "test" / "eda_temp.npy")
        X_hr_test = np.load(path / "test" / "hr.npy")
        y_test = np.load(path / "test" / "target.npy")
    else:
        from .utils import save_data_array

        X_bvp_train, X_acc_train, X_eda_temp_train, X_hr_train, y_train = [], [], [], [], []
        for client_i, patient_i in enumerate(train_idx):
            client_dir = path / f"client_{client_i}"
            save_data_array(client_dir / "train_bvp", bvps[patient_i])
            save_data_array(client_dir / "train_acc", accs[patient_i])
            save_data_array(client_dir / "train_eda_temp", eda_temps[patient_i])
            save_data_array(client_dir / "train_hr", hrs[patient_i])
            save_data_array(client_dir / "train_target", ys[patient_i])
            X_bvp_train.append(bvps[patient_i])
            X_acc_train.append(accs[patient_i])
            X_eda_temp_train.append(eda_temps[patient_i])
            X_hr_train.append(hrs[patient_i])
            y_train.append(ys[patient_i])

        X_bvp_test = np.concatenate([bvps[i] for i in test_idx])
        X_acc_test = np.concatenate([accs[i] for i in test_idx])
        X_eda_temp_test = np.concatenate([eda_temps[i] for i in test_idx])
        X_hr_test = np.concatenate([hrs[i] for i in test_idx])
        y_test = np.concatenate([ys[i] for i in test_idx])

        test_dir = path / "test"
        save_data_array(test_dir / "bvp", X_bvp_test)
        save_data_array(test_dir / "acc", X_acc_test)
        save_data_array(test_dir / "eda_temp", X_eda_temp_test)
        save_data_array(test_dir / "hr", X_hr_test)
        save_data_array(test_dir / "target", y_test)

    return (
        np.concatenate(X_bvp_train),
        np.concatenate(X_acc_train),
        np.concatenate(X_eda_temp_train),
        np.concatenate(X_hr_train),
        np.concatenate(y_train),
        X_bvp_test,
        X_acc_test,
        X_eda_temp_test,
        X_hr_test,
        y_test,
    )


def _preprocess_dreamt_multimodal(signals, labels):
    """Chunk raw signals into 30-second windows split by modality.

    Returns five per-patient lists: bvps, accs, eda_temps, hrs, ys.
    Arrays are already in (C, T) format ready for Conv1d.
    """
    fs = 64
    window_samples = fs * 30  # 1920

    all_labels = np.concatenate(
        [
            np.array([y_p[i * window_samples] for i in range(len(y_p) // window_samples)])
            for y_p in labels
        ]
    )
    classes = np.unique(all_labels)
    label_encoder = {val: idx for idx, val in enumerate(classes)}

    bvps, accs, eda_temps, hrs, ys = [], [], [], [], []
    for data, y_p in zip(signals, labels):
        n_windows = data.shape[0] // window_samples
        bvp_windows, acc_windows, eda_temp_windows, hr_windows, y_windows = [], [], [], [], []
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            bvp_windows.append(data[start:end, 0])
            acc_windows.append(data[start:end:2, 1:4])
            eda_temp_windows.append(data[start:end:16, 4:6])
            hr_windows.append(data[start:end:64, 6])
            y_windows.append(label_encoder[y_p[start]])

        bvps.append(np.expand_dims(np.stack(bvp_windows), axis=1).astype("float32"))
        accs.append(np.permute_dims(np.stack(acc_windows), axes=(0, 2, 1)).astype("float32"))
        eda_temps.append(
            np.permute_dims(np.stack(eda_temp_windows), axes=(0, 2, 1)).astype("float32")
        )
        hrs.append(np.stack(hr_windows).astype("float32"))
        ys.append(np.array(y_windows))

    return bvps, accs, eda_temps, hrs, ys


def _preprocess_dreamt(signals, labels, signal_len=64):
    signals_preprocessed = []
    labels_preprocessed = []
    for X_p, y_p in zip(signals, labels):
        signals_preprocessed.append(X_p[:-1].reshape(-1, signal_len, 7))
        labels_preprocessed.append(y_p[:-1].reshape(-1, signal_len)[:, 0])

    classes_n = np.unique(np.concat(labels_preprocessed, axis=0))
    label_encoder = {val: idx for idx, val in enumerate(classes_n)}
    labels_preprocessed_encoded = [
        np.array([label_encoder[val] for val in patient_labels])
        for patient_labels in labels_preprocessed
    ]

    return signals_preprocessed, labels_preprocessed_encoded
