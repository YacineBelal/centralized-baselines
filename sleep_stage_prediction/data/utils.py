from enum import Enum
from pathlib import Path

import numpy as np

__all__ = [
    "Workflow",
    "federate_data",
    "centralize_data",
    "patient_leave_out_split",
    "multimodal_cache_exists",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Workflow(Enum):
    FEDERATED_CROSS_DEVICE = 1
    FEDERATED_CROSS_SILO = 2
    CENTRALIZED = 3


def save_data_array(file: Path, arr):
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, arr)

def split_dataset_chronological(X, y, test_size=0.2):
    split = int(X.shape[0] * (1 - test_size))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    return X_train, X_test, y_train, y_test


def split_dataset(X, y, test_size=0.2, rng=None, shuffle=True):
    dataset_len = X.shape[0]
    idx = np.arange(dataset_len)
    if shuffle and rng:
        idx = rng.permutation(idx)

    X_train = X[idx[int(dataset_len * test_size) :]]

    y_train = y[idx[int(dataset_len * test_size) :]]

    X_test = X[idx[: int(dataset_len * test_size)]]
    y_test = y[idx[: int(dataset_len * test_size)]]

    return X_train, X_test, y_train, y_test


def centralize_data(X, y, dataset_name, rng, test_size=0.2):
    X = np.concat(X)
    y = np.concat(y)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, rng)
    folder = PROJECT_ROOT / "data" / "processed" / dataset_name
    save_data_array(
        folder / "server" / "train_data",
        np.permute_dims(X_train, axes=(0, 2, 1)).astype("float32"),
    )
    save_data_array(
        folder / "server" / "test_data",
        np.permute_dims(X_test, axes=(0, 2, 1)).astype("float32"),
    )
    save_data_array(
        folder / "server" / "train_target",
        y_train,
    )
    save_data_array(
        folder / "server" / "test_target",
        y_test,
    )

    return (X_train, X_test, y_train, y_test)


def federate_data(X, y, dataset_name, rng, test_size=0.2):
    split_data = [
        # TODO: add attribute for chronological testing
        split_dataset_chronological(
            X[i],
            y[i],
            test_size,
        )
        for i in range(len(X))
    ]
    folder = PROJECT_ROOT / "data" / "processed" / dataset_name
    for idx, (x_train, x_test, y_train, y_test) in enumerate(split_data):
        save_data_array(
            folder / f"client_{idx}" / "train_data",
            x_train.astype("float32"),
        )
        save_data_array(folder / f"client_{idx}" / "train_target", y_train)
        save_data_array(
            folder / f"client_{idx}" / "test_data",
            x_test.astype("float32"),
        )
        save_data_array(folder / f"client_{idx}" / "test_target", y_test)

    return split_data


def cache_exists(cache_dir, nb_patients):
    train_ok = all(
        (cache_dir / f"client_{i}" / f"{split}.npy").exists()
        for i in range(nb_patients)
        for split in ("train_data", "train_target", "test_data", "test_target")
    )

    test_ok = all((cache_dir / "test" / f"{mod}.npy").exists() for mod in ["data", "target"])
    return train_ok and test_ok


def patient_leave_out_split(nb_patients, test_size=0.2, rng=None):
    """Return (train_indices, test_indices) for a patient-level random split."""
    indices = np.arange(nb_patients)
    if rng is not None:
        indices = rng.permutation(indices)
    n_test = max(1, int(nb_patients * test_size))
    return indices[n_test:], indices[:n_test]


def multimodal_cache_exists(cache_dir, nb_train_patients):
    """Check all multi-modal npy files exist for train clients and shared test set."""
    train_ok = all(
        (cache_dir / f"client_{i}" / f"train_{mod}.npy").exists()
        for i in range(nb_train_patients)
        for mod in ("bvp", "acc", "eda_temp", "hr", "target")
    )
    test_ok = all(
        (cache_dir / "test" / f"{mod}.npy").exists()
        for mod in ("bvp", "acc", "eda_temp", "hr", "target")
    )
    return train_ok and test_ok
