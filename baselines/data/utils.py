from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

__all__ = [
    "Workflow",
    "federate_data",
    "centralize_data",
    "patient_leave_out_split",
    "aami_split",
    "multimodal_cache_exists",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataSplit:
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray | None = None

    @property
    def is_final(self):
        return self.val is None


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


def cache_exists(cache_dir, nb_patients, mode="design"):
    train_ok = all(
        (cache_dir / f"client_{i}" / f"{split}.npy").exists()
        for i in range(nb_patients)
        for split in ("train_data", "train_target")
    )

    test_ok = all((cache_dir / f"{mod}.npy").exists() for mod in ["test_data", "test_target"])

    val_ok = (
        all((cache_dir / f"{mod}.npy").exists() for mod in ["test_data", "test_target"])
        if mode == "design"
        else True
    )

    return train_ok and test_ok and val_ok


def aami_split(mode="design", val_size=0.2, rng=None):
    """
    Split patient records following the AAMI EC57 standard (DS1/DS2).
    DS1 is train, DS2 is test. In design mode a validation set is carved
    out of DS1.

    Returns
    -------
    design : DataSplit(train, val, test)
    test  : DataSplit(train, test)
    """

    DS1 = {
        "101",
        "106",
        "108",
        "109",
        "112",
        "114",
        "115",
        "116",
        "118",
        "119",
        "122",
        "124",
        "201",
        "203",
        "205",
        "207",
        "208",
        "209",
        "215",
        "220",
        "223",
        "230",
    }

    DS2 = {
        "100",
        "103",
        "105",
        "111",
        "113",
        "117",
        "121",
        "123",
        "200",
        "202",
        "210",
        "212",
        "213",
        "214",
        "219",
        "221",
        "222",
        "228",
        "231",
        "232",
        "233",
        "234",
    }

    train_records = np.array(sorted(DS1))
    test_records = np.array(sorted(DS2))

    if rng is not None:
        train_records = rng.permutation(train_records).tolist()

    if mode == "test" or mode == "CV":
        return DataSplit(train=train_records, test=test_records)

    n_val = max(1, int(len(train_records) * val_size))

    return DataSplit(
        train=train_records[n_val:],
        val=train_records[:n_val],
        test=test_records,
    )


def patient_leave_out_split(nb_patient, mode="design", test_size=0.2, val_size=0.1, rng=None):
    """
    Split patient indices into train/test (test mode) or train/val/test (design mode).

    Returns
    -------
    design : (train, val, test)
    test  : (train, test)
    """
    indices = np.arange(nb_patient)
    if rng is not None:
        indices = rng.permutation(indices)
    n_test = max(1, int(nb_patient * test_size))
    if mode == "test":
        return DataSplit(indices[n_test:], indices[:n_test])

    n_val = max(1, int(val_size * nb_patient))
    return DataSplit(
        train=indices[n_test + n_val :],
        test=indices[:n_test],
        val=indices[n_test : n_test + n_val],
    )


def multimodal_cache_exists(cache_dir, nb_train_patients, mode="design"):
    """Check all multi-modal npy files exist for train clients and shared test set."""
    train_ok = all(
        (cache_dir / f"client_{i}" / f"train_{mod}.npy").exists()
        for i in range(nb_train_patients)
        for mod in ("bvp", "acc", "eda_temp", "hr", "target")
    )
    test_ok = all(
        (cache_dir / f"{mod}.npy").exists()
        for mod in ("test_bvp", "test_acc", "test_eda_temp", "test_hr", "test_target")
    )

    val_ok = (
        all(
            (cache_dir / f"{mod}.npy").exists()
            for mod in ("val_bvp", "val_acc", "val_eda_temp", "val_hr", "val_target")
        )
        if mode == "design"
        else True
    )
    return train_ok and test_ok and val_ok
