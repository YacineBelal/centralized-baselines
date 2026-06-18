from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, firwin, resample_poly
from sklearn.model_selection import KFold
import wfdb

from .utils import aami_split

PROJECT_ROOT = "data/raw/mit-bih-arrhythmia-database-1.0.0/"
PROJECT_ROOT = Path(PROJECT_ROOT)
AAMI_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "S": "S",
    "J": "S",
    "V": "V",
    "E": "V",
    # "F":"F", #Removed F and Q as there are not relevant to arrhythmia detection
    # "/":'Q', "f": "Q", "Q":"Q",
}
FS = 360  # MIT-BIH sampling frequency


def load_mit_bih(window_len=64, val_size=0.1, preprocess=False, mode="design", k_folds=None):
    assert mode in ("design", "CV", "final")

    X_all, y_all, SYM_all, RR_all = _load_mit_bih(window_len=window_len, preprocess=preprocess)

    classes_n = np.unique(np.concatenate(list(y_all.values())))
    label_encoder = {val: idx for idx, val in enumerate(classes_n)}
    y_all_encoded = {
        pid: np.array([label_encoder[v] for v in labels]) for pid, labels in y_all.items()
    }

    beat_symbols = list(AAMI_MAP.keys())
    splits = aami_split(mode=mode, val_size=val_size)

    # test set is always the same
    X_test = np.concatenate([X_all[i] for i in splits.test])
    y_test = np.concatenate([y_all_encoded[i] for i in splits.test])
    RR_test = np.concatenate([RR_all[i] for i in splits.test])

    if mode == "CV" and isinstance(k_folds, int):
        kf = KFold(n_splits=k_folds, shuffle=True)
        folds = []
        for train_indices, val_indices in kf.split(splits.train):
            train_records = splits.train[train_indices]
            val_records = splits.train[val_indices]
            fold = _build_fold(
                train_records, val_records, X_all, y_all_encoded, RR_all, SYM_all, beat_symbols
            )
            # normalize test with this fold's stats
            fold["test"] = ((X_test - fold["x_mean"]) / (fold["x_std"] + 1e-8), RR_test, y_test)
            folds.append(fold)
        return {"folds": folds, "label_encoder": label_encoder}

    else:
        val_records = splits.val if splits.val is not None else None
        fold = _build_fold(
            splits.train, val_records, X_all, y_all_encoded, RR_all, SYM_all, beat_symbols
        )
        fold["test"] = ((X_test - fold["x_mean"]) / (fold["x_std"] + 1e-8), RR_test, y_test)
        fold["label_encoder"] = label_encoder
        return fold


def _build_fold(
    record_ids_train, record_ids_val, X_all, y_all_encoded, RR_all, SYM_all, beat_symbols
):
    """Given train and val record IDs, return processed arrays for one fold."""
    X_train = np.concatenate([X_all[pid] for pid in record_ids_train])
    y_train = np.concatenate([y_all_encoded[pid] for pid in record_ids_train])
    RR_train = np.concatenate([RR_all[pid] for pid in record_ids_train])
    sym_train = np.concatenate([SYM_all[pid] for pid in record_ids_train])

    # matched filters from raw train segments
    matched_filters = np.stack(
        [X_train[sym_train == s].mean(axis=0) for s in beat_symbols if (sym_train == s).any()]
    ).astype("float32")

    # normalize
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    X_train = (X_train - x_mean) / (x_std + 1e-8)

    fold = {
        "train": (X_train, RR_train, y_train),
        "matched_filters": matched_filters,
        "x_mean": x_mean,
        "x_std": x_std,
    }

    if record_ids_val is not None:
        X_val = np.concatenate([X_all[pid] for pid in record_ids_val])
        y_val = np.concatenate([y_all_encoded[pid] for pid in record_ids_val])
        RR_val = np.concatenate([RR_all[pid] for pid in record_ids_val])
        fold["val"] = ((X_val - x_mean) / (x_std + 1e-8), RR_val, y_val)
    else:
        fold["val"] = None

    return fold

def _preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """
    4-step FIR Kaiser pipeline from Sravan Kumar et al. (2015):
    1. HPF 0.5 Hz  -> remove baseline wander
    2. BSF 59.5-60.5 Hz -> remove power line interference
    3. LPF 100 Hz  -> remove EMG noise
    4. Moving average -> smooth
    """
    hp = firwin(57, cutoff=0.5, window=("kaiser", 8.6), pass_zero=False, fs=FS)
    signal = filtfilt(hp, 1.0, signal)

    bs = firwin(57, cutoff=[59.5, 60.5], window=("kaiser", 8.6), pass_zero=True, fs=FS)
    signal = filtfilt(bs, 1.0, signal)

    lp = firwin(57, cutoff=100.0, window=("kaiser", 8.6), pass_zero=True, fs=FS)
    signal = filtfilt(lp, 1.0, signal)

    kernel = np.ones(5) / 5
    signal = np.convolve(signal, kernel, mode="same")

    return signal


def _load_mit_bih(window_len, preprocess, target_frequency=128, extension="atr"):
    PACED_RECORDS = {"102", "104", "107", "217"}
    files = [f for f in PROJECT_ROOT.iterdir() if f.is_file() and f.suffix == ".hea"]
    y_all = {}
    X_all = {}
    SYM_all = {}
    RR_all = {}

    g = gcd(FS, target_frequency)
    up = target_frequency // g
    down = FS // g

    half_window_len = window_len // 2
    beat_symbols = list(AAMI_MAP.keys())

    for f in files:
        if f.stem in PACED_RECORDS:
            continue

        record = wfdb.rdrecord(record_name=f.with_suffix(""))
        annotation = wfdb.rdann(record_name=str(f.with_suffix("")), extension=extension)

        resampled_signal = resample_poly(record.p_signal[:, 0], up, down)
        clean_signal = _preprocess_ecg(resampled_signal) if preprocess else resampled_signal
        resampled_sig_len = len(clean_signal)

        resampled_samples = np.round(np.array(annotation.sample) * (target_frequency / FS)).astype(
            int
        )

        in_flutter = False
        in_flutter_indices = set()
        for i, sym in enumerate(annotation.symbol):
            if sym == "[":
                in_flutter = True
            elif sym == "]":
                in_flutter = False
            elif in_flutter:
                in_flutter_indices.add(i)

        valid_beats = []
        for i, sample_idx in enumerate(resampled_samples):
            if annotation.symbol[i] not in beat_symbols:
                continue
            if i in in_flutter_indices:
                continue
            if (
                sample_idx - half_window_len < 0
                or sample_idx + half_window_len > resampled_sig_len
            ):
                continue

            valid_beats.append((i, sample_idx, AAMI_MAP[annotation.symbol[i]]))

        x, y, sym, rr = [], [], [], []

        for pos, (i, sample_idx, label) in enumerate(valid_beats):
            pre_rr = (
                (valid_beats[pos][1] - valid_beats[pos - 1][1]) / target_frequency
                if pos > 0
                else 0.0
            )
            post_rr = (
                (valid_beats[pos + 1][1] - valid_beats[pos][1]) / target_frequency
                if pos < len(valid_beats) - 1
                else 0.0
            )
            local_mean_rr = (
                np.mean(np.diff([b[1] for b in valid_beats[max(0, pos - 80) : pos + 1]]))
                / target_frequency
                if pos > 0
                else pre_rr
            )
            global_mean_rr = (
                np.mean(np.diff([b[1] for b in valid_beats[max(0, pos - 400) : pos + 1]]))
                / target_frequency
                if pos > 0
                else pre_rr
            )

            pre_rr_local = pre_rr / local_mean_rr if local_mean_rr > 0 else 1.0
            post_rr_local = post_rr / local_mean_rr if local_mean_rr > 0 else 1.0
            pre_rr_global = pre_rr / global_mean_rr if global_mean_rr > 0 else 1.0
            post_rr_global = post_rr / global_mean_rr if global_mean_rr > 0 else 1.0
            rr.append([pre_rr_local, post_rr_local, pre_rr_global, post_rr_global])

            seg = clean_signal[sample_idx - half_window_len : sample_idx + half_window_len]
            x.append(seg)
            y.append(label)
            sym.append(annotation.symbol[i])

        X_all[f.stem] = np.expand_dims(np.stack(x), axis=1).astype("float32")
        y_all[f.stem] = np.array(y)
        SYM_all[f.stem] = np.array(sym)
        RR_all[f.stem] = np.array(rr, dtype="float32")

    return X_all, y_all, SYM_all, RR_all