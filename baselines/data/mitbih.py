from pathlib import Path

import numpy as np
from scipy.signal import filtfilt, firwin
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


def load_mit_bih(val_size=0.1, mode="design"):
    assert mode in ("design", "final"), f"mode must be 'design' or 'final', got {repr(mode)}"

    X_all, y_all, SYM_all, RR_all = _load_mit_bih()
    
    classes_n = np.unique(np.concatenate(list(y_all.values()), axis=0))
    label_encoder = {val: idx for idx, val in enumerate(classes_n)}
    y_all_encoded = {}
    for patient, patient_labels in y_all.items():
        y_all_encoded[patient] = np.array([label_encoder[val] for val in patient_labels])

    splits = aami_split(mode=mode, val_size=val_size)
    train_idx = splits.train 

    X_train = []
    X_val = []
    X_test = []
    
    RR_train = []
    RR_val = []
    RR_test = []

    y_train = []
    y_val = []
    y_test = []

    sym_train = []
    for client_id, patient_id in enumerate(train_idx):
        X_train.append(X_all[patient_id])
        y_train.append(y_all_encoded[patient_id])
        RR_train.append(RR_all[patient_id])
        sym_train.append(SYM_all[patient_id])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    RR_train = np.concatenate(RR_train)
    sym_train = np.concatenate(sym_train)

    beat_symbols = list(AAMI_MAP.keys())
    matched_filters = np.stack(
        [X_train[sym_train == s].mean(axis=0) for s in beat_symbols if (sym_train == s).any()]
    )


    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)

    X_test = np.concatenate([X_all[i] for i in splits.test])
    y_test = np.concatenate([y_all_encoded[i] for i in splits.test])
    RR_test = np.concatenate([RR_all[i] for i in splits.test])


    if splits.val is not None:
        X_val = np.concatenate([X_all[i] for i in splits.val])
        y_val = np.concatenate([y_all_encoded[i] for i in splits.val])
        RR_val = np.concatenate([RR_all[i] for i in splits.val])

    
    return {
        "train": ((X_train - x_mean) / (x_std + 1e-8), RR_train, y_train),
        "test": ((X_test - x_mean) / (x_std + 1e-8), RR_test, y_test),
        "val": ((X_val - x_mean) / (x_std + 1e-8), RR_val, y_val) if mode == "design" else None,
        "matched_filters": matched_filters,
    }





def preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    """
    4-step FIR Kaiser pipeline from Sravan Kumar et al. (2015):
    1. HPF 0.5 Hz  -> remove baseline wander
    2. BSF 59.5-60.5 Hz -> remove power line interference
    3. LPF 100 Hz  -> remove EMG noise
    4. Moving average -> smooth
    """
    hp = firwin(57, cutoff=0.5, window=('kaiser', 8.6), pass_zero=False, fs=FS)
    signal = filtfilt(hp, 1.0, signal)

    bs = firwin(57, cutoff=[59.5, 60.5], window=('kaiser', 8.6), pass_zero=True, fs=FS)
    signal = filtfilt(bs, 1.0, signal)

    lp = firwin(57, cutoff=100.0, window=('kaiser', 8.6), pass_zero=True, fs=FS)
    signal = filtfilt(lp, 1.0, signal)

    kernel = np.ones(5) / 5
    signal = np.convolve(signal, kernel, mode='same')

    return signal


def _load_mit_bih(window_len=64, extension="atr", preprocess=False):
    PACED_RECORDS = {'102', '104', '107', '217'}
    files = [f for f  in PROJECT_ROOT.iterdir() if f.is_file() and f.suffix == ".hea"]
    y_all = {}
    X_all = {}
    SYM_all = {}
    RR_all = {}

    half_window_len = window_len // 2
    beat_symbols = list(AAMI_MAP.keys())



    for f in files:
        if f.stem in PACED_RECORDS:
            continue

        record = wfdb.rdrecord(record_name=f.with_suffix(''))
        annotation = wfdb.rdann(record_name=str(f.with_suffix('')), extension=extension)

        clean_signal = (
            preprocess_ecg(record.p_signal[:, 0]) if preprocess else record.p_signal[:, 0]
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
        for i, sample_idx in enumerate(annotation.sample):
            if annotation.symbol[i] not in beat_symbols:
                continue
            if i in in_flutter_indices:
                continue
            if sample_idx - half_window_len < 0 or sample_idx + half_window_len > record.sig_len:
                continue

            valid_beats.append((i, sample_idx, AAMI_MAP[annotation.symbol[i]]))

        x, y, sym, rr = [], [], [], []
        
        for pos, (i, sample_idx, label) in enumerate(valid_beats):
            pre_rr  = (valid_beats[pos][1] - valid_beats[pos-1][1]) / FS if pos > 0 else 0.0
            post_rr = (valid_beats[pos+1][1] - valid_beats[pos][1]) / FS if pos < len(valid_beats)-1 else 0.0
            local_mean_rr = (
                np.mean(np.diff([b[1] for b in valid_beats[max(0, pos - 80) : pos + 1]])) / FS
                if pos > 0
                else pre_rr
            )
            global_mean_rr = (
                np.mean(np.diff([b[1] for b in valid_beats[max(0, pos - 400) : pos + 1]])) / FS
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