import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader


def test_model(model, test_dls, criterion, device=torch.device("cpu")):
    """Evaluate a model across one val/test DataLoaders.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).

    Args:
        test_dls: a single ``DataLoader`` or a list of ``DataLoader``s
                  (one per client for federated evaluation).
    """
    if isinstance(test_dls, DataLoader):
        test_dls = [test_dls]

    for test_dl in test_dls:
        results = _test_model(model, test_dl, criterion, device)
        print(results)
    return None


def _test_model(model, test_dl, criterion, pos_class=4, device=torch.device("cpu")):
    generalization_error = 0.0
    correct = 0
    predictions = []
    scores = []
    targets = []
    model.eval()

    with torch.no_grad():
        for *inputs, y in test_dl:
            inputs = [x.to(device) for x in inputs]
            y = y.to(device)
            logits = model(*inputs)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            generalization_error += loss.item()
            scores.append(torch.softmax(logits, dim=1).cpu())
            predictions.append(pred.cpu())
            targets.append(y.cpu())

    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)
    y_score = torch.cat(scores)

    generalization_error /= len(test_dl.dataset)
    accuracy = correct / len(test_dl.dataset)
    f1score = f1_score(y_true, y_pred, average="weighted")
    pos_f1_score = f1_score(y_true, y_pred, pos_label=pos_class, average="binary")
    fpr, tpr, _ = roc_curve(y_true, y_score[:, pos_class], pos_label=pos_class)
    auc = roc_auc_score(y_true, y_score[:, pos_class])

    return {
        "gen_error": generalization_error,
        "weighted_gen_error": None,
        "accuracy": accuracy,
        "weighted f1-score": f1score,
        "pos_class f1-score": pos_f1_score,
        "auc": auc,
    }


# todo will add fpr and tpr to plot roc curve


def _merge_labels(cls_to_merge: list, Y: torch.Tensor):
    """Merge a subset of classes into a single binary label.

    Classes in `cls_to_merge` are mapped to 0, all others to 1.
    Useful for binary evaluation metrics like ROC AUC.

    Parameters
    ----------
    cls_to_merge : list
        Class indices to merge into the negative class (0).
    Y : torch.Tensor
        Integer label tensor of shape (N,).

    Returns
    -------
    np.ndarray
        Binary label array of shape (N,) with dtype int32.
    """

    Y_out = np.ones_like(Y, dtype=np.int32)
    Y_out[np.isin(Y, cls_to_merge)] = 0

    return Y_out