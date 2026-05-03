from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader


def test_model(model, test_dls, criterion, pos_class=4, device=torch.device("cpu")):
    """Evaluate a model across one val/test DataLoaders.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).

    Args:
        test_dls: a single ``DataLoader`` or a list of ``DataLoader``s
                  (one per client for federated evaluation).
    """
    if isinstance(test_dls, DataLoader):
        test_dls = [test_dls]
    results = []

    for test_dl in test_dls:
        results.append(_test_model(model, test_dl, criterion, pos_class, device=device))

    return results


def _test_model(model, test_dl, criterion, pos_class, device=torch.device("cpu")):
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
    bin_f1_score = f1_score(
        (y_true == pos_class).long(),
        (y_pred == pos_class).long(),
        average="weighted",
    )
    fpr, tpr, _ = roc_curve(y_true, y_score[:, pos_class], pos_label=pos_class)
    auc = roc_auc_score(
        (y_true == pos_class).long(),
        (y_pred == pos_class).long(),
    )

    return {
        "gen_error": generalization_error,
        "Weighted_gen_error": None,
        "Accuracy": accuracy,
        "Weighted F1-score": f1score,
        "Binary F1-score": bin_f1_score,
        "Binary AUC": auc,
    }


# todo will add fpr and tpr to plot roc curve
