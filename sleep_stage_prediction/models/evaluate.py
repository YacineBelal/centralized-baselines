import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import auc, balanced_accuracy_score, confusion_matrix, f1_score, roc_curve
import torch
from torch.utils.data import DataLoader


def test_model(
    model, test_dls, criterion, pos_class=4, device=torch.device("cpu"), log_artifacts=False
):
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
        results.append(
            _test_model(
                model, test_dl, criterion, pos_class, device=device, log_artifacts=log_artifacts
            )
        )

    return results


def _test_model(
    model, test_dl, criterion, pos_class, device=torch.device("cpu"), log_artifacts=False
):
    generalization_error = 0.0
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
            generalization_error += loss.item()
            scores.append(torch.softmax(logits, dim=1).cpu())
            predictions.append(pred.cpu())
            targets.append(y.cpu())

    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)
    y_score = torch.cat(scores)

    generalization_error /= len(test_dl.dataset)
    accuracy = balanced_accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average="macro")
    binary_f1_score = f1_score(
        (y_true == pos_class).long(), (y_pred == pos_class).long(), average="binary"
    )

    fpr, tpr, _ = roc_curve(y_true, y_score[:, pos_class], pos_label=pos_class)
    auc_score = auc(fpr, tpr)
    if log_artifacts:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        mlflow.log_figure(fig, "roc_curve.png")
        plt.close(fig)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    return {
        "generalization_error": generalization_error,
        # TODO: add "Weighted_gen_error": None,
        "5_class_balanced_accuracy": accuracy,
        "macro_f1_score": f1score,
        "binary_f1_score": binary_f1_score,
        "auc": auc_score,
    }
