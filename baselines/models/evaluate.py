import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
import torch
from torch.utils.data import DataLoader


def test_model(
    model,
    test_dls,
    criterion,
    logger,
    normal_class=0,
    class_names=None,
    device=torch.device("cpu"),
    log_artifacts=False,
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
                model,
                test_dl,
                criterion,
                logger,
                normal_class,
                class_names=class_names,
                device=device,
                log_artifacts=log_artifacts,
            )
        )

    return results


def _test_model(
    model,
    test_dl,
    criterion,
    logger,
    normal_class,
    class_names=None,
    device=torch.device("cpu"),
    log_artifacts=False,
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

    binary_y_true = (y_true != normal_class).long()
    binary_y_pred = (y_pred != normal_class).long()
    abnormal_score = 1 - y_score[:, normal_class]

    binary_f1_score = f1_score(binary_y_true, binary_y_pred, average="binary")
    binary_accuracy = balanced_accuracy_score(binary_y_true, binary_y_pred)
    precision, recall, _ = precision_recall_curve(binary_y_pred, abnormal_score, pos_label=1)
    ap_score = average_precision_score(binary_y_true, abnormal_score)
    if log_artifacts:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP = {ap_score:.4f}")
        baseline = binary_y_true.float().mean().item()
        ax.axhline(y=baseline, linestyle="--", color="gray", label=f"Baseline = {baseline:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PRC Curve")
        ax.legend()

        logger.log_figure(fig, "prc_curve.png")
        plt.close(fig)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        logger.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    results = {
        "loss": generalization_error,
        "multiclass/balanced_accuracy": accuracy,
        "multiclass/macro_f1": f1score,
        "binary/f1": binary_f1_score,
        "binary/balanced_accuracy": binary_accuracy,
        "binary/ap_score": ap_score,
    }
    if class_names is not None:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            average=None,
            zero_division=0,
        )

        for c, name in enumerate(class_names):
            results[f"class_{name}/precision"] = precision[c]
            results[f"class_{name}/recall"] = recall[c]

    return results
