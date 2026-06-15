import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader


def test_model(
    model,
    test_dls,
    criterion,
    logger,
    label_encoder,
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
                label_encoder=label_encoder,
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
    label_encoder,
    device=torch.device("cpu"),
    log_artifacts=False,
):
    generalization_error = 0.0
    predictions = []
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
            predictions.append(pred.cpu())
            targets.append(y.cpu())

    y_pred = torch.cat(predictions)
    y_true = torch.cat(targets)

    generalization_error /= len(test_dl.dataset)

    results = {"loss": generalization_error, "mcc": matthews_corrcoef(y_true, y_pred)}

    cls_names = list(label_encoder.keys())
    labels = list(label_encoder.values())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division="warn",
    )

    for c, name in enumerate(cls_names):
        results[f"class_{name}/precision"] = precision[c]
        results[f"class_{name}/recall"] = recall[c]
        results[f"class_{name}/f1_score"] = f1[c]

    results["balanced_accuracy"] = sum(recall) / len(labels)
    results["macro_f1"] = sum(f1) / len(labels)
    if log_artifacts:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        logger.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    # TODO: add roc /auc curves
    return results
