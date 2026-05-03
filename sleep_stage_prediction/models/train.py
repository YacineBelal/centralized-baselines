import copy

import mlflow
import numpy as np
import torch
from tqdm import tqdm

from sleep_stage_prediction.models import test_model

__all__ = ["train_model"]


def train_model(
    model,
    train_dl,
    optimizer,
    criterion,
    epochs,
    val_dl=None,
    val_period=5,
    tolerated_steps=2,
    device=torch.device("cpu"),
):
    """Train a model for a fixed number of epochs.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).
    The last element of each batch is always treated as the target; all
    preceding elements are forwarded to the model via ``model(*inputs)``.
    """
    best_f1_score = -np.inf
    tolerated_steps_ctr = tolerated_steps
    best_model = copy.deepcopy(model.state_dict())
    for epoch in tqdm(range(epochs)):
        model.train()
        empirical_risk = 0.0
        for *inputs, y in train_dl:
            inputs = [x.to(device) for x in inputs]
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(*inputs)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            empirical_risk += loss.item()

        empirical_risk /= len(train_dl.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] | Train loss: {empirical_risk:.4f}")
        mlflow.log_metrics({"Train loss": empirical_risk}, step=epoch)
        if val_dl is not None and (epoch + 1) % val_period == 0:
            print(f"{'─' * 40}")
            print(f"  Validation @ epoch {epoch + 1}")
            results = test_model(model, val_dl, criterion, device=device)
            metrics = {f"val_{k}": v for k, v in results[0].items()}
            mlflow.log_metrics(metrics, step=epoch)
            if results[0]["Binary F1-score"] > best_f1_score:
                best_f1_score = results[0]["Binary F1-score"]
                best_model = copy.deepcopy(model.state_dict())
                tolerated_steps_ctr = tolerated_steps
            else:
                tolerated_steps_ctr -= 1

            print(
                f"  Binary F1:  {results[0]['Binary F1-score']:.4f}  (best: {best_f1_score:.4f}  patience: {tolerated_steps_ctr}/{tolerated_steps})"
            )
            print(f"{'─' * 40}")

            if tolerated_steps_ctr == 0:
                model.load_state_dict(best_model)
                break

    if val_dl is not None:
        model.load_state_dict(best_model)

    mlflow.pytorch.log_model(model, name="multiscale_cnn")