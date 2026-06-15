import copy

from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np
import optuna
import torch
from tqdm import tqdm

from baselines.models import test_model

__all__ = ["train_model"]


def train_model(
    model,
    train_dl,
    optimizer,
    criterion,
    epochs,
    logger,
    label_encoder,
    val_dl=None,
    val_period=5,
    tolerated_steps=3,
    trial: optuna.Trial = None,
    device=torch.device("cpu"),
):
    """Train a model for a fixed number of epochs.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).
    The last element of each batch is always treated as the target; all
    preceding elements are forwarded to the model via ``model(*inputs)``.
    """
    best_mcc = -np.inf
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
        logger.log_metrics({"train/loss": empirical_risk}, step=epoch)
        if val_dl is not None and (epoch + 1) % val_period == 0:
            print(f"{'─' * 40}")
            print(f"  Validation @ epoch {epoch + 1}")
            results = test_model(model, val_dl, criterion, logger, label_encoder, device=device)
            metrics = {f"val/{k}": v for k, v in results[0].items()}
            logger.log_metrics(metrics, step=epoch)
            if trial:
                trial.report(results[0]["mcc"], epoch)

            if results[0]["mcc"] > best_mcc:
                best_mcc = results[0]["mcc"]
                best_model = copy.deepcopy(model.state_dict())
                tolerated_steps_ctr = tolerated_steps
            else:
                tolerated_steps_ctr -= 1

            print(
                f"  loss: {metrics['val/loss']:.4f}  | balanced_accuracy: {metrics['val/balanced_accuracy']} | mcc:  {metrics['val/mcc']:.4f} | macro_f1:  {metrics['val/macro_f1']:.4f}  (best mcc: {best_mcc:.4f} patience: {tolerated_steps_ctr}/{tolerated_steps})"
            )
            print(f"{'─' * 40}")

            if tolerated_steps_ctr == 0:
                model.load_state_dict(best_model)
                break

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if val_dl is not None:
        model.load_state_dict(best_model)

    *inputs, y = next(iter(train_dl))
    numpy_inputs = [x.numpy() for x in inputs]
    input_schema = Schema(
        [
            TensorSpec(np.dtype("float32"), shape=inputs[0].shape, name="X"),
            TensorSpec(np.dtype("float32"), shape=inputs[1].shape, name="RR"),
        ]
    )
    output_schema = Schema([TensorSpec(np.dtype("float32"), shape=(1, 3), name="y")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    model.to("cpu")
    logger.log_model(
        model,
        input_example=numpy_inputs,
        signature=signature,
    )
    model.to(device)

    return best_mcc
