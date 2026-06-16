import copy

from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec
import numpy as np
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
    log_iter_metrics: bool = False,  # Whether to log val metrics every each val_period
    save_model: bool = False,
    device=torch.device("cpu"),
):
    """Train a model for a fixed number of epochs.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).
    The last element of each batch is always treated as the target; all
    preceding elements are forwarded to the model via ``model(*inputs)``.
    """
    best_mcc = -1
    best_f1 = 0
    best_bacc = 0
    best_min_f1_score = 0

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
        if log_iter_metrics:
            logger.log_metrics({"train/loss": empirical_risk}, step=epoch)
        if val_dl is not None and (epoch + 1) % val_period == 0:
            print(f"{'─' * 40}")
            print(f"  Validation @ epoch {epoch + 1}")
            results = test_model(model, val_dl, criterion, logger, label_encoder, device=device)
            min_f1_score = min(
                [results[0][f"class_{name}/f1_score"] for name in label_encoder.keys()]
            )

            if results[0]["macro_f1"] > best_f1:
                best_min_f1_score = min_f1_score
                best_mcc = results[0]["mcc"]
                best_f1 = results[0]["macro_f1"]
                best_bacc = results[0]["balanced_accuracy"]
                best_model = copy.deepcopy(model.state_dict())
                tolerated_steps_ctr = tolerated_steps
            else:
                tolerated_steps_ctr -= 1

            print(
                f"  loss: {results[0]['loss']:.4f}  macro_f1:  {results[0]['macro_f1']:.4f} | mcc:  {results[0]['mcc']:.4f} | min_f1_score: {min_f1_score} | balanced_acc: {results[0]['balanced_accuracy']} (best_macro_f1: {best_f1:.4f} patience: {tolerated_steps_ctr}/{tolerated_steps})"
            )
            print(f"{'─' * 40}")

            if log_iter_metrics:
                metrics = {f"val/{k}": v for k, v in results[0].items()}
                logger.log_metrics(metrics, step=epoch)

            if tolerated_steps_ctr == 0:
                model.load_state_dict(best_model)
                break

    if val_dl is not None:
        model.load_state_dict(best_model)

    if save_model:
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

    return {
        "min_f1": best_min_f1_score,
        "mcc": best_mcc,
        "macro_f1": best_f1,
        "balanced_acc": best_bacc,
    }
