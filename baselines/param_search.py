import fire
import numpy as np
import optuna
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.data import MitbihDataset, load_mit_bih
from baselines.models import (
    build_model,
    train_model,
)
from baselines.utils import MLFlowLogger, init_randomized_envs


def objective(
    trial,
    dataset_name="MIT-BIH",
    model_name="CNN",
    epochs=30,
    batch_size=128,
    val_size=0.1,
    val_period=5,
    LOGGER: MLFlowLogger = None,
    seed=42,
):
    init_randomized_envs(seed)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with LOGGER.start_run(nested=True):
        params = {
            "window_len": trial.suggest_int(
                "window_len",
                32,
                256,
                step=32,
            ),
            "trainable_conv": trial.suggest_categorical("trainable_conv", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "optimizer_name": trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]),
        }
        dataset = load_mit_bih(val_size=val_size, window_len=params["window_len"])
        train_dl = DataLoader(
            MitbihDataset(*dataset["train"]), batch_size=batch_size, shuffle=True
        )
        val_dl = DataLoader(
            MitbihDataset(*dataset["val"]),
            batch_size=1024,
            shuffle=False,
        )
        model = build_model(
            model_name,
            matched_filters=dataset["matched_filters"],
            trainable_conv=params["trainable_conv"],
        ).to(DEVICE)
        optimizer = getattr(torch.optim, params["optimizer_name"])(
            model.parameters(), lr=params["learning_rate"]
        )
        classes = np.array(list(dataset["label_encoder"].values()))
        weights = compute_class_weight("balanced", classes=classes, y=dataset["train"][-1])
        weights = torch.Tensor(weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")

        performance = train_model(
            model,
            train_dl,
            optimizer,
            criterion,
            epochs,
            LOGGER,
            dataset["label_encoder"],
            val_dl=val_dl,
            val_period=val_period,
            device=DEVICE,
            trial=trial,
        )
        LOGGER.log_params(params)
        LOGGER.log_metrics({"mcc": performance})

    return performance


def main(
    dataset_name="MIT-BIH",
    model_name="tinyCNN",
    epochs=30,
    batch_size=128,
    log_to_mlflow=True,
    seed=42,
):

    LOGGER = MLFlowLogger(log_to_mlflow, experiment_name=f"{dataset_name}/{model_name}_search")
    with LOGGER.start_run(experiment_id=LOGGER.experiment.experiment_id, nested=True):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial, dataset_name, model_name, epochs, batch_size, LOGGER=LOGGER
            ),
            n_trials=10,
        )

        LOGGER.log_params(study.best_params)
        LOGGER.log_metrics({"best_mcc": study.best_value})


if __name__ == "__main__":
    fire.Fire(main)
