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
    LOGGER: MLFlowLogger,
    model_name,
    epochs,
    batch_size,
    device=torch.device("cpu"),
):

    with LOGGER.start_run(nested=True):
        params = {
            "window_len": trial.suggest_int(
                "window_len",
                32,
                256,
                step=32,
            ),
            "trainable_conv": trial.suggest_categorical("trainable_conv", [True, False]),
            "preprocess": trial.suggest_categorical("preprocess", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "optimizer_name": trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]),
        }
        mean_performance = intra_fold_objective(
            trial, LOGGER, params, model_name, epochs, batch_size, device
        )
        LOGGER.log_params(params)
        LOGGER.log_metrics({"mcc": mean_performance})

        return mean_performance



def intra_fold_objective(trial, LOGGER, params, model_name, epochs, batch_size, device):

    dataset = load_mit_bih(
        mode="CV",
        window_len=params["window_len"],
        preprocess=params["preprocess"],
        k_folds=22,
    )
    performances = []
    for fold_idx, fold in enumerate(dataset["folds"]):
        train_dl = DataLoader(MitbihDataset(*fold["train"]), batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(
            MitbihDataset(*fold["val"]),
            batch_size=1024,
            shuffle=False,
        )
        model = build_model(
            model_name,
            matched_filters=fold["matched_filters"],
            trainable_conv=params["trainable_conv"],
        ).to(device)

        optimizer = getattr(torch.optim, params["optimizer_name"])(
            model.parameters(), lr=params["learning_rate"]
        )
        classes = np.array(list(dataset["label_encoder"].values()))
        weights = compute_class_weight("balanced", classes=classes, y=fold["train"][-1])
        weights = torch.Tensor(weights).to(device)
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
            device=device,
        )

        performances.append(performance)
        running_mean = np.mean(performances)
        trial.report(running_mean, step=fold_idx)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return running_mean


def main(
    dataset_name="MIT-BIH",
    model_name="tinyCNN",
    epochs=30,
    batch_size=128,
    log_to_mlflow=True,
    seed=42,
):
    init_randomized_envs(seed)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER = MLFlowLogger(log_to_mlflow, experiment_name=f"{dataset_name}/{model_name}_search")

    with LOGGER.start_run(experiment_id=LOGGER.experiment.experiment_id):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(
            lambda trial: objective(trial, LOGGER, model_name, epochs, batch_size, device=DEVICE),
            n_trials=10,
            n_jobs=10,
        )

        LOGGER.log_params(study.best_params)
        LOGGER.log_metrics({"best_mcc": study.best_value})


if __name__ == "__main__":
    fire.Fire(main)
