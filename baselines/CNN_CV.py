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

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
        "n_conv_layers": trial.suggest_int("n_conv_layers", 2, 4),
        "base_channels": trial.suggest_categorical("base_channels", [8, 16, 32]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "use_attention": trial.suggest_categorical("use_attention", [False]),
        "downsample": trial.suggest_categorical("downsample", ["maxpool", "stride"]),
        "head": trial.suggest_categorical("head", ["flatten", "gap"]),
    }

    run_name = (
        f"conv{params['n_conv_layers']}_attn={False}_"
        f"lr={params['learning_rate']:.2e}_opt={params['optimizer_name']}"
    )

    with LOGGER.start_run(nested=True, run_name=run_name):
        dataset = load_mit_bih(
            mode="CV",
            window_len=128,
            preprocess=True,
            k_folds=22,
        )
        print(f"Starting trial {trial.number}")
        mcc, f1, min_f1, acc = intra_fold_objective(
            trial, LOGGER, dataset, params, model_name, epochs, batch_size, device
        )
        LOGGER.log_params(params)
        LOGGER.log_metrics({"macro_f1": f1, "mcc": mcc, "min_f1": min_f1, "balanced_acc": acc})

        return mcc


def intra_fold_objective(trial, LOGGER, dataset, params, model_name, epochs, batch_size, device):

    min_f1s = []
    mccs = []
    f1s = []
    accs = []

    for fold_idx, fold in enumerate(dataset["folds"]):
        train_dl = DataLoader(
            MitbihDataset(*fold["train"], get_deriv=False),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            MitbihDataset(*fold["val"], get_deriv=False),
            batch_size=1024,
            shuffle=False,
        )
        model = build_model(
            model_name,
            window_len=128,
            n_conv_layers=params["n_conv_layers"],
            base_channels=params["base_channels"],
            kernel_size=params["kernel_size"],
            use_attention=False,
            downsample=params["downsample"],
            head=params["head"],
        ).to(device)

        optimizer = getattr(torch.optim, params["optimizer_name"])(
            model.parameters(), lr=params["learning_rate"]
        )
        classes = np.array(list(dataset["label_encoder"].values()))
        weights = compute_class_weight("balanced", classes=classes, y=fold["train"][-1])
        weights = torch.Tensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        results = train_model(
            model,
            train_dl,
            optimizer,
            criterion,
            epochs,
            LOGGER,
            dataset["label_encoder"],
            val_dl=val_dl,
            device=device,
            tolerated_steps=6,
        )

        mccs.append(results["mcc"])
        accs.append(results["balanced_acc"])
        f1s.append(results["macro_f1"])
        min_f1s.append(results["min_f1"])

        running_mean = np.mean(f1s)
        trial.report(running_mean, step=fold_idx)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(mccs), np.mean(f1s), np.mean(min_f1s), np.mean(accs)


def main(
    dataset_name="MIT-BIH",
    model_name="CNN",
    epochs=50,
    batch_size=128,
    log_to_mlflow=True,
    seed=42,
):
    init_randomized_envs(seed)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER = MLFlowLogger(log_to_mlflow, experiment_name=f"{dataset_name}/CV")

    with LOGGER.start_run(experiment_id=LOGGER.experiment.experiment_id, run_name=f"{model_name}"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
        )
        study.optimize(
            lambda trial: objective(trial, LOGGER, model_name, epochs, batch_size, device=DEVICE),
            n_trials=15,
            n_jobs=1,
        )

        print(study.best_params)
        print(study.best_value)
        LOGGER.log_params(study.best_params)


if __name__ == "__main__":
    fire.Fire(main)
