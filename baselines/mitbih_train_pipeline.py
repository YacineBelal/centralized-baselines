import fire
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.data import MitbihDataset, load_mit_bih
from baselines.models import (
    build_model,
    test_model,
    train_model,
)
from baselines.utils import MLFlowLogger, init_randomized_envs


def main(
    dataset_name="MIT-BIH",
    model_name="CNN",
    window_len=128,
    epochs=1,
    batch_size=128,
    normal_class=0,  # TODO: do not provide as argument, collect instead
    lr=0.001,
    mode="final",
    val_size=0.1,
    val_period=5,
    log_to_mlflow=True,
    seed=42,
):
    init_randomized_envs(seed)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_mit_bih(val_size=val_size, mode=mode)

    train_dl = DataLoader(MitbihDataset(*dataset["train"]), batch_size=batch_size, shuffle=True)

    test_dl = DataLoader(
        MitbihDataset(*dataset["test"]),
        batch_size=1024,
        shuffle=False,
    )

    LOGGER = MLFlowLogger(
        enabled=log_to_mlflow, experiment_name=f"{dataset_name}/{model_name}_{mode}"
    )

    with LOGGER.start_run():
        model = build_model(model_name).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        classes = np.unique(dataset["train"][-1])
        print(classes)
        print(dataset["classes_names"])
        weights = compute_class_weight("balanced", classes=classes, y=dataset["train"][-1])

        weights = torch.Tensor(weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        LOGGER.log_params(
            {
                "window_len": window_len,
                "batch_size": batch_size,
                "lr": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "val_size": val_size,
                "mode": mode,
            }
        )

        if mode == "design":
            val_dl = DataLoader(
                MitbihDataset(*dataset["val"]),
                batch_size=1024,
                shuffle=False,
            )
            train_model(
                model,
                train_dl,
                optimizer,
                criterion,
                epochs,
                logger=LOGGER,
                val_dl=val_dl,
                val_period=val_period,
                normal_class=normal_class,
                device=DEVICE,
            )
        else:
            train_model(
                model, train_dl, optimizer, criterion, epochs, logger=LOGGER, device=DEVICE
            )

        results = test_model(
            model,
            test_dl,
            criterion,
            LOGGER,
            normal_class=normal_class,
            class_names=dataset["classes_names"],
            device=DEVICE,
            log_artifacts=True,
        )
        test_metrics = {f"test/{k}": v for k, v in results[0].items()}
        print(test_metrics)
        LOGGER.log_metrics(test_metrics)
        LOGGER.log_text(str(model), "architecture.txt")
        LOGGER.set_tags({"model_type": model_name, "dataset": "MIT-BIH"})


if __name__ == "__main__":
    fire.Fire(main)


