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
    epochs=50,
    batch_size=128,
    lr=0.0001,
    mode="test",
    log_to_mlflow=True,
    seed=42,
):
    init_randomized_envs(seed)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_mit_bih(
        mode=mode,
        window_len=128
    )

    train_dl = DataLoader(MitbihDataset(*dataset["train"]), batch_size=batch_size, shuffle=True)

    test_dl = DataLoader(
        MitbihDataset(*dataset["test"]),
        batch_size=1024,
        shuffle=False,
    )

    LOGGER = MLFlowLogger(enabled=log_to_mlflow, experiment_name=f"{dataset_name}/{mode}")

    run_name = f"{model_name}"
    with LOGGER.start_run(run_name=run_name):
        model = build_model(
            name=model_name,
            n_conv_layers=4,
            base_channels=32,
            kernel_size=7,
            use_attention=True,
            downsample="stride",
            head="flatten",
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        classes = np.array(list(dataset["label_encoder"].values()))
        weights = compute_class_weight("balanced", classes=classes, y=dataset["train"][-1])
        weights = torch.Tensor(weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        LOGGER.log_params(
            {
                "window_len": window_len,
                "batch_size": batch_size,
                "lr": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
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
                LOGGER,
                dataset["label_encoder"],
                val_dl=val_dl,
                log_iter_metrics=True,
                device=DEVICE,
            )
        else:
            train_model(
                model,
                train_dl,
                optimizer,
                criterion,
                epochs,
                LOGGER,
                dataset["label_encoder"],
                log_iter_metrics=True,
                device=DEVICE,
            )

        results = test_model(
            model,
            test_dl,
            criterion,
            LOGGER,
            label_encoder=dataset["label_encoder"],
            device=DEVICE,
            log_artifacts=True,
        )
        test_metrics = {f"test/{k}": v for k, v in results[0].items()}
        print(test_metrics)
        LOGGER.log_metrics(test_metrics)
        LOGGER.log_text(str(model), "architecture.txt")


if __name__ == "__main__":
    fire.Fire(main)
