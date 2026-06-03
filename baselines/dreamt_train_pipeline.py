import fire
import mlflow
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.data import MultiModalDreamtDataset, load_dreamt_multimodal
from baselines.models import (
    MultiScaleCNN,
    test_model,
    train_model,
)


def main(
    nb_patients=100,
    frequency=64,
    test_size=0.2,
    val_size=0.1,
    n_fft=32,
    epochs=50,
    batch_size=32,
    lr=0.00005,
    mode="design",
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_dreamt_multimodal(
        nb_patients,
        frequency,
        val_size,
        test_size,
        seed,
        mode,
    )
    train_dl = DataLoader(
        MultiModalDreamtDataset(n_fft=n_fft, *dataset["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_dl = DataLoader(
        MultiModalDreamtDataset(n_fft=n_fft, *dataset["test"]),
        batch_size=128,
        shuffle=False,
    )
    mlflow.set_experiment("MULTICNN hyperparameterization")
    with mlflow.start_run():
        model = MultiScaleCNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        classes = np.unique(dataset["train"][-1])
        weights = compute_class_weight("balanced", classes=classes, y=dataset["train"][-1])
        weights = torch.Tensor(weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")

        mlflow.log_params(
            {
                "frequency": frequency,
                "n_fft": n_fft,
                "batch_size": batch_size,
                "lr": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "test_size": test_size,
                "val_size": val_size,
                "mode": mode,
            }
        )

        if mode == "design":
            val_dl = DataLoader(
                MultiModalDreamtDataset(n_fft=n_fft, *dataset["val"]),
                batch_size=128,
                shuffle=False,
            )
            train_model(
                model, train_dl, optimizer, criterion, epochs, val_dl=val_dl, device=DEVICE
            )
        else:
            train_model(model, train_dl, optimizer, criterion, epochs, device=DEVICE)

        results = test_model(model, test_dl, criterion, device=DEVICE, log_artifacts=True)
        test_metrics = {f"test/{k}": v for k, v in results[0].items()}
        mlflow.log_metrics(test_metrics)
        mlflow.log_text(str(model), "architecture.txt")
        mlflow.set_tags({"model_type": "MultiScaleCNN", "dataset": "DREAMT"})


if __name__ == "__main__":
    fire.Fire(main)
