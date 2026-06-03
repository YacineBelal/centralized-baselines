import fire
import mlflow
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.data import MitbihDataset, load_mit_bih
from baselines.models import (
    CNN,
    test_model,
    train_model,
)


def main(
        window_len = 128,
        epochs = 30,
        batch_size = 32,
        normal_class= 0, #TODO: do not provide as argument, collect instead
        lr = 0.0001,
        mode = "design",
        #only in mode=design
        val_size=0.1, 
        val_period=2,
        seed = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = load_mit_bih(val_size=val_size)

    train_dl = DataLoader(
        MitbihDataset(*dataset["train"]),
        batch_size=batch_size,
        shuffle=True)
    
    test_dl = DataLoader(
        MitbihDataset(*dataset["test"]),
        batch_size=1024,
        shuffle=False,
    )

    mlflow.set_experiment("CNN MIT-BIH")
    with mlflow.start_run():
        model = CNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        classes = np.unique(dataset["train"][-1])
        weights = compute_class_weight("balanced", classes=classes, y=dataset["train"][-1])
        weights = torch.Tensor(weights).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
        mlflow.log_params(
            {
                "window_len": window_len,
                "batch_size": batch_size,
                "lr": lr,
                "criterion": criterion.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "val_size": val_size,
                "mode": mode,
            })

        if mode == "design":
            val_dl = DataLoader(
                MitbihDataset(*dataset["val"]),
                batch_size=1024,
                shuffle=False,
            )
            train_model(
                model, train_dl, optimizer, criterion, epochs, val_dl=val_dl, val_period=val_period, normal_class=normal_class, device=DEVICE
            )
        else:
            train_model(model, train_dl, optimizer, criterion, epochs, device=DEVICE)

        results = test_model(model, test_dl, criterion, normal_class=normal_class, device=DEVICE, log_artifacts=True)
        test_metrics = {f"test/{k}": v for k, v in results[0].items()}
        print(test_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_text(str(model), "architecture.txt")
        mlflow.set_tags({"model_type": "CNN", "dataset": "MIT-BIH"})


if __name__ == "__main__":
    fire.Fire(main)


