import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sleep_stage_prediction.data import MultiModalDreamtDataset, load_dreamt_multimodal
from sleep_stage_prediction.models import (
    MultiScaleCNN,
    test_model,
    train_model,
)


def main(
    nb_patients=10,
    frequency=64,
    n_fft=32,
    epochs=10,
    batch_size=128,
    lr=0.001,
    momentum=0.9,
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
        seed,
        mode,
    )

    train_dl = DataLoader(
        MultiModalDreamtDataset(n_fft=n_fft, *dataset["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_dl = DataLoader(MultiModalDreamtDataset(n_fft=n_fft, *dataset["test"]), batch_size=1024)

    model = MultiScaleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    train_model(model, train_dl, optimizer, criterion, epochs, DEVICE)

    if mode == "design":
        val_dl = DataLoader(MultiModalDreamtDataset(n_fft=n_fft, *dataset["val"]), batch_size=1024)

    test_model(model, val_dl, criterion, device=DEVICE)
    test_model(model, test_dl, criterion, device=DEVICE)


if __name__ == "__main__":
    fire.Fire(main)
