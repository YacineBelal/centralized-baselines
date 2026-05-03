import torch
from tqdm import tqdm

__all__ = ["train_model"]


def train_model(model, train_dl, optimizer, criterion, epochs, device=torch.device("cpu")):
    """Train a model for a fixed number of epochs.

    Works with both single-modal loaders (batches of ``(X, y)``) and
    multi-modal loaders (batches of ``(x_bvp, x_acc, x_eda_temp, x_hr, y)``).
    The last element of each batch is always treated as the target; all
    preceding elements are forwarded to the model via ``model(*inputs)``.
    """
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
        print(f"Train loss: {empirical_risk}")
