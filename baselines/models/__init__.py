from .architectures import CNN, MultiScaleCNN
from .evaluate import test_model
from .train import train_model

__all__ = [
    "CNN",
    "MultiScaleCNN",
    "train_model",
    "test_model",
]
