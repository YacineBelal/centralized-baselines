import torch

MODEL_REGISTRY = {}


def register(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name, **kwargs) -> torch.nn.Module:
    return MODEL_REGISTRY[name](**kwargs)