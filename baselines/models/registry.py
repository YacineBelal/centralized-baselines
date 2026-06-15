from inspect import signature

import torch

MODEL_REGISTRY = {}


def register(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(name, **kwargs) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model class {name} is not registered. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    cls = MODEL_REGISTRY.get(name)
    valid_parameters = signature(cls).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_parameters}

    return MODEL_REGISTRY[name](**filtered_kwargs)