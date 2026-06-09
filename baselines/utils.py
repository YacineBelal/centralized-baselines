import random

import numpy as np
import torch


def init_randomized_envs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class _noLoggingContext:  # TODO add pretty printer here
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class MLFlowLogger:
    def __init__(self, enabled=True, experiment_name=None, **kwargs):
        self.enabled = enabled
        if self.enabled:
            import mlflow

            self.mlflow = mlflow

            self.mlflow.set_experiment(experiment_name=experiment_name, **kwargs)

    def start_run(self, **kwargs):
        return self.mlflow.start_run(**kwargs) if self.enabled else _noLoggingContext()

    def log_params(self, params):
        return self.mlflow.log_params(params) if self.enabled else _noLoggingContext()

    def log_text(self, text, artifact_file):
        return self.mlflow.log_text(text, artifact_file) if self.enabled else _noLoggingContext()

    def log_figure(self, figure, artifact_file):
        return (
            self.mlflow.log_figure(figure, artifact_file) if self.enabled else _noLoggingContext()
        )

    def log_metrics(self, metrics, step=None):
        return self.mlflow.log_metrics(metrics, step) if self.enabled else _noLoggingContext()

    def set_tags(self, tags):
        return self.mlflow.set_tags(tags) if self.enabled else _noLoggingContext()

    def log_model(self, model):
        return self.mlflow.pytorch.log_model(model) if self.enabled else _noLoggingContext()