import copy
from dataclasses import dataclass
from typing import Any

import mlflow
import torch

import maml_config


def log_configuration(hparams: "maml_config.MamlHyperParameters",
                      train_config: "maml_config.TrainConfig",
                      env_config: "maml_config.EnvConfig",
                      other_config: dict[str, Any]):
    """Logs the hparams and configuration.

    Logs them as params and as artifact files.
    This might become a problem if values are made non-serializable at some point.
    """
    mlflow.log_params(vars(hparams))
    mlflow.log_dict(vars(hparams), 'hparams.json')
    mlflow.log_params(vars(train_config))
    mlflow.log_dict(vars(train_config), 'train_config.json')
    env_config = vars(env_config)
    env_config['device'] = str(env_config['device'])
    mlflow.log_params(env_config)
    mlflow.log_dict(env_config, 'env_config.json')
    mlflow.log_params(other_config)
    mlflow.log_dict(other_config, 'other_config.json')


@dataclass
class MetricLog:
    key: str
    value: float
    step: int


class Logger:
    def __init__(self):
        self.metrics_logs_buffer: dict[str, MetricLog] = dict()

    def log_metric(self, key: str, value: float, step: int):
        self.metrics_logs_buffer[key] = MetricLog(key, value, step)

    def log_metrics_buffer_to_mlflow(self, episode: int):
        to_be_logged = dict()
        for k, l in self.metrics_logs_buffer.items():
            if l.step == episode:
                to_be_logged[l.key] = l.value
        mlflow.log_metrics(to_be_logged, episode)

    @staticmethod
    def log_dict(dictionary: dict, artifact_file: str):
        """Logs dictionary, after replacing its tensors with lists. Does not buffer, but log directly."""
        def rec_replace_tensors(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    rec_replace_tensors(v)
                elif isinstance(v, torch.Tensor):
                    d[k] = v.cpu().detach().numpy().tolist()

        dict_copy = copy.deepcopy(dictionary)
        rec_replace_tensors(dict_copy)  # in-place operation
        mlflow.log_dict(dict_copy, artifact_file)
