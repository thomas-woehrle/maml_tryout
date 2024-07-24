from typing import Dict, Any

import mlflow

import maml_config


def log_configuration(hparams: "maml_config.MamlHyperParameters", env_config: "maml_config.EnvConfig",
                      other_config: Dict[str, Any]):
    mlflow.log_params(vars(hparams))
    mlflow.log_params(vars(env_config))
    mlflow.log_params(other_config)


def log_file(local_path: str, artifact_path):
    mlflow.log_artifact(local_path, artifact_path)
