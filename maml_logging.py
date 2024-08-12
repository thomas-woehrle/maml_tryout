from typing import Dict, Any

import mlflow

import maml_config


def log_configuration(hparams: "maml_config.MamlHyperParameters", env_config: "maml_config.EnvConfig",
                      other_config: Dict[str, Any]):
    """Logs the hparams and configuration.

    Logs them as params and as artifact files.
    This might become a problem if values are made non-serializable at some point.
    """
    mlflow.log_params(vars(hparams))
    mlflow.log_dict(vars(hparams), 'hparams.json')
    env_config = vars(env_config)
    env_config['device'] = str(env_config['device'])
    mlflow.log_params(env_config)
    mlflow.log_dict(env_config, 'env_config.json')
    mlflow.log_params(other_config)
    mlflow.log_dict(other_config, 'other_config.json')
