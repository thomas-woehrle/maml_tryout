import json
from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class MamlHyperParameters():
    """
    Class representing hyperparameters needed for MAML

    Attributes:
        use_anil: Indicates whether ANIL should be used. Defaults to False.
        n_episodes: The number of episodes. Defaults to 10_000.
        meta_batch_size: Number of tasks per episode. Defaults to 32.
        inner_gradient_steps: Number of gradient steps in the inner loop. Defaults to 1.
        k: Number of samples per task. Defaults to 4.
        alpha: Inner learning rate. Defaults to 0.4.
        beta: Outer learning rate. Defaults to 0.001.

    """
    use_anil: bool = False
    n_episodes: int = 10_000
    meta_batch_size: int = 32
    inner_gradient_steps: int = 1
    k: int = 4
    alpha: float = 0.4
    beta: float = 0.001


@dataclass
class EnvConfig():
    """
    Class representing configuration variables needed in almost all cases.

    Attributes:
        device: A torch.device.
        ckpt_base: The base of the checkpoint directory.
        data_dir: The path to the data to be used.
        run_name: The name of this run.
    """
    device: torch.device
    ckpt_base: str
    data_dir: str
    run_name: Optional[str] = None


def load_configuration(file_path: str) -> tuple[MamlHyperParameters, EnvConfig, dict[str, Any]]:
    """Turns a .json file into a MamlHyperParameters object and a dictionary. 

    It expects the following form (fields that have a default can be omitted): 

    {
        "maml_hparams": {
            "use_anil": false,
            "n_episodes": 10000,
            "meta_batch_size": 32,
            "inner_gradient_steps": 1,
            "k": 4,
            "alpha": 0.4,
            "beta": 0.001
        },
        "env_config": {
            "device": "cpu",
            "ckpt_base": "checkpoints/rowfollow_maml",
            "data_dir": "/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new",
            "run_name": "some_name"
        },
        "other_config": {
            ...
        }
    }

    Args:
        file_path: Filepath to input .json file

    Returns:
        MamlHyperParameters object, EnvConfig object and dictionary representing the additional configuration values
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    maml_hparams = MamlHyperParameters(**data['maml_hparams'])
    data['env_config']['device'] = torch.device(data['env_config']['device'])
    env_config = EnvConfig(**data['env_config'])
    other_config = data.get('other_config', {})

    return maml_hparams, env_config, other_config
