import argparse
import json
from dataclasses import dataclass
from typing import Any, Optional

import torch

import maml_logging


@dataclass
class MamlHyperParameters():
    """
    Class representing hyperparameters needed for MAML

    Attributes:
        n_episodes: The number of episodes. Defaults to 10_000.
        meta_batch_size: Number of tasks per episode. Defaults to 32.
        inner_steps: Number of gradient steps in the inner loop. Defaults to 1.
        k: Number of samples per task. Defaults to 4.
        alpha: Inner learning rate. Defaults to 0.4.
        beta: Outer learning rate. Defaults to 0.001.
        use_anil: Indicates whether ANIL should be used. Defaults to False.
        use_msl: Whether to use Multi-Step Loss Optimization (MSL).
        use_ca: Whether to use Cosine Annealing of Meta-Optimizer Learning Rate (CA).
        use_da: Whether to use Derivative-Order Annealing (DA).
        use_lslr: Whether to learn Per-Layer Per-Step Learning Rates (LSLR).
        use_bnrs: Whether to use Per-Step Batch Normalization Running Statistics (BNRS).
            As of 08/26/2024 bnrs is used even if this is set to False.
        first_order_percentage_of_episodes: Percentage of episodes which use first order, if use_da=True
        msl_percentage_of_episodes: Percentage of episodes which use strong MSL, if use_msl=True
    """
    n_episodes: int = 10_000
    meta_batch_size: int = 32
    inner_steps: int = 1
    k: int = 4
    alpha: float = 0.4
    beta: float = 0.001
    min_beta: float = 0.00001
    use_anil: bool = False
    use_msl: bool = True
    use_ca: bool = True
    use_da: bool = True
    use_lslr: bool = True
    use_bnrs: bool = True
    first_order_percentage_of_episodes: float = 0.3
    msl_percentage_of_episodes: float = 0.1


@dataclass
class TrainConfig:
    """Represents configurations to the training process, which are not inherent to MAML

    log_val_loss_every_n_episodes: Frequency of logging validation losses.
    log_model_every_n_episodes: Frequency of logging models.
    n_val_iters: Frequency of validation iterations.
    """
    log_model_every_n_episodes: int = 1000
    log_val_loss_every_n_episodes: int = 1000


@dataclass
class EnvConfig:
    """
    Class representing configuration variables needed in almost all cases.

    Attributes:
        device: A torch.device.
        data_dir: The path to the data to be used.
        do_use_mlflow: Indicates whether to use MLFlow tracking or not.
        seed: Seed to be used for training. If this is None, then no seed is used.
        use_same_val_seed_for_all_episodes: Indicates whether to always use same seed for validation loss.
            If this is true, the seed used in the calculation of the val_loss at episode should be  EnvConfig.seed or 0,
            if EnvConfig.seed is None.
            If this is false, the seed used to calculate the val_loss at episode x should be x + EnvConfig.seed
            or x if EnvConfig.seed is None.
            This means that in any case, the seed used to calculate the val_loss at episode x can be inferred afterward 
            if one knows the hyperparameters.
    """
    device: torch.device
    data_dir: str
    do_use_mlflow: bool = False
    seed: Optional[int] = None
    use_same_val_seed_for_all_episodes: bool = True


def load_configuration(file_path: str) -> tuple[MamlHyperParameters, TrainConfig, EnvConfig, dict[str, Any]]:
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
            "data_dir": "/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new",
            "do_use_mlflow (Optional)": true/false
        },
        "other_config": {
            ...
            "key": "value",
            "not": "nested"
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
    train_config = data.get('train_config', {})
    train_config = TrainConfig(**train_config)
    env_config = EnvConfig(**data['env_config'])
    other_config = data.get('other_config', {})

    if env_config.do_use_mlflow:
        maml_logging.log_configuration(maml_hparams, train_config, env_config, other_config)

    return maml_hparams, train_config, env_config, other_config


def parse_maml_args() -> tuple[MamlHyperParameters, TrainConfig, EnvConfig, dict[str, Any]]:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_filepath", type=str)
    config_filepath: str = arg_parser.parse_args().config_filepath

    maml_hparams, train_config, env_config, other_config = load_configuration(
        config_filepath)
    return maml_hparams, train_config, env_config, other_config
