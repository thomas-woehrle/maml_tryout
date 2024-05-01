import json
from dataclasses import dataclass
from typing import Any


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


def load_configuration(file_path: str) -> tuple[MamlHyperParameters, dict[str, Any]]:
    """Turns a .json file into a MamlHyperParameters object and a dictionary. 
    It expects the following form: 
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
        "other_hparams": {...}
    }

    Args:
        file_path: Filepath to input .json file

    Returns:
        MamlHyperParameters object and dictionary representing the additional hyperparameters
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Parse MAML hyperparameters
    maml_hparams = MamlHyperParameters(**data['maml_hparams'])

    # Additional parameters can be whatever the user specifies
    add_hparams = data.get('other_hparams', {})

    return maml_hparams, add_hparams
