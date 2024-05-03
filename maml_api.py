import enum
from typing import Protocol

import torch
import torch.nn as nn


class SampleMode(enum.Enum):
    """Represents the mode in which sampling from a task should happen."""
    QUERY = enum.auto()
    SUPPORT = enum.auto()
    # Maybe add TEST or FINETUNE mode ?


class MamlModel(nn.Module):
    """The model which will trained using MAML. Can be treated like a normal nn.Module. 
    There might be situations where its base implementations need to be overridden.
    """

    def __init__(self):
        super().__init__()

    def get_state(self) -> tuple[dict[str, nn.Parameter], dict[str, torch.Tensor]]:
        """Used to retrieve the current state dict of the model, i.e. parameters and buffers.

        Returns:
            The parameters and buffers as tuple of dictionaries
        """
        params = {n: p for n, p in self.named_parameters()}
        buffers = {n: b for n, b in self.named_buffers()}
        return params, buffers

    def func_forward(self, x: torch.Tensor, params, buffers) -> torch.Tensor:
        """Functional pass through the model, using provided params and buffers. 

        Args:
            x: The input
            params: Params to be used in the pass
            buffers: Buffers to be used in the pass

        Returns:
            The output of the functional pass through the model
        """
        return torch.func.functional_call(self, (params, buffers), x)


class MamlTask(Protocol):
    """Protocol representing a task as it is used in MAML.
    """

    def sample(self, mode: SampleMode, current_ep: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from the task. 

        Args:
            mode: The current mode. Won't play a role in all use cases.
            current_ep: The current episode. Won't play a role in all use cases.

        Returns:
            A tuple of x and y, i.e. input and label
        """
        ...

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: SampleMode, current_ep: int) -> torch.Tensor:
        """Calculates the loss as specified by the task. 

        Args:
            y_hat: The prediction
            y: The target
            mode: The current mode. 
            current_ep: The current episode.

        Returns:
            A tensor containing the loss. Has to be a single value
        """
        ...
