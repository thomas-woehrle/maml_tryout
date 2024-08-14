import enum
from typing import Protocol

import torch
import torch.nn as nn


NamedParams = dict[str, torch.nn.parameter.Parameter]
NamedBuffers = dict[str, torch.Tensor]


class Stage(enum.Enum):
    """Represents the training stage of the MAML algorithm.

    TrainingStage and SampleMode combined, determine which datapool to use.
    TRAIN, VAL and TEST refer to datasets, which themselves are split into QUERY and SUPPORT data/tasks
    according to MAML theory"""
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class SetToSetType(enum.Enum):
    """Represents the phase of the MAML algorithm, common in set-to-set learning

    """
    SUPPORT = enum.auto()
    TARGET = enum.auto()


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

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from the task. 

        Returns:
            A tuple of x and y, i.e. input and label
        """
        ...

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, stage: Stage, sts_type: SetToSetType) -> torch.Tensor:
        """Calculates the loss as specified by the task. 

        Args:
            y_hat: The prediction
            y: The target
            stage: The stage ie TRAIN, VAL, TEST, etc.
            sts_type: The phase of the task ie Support or Target phase

        Returns:
            A tensor containing the loss. Has to be a single value
        """
        ...
