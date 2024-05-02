import enum
from typing import Protocol

import torch
import torch.nn as nn


class SampleMode(enum.Enum):
    QUERY = enum.auto()
    SUPPORT = enum.auto()
    # Maybe add TEST or FINETUNE mode ?


class MamlModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_state(self) -> tuple[dict[str, nn.Parameter], dict[str, torch.Tensor]]:
        params = {n: p for n, p in self.named_parameters()}
        buffers = {n: b for n, b in self.named_buffers()}
        return params, buffers

    def func_forward(self, x: torch.Tensor, params, buffers) -> torch.Tensor:
        return torch.func.functional_call(self, (params, buffers), x)


class MamlTask(Protocol):
    def sample(self, mode: SampleMode, current_ep: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: SampleMode, current_ep: int) -> torch.Tensor:
        ...
