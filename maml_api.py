from enum import Enum, auto
from typing import Protocol


import torch
import torch.nn as nn


class SampleMode(Enum):
    QUERY = auto()
    SUPPORT = auto()
    # Maybe add TEST or FINETUNE mode ?


class MamlModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_state(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        params = {n: p for n, p in self.named_parameters()}
        buffers = {n: b for n, b in self.named_buffers()}
        return params, buffers

    def func_forward(self, x: torch.Tensor, params: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.functional_call(self, (params, buffers), x)


class MamlTask(Protocol):
    def sample(self, mode: SampleMode, current_ep: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: SampleMode, current_ep: int) -> torch.Tensor:
        ...


# class old_MamlModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def get_initial_state(self):
#         """Should return params, buffers TODO documentation"""
#         raise RuntimeError('Subclass and implement this function')

#     def func_forward(self, x: torch.Tensor, params, buffers) -> torch.Tensor:
#         raise RuntimeError('Subclass and implement this function')


# class old_MamlTask():
#     def __init__(self):
#         pass

#     def sample(self, mode, current_ep) -> tuple[torch.Tensor, torch.Tensor]:
#         """mode: 'query' or 'support'"""
#         raise RuntimeError('Subclass and implement this function')

#     def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor, mode):
#         raise RuntimeError('Subclass and implement this function')
