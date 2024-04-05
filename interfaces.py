import torch
import torch.nn as nn


class MamlModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_initial_state(self):
        """Should return params, buffers TODO documentation"""
        raise RuntimeError('Subclass and implement this function')

    def func_forward(self, x: torch.Tensor, params, buffers) -> torch.Tensor:
        raise RuntimeError('Subclass and implement this function')


class MamlTask():
    def __init__(self):
        pass

    def sample(self, mode) -> tuple[torch.Tensor, torch.Tensor]:
        """mode: 'query' or 'support'"""
        raise RuntimeError('Subclass and implement this function')

    def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor, mode):
        raise RuntimeError('Subclass and implement this function')
