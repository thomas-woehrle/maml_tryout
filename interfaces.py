import torch
import torch.nn as nn


class MamlModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_init_params(self) -> list[torch.Tensor]:
        """Test doc string"""
        raise RuntimeError('Subclass and implement this function')

    def forward(self, x: torch.Tensor, theta: list[torch.Tensor], is_train=True) -> torch.Tensor:
        raise RuntimeError('Subclass and implement this function')


class MamlTask():
    def __init__(self):
        pass

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError('Subclass and implement this function')

    def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor):
        raise RuntimeError('Subclass and implement this function')
