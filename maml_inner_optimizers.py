import mlflow
import torch.nn as nn
import torch

import maml_api


# Adapted from: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/inner_loop_optimizers.py
class LSLRGradientDescentLearningRule(nn.Module):
    """Simple (stochastic) gradient descent learning rule.
    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form
        p[i] := p[i] - learning_rate * dE/dp[i]
    With `learning_rate` a positive scaling parameter.
    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, example_params: maml_api.NamedParams, inner_steps: int, init_lr: float,
                 use_learnable_learning_rates: bool, device: torch.device):
        super().__init__()
        assert init_lr > 0., 'learning_rate should be positive.'

        self.init_lr = init_lr
        self.inner_steps = inner_steps
        self.use_learnable_learning_rates = use_learnable_learning_rates

        # dictionary with names as keys and tensors of learning rates as values
        self.names_lrs_dict = nn.ParameterDict()
        for n, p in example_params.items():
            self.names_lrs_dict[n.replace(".", "-")] = nn.Parameter(
                # data=torch.ones(self.total_num_inner_loop_steps + 1) * self.init_learning_rate, why +1?
                data=torch.ones(self.inner_steps) * self.init_lr,
                requires_grad=self.use_learnable_learning_rates)

    def update_params(self, params: maml_api.NamedParams, names_grads_dict, num_step):
        return {
            n: p - self.names_lrs_dict[n.replace(".", "-")][num_step] * names_grads_dict[n]
            for n, p in params.items()
        }

    def log_lrs(self, episode: int):
        for n, lrs in self.names_lrs_dict.items():
            for i in range(self.inner_steps):
                mlflow.log_metric('step{}'.format(i) + n, lrs[i], episode)
