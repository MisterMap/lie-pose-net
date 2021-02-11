from liegroups.torch import SE3

from .se3_criterion import SE3Criterion
from ..utils.torch_math import *
from .poe_se3_criterion import POESE3Criterion
import torch


class ParametrizedPOESE3Criterion(POESE3Criterion):
    def __init__(self, head_count=10):
        super().__init__()
        self._head_count = head_count
        self._global_positions = torch.nn.Parameter(torch.randn(self._head_count, 7), requires_grad=True)

    def mean_matrix(self, predicted_position):
        local_matrix = super().mean_matrix(predicted_position)
        global_matrix = super().mean_matrix(self._global_positions)
        batch_size = local_matrix.shape[0] // self._head_count
        global_matrix = torch.repeat_interleave(global_matrix, batch_size, dim=0)
        print(global_matrix.shape)
        print(local_matrix.shape)
        return torch.bmm(global_matrix, inverse_pose_matrix(local_matrix))
