from liegroups.torch import SE3
import torch
from .torch_math import inverse_pose_matrix


def calculate_log_se3_delta(predicted_position, target_position):
    predicted_matrix = predicted_position.matrix
    target_matrix = target_position.matrix
    delta_matrix = torch.bmm(inverse_pose_matrix(predicted_matrix), target_matrix)
    delta_log = SE3.log(SE3.from_matrix(delta_matrix, normalize=False, check=False))
    if delta_log.dim() < 2:
        delta_log = delta_log[None]
    return delta_log
