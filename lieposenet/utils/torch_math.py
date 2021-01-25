import kornia
import torch


def logq_position_from_matrix(matrix_position):
    translation = matrix_position[:, :3, 3]
    rotation_matrix = matrix_position[:, :3, :3].clone()
    quaternion = kornia.rotation_matrix_to_quaternion(rotation_matrix)
    logq = kornia.quaternion_exp_to_log(quaternion)
    position = torch.cat([translation, logq], dim=1)
    return position
