import kornia
import torch
import math


def logq_position_from_matrix(matrix_position):
    translation = matrix_position[:, :3, 3]
    rotation_matrix = matrix_position[:, :3, :3].clone()
    quaternion = kornia.rotation_matrix_to_quaternion(rotation_matrix)
    logq = kornia.quaternion_exp_to_log(quaternion)
    position = torch.cat([translation, logq], dim=1)
    return position


def quaternion_from_matrix(matrix_position):
    rotation_matrix = matrix_position[:, :3, :3].clone()
    return kornia.rotation_matrix_to_quaternion(rotation_matrix)


def quaternion_from_logq(logq):
    return kornia.quaternion_log_to_exp(logq)


def quaternion_angular_error(q1, q2):
    dot = torch.sum(q1 * q2, dim=1)
    d = torch.abs(dot)
    d = torch.clamp(d, -1, 1)
    theta = 2 * torch.acos(d) * 180 / math.pi
    return theta
