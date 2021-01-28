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


def matrix_from_logq_position(logq_position):
    quaternion = kornia.quaternion_log_to_exp(logq_position[:, 3:])
    matrix = torch.zeros(logq_position.shape[0], 4, 4, device=logq_position.device)
    matrix[:, :3, :3] = kornia.quaternion_to_rotation_matrix(quaternion)
    matrix[:, :3, 3] = logq_position[:, :3]
    matrix[:, 3, 3] = 1
    return matrix


def matrix_from_q_position(q_position):
    matrix = torch.zeros(q_position.shape[0], 4, 4, device=q_position.device)
    matrix[:, :3, :3] = kornia.quaternion_to_rotation_matrix(torch.nn.functional.normalize(q_position[:, 3:]))
    matrix[:, :3, 3] = q_position[:, :3]
    matrix[:, 3, 3] = 1
    return matrix


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


def inverse_pose_matrix(matrix):
    result = torch.zeros_like(matrix)
    rotation_part = matrix[:, :3, :3]
    translation_part = matrix[:, :3, 3]
    rotation_part_transposed = torch.transpose(rotation_part, 1, 2)
    result[:, :3, :3] = rotation_part_transposed
    result[:, :3, 3] = -torch.bmm(rotation_part_transposed, translation_part[:, :, None])[:, :, 0]
    result[:, 3, 3] = 1
    return result
