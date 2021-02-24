from liegroups.torch import SE3

from .base_pose_criterion import BasePoseCriterion
from ..utils.torch_math import *
import torch.nn as nn


class SE3Criterion(BasePoseCriterion):
    def __init__(self, rotation_koef=None, translation_koef=None, zero_covariance=False):
        super().__init__()
        self._logvar = None
        self._zero_covariance = zero_covariance
        if rotation_koef is not None and translation_koef is not None:
            x = torch.zeros(6)
            x[:3] = translation_koef
            x[3:6] = rotation_koef
            self._logvar = nn.Parameter(x)

    @property
    def position_dimension(self):
        return 7 + 6 + 15

    @staticmethod
    def mean_matrix(predicted_position):
        return matrix_from_q_position(predicted_position[:, :7])

    def forward(self, predicted_position, target_position):
        logvar = predicted_position[:, 7:]
        mean_matrix = self.mean_matrix(predicted_position)
        return self.log_prob(target_position, mean_matrix, logvar)

    def log_prob(self, value_matrix, mean_matrix, logvar):
        if logvar.dim() < 2:
            logvar = logvar[None].expand(mean_matrix.shape[0], logvar.shape[0])
        delta_matrix = torch.bmm(inverse_pose_matrix(mean_matrix), value_matrix)
        delta_log = SE3.log(SE3.from_matrix(delta_matrix, normalize=False, check=False))
        if delta_log.dim() < 2:
            delta_log = delta_log[None]
        inverse_sigma_matrix = self.get_inverse_sigma_matrix(logvar).expand(delta_log.shape[0], delta_log.shape[1],
                                                                            delta_log.shape[1])
        delta_log = torch.bmm(inverse_sigma_matrix, delta_log[:, :, None])[:, :, 0]
        log_determinant = self.get_logvar_determinant(logvar)

        log_prob = torch.sum(delta_log ** 2 / 2., dim=1) + 0.5 * log_determinant
        return torch.mean(log_prob)

    @staticmethod
    def get_sigma_matrix(logvar, dim=6):
        matrix = []
        for i in range(dim):
            matrix.append([])
            for j in range(dim):
                matrix[i].append(torch.zeros(logvar.shape[0], dtype=torch.float, device=logvar.device))
        k = 0
        for i in range(dim):
            matrix[i][i] = torch.exp(0.5 * logvar[:, k])
            k += 1
        auxiliary_matrix = torch.zeros(logvar.shape[0], dim, dim, device=logvar.device)
        for i in range(dim):
            for j in range(i + 1, dim):
                auxiliary_matrix[:, i, j] = torch.sinh(logvar[:, k])
                k += 1
        for i in range(dim - 1, -1, -1):
            for j in range(i + 1, dim):
                vector = torch.zeros(logvar.shape[0], j + 1 - i, device=logvar.device)
                for k in range(i, j + 1):
                    vector[:, k - i] = -auxiliary_matrix[:, i, k] * matrix[k][j]
                matrix[i][j] = torch.sum(vector, dim=1)
        result = torch.zeros(logvar.shape[0], dim, dim, device=logvar.device)
        for i in range(dim):
            for j in range(dim):
                result[:, i, j] = matrix[i][j]
        return result

    def get_logvar_determinant(self, logvar):
        return self._get_logvar_determinant(self.calculate_logvar(logvar))

    def calculate_logvar(self, logvar):
        if self._logvar is not None:
            old_logvar = logvar
            logvar = torch.zeros_like(logvar)
            base_logvar = torch.repeat_interleave(self._logvar[None, :6], old_logvar.shape[0], dim=0)
            stacked_logvar = torch.stack([old_logvar[:, :6], base_logvar], dim=2)
            # logvar[:, :6] = torch.logsumexp(stacked_logvar, dim=2)
            logvar[:, :6] = base_logvar
        if self._zero_covariance:
            logvar[:, 6:] = 0
        return logvar

    @staticmethod
    def _get_logvar_determinant(logvar):
        return torch.sum(logvar[:, :6], dim=1)

    def get_inverse_sigma_matrix(self, logvar, dim=6):
        return self._get_inverse_sigma_matrix(self.calculate_logvar(logvar),  dim=dim)

    @staticmethod
    def _get_inverse_sigma_matrix(logvar, dim=6):
        matrix = torch.zeros(logvar.shape[0], dim, dim, device=logvar.device)
        k = 0
        for i in range(dim):
            matrix[:, i, i] = torch.exp(-0.5 * logvar[:, k])
            k += 1
        for i in range(dim):
            for j in range(i + 1, dim):
                matrix[:, i, j] = torch.exp(-0.5 * logvar[:, i]) * logvar[:, k]
                k += 1
        return matrix

    def translation(self, predicted_position):
        return predicted_position[:, :3]

    def rotation(self, predicted_position):
        return torch.nn.functional.normalize(predicted_position[:, 3:7])

    def saved_data(self, predicted_position):
        logvar = predicted_position[:, 7:]
        mean_matrix = self.mean_matrix(predicted_position)
        inverse_sigma_matrix = self.get_inverse_sigma_matrix(logvar)
        return {"mean_matrix": mean_matrix,
                "inverse_sigma_matrix": inverse_sigma_matrix}
