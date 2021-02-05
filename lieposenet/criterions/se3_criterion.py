from liegroups.torch import SE3

from .base_pose_criterion import BasePoseCriterion
from ..utils.torch_math import *


class SE3Criterion(BasePoseCriterion):
    def __init__(self, rotation_base_error=0, translation_base_error=0):
        super().__init__()
        self._base_covariance = torch.tensor([translation_base_error ** 2, translation_base_error ** 2,
                                              translation_base_error ** 2, rotation_base_error ** 2,
                                              rotation_base_error ** 2, rotation_base_error ** 2])

    @property
    def position_dimension(self):
        # return 6 + 21
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

        self._base_covariance = self._base_covariance.to(value_matrix.device)
        trace = torch.sum(inverse_sigma_matrix * inverse_sigma_matrix, dim=2) * self._base_covariance
        log_prob = torch.sum(delta_log ** 2 / 2., dim=1
                             ) + 0.5 * log_determinant + 6 * math.log(math.sqrt(2 * math.pi)) + torch.sum(trace, dim=1)

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

    @staticmethod
    def get_logvar_determinant(logvar):
        return torch.sum(logvar[:, :6], dim=1)

    @staticmethod
    def get_inverse_sigma_matrix(logvar, dim=6):
        matrix = torch.zeros(logvar.shape[0], dim, dim, device=logvar.device)
        k = 0
        for i in range(dim):
            matrix[:, i, i] = torch.exp(-0.5 * logvar[:, k])
            k += 1
        for i in range(dim):
            for j in range(i + 1, dim):
                matrix[:, i, j] = torch.exp(-0.5 * logvar[:, i]) * torch.sinh(logvar[:, k])
                k += 1
        return matrix

    def translation(self, predicted_position):
        return predicted_position[:, :3]

    def rotation(self, predicted_position):
        # return quaternion_from_logq(predicted_position[:, 3:6])
        return torch.nn.functional.normalize(predicted_position[:, 3:7])
