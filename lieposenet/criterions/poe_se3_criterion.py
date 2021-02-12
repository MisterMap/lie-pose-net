from liegroups.torch import SE3

from .se3_criterion import SE3Criterion
from ..utils.torch_math import *


class POESE3Criterion(SE3Criterion):
    def __init__(self, head_count=10, **kwargs):
        super().__init__(**kwargs)
        self._head_count = head_count

    @property
    def position_dimension(self):
        return super().position_dimension * self._head_count

    def forward(self, predicted_position, target_position):
        batch_size = predicted_position.shape[0]
        predicted_position = predicted_position.reshape(batch_size * self._head_count,
                                                        predicted_position.shape[1] // self._head_count)
        target_position = torch.repeat_interleave(target_position, self._head_count, 0)
        return super().forward(predicted_position, target_position)

    def mean_position(self, predicted_position):
        batch_size = predicted_position.shape[0]
        predicted_position = predicted_position.reshape(batch_size * self._head_count,
                                                        predicted_position.shape[1] // self._head_count)
        logvar = predicted_position[:, 7:]
        mean_matrix = self.mean_matrix(predicted_position)
        log_mean = SE3.log(SE3.from_matrix(mean_matrix, normalize=False, check=False))
        if log_mean.dim() < 2:
            log_mean = log_mean[None]
        inverse_sigma_matrix = self.get_inverse_sigma_matrix(logvar)
        inverse_covariance_matrix = torch.bmm(inverse_sigma_matrix.transpose(1, 2), inverse_sigma_matrix)
        result_inverse_covariance_matrix = torch.sum(inverse_covariance_matrix.reshape(-1, self._head_count, 6, 6),
                                                     dim=1)
        # print("result_inverse_covariance_matrix", result_inverse_covariance_matrix)
        result_covariance_matrix = torch.inverse(result_inverse_covariance_matrix)
        # print("result_covariance_matrix", result_covariance_matrix)
        factors = torch.bmm(result_covariance_matrix.repeat_interleave(self._head_count, 0), inverse_covariance_matrix)
        scaled_log_mean = torch.bmm(factors, log_mean[:, :, None])[:, :, 0]
        result_log_mean = torch.sum(scaled_log_mean.reshape(-1, self._head_count, 6), dim=1)
        mean_matrix = SE3.exp(result_log_mean).as_matrix()
        if mean_matrix.dim() < 3:
            mean_matrix = mean_matrix[None]
        return mean_matrix

    def translation(self, predicted_position):
        mean_matrix = self.mean_position(predicted_position)
        return mean_matrix[:, :3, 3]

    def rotation(self, predicted_position):
        mean_matrix = self.mean_position(predicted_position)
        return quaternion_from_matrix(mean_matrix[:, :3, :3])
