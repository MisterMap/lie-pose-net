from .poe_se3_criterion import POESE3Criterion
import torch
import torch.nn as nn
from ..utils.torch_math import *
from liegroups.torch import SE3


class DNCLPoESE3Criterion(POESE3Criterion):
    def __init__(self, head_count=10, lambda_koef=0.5, **kwargs):
        super().__init__(head_count, **kwargs)
        self._batches = 0
        self._lambda = lambda_koef
        self._loss = nn.MSELoss()
        self._dncl_loss = None

    def forward(self, predicted_position, target_position):
        # print("predicted_position", predicted_position)
        mean_position = self.mean_position(predicted_position).detach()
        # print("mean position", mean_position)
        loss = super().forward(predicted_position, target_position)
        self._dncl_loss = self.dncl_loss(predicted_position, mean_position)
        # print("dncl_loss", dncl_loss)
        # print("loss", loss)
        if self._dncl_loss != self._dncl_loss:
            raise Exception()
        # return loss - self._lambda * dncl_loss
        return loss - self._lambda * self._dncl_loss

    def dncl_loss(self, predicted_position, mean_position):
        batch_size = predicted_position.shape[0]
        predicted_position = predicted_position.reshape(batch_size * self._head_count,
                                                        predicted_position.shape[1] // self._head_count)
        predicted_translations = predicted_position[:, :3]
        # predicted_translations = predicted_translations.reshape(batch_size, self._head_count, 3)
        mean_translations = mean_position[:, :3, 3]
        mean_translations = mean_translations[:, :].repeat_interleave(self._head_count, dim=0)
        # delta_matrix = predicted_translations, mean_translations
        inverse_sigma_matrix = self.get_inverse_sigma_matrix(predicted_position[:, 7:]).expand(
            predicted_position.shape[0], 6, 6).detach()
        deltas = torch.bmm(inverse_sigma_matrix[:, :3, :3], (predicted_translations - mean_translations)[:, :, None])[:, :, 0]
        # predicted_position = predicted_position.reshape
        # mean_matrix = self.mean_matrix(predicted_position)
        # delta_matrix = torch.bmm(inverse_pose_matrix(mean_matrix), mean_position)
        # delta_log = SE3.log(SE3.from_matrix(delta_matrix, normalize=False, check=False))
        # if delta_log.dim() < 2:
        #     delta_log = delta_log[None]
        # print("delta_log", delta_log.max())
        # deltas = (predicted_translations -mean_translations) / variances
        # return self._loss(predicted_translations, mean_translations)
        return self._loss(deltas, torch.zeros_like(deltas))

    def additional_losses(self):
        return {"dncl": self._dncl_loss}
