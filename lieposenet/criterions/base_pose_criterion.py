import torch
import torch.nn as nn
from ..utils.se3_position import SE3Position


class BasePoseCriterion(nn.Module):
    @property
    def position_dimension(self):
        raise NotImplementedError

    def forward(self, predicted_position, target_position):
        raise NotImplementedError

    def translation(self, predicted_position):
        raise NotImplementedError

    def rotation(self, predicted_position):
        raise NotImplementedError

    def se3_position(self, predicted_position_parametrization):
        rotation = self.rotation(predicted_position_parametrization)
        translation = self.translation(predicted_position_parametrization)
        return SE3Position.from_q_position(torch.cat([translation, rotation], dim=1))

    def saved_data(self, predicted_position):
        return {}
