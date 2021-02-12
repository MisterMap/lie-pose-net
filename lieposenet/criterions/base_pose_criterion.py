import torch.nn as nn


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

    def saved_data(self, predicted_position):
        return {}

    def additional_losses(self):
        return {}

