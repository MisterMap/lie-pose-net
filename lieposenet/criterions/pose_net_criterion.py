import torch.nn as nn

from ..utils.torch_math import *
from .base_pose_criterion import BasePoseCriterion


class PoseNetCriterion(BasePoseCriterion):
    def __init__(self, translation_loss_function=None, rotation_loss_function=None, translation_koef=0.0,
                 rotation_koef=0.0, requires_grad=True):
        super(PoseNetCriterion, self).__init__()
        if translation_loss_function is None:
            translation_loss_function = nn.L1Loss()
        if rotation_loss_function is None:
            rotation_loss_function = nn.L1Loss()
        self.translation_loss_function = translation_loss_function
        self.rotation_loss_function = rotation_loss_function
        self.translation_koef = nn.Parameter(torch.tensor([translation_koef]), requires_grad=requires_grad)
        self.rotation_koef = nn.Parameter(torch.tensor([rotation_koef]), requires_grad=requires_grad)

    @property
    def position_dimension(self):
        return 6

    def forward(self, predicted_position, target_position):
        """
        :param predicted_position: N x 6
        :param target_position: N x 4 x 4
        :return:
        """
        target_position = logq_position_from_matrix(target_position)
        translation_loss = self.translation_loss_function(predicted_position[:, :3], target_position[:, :3])
        rotation_loss = self.rotation_loss_function(predicted_position[:, 3:], target_position[:, 3:])
        loss = torch.exp(-self.translation_koef) * translation_loss + self.translation_koef + \
               torch.exp(-self.rotation_koef) * rotation_loss + self.rotation_koef
        return loss

    def translation(self, predicted_position):
        return predicted_position[:, :3]

    def rotation(self, predicted_position):
        return quaternion_from_logq(predicted_position[:, 3:6])

    def metrics(self):
        return {
            "rotation_koef": self.rotation_koef,
            "translation_koef": self.translation_koef
        }
