from .base_pose_criterion import BasePoseCriterion
from torch.nn import Parameter
import torch.nn.functional
import torch
import torch.nn as nn
from ..utils.se3_position import SE3Position
from ..utils.torch_se3_math import calculate_log_se3_delta

LOSS_TYPES = {
    "l2": nn.MSELoss,
    "l1": nn.L1Loss
}


class SimpleSE3Criterion(BasePoseCriterion):
    def __init__(self, rotation_koef=0., translation_koef=0., koef_requires_grad=True, use_se3_translation=True,
                 loss_type="l2", lr=None):
        super().__init__(lr)
        self._rotation_koef = Parameter(torch.tensor(rotation_koef), requires_grad=koef_requires_grad)
        self._translation_koef = Parameter(torch.tensor(translation_koef), requires_grad=koef_requires_grad)
        self._use_se3_translation = use_se3_translation
        self._loss = LOSS_TYPES[loss_type]()
        print("Loss function ", self._loss)

    def forward(self, predicted_position, target_position):
        predicted_position = SE3Position.from_q_position(predicted_position)
        target_position = SE3Position.from_matrix_position(target_position)
        log_se3_delta = calculate_log_se3_delta(predicted_position, target_position)
        if self._use_se3_translation:
            translation_part = log_se3_delta[:, :3]
        else:
            translation_part = predicted_position.translation - target_position.translation
        rotation_part = log_se3_delta[:, 3:]
        return self.calculate_weighted_loss(translation_part, rotation_part)

    def calculate_weighted_loss(self, translation_part, rotation_part):
        translation_loss = self._loss(translation_part, torch.zeros_like(translation_part))
        rotation_loss = self._loss(rotation_part, torch.zeros_like(rotation_part))
        translation_loss = torch.exp(-self._translation_koef) * translation_loss + self._translation_koef
        rotation_loss = torch.exp(-self._rotation_koef) * rotation_loss + self._rotation_koef
        return translation_loss + rotation_loss

    @property
    def position_dimension(self):
        return 7

    def translation(self, predicted_position):
        return predicted_position[:, :3]

    def rotation(self, predicted_position):
        return torch.nn.functional.normalize(predicted_position[:, 3:7])

    def metrics(self):
        return {
            "rotation_koef": self._rotation_koef,
            "translation_koef": self._translation_koef
        }
