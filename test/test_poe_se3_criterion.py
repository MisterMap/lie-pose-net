import os
import unittest

import torch
from lieposenet.criterions import POESE3Criterion


class TestPOESE3Criterion(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        self._criterion = POESE3Criterion()
        self._position = (torch.arange(1, self._criterion.position_dimension + 1, requires_grad=True,
                                       dtype=torch.float, device="cuda:0") * 0.3)[None]
        self._position = torch.repeat_interleave(self._position, 2, dim=0)
        self._position[:6] = 0
        self._target_position = torch.eye(4, 4, device="cuda:0")[None]
        self._target_position = torch.repeat_interleave(self._target_position, 2, dim=0)

    def test_forward(self):
        loss = self._criterion.forward(self._position, self._target_position)
        self.assertIsNotNone(loss)
        loss.backward()

    def test_translation(self):
        translation = self._criterion.translation(self._position)
        self.assertEqual(translation.shape, torch.Size([2, 3]))

    def test_rotation(self):
        rotation = self._criterion.rotation(self._position)
        self.assertEqual(rotation.shape, torch.Size([2, 4]))
