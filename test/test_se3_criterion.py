import os
import unittest

import torch
from lieposenet.criterions import SE3Criterion


class TestLieCriterion(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        self._criterion = SE3Criterion()
        self._logvar = (torch.arange(1, 22, requires_grad=True, dtype=torch.float) * 0.1)[None]
        self._position = (torch.arange(1, self._criterion.position_dimension + 1, requires_grad=True,
                                       dtype=torch.float) * 0.1)[None]
        self._position[:6] = 0
        self._target_position = torch.eye(4, 4)[None]

    def test_get_sigma_matrix(self):
        sigma_matrix = self._criterion.get_sigma_matrix(self._logvar)
        self.assertEqual(sigma_matrix.shape, torch.Size([1, 6, 6]))
        for i in range(6):
            for j in range(0, i):
                self.assertAlmostEqual(sigma_matrix[0, i, j], 0)

    def test_get_inverse_sigma_matrix(self):
        inverse_sigma_matrix = self._criterion.get_inverse_sigma_matrix(self._logvar)
        self.assertEqual(inverse_sigma_matrix.shape, torch.Size([1, 6, 6]))
        sigma_matrix = self._criterion.get_sigma_matrix(self._logvar)
        for i in range(6):
            for j in range(0, i):
                self.assertAlmostEqual(inverse_sigma_matrix[0, i, j], 0)
        matrix_product = torch.bmm(sigma_matrix, inverse_sigma_matrix)
        for i in range(6):
            for j in range(6):
                if i == j:
                    self.assertAlmostEqual(matrix_product[0, i, j].item(), 1., 6)
                else:
                    self.assertAlmostEqual(matrix_product[0, i, j].item(), 0., 6)

    def test_forward(self):
        loss = self._criterion.forward(self._position, self._target_position)
        self.assertIsNotNone(loss)
        loss.backward()
