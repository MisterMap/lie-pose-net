import unittest
import os
from lieposenet.data import SevenScenesDataModule
import torch


class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "data", "7scenes")
        self._data_module = SevenScenesDataModule("chess", dataset_folder, 32, 4)

    def test_load(self):
        self.assertEqual(len(self._data_module._train_dataset), 4000)
        self.assertEqual(len(self._data_module._test_dataset), 2000)
        self.assertEqual(len(self._data_module._train_subset), 4000 * 0.9)
        self.assertEqual(len(self._data_module._validation_subset), 4000 * 0.1)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["image"].shape, torch.Size([32, 3, 224, 224]))
            self.assertEqual(batch["position"].shape, torch.Size([32, 4, 4]))
            break
