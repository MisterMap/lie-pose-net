import os
import unittest

import torch
from pytorch_lightning.utilities.parsing import AttributeDict

from lieposenet.data import SevenScenesDataModule


class TestSevenScenesDataModule(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "data", "7scenes")
        self._params = AttributeDict(
            name="seven_scenes",
            scene="chess",
            data_path=dataset_folder,
            batch_size=32,
            use_test=True,
            num_workers=4,
            seed=0
        )

    def test_seven_scenes(self):
        self._data_module = SevenScenesDataModule(self._params)
        self.assertEqual(len(self._data_module._train_dataset), 4000)
        self.assertEqual(len(self._data_module._test_dataset), 2000)
        self.assertEqual(len(self._data_module._train_subset), 4000)
        self.assertEqual(len(self._data_module._validation_subset), 2000)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["image"].shape, torch.Size([32, 3, 256, 341]))
            self.assertEqual(batch["position"].shape, torch.Size([32, 4, 4]))
            self.assertEqual(batch["sequence"].shape, torch.Size([32]))
            self.assertEqual(batch["index"].shape, torch.Size([32]))
            break

    def test_odometry_seven_scenes(self):
        self._params.name = "odom_seven_scenes"
        self._data_module = SevenScenesDataModule(self._params)
        self.assertEqual(len(self._data_module._train_dataset), 6000)
        self.assertEqual(len(self._data_module._test_dataset), 2000)
        self.assertEqual(len(self._data_module._train_subset), 6000)
        self.assertEqual(len(self._data_module._validation_subset), 2000)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["image"].shape, torch.Size([32, 3, 256, 341]))
            self.assertEqual(batch["position"].shape, torch.Size([32, 4, 4]))
            self.assertEqual(batch["sequence"].shape, torch.Size([32]))
            self.assertEqual(batch["index"].shape, torch.Size([32]))
            break

