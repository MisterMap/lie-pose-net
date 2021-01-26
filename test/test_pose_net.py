import os
import unittest

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import AttributeDict

from lieposenet import ModelFactory
from lieposenet.data import SevenScenesDataModule
from lieposenet.utils import TensorBoardLogger


class TestPoseNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "data", "7scenes")
        self._data_module = SevenScenesDataModule("chess", dataset_folder, 32, 4)
        params = AttributeDict(
            name="pose_net",
            optimizer=AttributeDict(),
            feature_extractor=AttributeDict(
                name="resnet34",
                pretrained=True,
            ),
            criterion=AttributeDict(
                name="pose_net_criterion",
            ),
            feature_dimension=2048,
            drop_rate=0.5,
        )
        self._model = ModelFactory().make_model(params)

    # noinspection PyTypeChecker
    def test_training(self):
        trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        trainer.fit(self._model, self._data_module)

    def test_testing(self):
        trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        trainer.test(self._model, self._data_module.test_dataloader())