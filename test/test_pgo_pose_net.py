import os
import unittest

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import AttributeDict

from lieposenet import ModelFactory
from lieposenet.data import SevenScenesDataModule
from lieposenet.utils import TensorBoardLogger
from lieposenet.utils.data_module_mock import DataModuleMock


# noinspection PyTypeChecker
class TestPoseNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(os.path.dirname(current_folder), "data", "7scenes")
        self._data_module_params = AttributeDict(
            name="odom_seven_scenes",
            scene="chess",
            data_path=dataset_folder,
            batch_size=32,
            use_test=True,
            num_workers=4,
            seed=0
        )
        self._data_module = DataModuleMock(SevenScenesDataModule(self._data_module_params))
        self._params = AttributeDict(
            name="pgo_pose_net",
            optimizer=AttributeDict(),
            feature_extractor=AttributeDict(
                name="resnet34",
                pretrained=True,
            ),
            criterion=AttributeDict(
                name="se3",
            ),
            pgo_optimizer=AttributeDict(
                linear_sigma=0.02,
                angular_sigma=0.02
            ),
            feature_dimension=2048,
            drop_rate=0.5,
            bias=True,
        )
        self._trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1,
                                   log_every_n_steps=1)

    def test_training(self):
        model = ModelFactory().make_model(self._params, train_dataset=self._data_module.get_train_dataset())
        self._trainer.fit(model, self._data_module)

    def test_testing(self):
        trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        model = ModelFactory().make_model(self._params)
        trainer.test(model, self._data_module.test_dataloader())
