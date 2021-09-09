import os
import unittest

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.parsing import AttributeDict

from lieposenet.utils.universal_factory import UniversalFactory
from lieposenet.data import SevenScenesDataModule
from lieposenet.utils.data_module_mock import DataModuleMock
from lieposenet.models.pose_net import PoseNet
from lieposenet.criterions.simple_se3_criterion import SimpleSE3Criterion
from lieposenet.criterions.se3_criterion import SE3Criterion
from lieposenet.criterions.pose_net_criterion import PoseNetCriterion


# noinspection PyTypeChecker
class TestPoseNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        dataset_folder = "/media/mikhail/Data3T/7scenes"
        self._data_module = DataModuleMock(SevenScenesDataModule("chess", dataset_folder, 2, 4))
        self._params = AttributeDict(
            name="PoseNet",
            optimizer=AttributeDict(),
            feature_extractor=AttributeDict(
                pretrained=True,
            ),
            criterion=AttributeDict(
                name="PoseNetCriterion",
                lr=0.1,
            ),
            feature_dimension=2048,
            drop_rate=0.5,
            bias=True,
            activation="relu"
        )
        self._trainer = pl.Trainer(max_epochs=1, gpus=1)
        self._factory = UniversalFactory([PoseNet, PoseNetCriterion, SimpleSE3Criterion, SE3Criterion])

    def test_training(self):
        model = self._factory.make_from_parameters(self._params)
        self._trainer.fit(model, self._data_module)

    def test_testing(self):
        model = self._factory.make_from_parameters(self._params)
        self._trainer.test(model, self._data_module.test_dataloader())

    def test_training_se3_criterion(self):
        self._params.criterion.name = "SimpleSE3Criterion"
        model = self._factory.make_from_parameters(self._params)
        self._trainer.fit(model, self._data_module)

    def test_testing_se3_criterion(self):
        self._params.criterion.name = "SimpleSE3Criterion"
        model = self._factory.make_from_parameters(self._params)
        self._trainer.test(model, self._data_module.test_dataloader())