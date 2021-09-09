import os

import pytorch_lightning as pl
from clearml import Task, Logger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.parsing import AttributeDict

from lieposenet.criterions.pose_net_criterion import PoseNetCriterion
from lieposenet.criterions.se3_criterion import SE3Criterion
from lieposenet.criterions.simple_se3_criterion import SimpleSE3Criterion
from lieposenet.data import SevenScenesDataModule
from lieposenet.models.pose_net import PoseNet
from lieposenet.utils.universal_factory import UniversalFactory

DEFAULT_MODEL_PARAMS = AttributeDict(**{
    "name": "PoseNet",
    "feature_extractor": AttributeDict(
        pretrained=True
    ),
    "criterion": {
        "name": "SE3Criterion",
        "rotation_koef": -3.0,
        "translation_koef": -3.0,
        "use_se3_translation": True,
        "loss_type": "l2",
        "koef_requires_grad": True,
        "lr": 0.0001
    },
    "feature_dimension": 2048,
    "drop_rate": 0,
    "optimizer": AttributeDict(
        betas="0.9 0.999",
        lr=0.0001,
        weight_decay=0.0005,
    ),
    "scheduler": {
        "step_size": 20,
        "gamma": 0.5,
    },
    "bias": True,
    "activation": "tanh",
    "pretrained": True
})

DEFAULT_DATA_MODEL_PARAMS = {
    "batch_size": 64,
    "use_test": True,
    "num_workers": 4,
    "image_size": 256,
    "scene": "fire",
    "data_path": "/media/mikhail/Data3T/7scenes"
}

DEFAULT_TRAINER_PARAMS = {
    "max_epochs": 1,
    "checkpoint_every_n_val_epochs": 10,
    "gpus": 1,
    "check_val_every_n_epoch": 2
}


class MlFlowExperiment(object):
    def __init__(self):
        self._task = Task.init(project_name="lie-pose-net", task_name="LiePoseNet on local machine")
        self._factory = UniversalFactory([PoseNet, PoseNetCriterion, SE3Criterion, SimpleSE3Criterion])
        self._scene = None

    def run_experiment(self):
        data_module = self.prepare_data_module()
        model = self.prepare_model()
        trainer = self.prepare_trainer()
        trainer.fit(model, data_module)
        trainer.test(model, data_module.test_dataloader())
        self.log_trajectories(model)

    def prepare_data_module(self):
        data_module_params = DEFAULT_DATA_MODEL_PARAMS
        self._task.connect(data_module_params)
        self._scene = data_module_params["scene"]
        return SevenScenesDataModule(**data_module_params)

    def prepare_model(self):
        model_params = DEFAULT_MODEL_PARAMS
        self._task.connect(model_params)
        print(model_params)
        return self._factory.make_from_parameters(model_params)

    def prepare_trainer(self):
        # noinspection PyTypeChecker
        logger_path = os.path.join(os.path.dirname(self._task.cache_dir), "lightning_logs", "lieposenet")
        trainer_params = DEFAULT_TRAINER_PARAMS
        self._task.connect(trainer_params)
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                        every_n_val_epochs=trainer_params[
                                                            "checkpoint_every_n_val_epochs"])
        trainer = self._factory.kwargs_function(pl.Trainer)(
            logger=TensorBoardLogger(logger_path, name=self._scene),
            callbacks=[model_checkpoint],
            **trainer_params
        )
        return trainer

    @staticmethod
    def log_trajectories(model):
        truth_position = model.data_saver["truth_position"]
        predicted_position = model.data_saver["predicted_position"]

        Logger.current_logger().report_scatter3d(title="trajectory", series="truth_positions", iteration=1,
                                                 scatter=truth_position,
                                                 xaxis="x", yaxis="y", zaxis="z", mode="lines")
        Logger.current_logger().report_scatter3d(title="trajectory", series="predicted_position", iteration=1,
                                                 scatter=predicted_position,
                                                 xaxis="x", yaxis="y", zaxis="z", mode="lines")
