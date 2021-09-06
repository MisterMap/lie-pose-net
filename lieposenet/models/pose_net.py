import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_squared_error

from .base_lightning_module import BaseLightningModule
from ..utils.pose_net_result_evaluator import calculate_metrics, show_3d_trajectories
from ..utils.torch_math import *
from ..utils.data_saver import DataSaver
from torchvision import models
from ..utils.se3_position import SE3Position
from ..utils.torch_se3_math import calculate_log_se3_delta


class PoseNet(BaseLightningModule):
    def __init__(self, parameters, criterion, data_saver_path="trajectories.npy"):
        super(PoseNet, self).__init__(parameters)

        self.criterion = criterion
        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=parameters.feature_extractor.pretrained)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_extractor_output_dimension = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(feature_extractor_output_dimension, parameters.feature_dimension)

        self.final_fc = nn.Linear(parameters.feature_dimension, criterion.position_dimension, bias=parameters.bias)

        # initialize
        if parameters.feature_extractor.pretrained:
            init_modules = [self.feature_extractor.fc]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        self._data_saver = DataSaver(data_saver_path)

    def save_test_data(self, batch, output, losses):
        self._data_saver.add("truth_position", batch["position"][:, :3, 3])
        self._data_saver.add("truth_rotation", quaternion_from_matrix(batch["position"]))
        self._data_saver.add("predicted_position", self.criterion.translation(output))
        self._data_saver.add("predicted_rotation", self.criterion.rotation(output))
        self._data_saver.add("output", output)
        for key, value in self.criterion.saved_data(output).items():
            self._data_saver.add(key, value)

    def show_images(self):
        figure = show_3d_trajectories([self._data_saver["truth_position"],
                                       self._data_saver["predicted_position"]])
        # self.logger.log_figure("3d_trajectories", figure, self.global_step)

    def additional_metrics(self):
        metrics = calculate_metrics(self._data_saver)
        result = {}
        for key, value in metrics.items():
            result[key] = torch.tensor(value)
        return result

    def on_test_epoch_end(self):
        metrics = self.additional_metrics()
        self.log_dict(metrics)
        self.show_images()
        self._data_saver.save()

    def forward(self, x):
        x = self.feature_extractor(x)
        if self.hparams.activation == "tanh":
            x = F.tanh(x)
        else:
            x = F.relu(x)
        if self.hparams.drop_rate > 0:
            x = F.dropout(x, p=self.hparams.drop_rate, training=self.training)

        x = self.final_fc(x)
        return x

    def loss(self, batch):
        image = batch["image"]
        predicted_position = self.forward(image)
        target_position = batch["position"]
        loss = self.criterion(predicted_position, target_position)
        return predicted_position, {"loss": loss}

    def metrics(self, batch, output):
        target_position = SE3Position.from_matrix_position(batch["position"])
        predicted_position = self.criterion.se3_position(output)
        mean_translation_error = torch.mean(
            torch.norm(predicted_position.translation - target_position.translation, dim=1, p=2))
        mean_rotation_error = torch.mean(
            quaternion_angular_error(predicted_position.q_rotation, target_position.q_rotation)
        )
        translation_error = mean_squared_error(predicted_position.translation, target_position.translation)
        rotation_error = mean_squared_error(predicted_position.log_q_rotation, target_position.log_q_rotation)
        log_se3_delta = calculate_log_se3_delta(predicted_position, target_position)
        log_se3_translation_error = torch.mean(log_se3_delta[:, :3] ** 2)
        log_se3_rotation_error = torch.mean(log_se3_delta[:, 3:] ** 2)
        log_se3_error = torch.mean(log_se3_delta ** 2)

        metrics = {
            "mean_translation": mean_translation_error,
            "mean_rotation": mean_rotation_error,
            "mse_translation": translation_error,
            "mse_rotation": rotation_error,
            "mse_log_se3": log_se3_error,
            "mse_log_se3_rotation": log_se3_rotation_error,
            "mse_log_se3_translation": log_se3_translation_error
        }
        metrics.update(self.criterion.metrics())
        return metrics
