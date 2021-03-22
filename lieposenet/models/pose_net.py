"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
implementation of PoseNet and MapNet networks 
"""
import torch.nn as nn
import torch.nn.functional as F

from .base_lightning_module import BaseLightningModule
from ..utils.pose_net_result_evaluator import *
from ..utils.torch_math import *
from ..utils.data_saver import DataSaver


class PoseNet(BaseLightningModule):
    def __init__(self, parameters, feature_extractor, criterion, data_saver_path="trajectories.npy"):
        super(PoseNet, self).__init__(parameters)

        self.criterion = criterion
        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
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
        self.logger.log_figure("3d_trajectories", figure, self.global_step)

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
        truth_position = batch["position"][:, :3, 3]
        truth_rotation = quaternion_from_matrix(batch["position"])
        predicted_position = self.criterion.translation(output)
        predicted_rotation = self.criterion.rotation(output)
        metrics = {
            "position_error": torch.mean(torch.sqrt(torch.sum((truth_position - predicted_position) ** 2, dim=1))),
            "rotation_error": torch.mean(quaternion_angular_error(truth_rotation, predicted_rotation))
        }
        return metrics
