"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
implementation of PoseNet and MapNet networks 
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils.pose_net_result_evaluator import *
from ..utils.torch_math import *


from .base_lightning_module import BaseLightningModule


class PoseNet(BaseLightningModule):
    def __init__(self, parameters, feature_extractor, criterion):
        super(PoseNet, self).__init__(parameters)

        self.criterion = criterion
        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_extractor_output_dimension = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(feature_extractor_output_dimension, parameters.feature_dimension)

        self.final_fc = nn.Linear(parameters.feature_dimension, criterion.position_dimension)

        # initialize
        if parameters.feature_extractor.pretrained:
            init_modules = [self.feature_extractor.fc, self.final_fc]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        self._truth_trajectory = np.zeros([0, 3])
        self._predicted_trajectory = np.zeros([0, 3])

    def save_test_data(self, batch, output, losses):
        truth_position = batch["position"][:, :3, 3].detach().cpu().numpy()
        predicted_position = output[:, :3].detach().cpu().numpy()
        self._truth_trajectory = np.concatenate([self._truth_trajectory, truth_position], axis=0)
        self._predicted_trajectory = np.concatenate([self._predicted_trajectory, predicted_position], axis=0)

    def show_images(self):
        figure = show_3d_trajectories([self._truth_trajectory, self._predicted_trajectory])
        self.logger.log_figure("3d_trajectories", figure, self.global_step)

    def on_test_epoch_end(self):
        self.show_images()
        save_trajectories([self._truth_trajectory, self._predicted_trajectory])

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.hparams.drop_rate > 0:
            x = F.dropout(x, p=self.hparams.drop_rate)

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
        predicted_position = output[:, :3]
        predicted_rotation = quaternion_from_logq(output[:, 3:])
        metrics = {
            "position_error": torch.mean(torch.sqrt((truth_position - predicted_position) ** 2)),
            "rotation_error": torch.mean(quaternion_angular_error(truth_rotation, predicted_rotation))
        }
        return metrics
