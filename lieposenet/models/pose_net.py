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
from ..criterions.poe_se3_criterion import POESE3Criterion

class PoseNet(BaseLightningModule):
    def __init__(self, parameters, feature_extractor, criterion, data_saver_path="trajectories.npy"):
        super(PoseNet, self).__init__(parameters)

        self.criterion = criterion
        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        previous_conv1 = self.feature_extractor.conv1
        # self.feature_extractor.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
        #     bias=False)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        feature_extractor_output_dimension = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(feature_extractor_output_dimension, parameters.feature_dimension)

        self.first_final_fc = nn.Linear(parameters.feature_dimension, parameters.feature_dimension)
        # self.final_fc = nn.Linear(parameters.feature_dimension + 2, criterion.position_dimension, bias=parameters.bias)
        self.final_fc = nn.Linear(parameters.feature_dimension, criterion.position_dimension, bias=parameters.bias)

        # initialize
        if parameters.feature_extractor.pretrained:
            # init_modules = [self.feature_extractor.fc, self.final_fc]
            # init_modules = [self.feature_extractor.fc, self.feature_extractor.conv1]
            init_modules = [self.feature_extractor.fc]
            # init_modules = [self.feature_extractor.fc, self.first_final_fc]
        else:
            init_modules = self.modules()
        # init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        # nn.init.constant_(self.feature_extractor.conv1.weight.data, 0)
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
        self.show_images()
        self._data_saver.save()

    def forward(self, x, shift_x, shift_y):
        px = torch.linspace(-1, 1, x.shape[2], device=x.device)
        px = px[:, None].repeat_interleave(x.shape[3], dim=1)[None].repeat_interleave(x.shape[0], dim=0)
        py = torch.linspace(-1, 1, x.shape[3], device=x.device)
        py = py[None, :].repeat_interleave(x.shape[2], dim=0)[None].repeat_interleave(x.shape[0], dim=0)
        # x = torch.cat([x, px[:, None], py[:, None]], dim=1)
        # if self.current_epoch > 0:
        #     x = F.dropout(x, p=0.01 * self.current_epoch, training=self.training)
        x = self.feature_extractor(x)
        x = F.tanh(x)
        if self.hparams.drop_rate > 0:
            x = F.dropout(x, p=self.hparams.drop_rate, training=self.training)
        # x = self.first_final_fc(x)
        # x = F.tanh(x)
        # x = torch.cat([x, shift_x, shift_y], dim=1)
        x = self.final_fc(x)
        return x

    def test_forward(self, x):
        images = []
        for i in range(10):
            x1 = int(np.random.rand() * x.shape[2])
            x2 = int(np.random.rand() * x.shape[2])
            y1 = int(np.random.rand() * x.shape[3])
            y2 = int(np.random.rand() * x.shape[3])
            mask = torch.zeros_like(x)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1 = max(0, x1 - 64)
            x2 = min(x.shape[2], x2 + 64)
            y1 = max(0, y1 - 64)
            y2 = min(x.shape[3], y2 + 64)
            mask[:, x1:x2, y1:y2] = 1. * x.shape[3] * x.shape[3] / (x2 - x1) / (y2 - y1)
            images.append(mask * x)
        x = torch.stack(images, dim=0).reshape(-1, x.shape[1], x.shape[2], x.shape[3])
        x = self.forward(x, 0, 0)
        x = x.reshape(10, -1, x.shape[1])
        final_x = torch.zeros(x.shape[1], x.shape[2], device=x.device)
        for i in range(10):
            final_x[:, i * 28:i * 28 + 28] = x[i, :, i * 28:i * 28 + 28]
        return final_x

    # def test_forward(self, x):
    #     shift_xs = np.linspace(0, 1, 4)
    #     shift_ys = np.linspace(0, 1, 4)
    #     images = []
    #     shift_xs_list = []
    #     shift_ys_list = []
    #     w0 = 128
    #     for shift_x in shift_xs:
    #         for shift_y in shift_ys:
    #             x0 = int((x.shape[2] - w0) * shift_x)
    #             y0 = int((x.shape[3] - w0) * shift_y)
    #             images.append(x[:, :, x0:x0 + w0, y0:y0+w0])
    #             shift_xs_list.append(torch.ones(x.shape[0], device=x.device) * shift_x)
    #             shift_ys_list.append(torch.ones(x.shape[0], device=x.device) * shift_y)
    #     x = torch.stack(images, dim=0).reshape(-1, 3, w0, w0)
    #     shift_x = torch.stack(shift_xs_list, dim=0).reshape(-1, 1) * 2 - 1
    #     shift_y = torch.stack(shift_ys_list, dim=0).reshape(-1, 1) * 2 - 1
    #     x = self.feature_extractor(x)
    #     x = F.tanh(x)
    #     if self.hparams.drop_rate > 0:
    #         x = F.dropout(x, p=self.hparams.drop_rate, training=self.training)
    #     x = self.first_final_fc(x)
    #     x = torch.cat([x, shift_x, shift_y], dim=1)
    #     x = self.final_fc(x)
    #     x = x.reshape(16, x.shape[0] // 16, -1)
    #     final_x = torch.zeros(x.shape[1], x.shape[2], device=x.device)
    #     for i in range(16):
    #         final_x[:, i * 28:i * 28 + 28] = x[i, :, i * 28:i * 28 + 28]
    #     return final_x

    def loss(self, batch):
        image = batch["image"]
        predicted_position = self.forward(image, batch["shift_x"], batch["shift_y"])
        # if self.training:
        #     predicted_position = self.forward(image, batch["shift_x"], batch["shift_y"])
        # else:
        #     predicted_position = self.test_forward(image)
        target_position = batch["position"]
        loss = self.criterion(predicted_position, target_position)
        losses = self.criterion.additional_losses()
        losses["loss"] = loss
        return predicted_position, losses

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
