from .pose_net import PoseNet
import numpy as np


class PGOPoseNet(PoseNet):
    def __init__(self, params, pgo_optimizer, train_dataset, **kwargs):
        super().__init__(params, **kwargs)
        self._pgo_optimizer = pgo_optimizer
        self._train_dataset = train_dataset
        self._is_optimize_graph = False

    def on_train_epoch_end(self, outputs) -> None:
        super().on_train_epoch_end(outputs)
        self._is_optimize_graph = True

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self.show_images()
        if not self._is_optimize_graph:
            return
        sequences = self._train_dataset.get_test_sequences()
        for sequence in sequences:
            data = self._data_saver.get_sequence(sequence)
            mean_matrix = data["mean_matrix"].astype(np.float64)
            inverse_sigma_matrix = data["inverse_sigma_matrix"].astype(np.float64)
            index = data["index"].astype(np.int)
            print(index)
            odometry_positions, odometry_indexes = self._train_dataset.get_odometry(sequence)
            if len(index) > 0:
                self._pgo_optimizer.make_graph(mean_matrix, inverse_sigma_matrix, index,
                                               odometry_positions, odometry_indexes)
                optimized_trajectory = self._pgo_optimizer.optimize()
                self._train_dataset.set_positions(sequence, index, optimized_trajectory)
