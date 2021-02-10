from .seven_scenes import SevenScenes
import os.path as osp
import numpy as np


class OdometrySevenScenes(SevenScenes):
    def __init__(self, odometry_path="dso_poses", **kwargs):
        super().__init__(**kwargs)
        # self._test_sequences = self.get_sequences(self._base_directory, False)
        self._test_sequences = [3]
        odometry_path = osp.join(self._base_directory, odometry_path)
        self._odometry, self._odometry_indexes = self.load_odometry(odometry_path, self._test_sequences)
        self.load_init_positions()

    @staticmethod
    def load_odometry(odometry_directory, sequences):
        odometry = {}
        odometry_indexes = {}
        for sequence in sequences:
            data_path = osp.join(odometry_directory, 'seq-{:02d}.txt'.format(sequence))
            data = np.loadtxt(data_path)
            odometry_indexes[sequence] = data[:, 0].astype(np.int)
            odometry[sequence] = np.zeros((data.shape[0], 4, 4))
            odometry[sequence][:, :3, :] = data[:, 1:].reshape(-1, 3, 4)
            odometry[sequence][:, 3, 3] = 1.
        return odometry, odometry_indexes

    def get_sequences(self, base_directory, train):
        if train:
            # return super().get_sequences(base_directory, True) + super().get_sequences(base_directory, False)
            return super().get_sequences(base_directory, True) + [3]
        return [3]

    def set_positions(self, sequence, index, positions):
        self.positions[sequence][index] = positions

    def get_odometry(self, sequence):
        return self._odometry[sequence], self._odometry_indexes[sequence]

    def get_test_sequences(self):
        return self._test_sequences

    def odometry_init_test_positions(self):
        for sequence in self._test_sequences:
            previous_odometry = self._odometry[sequence][0]
            j = 0
            for i in range(len(self.positions[sequence])):
                if self._odometry_indexes[sequence][j] == i:
                    self.positions[sequence][i] = self._odometry[sequence][j]
                    previous_odometry = self._odometry[sequence][j]
                    j += 1
                else:
                    self.positions[sequence][i] = previous_odometry

    def load_init_positions(self):
        for sequence in self._test_sequences:
            path = osp.join("notebooks", 'seq-{:02d}.npy'.format(sequence))
            positions = np.load(path)
            print(np.all(positions == positions))
            self.positions[sequence] = positions
