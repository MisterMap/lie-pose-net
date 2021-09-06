from .torch_math import  *


class SE3Position(object):
    def __init__(self):
        self._log_q_position = None
        self._q_position = None
        self._matrix_position = None

    @staticmethod
    def from_matrix_position(matrix_position):
        position = SE3Position()
        position._matrix_position = matrix_position
        return position

    @staticmethod
    def from_q_position(q_position):
        position = SE3Position()
        position._q_position = q_position
        return position

    @staticmethod
    def from_log_q_position(log_q_position):
        position = SE3Position()
        position._log_q_position = log_q_position
        return position

    @property
    def translation(self):
        if self._log_q_position is not None:
            return self._log_q_position[:, :3]
        if self._q_position is not None:
            return self._q_position[:, :3]
        return self._matrix_position[:, :3, 3]

    @property
    def log_q_rotation(self):
        if self._log_q_position is not None:
            return self._log_q_position[:, 3:6]
        if self._q_position is not None:
            return logq_from_quaternion(self._q_position[:, 3:7])
        log_q_position = logq_position_from_matrix(self._matrix_position)
        return log_q_position[:, 3:6]

    @property
    def q_rotation(self):
        if self._log_q_position is not None:
            return quaternion_from_logq(self._log_q_position[:, 3:6])
        if self._q_position is not None:
            return self._q_position[:, 3:7]
        q_position = quaternion_from_matrix(self._matrix_position)
        return q_position

    @property
    def matrix(self):
        if self._log_q_position is not None:
            return matrix_from_logq_position(self._log_q_position)
        if self._q_position is not None:
            return matrix_from_q_position(self._q_position)
        return self._matrix_position[:, :, :]
