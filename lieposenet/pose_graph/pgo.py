import numpy as np
from minisam import *


class PGO(object):
    def __init__(self, linear_sigma=0.02, angular_sigma=0.02):
        self._linear_sigma = linear_sigma
        self._angular_sigma = angular_sigma
        self._graph = None
        self._initials = None
        self._measurement_indexes = []

    @staticmethod
    def se3_position(matrix):
        u, d, v = np.linalg.svd(matrix[:3, :3])
        matrix[:3, :3] = u @ v
        return SE3(matrix)

    # noinspection PyArgumentList,PyCallByClass
    def make_graph(self, mean_matrix, inverse_sigma_matrix, measurement_indexes,
                   odometry_positions, odometry_indexes):
        self._graph = FactorGraph()
        self._initials = Variables()
        self._measurement_indexes = []
        for i, position, sqrt_information in zip(measurement_indexes, mean_matrix, inverse_sigma_matrix):
            position = self.se3_position(position)
            prior_loss = GaussianLoss.SqrtInformation(sqrt_information)
            self._graph.add(PriorFactor(key('x', i), position, prior_loss))
            self._initials.add(key('x', i), position)
            self._measurement_indexes.append(key("x", i))

        linear_sigma = self._linear_sigma
        angular_sigma = self._angular_sigma
        odometry_loss = DiagonalLoss.Sigmas(np.array([linear_sigma, linear_sigma, linear_sigma,
                                                      angular_sigma, angular_sigma, angular_sigma]))
        for i in range(len(odometry_positions) - 1):
            odometry_delta = odometry_positions[i + 1] @ np.linalg.inv(odometry_positions[i])
            odometry_delta = self.se3_position(odometry_delta)
            self._graph.add(BetweenFactor(key('x', int(odometry_indexes[i])), key('x', int(odometry_indexes[i + 1])),
                                          odometry_delta, odometry_loss))
            # if not self._initials.exists(key("x", int(odometry_indexes[i]))):
            #     self._initials.add(key("x", int(odometry_indexes[i])),
            #                        self.se3_position(odometry_positions[i]))
            # if not self._initials.exists(key("x", int(odometry_indexes[i + 1]))):
            #     self._initials.add(key("x", int(odometry_indexes[i + 1])),
            #                        self.se3_position(odometry_positions[i + 1]))

    def optimize(self):
        optimizer_param = LevenbergMarquardtOptimizerParams()
        optimizer = LevenbergMarquardtOptimizer(optimizer_param)

        results = Variables()
        status = optimizer.optimize(self._graph, self._initials, results)
        if status != NonlinearOptimizationStatus.SUCCESS:
            print("optimization error: ", status)
        optimized_trajectory = [results.at(key('x', i)).matrix() for i in self._measurement_indexes]
        optimized_trajectory = np.array(optimized_trajectory)
        return optimized_trajectory
