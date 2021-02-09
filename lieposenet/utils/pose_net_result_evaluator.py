import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def show_3d_trajectories(truth_trajectory, predicted_trajectory):
    figure = plt.figure(figsize=(35, 15))
    ax = figure.add_subplot(1, 2, 2, projection='3d')
    ss = 1
    pred_poses = predicted_trajectory
    targ_poses = truth_trajectory
    z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
    for xx, yy, zz in zip(x.T, y.T, z.T):
        ax.plot(xx, yy, zs=zz, c='b')
    ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0)
    ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0)
    ax.view_init(azim=119, elev=13)
    return figure


def save_trajectories(trajectories, file="trajectories.npy"):
    np.save(file, trajectories)


def calculate_metrics(data_saver):
    truth_trajectory = data_saver["truth_position"]
    predicted_trajectory = data_saver["predicted_position"]
    truth_rotation = data_saver["truth_rotation"]
    predicted_rotation = data_saver["predicted_rotation"]
    position_errors = np.linalg.norm(truth_trajectory - predicted_trajectory, axis=1)
    rotation_errors = [quaternion_angular_error(q1, q2) for q1, q2 in zip(truth_rotation, predicted_rotation)]
    rotation_errors = np.array(rotation_errors)
    results = {
        "median_position_error": float(np.median(position_errors)),
        "median_rotation_error": float(np.median(rotation_errors))
    }
    return results
