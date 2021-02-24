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


def show_3d_trajectories(trajectories):
    figure = plt.figure()
    ax = figure.add_subplot(1, 2, 2, projection='3d')
    for trajectory in trajectories:
        x, y, z = trajectory.T
        ax.plot(x, y, z)
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


def show_points(image, px, py):
    figure = plt.figure(dpi=200)
    plt.imshow(np.clip((image.cpu().detach().permute(1, 2, 0).numpy() + 1) / 2, 0, 1))
    px = ((px.cpu().detach().numpy() + 1) / 2) * image.shape[1]
    py = ((py.cpu().detach().numpy() + 1) / 2) * image.shape[2]
    plt.scatter(py, px, c="red")
    return figure
