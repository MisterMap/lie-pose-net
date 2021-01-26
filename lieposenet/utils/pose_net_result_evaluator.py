import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def show_3d_trajectories(trajectories):
    figure = plt.figure()
    ax = figure.add_subplot(1, 2, 2, projection='3d')
    for trajectory in trajectories:
        x, y, z = trajectory.T
        ax.plot(x, y, z)
    return figure


def save_trajectories(trajectories, file="trajectories.npy"):
    np.save(file, trajectories)
