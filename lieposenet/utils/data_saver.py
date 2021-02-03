import numpy as np


class DataSaver(dict):
    def add(self, key, tensor):
        tensor = tensor.detach().cpu().numpy()
        if key not in self.keys():
            self[key] = tensor
            return

        self[key] = np.concatenate([self[key], tensor], axis=0)

    def save(self, file="trajectories.npy"):
        dictionary = {}
        for key, value in self.items():
            dictionary[key] = value
        np.save(file, dictionary)

