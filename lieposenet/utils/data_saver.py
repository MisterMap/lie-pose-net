import numpy as np


class DataSaver(dict):
    def __init__(self, data_saver_path):
        self._data_saver_path = data_saver_path
        super().__init__()

    def add(self, key, tensor):
        tensor = tensor.detach().cpu().numpy()
        if key not in self.keys():
            self[key] = tensor
            return

        self[key] = np.concatenate([self[key], tensor], axis=0)

    def save(self):
        dictionary = {}
        for key, value in self.items():
            dictionary[key] = value
        np.save(self._data_saver_path, dictionary)
