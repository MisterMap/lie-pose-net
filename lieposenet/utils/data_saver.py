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

    def get_sequence(self, sequence):
        if "sequence" not in self.keys():
            raise KeyError("[DataSaver] - sequence not in keys")
        if "index" not in self.keys():
            raise KeyError("[DataSaver] - indexes not in keys")
        mask = self["sequence"] == sequence
        index = self["index"][mask]
        reverse_index = np.argsort(index)
        data = {}
        for key, value in self.items():
            data[key] = value[mask][reverse_index]
        return data
