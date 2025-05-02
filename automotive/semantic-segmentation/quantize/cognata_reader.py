
import numpy as np
import os
import pickle
from onnxruntime.quantization import CalibrationDataReader
import pickle


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class Cognata(CalibrationDataReader):
    def __init__(self, data_root, num_items):
        self.data_root = data_root
        self.length = num_items
        self.data = []
        for i in range(num_items):
            file_path = os.path.join(self.data_root, f'calib_{i}.pkl')
            with open(file_path, 'rb') as f:
                item = pickle.load(f)

                self.data.append(
                    {'input_image': to_numpy(item['img'].unsqueeze(0))})
        self.enum_data = iter(self.data)

    def __len__(self):
        return len(self.data)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter(self.data)
