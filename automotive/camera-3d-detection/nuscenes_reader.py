
import numpy as np
import os
import pickle
from onnxruntime.quantization import CalibrationDataReader
import pickle

class Nuscenes(CalibrationDataReader):
    def __init__(self, pkl_file):
        self.enum_data = None
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        

    def __len__(self):
        return len(self.data)
    
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
