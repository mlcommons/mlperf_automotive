"""
Modifications by MLCommons from SSD-Pytorch (https://github.com/uvipen/SSD-pytorch) author: Viet Nguyen (nhviet1009@gmail.com)
Copyright 2024 MLCommons Association and Contributors

MIT License

Copyright (c) 2021 Viet Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from torch.utils.data import Dataset
import os
import pickle
import torch


class Nuscenes(Dataset):
    def __init__(self, data_root, length):
        self.data_root = data_root
        self.preloaded = {}
        self.length = length

    def __len__(self):
        return self.length

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_item(self, index, prefix='val'):
        file_path = os.path.join(self.data_root, f'{prefix}_{index}.pkl')
        with open(file_path, 'rb') as f:
            item = pickle.load(f)
        return item

    def load_query_samples(self, sample_list):
        for sample in sample_list:
            self.preloaded[sample] = self.load_item(sample)

    def unload_query_samples(self, sample_list):
        for sample in sample_list:
            del self.preloaded[sample]

    def get_samples(self, id_list):
        data = []
        labels = []
        for id in id_list:
            item = self.get_item(id)
            data.append(item)
            labels.append(None)
        return data, labels

    def get_item(self, index):
        return self.preloaded[index]

    def get_item_count(self):
        return self.length
    
    def get_item_loc(self, index):
        file_path = os.path.join(self.data_root, f'val_{index}.pkl')
        src = os.path.join(self.data_path, file_path)
        return src


class PostProcessNuscenes:
    def __init__(
        self,  # Postprocess parameters
    ):
        self.content_ids = []
        # TODO: Init Postprocess parameters
        self.results = []

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, content_id, inputs, result_dict):
        self.content_ids.extend(content_id)
        processed_results = []
        for idx in range(len(content_id)):
            processed_results.append(torch.stack(results).cpu())
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):

        return result_dict
