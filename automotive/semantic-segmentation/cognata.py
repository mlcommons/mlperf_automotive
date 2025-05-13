import os
import logging
import dataset
import torch
import pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cognata")


class Cognata(dataset.Dataset):
    def __init__(self, data_root, length):
        self.data_root = data_root
        self.preloaded = {}
        self.length = length

    def get_item(self, index):
        return self.preloaded[index]

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
            data.append(item['img'])
            labels.append(item['label'])
        return torch.stack(data), torch.stack(labels)

    def __len__(self):
        return self.length

    def get_item_count(self):
        return self.length


class PostProcessCognata:
    def __init__(
        self,  # Postprocess parameters
    ):
        self.content_ids = []
        self.results = []

    def add_results(self, results):
        pass
        # self.results.extend(results)

    def __call__(self, results, content_id, inputs, result_dict):
        self.content_ids.extend(content_id)
        processed_results = []
        for idx in range(len(content_id)):
            processed_results.append([])
            processed_results[idx].append(results.cpu())
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):
        pass
