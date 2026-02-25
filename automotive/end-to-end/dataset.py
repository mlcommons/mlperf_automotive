import os
from torch.utils.data import Dataset

from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv import Config
from mmcv.parallel import collate

class Nuscenes(Dataset):
    def __init__(self, dataset_path, config, length):
        cfg = Config.fromfile(config)
        self.dataset_path = dataset_path
        self.preloaded = {}
        dataset = build_dataset(cfg.data.test)
        self.length = length if length > 0 else len(dataset)
        samples_per_gpu = 1
        
        self.data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )

    def __len__(self):
        return self.length

    def get_item_count(self):
        return self.length

    def load_item(self, index):
        item = self.data_loader.dataset[index]
        return item

    def get_samples(self, id_list):
        data = []
        for id in id_list:
            if id in self.preloaded:
                item = self.preloaded[id]
            else:
                item = self.load_item(id)
                
            # UniAD evaluates temporally and expects batch size 1. 
            # Collating multiple samples with varying sizes crashes torch.stack.
            # So we collate each item individually to maintain BS=1.
            collated_item = collate([item], samples_per_gpu=1)
            data.append(collated_item)
            
        return data

    def load_query_samples(self, sample_list):
        for sample in sample_list:
            self.preloaded[sample] = self.load_item(sample)

    def unload_query_samples(self, sample_list):
        if sample_list is None:
            self.preloaded.clear()
            return
            
        for sample in sample_list:
            if sample in self.preloaded:
                del self.preloaded[sample]