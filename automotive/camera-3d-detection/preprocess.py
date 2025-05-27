import argparse
import sys
import os
from tools import nuscenes_raw
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import torch
import numpy as np
import importlib
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class Process():
    def __init__(self, dataset_root, output, cfg, prefix='val'):
        self.dataset_root = dataset_root
        self.ds = nuscenes_raw.Nuscenes(self.dataset_root)
        self.preprocessed_directory = output
        self.prefix = prefix
        self.prev_frame_info = {
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.prev_bev = torch.zeros(cfg.bev_h_ * cfg.bev_w_, 1, cfg._dim_)
        os.makedirs(self.preprocessed_directory, exist_ok=True)

    def process_item(self, i):
        input_dict = self.ds.get_item(i)
        tmp_pos = (input_dict['can_bus'][0][:3]).copy()
        tmp_angle = (input_dict['can_bus'][0][-1]).copy()
        if input_dict["scene_token"] != self.prev_frame_info["scene_token"]:
            use_prev_bev = torch.tensor(0.0)
            # prev_bev = None
            input_dict["can_bus"][0][-1] = 0
            input_dict["can_bus"][0][:3] = 0
        else:
            use_prev_bev = torch.tensor(1.0)
            input_dict["can_bus"][0][:3] -= self.prev_frame_info["prev_pos"]
            input_dict["can_bus"][0][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["scene_token"] = input_dict["scene_token"]
        can_bus = input_dict["can_bus"][0].astype(np.float32)
        lidar2img = np.stack(
            input_dict['lidar2img'][0]).astype(
            np.float32)
        img = input_dict['img'][0]
        input_data = {'img': np.expand_dims(to_numpy(img), 0),
                'use_prev_bev': to_numpy(use_prev_bev),
                'can_bus': can_bus,
                'lidar2img': np.expand_dims(lidar2img, 0),
                }
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        output_file = os.path.join(
            self.preprocessed_directory,
            f"{self.prefix}_{i}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(input_data, f)

    def num_items(self):
        return len(self.ds)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess semantic segmentation dataset.")
    parser.add_argument('--dataset-root', type=str, required=True,
                        help="Path to the root directory of the dataset.")
    parser.add_argument('--output', type=str, required=True,
                        help="output directory for pkl files.")
    parser.add_argument('--calibration-set', action='store_true',
                        help="Flag to indicate if the calibration set is being processed.")
    parser.add_argument("--config", type=str, required=True, help="bevformer configuration file path")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    # Add your preprocessing logic here
    spec = importlib.util.spec_from_file_location(
        'bevformer_tiny', str(args.config))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    if args.calibration_set:
        prefix = 'calib'
    else:
        prefix = 'val'
    proc = Process(
        dataset_root,
        args.output,
        cfg,
        prefix)

    for i in tqdm(range(proc.num_items()), desc="Preprocessing dataset",
                total=proc.num_items()):
        proc.process_item(i)

if __name__ == "__main__":
    main()
