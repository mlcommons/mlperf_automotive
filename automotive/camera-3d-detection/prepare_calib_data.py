#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import copy
import numpy as np
from collections import defaultdict
import onnxruntime as ort
import os
import sys
from post_process import PostProcess
import nuscenes_inf
import importlib
import csv
import torch
import pickle

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Create calibration data.")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument(
        "--onnx-path", required=True, type=str, help="Input onnx model without Q/DQ nodes."
    )
    parser.add_argument("--output", default=None, help="Path to save calibration data.")
    parser.add_argument("--verbose", action="store_true", help="If verbose, print all the debug info.")
    parser.add_argument(
    "--dataset-path",
    required=True,
    help="path to the dataset")
    args = parser.parse_args()
    return args


def generate_calibration_indices(calibration_list):
        calibration_indexes = []
        for row in calibration_list:
            # Assuming the first column is the index
            start = int(row[0])
            indices = range(start,start + int(row[1]))
            calibration_indexes.extend(indices)
        return calibration_indexes


def main():
    args = parse_args()
    spec = importlib.util.spec_from_file_location('bevformer_tiny', str(args.config))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    dataset = nuscenes_inf.Nuscenes(cfg, args.dataset_path, 'train')

    ort_sess = ort.InferenceSession(args.onnx_path)
    input_img_name = ort_sess.get_inputs()[0].name
    input_prev_bev_name = ort_sess.get_inputs()[1].name
    input_use_prev_bev_name = ort_sess.get_inputs()[2].name
    input_can_bus_name = ort_sess.get_inputs()[3].name
    input_lidar2img_name = ort_sess.get_inputs()[4].name
    calibration_file = "calibration_list.csv"
    calibration_list= []
    with open(calibration_file, 'r') as f:
        reader = csv.reader(f)
        calibration_header = next(reader)  # Read the header
        calibration_list = [row for row in reader]
        calibration_list.extend(calibration_list)
    calibration_indexes = generate_calibration_indices(calibration_list)
    post_process = PostProcess(
        num_classes=10, max_num=300, pc_range=[
            -51.2, -51.2, -5.0, 51.2, 51.2, 3.0], post_center_range=[
            -61.2, -61.2, -10.0, 61.2, 61.2, 10.0], score_threshold=None)


    prev_bev = torch.zeros(cfg.bev_h_ * cfg.bev_w_, 1, cfg._dim_)
    prev_frame_info = {
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    inputs_dict = defaultdict(lambda: [])
    num_samples = 0
    calibration_input = []
    for idx in calibration_indexes:
        input_dict = dataset.get_item(idx)
        tmp_pos = (input_dict['can_bus'][0][:3]).copy()
        tmp_angle = (input_dict['can_bus'][0][-1]).copy()
        if input_dict["scene_token"] != prev_frame_info["scene_token"]:
            use_prev_bev = torch.tensor(0.0)
            # prev_bev = None
            input_dict["can_bus"][0][-1] = 0
            input_dict["can_bus"][0][:3] = 0
        else:
            use_prev_bev = torch.tensor(1.0)
            input_dict["can_bus"][0][:3] -= prev_frame_info["prev_pos"]
            input_dict["can_bus"][0][-1] -= prev_frame_info["prev_angle"]
        prev_frame_info["scene_token"] = input_dict["scene_token"]
        can_bus = input_dict["can_bus"][0].astype(np.float32)
        lidar2img = np.stack(
            input_dict['lidar2img'][0]).astype(
            np.float32)
        img = input_dict['img'][0]
        input_data = {input_img_name: np.expand_dims(to_numpy(img), 0),
                      input_prev_bev_name: to_numpy(prev_bev),
                      input_use_prev_bev_name: to_numpy(use_prev_bev),
                      input_can_bus_name: can_bus,
                      input_lidar2img_name: np.expand_dims(lidar2img, 0),
                      }
        calibration_input.append(input_data)
        result = ort_sess.run(None, input_data)
        bev_embed = torch.from_numpy(result[0])
        outputs_classes = torch.from_numpy(result[1])
        outputs_coords = torch.from_numpy(result[2])
        result = post_process.process(outputs_classes, outputs_coords)
        prev_bev = bev_embed
        prev_frame_info["prev_pos"] = tmp_pos
        prev_frame_info["prev_angle"] = tmp_angle
    # Save the dictionary to an .npz file
    pickle.dump(calibration_input, open(args.output, 'wb'))
    print(f"Calibration data saved to {args.output}")


if __name__ == "__main__":
    main()