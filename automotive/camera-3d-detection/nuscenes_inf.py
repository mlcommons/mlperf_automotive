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
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import pickle
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import numpy as np
from custom import LoadMultiViewImageFromFiles, NormalizeMultiviewImage, MultiScaleFlipAug3D, RandomScaleImageMultiViewImage, PadMultiViewImage
from formatting import DefaultFormatBundle


def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class Nuscenes(Dataset):
    def __init__(self, cfg, dataset_path):
        self.pipeline = []
        self.pipeline.append(LoadMultiViewImageFromFiles(to_float32=True))
        self.pipeline.append(
            NormalizeMultiviewImage(
                mean=[
                    123.675, 116.28, 103.53], std=[
                    58.395, 57.12, 57.375], to_rgb=True))
        transforms = []
        transforms.append(RandomScaleImageMultiViewImage(scales=[0.5]))
        transforms.append(PadMultiViewImage(size_divisor=32))
        transforms.append(DefaultFormatBundle())
        self.pipeline.append(
            MultiScaleFlipAug3D(
                img_scale=(
                    1600,
                    900),
                pts_scale_ratio=1,
                flip=False,
                transforms=transforms))

        with open(os.path.join(dataset_path, cfg.data.test['ann_file']), 'rb') as f:
            data = pickle.load(f)
            self.data_infos = list(
                sorted(
                    data['infos'],
                    key=lambda e: e['timestamp']))

    def __len__(self):
        return len(self.data_infos)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        # TODO: Load queries into memory, if needed
        pass

    def unload_query_samples(self, sample_list):
        # TODO: Unload queries from memory, if needed
        pass

    def get_samples(self, id_list):
        data = []
        labels = []
        for id in id_list:
            item = self.get_item(id)
            data.append(item)
            labels.append(None)
        return data, labels

    def get_item(self, idx):
        data_info = self.get_data_info(idx)
        info = self.compose(data_info)
        return info

    def get_item_count(self):
        return len(self.data_infos)

    def compose(self, input_dict):
        for t in self.pipeline:
            input_dict = t(input_dict)
        return input_dict

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict


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
            processed_results.append([])
            detection_num = len(results[idx][0])
            if detection_num == 0:
                processed_results[idx].append([
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    content_id[idx]
                ])
            for detection in range(0, detection_num):
                processed_results[idx].append([
                    results[idx][0][detection][0],
                    results[idx][0][detection][1],
                    results[idx][0][detection][2],
                    results[idx][0][detection][3],
                    results[idx][0][detection][4],
                    results[idx][0][detection][5],
                    results[idx][0][detection][6],
                    results[idx][0][detection][7],
                    results[idx][0][detection][8],
                    results[idx][1][detection],
                    results[idx][2][detection],
                    content_id[idx]
                ])
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):

        return result_dict
