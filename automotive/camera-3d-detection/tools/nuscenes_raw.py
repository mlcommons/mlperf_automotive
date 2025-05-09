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
from pyquaternion import Quaternion
import numpy as np
from custom import LoadMultiViewImageFromFiles, NormalizeMultiviewImage, MultiScaleFlipAug3D, RandomScaleImageMultiViewImage, PadMultiViewImage
from formatting import DefaultFormatBundle


class Nuscenes(Dataset):
    def __init__(self, dataset_path):
        self.pipeline = []
        self.pipeline.append(
            LoadMultiViewImageFromFiles(
                to_float32=True,
                data_root=dataset_path))
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

        with open(os.path.join(dataset_path, 'nuscenes', 'nuscenes_infos_temporal_val.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.data_infos = list(
                sorted(
                    data['infos'],
                    key=lambda e: e['timestamp']))

    def __len__(self):
        return len(self.data_infos)

    def get_item(self, idx):
        data_info = self.get_data_info(idx)
        info = self.compose(data_info)
        return info

    def compose(self, input_dict):
        for t in self.pipeline:
            input_dict = t(input_dict)
        return input_dict

    def quaternion_yaw(self, q: Quaternion) -> float:
        # From nuScenes devkit
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        return yaw

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
        patch_angle = self.quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict
