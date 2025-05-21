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
import dataset
import os
from PIL import Image
import csv
import ast
import random
import pickle


class Cognata(dataset.Dataset):
    def __init__(self, data_root, length):
        self.data_root = data_root
        self.preloaded = {}
        self.length = length

    def __len__(self):
        return self.length

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        for sample in sample_list:
            item = self.load_item(sample)
            self.preloaded[sample] = {
                'img': item['img'],
                'idx': item['idx'],
                'img_size': item['img_size'],
            }

    def unload_query_samples(self, sample_list):
        for sample in sample_list:
            del self.preloaded[sample]

    def get_samples(self, id_list):
        data = []
        for id in id_list:
            item = self.get_item(id)
            data.append((item['img'], item['idx'], item['img_size']))
        return data, None

    def get_item(self, idx):
        return self.preloaded[idx]

    def load_item(self, index, prefix='val'):
        file_path = os.path.join(self.data_root, f'{prefix}_{index}.pkl')
        with open(file_path, 'rb') as f:
            item = pickle.load(f)
        return item

    def get_item_count(self):
        return self.length


def object_labels(files, ignore_classes):
    counter = 1
    label_map = {}
    label_info = {}
    label_info[0] = "background"
    label_map[0] = 0
    for file in files:
        with open(file['ann']) as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            annotations = rows[1:]
            class_index = header.index('object_class')
            class_name_index = header.index('object_class_name')
            for annotation in annotations:
                label = ast.literal_eval(annotation[class_index])
                if label not in label_map and not int(label) in ignore_classes:
                    label_map[label] = counter
                    label_info[counter] = annotation[class_name_index]
                    counter += 1
    return label_map, label_info


def prepare_cognata(root, folders, cameras, ignore_classes=[2, 25, 31]):
    files = []
    for folder in folders:
        for camera in cameras:
            ann_folder = os.path.join(root, folder, camera + '_ann')
            img_folder = os.path.join(root, folder, camera + '_png')
            ann_files = sorted([os.path.join(ann_folder, f) for f in os.listdir(
                ann_folder) if os.path.isfile(os.path.join(ann_folder, f))])
            img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(
                img_folder) if os.path.isfile(os.path.join(img_folder, f))])
            for i in range(len(ann_files)):
                with open(ann_files[i]) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    header = rows[0]
                    annotations = rows[1:]
                    bbox_index = header.index('bounding_box_2D')
                    class_index = header.index('object_class')
                    distance_index = header.index('center_distance')
                    for annotation in annotations:
                        bbox = annotation[bbox_index]
                        bbox = ast.literal_eval(bbox)
                        object_width = bbox[2] - bbox[0]
                        object_height = bbox[3] - bbox[1]
                        object_area = object_width * object_height
                        label = ast.literal_eval(annotation[class_index])
                        distance = ast.literal_eval(annotation[distance_index])
                        if object_area >= 50 and int(
                                label) not in ignore_classes and object_height >= 8 and object_width >= 8 and distance <= 300:
                            files.append(
                                {'img': img_files[i], 'ann': ann_files[i]})
                            break

    label_map, label_info = object_labels(files, ignore_classes)
    return files, label_map, label_info


def train_val_split(files, calibration_length=200):
    random.Random(5).shuffle(files)
    val_index = round(len(files) * 0.8)
    calibration_index = len(files) - calibration_length
    return {'train': files[:val_index], 'val': files[val_index:calibration_index],
            'calibration': files[calibration_index:]}


class PostProcessCognata:
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
            detection_num = len(results[0])
            if detection_num == 0:
                processed_results[idx].append([
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
                    results[0][detection][0],
                    results[0][detection][1],
                    results[0][detection][2],
                    results[0][detection][3],
                    results[1][detection],
                    results[2][detection],
                    content_id[idx]
                ])
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):

        return result_dict
