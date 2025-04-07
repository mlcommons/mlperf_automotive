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
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
from PIL import Image
import csv
import ast
import random


def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class Cognata(Dataset):
    def __init__(self, label_map, label_info, files,
                 ignore_classes=[2, 25, 31], transform=None):
        self.label_map = label_map
        self.label_info = label_info
        self.transform = transform
        self.files = files
        self.ignore_classes = ignore_classes

    def __len__(self):
        return len(self.files)

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
            data.append((item[0], item[1], item[2]))
            labels.append((item[3], item[4], item[5]))
        return data, labels

    def get_item(self, idx):
        img = Image.open(self.files[idx]['img']).convert('RGB')
        width, height = img.size
        boxes = []
        labels = []
        gt_boxes = []
        with open(self.files[idx]['ann']) as f:
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
                if object_area < 50 or int(
                        label) in self.ignore_classes or object_height < 8 or object_width < 8 or distance > 300:
                    continue
                boxes.append([bbox[0] / width, bbox[1] / height,
                             bbox[2] / width, bbox[3] / height])
                label = self.label_map[label]
                gt_boxes.append(
                    [bbox[0], bbox[1], bbox[2], bbox[3], label, 0, 0])
                labels.append(label)

            boxes = torch.tensor(boxes)
            labels = torch.tensor(labels)
            gt_boxes = torch.tensor(gt_boxes)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(
                img, (height, width), boxes, labels, max_num=500)
        return image, idx, (height, width), boxes, labels, gt_boxes

    def get_item_count(self):
        return len(self.files)


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
    return {'train': files[:val_index], 'val': files[val_index:calibration_index], 'calibration': files[calibration_index:]}


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
