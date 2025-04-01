"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""
import argparse
import json
import os
import pprint
import numpy as np
from cognata import collate_fn, Cognata, prepare_cognata, train_val_split
from transform import SSDTransformer
import importlib
import torch
from utils import generate_dboxes, Encoder
from torchmetrics.detection.mean_ap import MeanAveragePrecision
# pylint: disable=missing-docstring


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlperf-accuracy-file",
        required=True,
        help="path to mlperf_log_accuracy.json")
    parser.add_argument(
        "--cognata-dir",
        required=True,
        help="cognata dataset directory")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose messages")
    parser.add_argument(
        "--output-file",
        default="openimages-results.json",
        help="path to output file")
    parser.add_argument(
        "--use-inv-map",
        action="store_true",
        help="use inverse label map")
    parser.add_argument("--config", help="config file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    detections = {}
    image_ids = set()
    seen = set()
    no_results = 0
    config = importlib.import_module('config.' + args.config)
    folders = config.dataset['folders']
    cameras = config.dataset['cameras']
    ignore_classes = [2, 25, 31]
    if 'ignore_classes' in config.dataset:
        ignore_classes = config.dataset['ignore_classes']
    files, label_map, label_info = prepare_cognata(
        args.cognata_dir, folders, cameras, ignore_classes)
    files = train_val_split(files)
    dboxes = generate_dboxes(config.model, model="ssd")
    image_size = config.model['image_size']
    val_set = Cognata(
        label_map,
        label_info,
        files['val'],
        ignore_classes,
        SSDTransformer(
            dboxes,
            image_size,
            val=True))
    preds = []
    targets = []
    for j in results:
        idx = j['qsl_idx']
        # de-dupe in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)

        # reconstruct from mlperf accuracy log
        # what is written by the benchmark is an array of float32's:
        # id, box[0], box[1], box[2], box[3], score, detection_class
        # note that id is a index into instances_val2017.json, not the actual
        # image_id
        data = np.frombuffer(bytes.fromhex(j['data']), np.float32)
        current_id = -1
        predictions = {}
        dts = []
        labels = []
        scores = []
        ids = []
        for i in range(0, len(data), 7):
            box = [float(x) for x in data[i:i + 4]]
            label = int(data[i + 4])
            score = float(data[i + 5])
            image_idx = int(data[i + 6])
            if image_idx not in predictions:
                predictions[image_idx] = {
                    'dts': [], 'labels': [], 'scores': []}
                ids.append(image_idx)
            predictions[image_idx]['dts'].append(box)
            predictions[image_idx]['labels'].append(label)
            predictions[image_idx]['scores'].append(score)
        for id in ids:
            preds.append({'boxes': torch.tensor(predictions[id]['dts']), 'labels': torch.tensor(
                predictions[id]['labels']), 'scores': torch.tensor(predictions[id]['scores'])})
            _, _, _, _, _, gt_boxes = val_set.get_item(id)
            targets.append(
                {'boxes': gt_boxes[:, :4], 'labels': gt_boxes[:, 4].to(dtype=torch.int32)})
    metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        backend='faster_coco_eval')
    metric.update(preds, targets)
    metrics = metric.compute()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(metrics)


if __name__ == "__main__":
    main()
