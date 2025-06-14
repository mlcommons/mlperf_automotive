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
from cognata import Cognata, prepare_cognata
import cognata_labels
from transform import SSDTransformer
import importlib
import torch
from utils import generate_dboxes, read_dataset_csv
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
        "--dataset-path",
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

    seen = set()
    files = read_dataset_csv(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "val_set.csv"))
    val_set = Cognata(args.dataset_path, len(files))
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
        # box[0], box[1], box[2], box[3], detection_class, score
        data = np.frombuffer(bytes.fromhex(j['data']), np.float32)
        predictions = {}
        ids = []
        for i in range(0, len(data), 6):
            box = [float(x) for x in data[i:i + 4]]
            label = int(data[i + 4])
            score = float(data[i + 5])
            if idx not in predictions:
                predictions[idx] = {
                    'dts': [], 'labels': [], 'scores': []}
                ids.append(idx)
            predictions[idx]['dts'].append(box)
            predictions[idx]['labels'].append(label)
            predictions[idx]['scores'].append(score)
        for id in ids:
            preds.append({'boxes': torch.tensor(predictions[id]['dts']), 'labels': torch.tensor(
                predictions[id]['labels']), 'scores': torch.tensor(predictions[id]['scores'])})
            gt_boxes = torch.from_numpy(val_set.load_item(id)['gt_boxes'])
            targets.append(
                {'boxes': gt_boxes[:, :4], 'labels': gt_boxes[:, 4].to(dtype=torch.int32)})
    metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        backend='faster_coco_eval')
    metric.update(preds, targets)
    metrics = metric.compute()
    print(f"mAP: {metrics['map'].item()}")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(metrics)


if __name__ == "__main__":
    main()
