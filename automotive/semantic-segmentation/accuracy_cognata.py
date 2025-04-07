"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""
import argparse
import json
import os
import numpy as np
from cognata import Cognata, prepare_cognata, train_val_split
import torch
import cognata_scenarios
from utils import StreamSegMetrics, read_dataset_csv
from utils import ext_transforms as et
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
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(2160,3840),
        help="image size as two integers: width and height")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    num_classes = 19
    metrics = StreamSegMetrics(num_classes)
    metrics.reset()
    files = read_dataset_csv("val_set.csv")
    files = [{'img': os.path.join(args.dataset_path, f['img']), 'label': os.path.join(args.dataset_path, f['label'])} for f in files]
    image_size = args.image_size
    val_transform = et.ExtCompose([
        et.ExtResize((image_size[0], image_size[1])),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_loader = Cognata(files, transform=val_transform)
    for j in results:
        idx = j['qsl_idx']
        # de-dupe in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)
        _, target = val_loader.get_item(idx)
        # reconstruct from mlperf accuracy log
        # what is written by the benchmark is an array of float32's:
        # id, box[0], box[1], box[2], box[3], score, detection_class
        # note that id is a index into instances_val2017.json, not the actual
        # image_id
        prediction = np.frombuffer(bytes.fromhex(j['data']), np.float32).astype(int)
        metrics.update(target.cpu().to(dtype=torch.int32).numpy().reshape(1, image_size[0], image_size[1]), prediction.reshape(1, image_size[0], image_size[1]))
    
    score = metrics.get_results()
    print(metrics.to_str(score))


if __name__ == "__main__":
    main()
