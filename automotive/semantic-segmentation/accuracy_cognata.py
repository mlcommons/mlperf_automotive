"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""
import argparse
import ijson
import os
import numpy as np
from cognata import Cognata
import torch
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
    parser.add_argument("--config", help="config file")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(2160, 3840),
        help="image size as two integers: width and height")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    seen = set()
    num_classes = 19
    metrics = StreamSegMetrics(num_classes)
    metrics.reset()
    files = read_dataset_csv(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            "val_set.csv"))
    image_size = args.image_size
    val_loader = Cognata(args.dataset_path, length=len(files))
    with open(args.mlperf_accuracy_file, "r") as f:
        for j in ijson.items(f, 'item'):
            idx = j['qsl_idx']
            # de-dupe in case loadgen sends the same image multiple times
            if idx in seen:
                continue
            seen.add(idx)
            item = val_loader.load_item(idx)

            prediction = np.frombuffer(
                bytes.fromhex(
                    j['data']),
                np.uint8)
            metrics.update(
                item['label'].astype(np.uint8).reshape(
                    1, image_size[0], image_size[1]), prediction.reshape(
                    1, image_size[0], image_size[1]))

    score = metrics.get_results()
    print(metrics.to_str(score))


if __name__ == "__main__":
    main()
