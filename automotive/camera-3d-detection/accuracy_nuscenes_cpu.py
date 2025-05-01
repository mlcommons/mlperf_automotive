"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""
import argparse
import json
import numpy as np
# pylint: disable=missing-docstring
import torch
from eval.evaluate import NuScenesEvaluate
import nuscenes_inf
import os


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlperf-accuracy-file",
        required=True,
        help="path to mlperf_log_accuracy.json")
    parser.add_argument(
        "--nuscenes-dir",
        required=True,
        help="nuscenes dataset directory")
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
    parser.add_argument(
        "--config",
        required=True,
        help="bevformer config file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    predictions = {}
    ids = []
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

        for i in range(0, len(data), 12):
            box = [float(x) for x in data[i:i + 9]]
            score = float(data[i + 9])
            label = int(data[i + 10])
            id = int(data[i + 11])
            if id not in predictions:
                predictions[id] = {
                    'bboxes': [], 'labels': [], 'scores': []}
                ids.append(id)
            predictions[id]['bboxes'].append(box)
            predictions[id]['labels'].append(label)
            predictions[id]['scores'].append(score)

    sorted_predictions = []
    for i in range(len(predictions)):
        sorted_predictions.append([torch.tensor(predictions[i]['bboxes']), torch.tensor(
            predictions[i]['scores']), torch.tensor(predictions[i]['labels'])])
    result_list = []
    for i in range(len(sorted_predictions)):
        for bboxes, scores, labels in [sorted_predictions[i]]:
            result_dict = dict(
                boxes_3d=bboxes.to('cpu'),
                scores_3d=scores.to('cpu'),
                labels_3d=labels.to('cpu'))
            result_list.append(result_dict)
        results.extend(result_list)

    nusc_data = nuscenes_inf.Nuscenes(cfg=None, dataset_path=args.nuscenes_dir)
    nusc_eval = NuScenesEvaluate(
        data_infos=nusc_data.data_infos,
        data_root=os.path.join(
            args.nuscenes_dir,
            'nuscenes'))
    print(nusc_eval.evaluate(results=result_list))


if __name__ == "__main__":
    main()
