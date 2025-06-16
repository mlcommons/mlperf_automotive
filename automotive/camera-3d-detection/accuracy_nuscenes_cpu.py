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
from tools import nuscenes_raw
import os
import post_process


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
        "--config",
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
    post_proc = post_process.PostProcess(
        num_classes=10, max_num=300, pc_range=[
            -51.2, -51.2, -5.0, 51.2, 51.2, 3.0], post_center_range=[
            -61.2, -61.2, -10.0, 61.2, 61.2, 10.0], score_threshold=None)
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
        prediction = np.frombuffer(bytes.fromhex(j['data']), np.float32)
        # reformat to stacked tensors
        prediction = torch.from_numpy(
            prediction.copy().reshape(
                2, 6, 1, 900, 10))
        result = post_proc.process(prediction[0], prediction[1])[0]
        if idx not in predictions:
            ids.append(idx)
        predictions[idx] = {
            'bboxes': result[0],
            'scores': result[1],
            'labels': result[2]}

    sorted_predictions = []
    for i in range(len(predictions)):
        sorted_predictions.append([predictions[i]['bboxes'],
                                   predictions[i]['scores'], predictions[i]['labels']])
    result_list = []
    for i in range(len(sorted_predictions)):
        for bboxes, scores, labels in [sorted_predictions[i]]:
            result_dict = dict(
                boxes_3d=bboxes.to('cpu'),
                scores_3d=scores.to('cpu'),
                labels_3d=labels.to('cpu'))
            result_list.append(result_dict)
        results.extend(result_list)

    nusc_data = nuscenes_raw.Nuscenes(dataset_path=args.nuscenes_dir)
    nusc_eval = NuScenesEvaluate(
        data_infos=nusc_data.data_infos,
        data_root=os.path.join(
            args.nuscenes_dir,
            'nuscenes'))
    print(nusc_eval.evaluate(results=result_list))


if __name__ == "__main__":
    main()
