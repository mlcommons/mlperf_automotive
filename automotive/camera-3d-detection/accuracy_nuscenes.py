"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""
import argparse
import json
import numpy as np
from mmdet3d.datasets import build_dataset
# pylint: disable=missing-docstring
from mmdet3d.core import bbox3d2result
from mmcv import Config
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import torch


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
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    dataset = data_loader.dataset
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
            code_size = bboxes.shape[-1]
            img_metas = dataset[i]['img_metas']
            bboxes = img_metas[0].data['box_type_3d'](bboxes, code_size)
            result_list.append(bbox3d2result(bboxes, scores, labels))
        results.extend(result_list)
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    kwargs = {}
    eval_kwargs.update(dict(metric='bbox', **kwargs))

    print(dataset.evaluate(result_list, **eval_kwargs))


if __name__ == "__main__":
    main()
