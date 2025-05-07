import argparse
import sys
import os
import importlib
from utils import read_dataset_csv, generate_dboxes
from data_utils import cognata_raw
from cognata import prepare_cognata
from transform import SSDTransformer
import cognata_labels
import pickle
from tqdm import tqdm
from multiprocessing import Pool


class Process():
    def __init__(self, dataset_root, image_size, files, transformer,
                 label_map, label_info, output, prefix='val'):
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.ds = cognata_raw.Cognata(label_map=label_map,
                                      label_info=label_info,
                                      files=files,
                                      ignore_classes=[2, 25, 31],
                                      transform=transformer)
        self.preprocessed_directory = output
        self.prefix = prefix
        os.makedirs(self.preprocessed_directory, exist_ok=True)

    def process_item(self, i):
        image, idx, (height, width), boxes, labels, gt_boxes = self.ds.get_item(i)
        output_data = {
            'img': image,
            'idx': idx,
            'img_size': (
                height,
                width),
            'boxes': boxes,
            'labels': labels,
            'gt_boxes': gt_boxes}
        output_file = os.path.join(
            self.preprocessed_directory,
            f"{self.prefix}_{i}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess semantic segmentation dataset.")
    parser.add_argument('--dataset-root', type=str, required=True,
                        help="Path to the root directory of the dataset.")
    parser.add_argument('--csv-file', type=str, default="val_set.csv",
                        help="path to csv containing cognata file paths.")
    # parser.add_argument('--workers', type=int, default=4,
    #                    help="Number of workers to use for preprocessing.")
    parser.add_argument('--output', type=str, required=True,
                        help="output directory for pkl files.")
    parser.add_argument('--calibration-set', action='store_true',
                        help="Flag to indicate if the calibration set is being processed.")
    parser.add_argument(
        "--config",
        type=str,
        help="config file",
        required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    # Add your preprocessing logic here
    config = importlib.import_module('config.' + args.config)
    image_size = config.model['image_size']
    dboxes = generate_dboxes(config.model, model="ssd")
    transformer = SSDTransformer(dboxes, image_size, val=True)
    folders = config.dataset['folders']
    cameras = config.dataset['cameras']

    if config.dataset['use_label_file']:
        label_map = cognata_labels.label_map
        label_info = cognata_labels.label_info
    else:
        _, label_map, label_info = prepare_cognata(
            args.dataset_root, folders, cameras)
    files = read_dataset_csv("val_set.csv")
    files = [{'img': os.path.join(args.dataset_root, f['img']), 'ann': os.path.join(
        args.dataset_root, f['ann'])} for f in files]
    if args.calibration_set:
        prefix = 'calib'
    else:
        prefix = 'val'

    proc = Process(
        dataset_root,
        image_size,
        files,
        transformer,
        label_map,
        label_info,
        args.output,
        prefix)
    # with Pool(args.workers) as pool:
    #    list(tqdm(pool.imap(proc.process_item, range(len(files))), desc="Preprocessing dataset", total=len(files)))
    # multiprocessing is failing, possibly due to pickling issues with the transformer
    # using a single process for loop
    for i in tqdm(range(len(files)), desc="Preprocessing dataset",
                  total=len(files)):
        proc.process_item(i)


if __name__ == "__main__":
    main()
