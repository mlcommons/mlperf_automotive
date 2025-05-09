import argparse
import importlib
from utils import cognata_raw
import cognata_scenarios
import csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="path to the dataset")
    parser.add_argument(
        "--calibration-size",
        default=200,
        type=int,
        help="number of items in calibration dataset",
    )
    parser.add_argument(
        "--validation-size",
        default=6000,
        type=int,
        help="number of items in validation dataset",
    )
    args = parser.parse_args()
    return args


def write_data_to_csv(file_path, files):
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['img', 'label'])
        for file in files:
            writer.writerow([file['img'], file['label']])


def main():
    args = get_args()
    files = cognata_raw.prepare_cognata(
        args.dataset_path, cognata_scenarios.folders, cognata_scenarios.cameras)
    files = cognata_raw.train_val_split(files, args.calibration_size)
    for file_set in files.values():
        for file in file_set:
            file['img'] = file['img'].replace(args.dataset_path, '')
            file['label'] = file['label'].replace(args.dataset_path, '')
    write_data_to_csv('val_set.csv', files['val'][:args.validation_size])
    write_data_to_csv('calibration_set.csv', files['calibration'])


main()
