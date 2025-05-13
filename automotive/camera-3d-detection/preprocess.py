import argparse
import sys
import os
from tools import nuscenes_raw
import pickle
from tqdm import tqdm
from multiprocessing import Pool


class Process():
    def __init__(self, dataset_root, output, prefix='val'):
        self.dataset_root = dataset_root
        self.ds = nuscenes_raw.Nuscenes(self.dataset_root)
        self.preprocessed_directory = output
        self.prefix = prefix
        os.makedirs(self.preprocessed_directory, exist_ok=True)

    def process_item(self, i):
        info = self.ds.get_item(i)
        output_file = os.path.join(
            self.preprocessed_directory,
            f"{self.prefix}_{i}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(info, f)

    def num_items(self):
        return len(self.ds)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess semantic segmentation dataset.")
    parser.add_argument('--dataset-root', type=str, required=True,
                        help="Path to the root directory of the dataset.")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of workers to use for preprocessing.")
    parser.add_argument('--output', type=str, required=True,
                        help="output directory for pkl files.")
    parser.add_argument('--calibration-set', action='store_true',
                        help="Flag to indicate if the calibration set is being processed.")

    args = parser.parse_args()

    dataset_root = args.dataset_root
    # Add your preprocessing logic here

    if args.calibration_set:
        prefix = 'calib'
    else:
        prefix = 'val'
    proc = Process(
        dataset_root,
        args.output,
        prefix)
    with Pool(args.workers) as pool:
        list(tqdm(pool.imap(
            proc.process_item,
            range(proc.num_items())),
            desc="Preprocessing dataset",
            total=proc.num_items()))


if __name__ == "__main__":
    main()
