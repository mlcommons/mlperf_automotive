import argparse
import sys
import os
from utils import ext_transforms as et, read_dataset_csv, cognata_raw
import pickle
from tqdm import tqdm
from multiprocessing import Pool


class Process():
    def __init__(self, dataset_root, image_size, files,
                 val_transform, output, prefix='val'):
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.ds = cognata_raw.Cognata(files=files, transform=val_transform)
        self.preprocessed_directory = output
        self.prefix = prefix
        os.makedirs(self.preprocessed_directory, exist_ok=True)

    def process_item(self, i):
        img, label = self.ds.get_item(i)
        output_data = {'img': img, 'label': label}
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
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(2160, 3840),
        help="image size as two integers: width and height")
    parser.add_argument('--csv-file', type=str, default="val_set.csv",
                        help="path to csv containing cognata file paths.")
    parser.add_argument('--workers', type=int, default=4,
                        help="Number of workers to use for preprocessing.")
    parser.add_argument('--output', type=str, required=True,
                        help="output directory for pkl files.")
    parser.add_argument('--calibration-set', action='store_true',
                        help="Flag to indicate if the calibration set is being processed.")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    # Add your preprocessing logic here
    image_size = args.image_size
    val_transform = et.ExtCompose([
        et.ExtResize((image_size[0], image_size[1])),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    files = read_dataset_csv(args.csv_file)
    files = [{'img': os.path.join(dataset_root, f['img']), 'label': os.path.join(
        dataset_root, f['label'])} for f in files]
    if args.calibration_set:
        prefix = 'calib'
    else:
        prefix = 'val'
    proc = Process(
        dataset_root,
        image_size,
        files,
        val_transform,
        args.output,
        prefix)
    with Pool(args.workers) as pool:
        list(
            tqdm(
                pool.imap(
                    proc.process_item,
                    range(
                        len(files))),
                desc="Preprocessing dataset",
                total=len(files)))


if __name__ == "__main__":
    main()
