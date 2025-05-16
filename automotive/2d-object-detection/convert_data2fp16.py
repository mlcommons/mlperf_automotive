import pickle
import torch
import sys
from pathlib import Path
import argparse
from cognata import Cognata
import torch
from utils import read_dataset_csv
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Convert fp32 tensors in pkl files to fp16.")
    parser.add_argument("input", help="Input .pkl file or folder containing .pkl files")
    parser.add_argument("output", help="Output .pkl file or folder for converted files")
    args = parser.parse_args()

    files = read_dataset_csv("val_set.csv")
    val_loader = Cognata(args.input, length=len(files))
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(val_loader)), desc="Converting to fp16"):
        item = val_loader.load_item(i)
        item['img'] = item['img'].to(torch.float16)
        item['boxes'] = item['boxes'].to(torch.float16)
        item['gt_boxes'] = item['gt_boxes'].to(torch.float16)
        with open(os.path.join(output_path, f'val_{i}.pkl'), 'wb') as f:
            pickle.dump(item, f)

if __name__ == "__main__":
    main()