import argparse
import os
import pickle
import torch
import pandas as pd
from mmcv import Config
from mmcv.parallel import collate, DataContainer
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
# Add this import! It triggers the @MODELS.register_module()
import projects.mmdet3d_plugin.uniad

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess NuScenes training dataset for calibration")
    parser.add_argument("--config", help="Path to model config file", required=True)
    parser.add_argument("--csv-file", help="Path to CSV containing Start and Length columns", required=True)
    parser.add_argument("--output-dir", help="Directory to save preprocessed .pkl files", default="preprocessed_calib")
    return parser.parse_args()

def unwrap_and_format(batch_data):
    """Strips DataContainers and rigidly formats shapes offline."""
    # 1. Robustly unwrap DataContainers
    def unwrap(obj):
        if type(obj).__name__ == 'DataContainer' or isinstance(obj, DataContainer):
            return unwrap(obj.data)
        elif isinstance(obj, list):
            return [unwrap(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(unwrap(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: unwrap(v) for k, v in obj.items()}
        return obj

    batch_data = unwrap(batch_data)

    # 2. Ensure 'img' is strictly formatted as a 5D batched tensor: [B, N, C, H, W]
    if 'img' in batch_data:
        img_data = batch_data['img']
        if isinstance(img_data, list) and len(img_data) > 0:
            if isinstance(img_data[0], torch.Tensor):
                batch_data['img'] = torch.stack(img_data, dim=0)
            elif isinstance(img_data[0], list) and isinstance(img_data[0][0], torch.Tensor):
                batch_data['img'] = torch.stack(img_data[0], dim=0).unsqueeze(0)
        
        if isinstance(batch_data['img'], torch.Tensor) and batch_data['img'].dim() == 4:
            batch_data['img'] = batch_data['img'].unsqueeze(0)

    # 3. Ensure 'img_metas' is flattened to a pure list of dictionaries
    if 'img_metas' in batch_data:
        metas = batch_data['img_metas']
        flat_metas = []
        def get_dicts(obj):
            if isinstance(obj, dict):
                flat_metas.append(obj)
            elif isinstance(obj, (list, tuple)):
                for i in obj: get_dicts(i)
        get_dicts(metas)
        batch_data['img_metas'] = flat_metas
        
    return batch_data

def main():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)
    
    # We build the training dataset specifically for calibration extraction
    print("Building training dataset...")
    dataset = build_dataset(cfg.data.train)
    dataset_len = len(dataset)
    
    print(f"Reading calibration sequences from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    # Ensure columns exist
    if 'Start' not in df.columns or 'Length' not in df.columns:
        raise ValueError("CSV must contain 'Start' and 'Length' columns")
        
    out_idx = 0
    for index, row in df.iterrows():
        start = int(row['Start'])
        length = int(row['Length'])
        
        print(f"Processing sequence starting at index {start} for {length} samples...")
        
        for i in range(start, start + length):
            if i >= dataset_len:
                print(f" Warning: Index {i} is out of bounds for dataset of size {dataset_len}. Skipping.")
                continue
                
            # Grab the item directly from the dataset (this runs the mmcv pipeline)
            item = dataset[i]
            
            # Collate individual item
            collated_item = collate([item], samples_per_gpu=1)
            
            # Format shapes and strip containers entirely
            formatted_item = unwrap_and_format(collated_item)
            
            filepath = os.path.join(args.output_dir, f"{out_idx}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(formatted_item, f)
                
            out_idx += 1

    print("-" * 50)
    print(f"Calibration preprocessing complete! Saved {out_idx} total items to {args.output_dir}/")

if __name__ == "__main__":
    main()