import argparse
import os
import pickle
from mmcv import Config
from mmcv.parallel import collate
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess NuScenes dataset for offline LoadGen execution")
    parser.add_argument("--config", help="Path to model config file", required=True)
    parser.add_argument("--output-dir", help="Directory to save preprocessed .pkl files", default="preprocessed_data")
    return parser.parse_args()

def main():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)
    
    print("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    
    print("Building dataloader...")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    
    num_items = len(data_loader.dataset)
    print(f"Preprocessing {num_items} items into {args.output_dir}/ ...")
    
    for i in range(num_items):
        item = data_loader.dataset[i]
        
        # Collate individual item with BS=1 to match the expected backend input
        collated_item = collate([item], samples_per_gpu=1)
        
        filepath = os.path.join(args.output_dir, f"{i}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(collated_item, f)
            
        if (i + 1) % 50 == 0:
            print(f" Processed {i + 1}/{num_items} items")
            
    print("Preprocessing complete!")
    print(f"You can now run main.py and pass --dataset-path {args.output_dir}")

if __name__ == "__main__":
    main()