import argparse
import os
import pickle
import torch
from mmcv import Config
from mmcv.parallel import collate, DataContainer
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess NuScenes dataset for offline LoadGen execution")
    parser.add_argument("--config", help="Path to model config file", required=True)
    parser.add_argument("--output-dir", help="Directory to save preprocessed .pkl files", default="preprocessed_data")
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
        
        # Collate individual item
        collated_item = collate([item], samples_per_gpu=1)
        
        # Format shapes and strip containers entirely
        formatted_item = unwrap_and_format(collated_item)
        
        filepath = os.path.join(args.output_dir, f"{i}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(formatted_item, f)
            
        if (i + 1) % 50 == 0:
            print(f" Processed {i + 1}/{num_items} items")
            
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()