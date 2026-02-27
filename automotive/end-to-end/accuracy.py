import argparse
import json
import pickle
import numpy as np
import os
import torch
import torch.nn.functional as F

from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import PlanningMetric

def get_args():
    parser = argparse.ArgumentParser(description="UniAD Accuracy Checker (Planning Metric)")
    parser.add_argument("--log-file", type=str, default="results/mlperf_log_accuracy.json", help="Path to MLPerf accuracy log")
    parser.add_argument("--dataset-path", help="Path to preprocessed NuScenes dataset (.pkl files)", required=True)
    return parser.parse_args()

def unwrap(obj):
    if isinstance(obj, dict):
        return {k: unwrap(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [unwrap(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(unwrap(x) for x in obj)
    return obj

def to_tensor(obj):
    """Recursively converts structures to CPU PyTorch tensors for native torchmetrics."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).cpu()
    elif isinstance(obj, (list, tuple)):
        return [to_tensor(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_tensor(v) for k, v in obj.items()}
    return obj

def main():
    args = get_args()
    
    print(f"Parsing accuracy log: {args.log_file}")
    try:
        with open(args.log_file, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.log_file} not found. Did you run main.py with --accuracy?")
        return
        
    raw_outputs_dict = {}
    for entry in log_data:
        qsl_idx = entry["qsl_idx"]
        data_hex = entry["data"]
        prediction_orig = pickle.loads(bytes.fromhex(data_hex))
        raw_outputs_dict[qsl_idx] = prediction_orig
        
    ordered_keys = sorted(raw_outputs_dict.keys())
    
    planning_metrics = PlanningMetric(conf={
        'xbound': [-12.5, 12.5, 0.5],
        'ybound': [-12.5, 12.5, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
    })

    print("\n" + "="*50)
    print(" UniAD Planning Accuracy Results")
    print("="*50)

    for qsl_idx in ordered_keys:
        prediction = unwrap(raw_outputs_dict[qsl_idx])
        
        if 'planning' not in prediction:
            continue
            
        try:
            # 1. Load Ground Truth data from the preprocessed offline dataset
            filepath = os.path.join(args.dataset_path, f"{qsl_idx}.pkl")
            with open(filepath, 'rb') as f:
                gt_data = pickle.load(f)
            
            # 2. Extract Model predictions
            res_plan = prediction['planning']['result_planning']
            pred_sdc_traj = to_tensor(res_plan['sdc_traj'])
            
            # 3. Extract Ground Truth directly from the dataset item
            sdc_planning = to_tensor(gt_data['sdc_planning'])
            sdc_planning_mask = to_tensor(gt_data['sdc_planning_mask'])
            

            if 'segmentation' in gt_data:
                segmentation = to_tensor(gt_data['segmentation'])
            elif 'gt_segmentation' in gt_data:
                segmentation = to_tensor(gt_data['gt_segmentation'])
            else:
                plan_gt = prediction['planning']['planning_gt']
                segmentation = to_tensor(plan_gt['segmentation'])
            
            # Apply the exact shape slices from the test.py script
            pred_sliced = pred_sdc_traj[:, :6, :2]
            gt_sliced = sdc_planning[0][0, :, :6, :2]
            mask_sliced = sdc_planning_mask[0][0, :, :6, :2]
            seg_sliced = segmentation[0][:, [1,2,3,4,5,6]]
            
            planning_metrics(pred_sliced, gt_sliced, mask_sliced, seg_sliced)
            
        except Exception as e:
            print(f"Warning: Failed to process QSL Index {qsl_idx}. Error: {e}")
            continue
            
    # Compute and print the finalized metrics
    try:
        planning_results = planning_metrics.compute()
        
        if 'L2' in planning_results:
            l2_val = planning_results['L2']
            if isinstance(l2_val, torch.Tensor):
                l2_list = l2_val.tolist()
            else:
                l2_list = list(l2_val)
                
            avg_l2 = sum(l2_list) / len(l2_list)
            
            # Generate time labels (0.5s, 1.0s, ...)
            time_labels = [f"{(i+1)*0.5:.1f}s" for i in range(len(l2_list))]
            
            # Format output table
            header = "        " + "".join(f"{label:>8}" for label in time_labels)
            row = " L2:    " + "".join(f"{val:>8.4f}" for val in l2_list)
            
            print(header)
            print(row)
            print(f"Avg: {avg_l2}")
        else:
            print(" L2 metric not found in results.")
            
    except Exception as e:
        print(f"Error computing final metrics: {e}")

if __name__ == "__main__":
    main()