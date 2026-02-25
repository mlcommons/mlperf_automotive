import argparse
import json
import pickle
import numpy as np
import torch
from dataset import Nuscenes
from mmcv.parallel import DataContainer

def get_args():
    parser = argparse.ArgumentParser(description="UniAD Accuracy Checker (Planning L2 Metric)")
    parser.add_argument("--log-file", type=str, default="results/mlperf_log_accuracy.json", help="Path to MLPerf accuracy log")
    parser.add_argument("--dataset-path", help="Path to NuScenes dataset", required=True)
    parser.add_argument("--config", help="Path to model config file", required=True)
    parser.add_argument("--length", type=int, default=100, help="Benchmark dataset size")
    return parser.parse_args()

def unwrap(obj):
    """Safely unwraps mmcv DataContainers if present."""
    if isinstance(obj, DataContainer):
        return unwrap(obj.data)
    return obj

def recursive_cpu_numpy(obj):
    """Recursively converts any PyTorch tensors in a nested structure to numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, (list, tuple)):
        return [recursive_cpu_numpy(x) for x in obj]
    return obj

def to_numpy_coords(traj):
    """Converts raw trajectory data (list/tensor) to a clean numpy array and slices X,Y coordinates."""
    if traj is None:
        return None
    
    traj = unwrap(traj)
    
    # Safely convert any nested GPU tensors to CPU numpy arrays before stacking
    traj = recursive_cpu_numpy(traj)
    traj = np.array(traj)
        
    traj = np.squeeze(traj)
    
    # Ensure the array actually has coordinate dimensions
    if len(traj.shape) == 0:
        return None
        
    # We only want the first 2 coordinates (X, Y) to calculate L2 distance
    if len(traj.shape) >= 1 and traj.shape[-1] >= 2:
        traj = traj[..., :2]
    else:
        return None
        
    # If we accidentally grab a batch/multi-agent array like (6, 16, 2), 
    # force it down to 2D by taking the first element of extra dimensions.
    # The ego trajectory must strictly be 2D: (timesteps, 2)
    while len(traj.shape) > 2:
        traj = traj[0]
        
    return traj

def main():
    args = get_args()
    
    print("Initializing Nuscenes Dataset for Ground Truth extraction...")
    dataset = Nuscenes(dataset_path=args.dataset_path, config=args.config, length=args.length)
    
    print(f"Parsing accuracy log: {args.log_file}")
    try:
        with open(args.log_file, "r") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.log_file} not found. Did you run main.py with --accuracy?")
        return
        
    l2_errors = []
    
    for entry in log_data:
        qsl_idx = entry["qsl_idx"]
        data_hex = entry["data"]
        
        # 1. Unpickle Prediction
        prediction = pickle.loads(bytes.fromhex(data_hex))
        prediction = unwrap(prediction)
        
        # 2. Extract Prediction & GT Trajectories
        pred_traj_raw = None
        gt_traj_raw = None
        
        if 'planning' in prediction:
            plan_data = unwrap(prediction['planning'])
            # Navigate the exact UniAD 'planning' dictionary structure
            if isinstance(plan_data, dict):
                # --- PREDICTION ---
                if 'result_planning' in plan_data and isinstance(plan_data['result_planning'], dict):
                    res_plan = plan_data['result_planning']
                    # 'sdc_traj' (Self-Driving Car trajectory) is our target
                    if 'sdc_traj' in res_plan:
                        pred_traj_raw = res_plan['sdc_traj']
                    elif 'sdc_traj_all' in res_plan:
                        pred_traj_raw = res_plan['sdc_traj_all']
                elif 'traj' in plan_data:
                    pred_traj_raw = plan_data['traj']
                elif 'planning_traj' in plan_data:
                    pred_traj_raw = plan_data['planning_traj']
                    
                # --- GROUND TRUTH ---
                # UniAD conveniently packages the GT ego trajectory inside the prediction dict!
                if 'planning_gt' in plan_data and isinstance(plan_data['planning_gt'], dict):
                    plan_gt = plan_data['planning_gt']
                    if 'sdc_planning' in plan_gt:
                        gt_traj_raw = plan_gt['sdc_planning']
            else:
                # 'planning' might just be the trajectory tensor itself
                pred_traj_raw = plan_data
                
        pred_traj = to_numpy_coords(pred_traj_raw)
        gt_traj = to_numpy_coords(gt_traj_raw)
        
        # 3. Handle missing data cleanly
        if pred_traj is None or gt_traj is None:
            missing = []
            if pred_traj is None: missing.append("PRED")
            if gt_traj is None: missing.append("GT")
            print(f"Warning: Missing {' and '.join(missing)} trajectory for QSL Index {qsl_idx}. Skipping...")
            continue
            
        # 4. Compute L2 Distance (Euclidean)
        # Ensure temporal sequence lengths match before calculating (e.g. 6 frames)
        min_len = min(pred_traj.shape[0], gt_traj.shape[0])
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]
        
        diff = pred_traj - gt_traj
        l2_dist = np.linalg.norm(diff, axis=-1)
        l2_errors.append(l2_dist)
        
    if len(l2_errors) == 0:
        print("\nNo valid trajectories could be evaluated.")
        return
        
    # Average the errors across all samples
    l2_errors = np.array(l2_errors)
    mean_l2_per_step = np.mean(l2_errors, axis=0)
    overall_mean_l2 = np.mean(mean_l2_per_step)
    
    # 5. Print Metrics
    print("\n" + "="*50)
    print(" UniAD Planning Accuracy Results (L2 Metric)")
    print("="*50)
    
    for i, val in enumerate(mean_l2_per_step):
        # Typical autonomous driving benchmarks evaluate at 2Hz (0.5s per step)
        time_step = (i + 1) * 0.5 
        print(f" L2 Error @ {time_step:.1f}s: \t {val:.4f} m")
        
    print("-" * 50)
    print(f" Average L2 Error: \t {overall_mean_l2:.4f} m")
    print("="*50)

if __name__ == "__main__":
    main()