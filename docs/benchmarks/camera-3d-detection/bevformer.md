---
hide:
  - toc
---


# 3D Object Detection using BEVFormer


=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "bevformer", "reference", devices = ["CPU"]) }}


<!-- # BEVFormer Camera-Based 3D Object Detection

## Overview
BEVFormer (tiny) is used for camera-based 3D object detection in the MLPerf Automotive benchmark suite. This implementation uses ONNX as a backend.

| Metric | Value |
| ---- | ---- |
| Model | BEVFormer (tiny) |
| Accuracy | 0.2683556 mAP/0.37884288 NDS |
| Dataset | NuScenes |
| Model Source | [BEVFormer](https://github.com/rod409/BEVFormer) |
| Precision | fp32 |
| Latency Target | 99.9% |
| Accuracy Constraint | 99% |

## Quick Start Guide

### Prerequisites
1. MLCommons membership and access to NuScenes dataset
2. Python environment with MLCFlow installed (`pip install mlc-scripts`)
3. Docker for containerized execution

### Dataset and Model Setup

1. Download the model:
   ```bash
   # ONNX version
   mlcr get,ml-model,bevformer,_mlc,_rclone,_onnx --outdirname=<path_to_download>
   
   # PyTorch version
   mlcr get,ml-model,bevformer,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
   ```

2. Download the dataset:
   ```bash
   # Preprocessed validation data
   mlcr get,preprocessed,dataset,nuscenes,_mlc,_validation --outdirname=<path_to_download>
   
   # Preprocessed calibration data
   mlcr get,preprocessed,dataset,nuscenes,_mlc,_calibration --outdirname=<path_to_download>
   
   # Raw dataset (optional)
   mlcr get,dataset,nuscenes,_mlc,_rclone --outdirname=<path_to_download>
   ```

### Running the Benchmark

#### Using MLCFlow (Recommended)

1. CPU Execution:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_full --model=bevformer --docker --quiet \
       --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes \
       --device=cpu --implementation=reference --framework=onnxruntime \
       --scenario=SingleStream
   ```

#### Performance Mode

```bash
mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=bevformer \
    --quiet --device=cpu --implementation=reference \
    --framework=onnxruntime --scenario=SingleStream
```

- Use `--performance_sample_count` to adjust the performance sample count value (default: 1024)

#### Accuracy Mode

```bash
mlcr run-abtf-inference,reference,_v0.5,_accuracy-only --model=bevformer \
    --quiet --device=cpu --implementation=reference \
    --framework=onnxruntime --scenario=SingleStream
```

### Evaluating Accuracy

```bash
mlcr process,mlperf,accuracy,_nuscenes \
    --result_dir=<Path to benchmark results>
```

## Advanced Usage

### Native Execution (Without MLCFlow)

1. Clone and setup:
   ```bash
   git clone git@github.com:mlcommons/mlperf_automotive.git
   cd mlperf_automotive/automotive/camera-3d-detection
   ```

2. Build Docker container:
   ```bash
   docker build -t bevformer_inference -f dockerfile.cpu .
   ```

3. Run container:
   ```bash
   docker run -it \
       -v ./mlperf_automotive:/mlperf_automotive \
       -v <path to nuscenes dataset>:/nuscenes_data --rm bevformer_inference
   ```

4. Execute benchmark:
   ```bash
   # Performance mode
   cd /mlperf_automotive/automotive/camera-3d-detection
   python main.py --dataset nuscenes \
       --dataset-path /nuscenes_data/val_3d \
       --checkpoint /nuscenes_data/bevformer_tiny.onnx \
       --config ./projects/configs/bevformer/bevformer_tiny.py

   # Accuracy mode
   python main.py --dataset nuscenes \
       --dataset-path /nuscenes_data/val_3d \
       --checkpoint /nuscenes_data/bevformer_tiny.onnx \
       --config ./projects/configs/bevformer/bevformer_tiny.py --accuracy
   ```

### Data Preprocessing

To preprocess the dataset manually:
```bash
python preprocess.py \
    --dataset-root /nuscenes_data/ \
    --workers <num workers> \
    --output /nuscenes_data/val_3d
```

## Important Notes

1. The benchmark workflow is tested only for SingleStream runs.
2. For valid benchmark runs, use `--execution_mode=valid`. Default is test mode.
3. MLCFlow handles dataset and model downloads automatically when using Docker.
4. RClone login is required with the email account having model file access.
5. If you mounted the main NuScenes directory differently than the instructions, you can add the flag `--nuscenes-root` to specify the location.  -->