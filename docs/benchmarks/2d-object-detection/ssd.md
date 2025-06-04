---
hide:
  - toc
---


# 2D Object Detection using SSD ResNet50


=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "ssd", "reference", devices = ["CPU", "CUDA"]) }}

<!-- # SSD-ResNet50 2D Object Detection

## Overview
The SSD-ResNet50 model is used for 2D object detection in the MLPerf Automotive benchmark suite. This implementation uses ONNX as a backend and achieves a 99% latency of 0.862741101 seconds on an Nvidia L4 GPU.

| Metric | Value |
| ---- | ---- |
| Model | SSD-ResNet50 |
| Accuracy | 0.7179 mAP |
| Dataset | Cognata |
| Model Source | [abtf-ssd-pytorch](https://github.com/mlcommons/abtf-ssd-pytorch) |
| Precision | fp32 |
| Latency Target | 99.9% |
| Accuracy Constraint | 99.9% |

## Quick Start Guide

### Prerequisites
1. MLCommons membership and EULA signing for Cognata dataset access
2. Python environment with MLCFlow installed (`pip install mlc-scripts`)
3. Docker for containerized execution

### Dataset and Model Setup

1. Download the model:
   ```bash
   # ONNX version
   mlcr get,ml-model,ssd,resnet50,_mlc,_rclone,_onnx --outdirname=<path_to_download>
   
   # PyTorch version
   mlcr get,ml-model,ssd,resnet50,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
   ```

2. Download the dataset:
   ```bash
   # Preprocessed validation data
   mlcr get,preprocessed,dataset,cognata,_mlc,_2d_obj_det,_validation --outdirname=<path_to_download>
   
   # Preprocessed calibration data
   mlcr get,preprocessed,dataset,cognata,_mlc,_2d_obj_det,_calibration --outdirname=<path_to_download>
   
   # Raw dataset (optional)
   mlcr get,raw,dataset,cognata,_mlc,_rclone --outdirname=<path_to_download>
   ```

### Running the Benchmark

#### Using MLCFlow (Recommended)

1. CPU Execution:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_full --model=ssd --docker --quiet \
       --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes \
       --device=cpu --implementation=reference --framework=onnxruntime \
       --scenario=SingleStream
   ```

2. GPU Execution:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_full --model=ssd --docker --quiet \
       --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes \
       --device=cuda --implementation=reference --framework=pytorch \
       --scenario=SingleStream
   ```

#### Performance Mode

1. Using ONNX:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=ssd \
       --quiet --device=cpu --implementation=reference \
       --framework=onnxruntime --scenario=SingleStream
   ```

2. Using PyTorch:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=ssd \
       --quiet --device=cpu --implementation=reference \
       --framework=pytorch --scenario=SingleStream
   ```

   - Use `--device=gpu` for GPU execution
   - Use `--performance_sample_count` to adjust the performance sample count (default: 128)

#### Accuracy Mode

```bash
mlcr run-abtf-inference,reference,_v0.5,_accuracy-only --model=ssd \
    --quiet --device=cpu --implementation=reference \
    --framework=onnxruntime --scenario=SingleStream
```

- Use `--device=gpu` for GPU execution
- Use `--framework=pytorch` to use the PyTorch framework

### Evaluating Accuracy

```bash
mlcr process,mlperf,accuracy,_cognata_ssd \
    --result_dir=<Path to benchmark results>
```

## Advanced Usage

### Native Execution (Without MLCFlow)

1. Clone and setup:
   ```bash
   git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
   cd mlperf_automotive/automotive/2d-object-detection
   ```

2. Build Docker container:
   ```bash
   # CPU version
   docker build -t ssd_inference -f dockerfile.cpu .
   
   # GPU version
   docker build -t ssd_inference -f dockerfile.gpu .
   ```

3. Run container:
   ```bash
   docker run -it \
       -v <your repo path>/mlperf_automotive:/mlperf_automotive \
       -v <path to cognata>:/cognata/ ssd_inference
   ```

4. Execute benchmark:
   ```bash
   # Performance mode (ONNX)
   python main.py --backend onnx \
       --config baseline_8MP_ss_scales_fm1_5x5_all \
       --dataset cognata --dataset-path /cognata/val_2d \
       --checkpoint /cognata/ssd_resnet50.onnx

   # Accuracy mode
   python main.py --backend onnx \
       --config baseline_8MP_ss_scales_fm1_5x5_all \
       --dataset cognata --dataset-path /cognata/val_2d \
       --checkpoint /cognata/ssd_resnet50.onnx --accuracy
   ```

### Data Preprocessing

To preprocess the dataset manually:
```bash
python preprocess.py \
    --config baseline_8MP_ss_scales_fm1_5x5_all \
    --dataset-root /cognata/ \
    --output /cognata/val_2d
```

## Important Notes

1. The benchmark workflow is tested only for SingleStream runs.
2. For valid benchmark runs, use `--execution_mode=valid`. Default is test mode.
3. MLCFlow handles dataset and model downloads automatically when using Docker.
4. RClone login is required with the email account having model file access.  -->