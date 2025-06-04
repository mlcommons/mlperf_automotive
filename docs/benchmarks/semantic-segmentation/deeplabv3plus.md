---
hide:
  - toc
---


# Semantic Segmentation using DeepLabv3+


=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "deeplabv3plus", "reference", devices = ["CPU", "GPU"]) }}

<!-- # DeepLabv3+ Semantic Segmentation

## Overview
DeepLabv3+ is used for semantic segmentation in the MLPerf Automotive benchmark suite. This implementation provides both ONNX and PyTorch backends.

| Metric | Value |
| ---- | ---- |
| Model | DeepLabv3+ |
| Accuracy | 0.924355 mIOU |
| Resolution | 8MP |
| Dataset | Cognata |
| Model Source | [deeplabv3plus](https://github.com/rod409/pp/tree/main/deeplabv3plus) |
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
   mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_onnx --outdirname=<path_to_download>
   
   # PyTorch version
   mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
   ```

2. Download the dataset:
   ```bash
   # Preprocessed validation data
   mlcr get,preprocessed,dataset,cognata,_mlc,_segmentation,_validation --outdirname=<path_to_download>
   
   # Preprocessed calibration data
   mlcr get,preprocessed,dataset,cognata,_mlc,_segmentation,_calibration --outdirname=<path_to_download>
   
   # Raw dataset (optional)
   mlcr get,raw,dataset,cognata,_mlc,_rclone --outdirname=<path_to_download>
   ```

### Running the Benchmark

#### Using MLCFlow (Recommended)

1. CPU Execution:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_full --model=deeplabv3plus --docker --quiet \
       --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes \
       --device=cpu --implementation=reference --framework=onnxruntime \
       --scenario=SingleStream
   ```

2. GPU Execution:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_full --model=deeplabv3plus --docker --quiet \
       --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes \
       --device=cuda --implementation=reference --framework=pytorch \
       --scenario=SingleStream
   ```

#### Performance Mode

1. Using ONNX:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=deeplabv3plus \
       --quiet --device=cpu --implementation=reference \
       --framework=onnxruntime --scenario=SingleStream
   ```

2. Using PyTorch:
   ```bash
   mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=deeplabv3plus \
       --quiet --device=cpu --implementation=reference \
       --framework=pytorch --scenario=SingleStream
   ```

   - Use `--device=gpu` to run on GPU

#### Accuracy Mode

```bash
mlcr run-abtf-inference,reference,_v0.5,_accuracy-only --model=deeplabv3plus \
    --quiet --device=cpu --implementation=reference \
    --framework=onnxruntime --scenario=SingleStream
```

- Use `--framework=pytorch` to run using the PyTorch framework

### Evaluating Accuracy

```bash
mlcr process,mlperf,accuracy,_cognata_deeplabv3plus \
    --result_dir=<Path to benchmark results>
```

## Advanced Usage

### Native Execution (Without MLCFlow)

1. Clone and setup:
   ```bash
   git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
   cd mlperf_automotive/automotive/semantic-segmentation
   ```

2. Build Docker container:
   ```bash
   # CPU version
   docker build -t deeplab_inference -f dockerfile.cpu .
   
   # GPU version
   docker build -t deeplab_inference -f dockerfile.gpu .
   ```

3. Run container:
   ```bash
   docker run -it \
       -v ./mlperf_automotive:/mlperf_automotive \
       -v <path to cognata>:/cognata deeplab_inference
   ```

4. Execute benchmark:
   ```bash
   # Using ONNX backend
   python main.py --backend onnx \
       --checkpoint /cognata/deeplabv3+_8mp.onnx \
       --dataset-path /cognata/val_seg \
       --dataset cognata

   # Using PyTorch backend
   python main.py \
       --checkpoint /cognata/latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth \
       --dataset-path /cognata/val_seg \
       --dataset cognata

   # Accuracy mode (add --accuracy flag)
   python main.py --backend onnx \
       --checkpoint /cognata/deeplabv3+_8mp.onnx \
       --dataset-path /cognata/val_seg \
       --dataset cognata --accuracy
   ```

### Data Preprocessing

To preprocess the dataset manually:
```bash
python preprocess.py \
    --dataset-root /cognata/ \
    --workers <num of processes> \
    --output /cognata/val_seg
```

You can add the `--image-size` flag for different resolutions. In the accuracy checker, you will need to include `--image-size` with the same dimensions used during preprocessing.

## Model Variants

The implementation provides two ONNX model variants:
1. `deeplabv3+_8mp.onnx` - Optimized for 8MP images
2. `deeplabv3+_dynamic.onnx` - Supports dynamic input resolutions

## Important Notes

1. The benchmark workflow is tested only for SingleStream runs.
2. For valid benchmark runs, use `--execution_mode=valid`. Default is test mode.
3. MLCFlow handles dataset and model downloads automatically when using Docker.
4. RClone login is required with the email account having model file access.
5. When preprocessing data, ensure to use consistent image sizes between preprocessing and accuracy checking.  -->