# MLPerf Inference Benchmarks

## Overview
The currently valid [MLPerf Inference Benchmarks](index_gh.md) as of MLPerf inference v5.0 round are listed below, categorized by tasks. Under each model you can find its details like the dataset used, reference accuracy, server latency constraints etc.

---

## 2D Object Detection
### [SSD-ResNet50](benchmarks/2d-object-detection/ssd.md)
- **Dataset**: Cognata
    - **Dataset Size**: TBD
    - **QSL Size**: 128 (default, minimum needed)
- **Number of Parameters**: TBD
- **FLOPs**: TBD
- **Reference Model Accuracy**: 0.7179 mAP
- **Latency Target**: 99.9%
- **Accuracy Constraint**: 99.9%
- **Framework Support**: ONNX, PyTorch
- **Submission Category**: Edge

## Camera-Based 3D Object Detection
### [BEVFormer (tiny)](benchmarks/camera-3d-detection/bevformer.md)
- **Dataset**: NuScenes
    - **Dataset Size**: TBD
    - **QSL Size**: 1024 (default), 512 (minimum needed)
- **Number of Parameters**: TBD
- **FLOPs**: TBD
- **Reference Model Accuracy**: 0.2683556 mAP / 0.37884288 NDS
- **Latency Target**: 99.9%
- **Accuracy Constraint**: 99%
- **Framework Support**: ONNX, PyTorch
- **Submission Category**: Edge

## Semantic Segmentation
### [DeepLabv3+](benchmarks/semantic-segmentation/deeplabv3plus.md)
- **Dataset**: Cognata
    - **Dataset Size**: TBD
    - **QSL Size**: 128 (default, minimum needed)
- **Number of Parameters**: TBD
- **FLOPs**: TBD
- **Reference Model Accuracy**: 0.924355 mIOU
- **Resolution**: 8MP
- **Latency Target**: 99.9%
- **Accuracy Constraint**: 99.9%
- **Framework Support**: ONNX (8MP & Dynamic), PyTorch
- **Submission Category**: Edge

---

## Submission Categories
- **Edge Category**: All benchmarks are applicable to the edge category for the automotive inference v0.5

## High Accuracy Variants
- **Benchmarks**: All the benchmarks for submission round v0.5 have high accuracy variant.
- **Requirement**: Must achieve at least 99.9% of the reference model accuracy.
