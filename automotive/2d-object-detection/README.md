# ABTF 2D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| SSD-ResNet50 | 0.7146 mAP | Cognata | https://github.com/mlcommons/abtf-ssd-pytorch | fp32 |

Achieved a 99% latency of 0.862741101 seconds on an Nvidia L4 GPU.

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the SSD model checkpoints.
After downloading the datasets extract the compressed files.

## Build and run the Docker container
CPU only
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.gpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata ssd_inference
```

GPU enabled
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.gpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata ssd_inference
```
## Run the model in performance mode
Using the ONNX backend
```
python main.py --backend onnx --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --checkpoint /cognata/ssd_resnet50.onnx
```

Using PyTorch
```
python main.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --checkpoint /cognata/baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth
```
## Run the model in accuracy mode and run the accuracy checker
Add the --accuracy flag to run in accuracy mode.

```
python main.py --backend onnx --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --checkpoint /cognata/ssd_resnet50.onnx.pth --accuracy
python accuracy_cognata.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --mlperf-accuracy-file ./output/mlperf_log_accuracy.json
```
