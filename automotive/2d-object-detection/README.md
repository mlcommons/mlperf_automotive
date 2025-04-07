# ABTF 2D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| SSD-ResNet50 | 0.7146 mAP | Cognata | https://github.com/mlcommons/abtf-ssd-pytorch | fp32 |

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the SSD model checkpoints.
After downloading the datasets extract the compressed files.

## Build and run the Docker container
```
git clone -b v0.5abtf git@github.com:rod409/inference.git
cd inference/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.gpu .
docker run -it -v ./inference:/inference -v <path to cognata>:/cognata ssd_inference
```
## Run the model in performance mode
```
python main.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --checkpoint /cognata/baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth
```
## Run the model in accuracy mode and run the accuracy checker
```
python main.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --checkpoint /cognata/baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth --accuracy
python accuracy_cognata.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/ --mlperf-accuracy-file ./output/mlperf_log_accuracy.json
```
