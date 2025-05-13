# ABTF 2D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend. This model requires a 99.9% latency target and a 99.9% accuracy constraint of the reference.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| SSD-ResNet50 | 0.7179 mAP | Cognata | https://github.com/mlcommons/abtf-ssd-pytorch | fp32 |

Achieved a 99% latency of 0.862741101 seconds on an Nvidia L4 GPU.

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the SSD model checkpoints. You do not need the entire dataset, the mlc_cognata_dataset folder, to run the benchmark. You can download the preprocessed validation data val_2d.tar.gz along with model checkpoints ssd_resent50.onnx and baseline_8MP_ss_scales_fm1_5x5_all.pth. All our under the mlc_cognata_dataset directory. You should have a cognata directory with the model checkpoints and the val_2d folder extracted directly within it.


## Build and run the Docker container


CPU only
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.cpu .
docker run -it -v <your repo path>/mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata/ ssd_inference
```

GPU enabled
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.gpu .
docker run -it -v <your repo path>/mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata/ ssd_inference
```
## Run the model in performance mode
Using the ONNX backend
```
python main.py --backend onnx --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --checkpoint /cognata/ssd_resnet50.onnx
```

Using PyTorch
```
python main.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --checkpoint /cognata/baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth
```
## Run the model in accuracy mode and run the accuracy checker
Add the --accuracy flag to run in accuracy mode.

```
python main.py --backend onnx --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --checkpoint /cognata/ssd_resnet50.onnx --accuracy
```
Run the accuracy checker. If you changed the output log location in accuracy mode the modify the file location.
```
python accuracy_cognata.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --mlperf-accuracy-file ./output/mlperf_log_accuracy.json
```

## Preprocessing data
If you wanted to preprocess the dataset yourself you can use preprocess.py within the docker container. You will need to download the entire cognata dataset and extract the compressed files first. Then run the preprocessing script.
```
python preprocess.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset-root /cognata/ --output /cognata/val_2d
```