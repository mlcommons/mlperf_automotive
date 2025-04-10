# ABTF Semantic Segmentation Benchmark

This is the reference implementation for the ABTF semantic segmentation benchmark. The reference uses ONNX as a backend. A pytorch implementation is provided as well.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| DeepLabv3+ | TBD | Cognata | https://github.com/rod409/pp/tree/main/deeplabv3plus | fp32 |

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the DeepLabv3+ onnx and pytorch model checkpoints.
After downloading the datasets extract the compressed files.

You will need to perform some dataset preprocessing for segmentation. Follow the instructions on the [source repo](https://github.com/rod409/pp/tree/main/deeplabv3plus) to process the dataset.

> [!Note]
> Providing preprocessed data is in progress.

## Build and run the Docker container
```
git clone -b v0.5abtf git@github.com:rod409/inference.git
cd inference/automotive/semantic-segmentation
docker build -t deeplab_inference .
docker run -it -v ./inference:/inference -v <path to cognata>:/cognata deeplab_inference
```
## Run the model in performance mode
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+.onnx --dataset-path /cognata/ --dataset cognata
```
## Run the model in accuracy mode and run the accuracy checker
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+.onnx --dataset-path /cognata/ --dataset cognata --accuracy
python accuracy_cognata.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --dataset-path /cognata/
