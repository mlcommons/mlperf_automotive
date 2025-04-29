# ABTF Semantic Segmentation Benchmark

This is the reference implementation for the ABTF semantic segmentation benchmark. The reference uses ONNX as a backend. A pytorch implementation is provided as well.

| model | accuracy | resolution | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| DeepLabv3+ | 0.8959 mIOU | 4MP | Cognata | https://github.com/rod409/pp/tree/main/deeplabv3plus | fp32 |
| DeepLabv3+ | 0.9242 mIOU | 8MP | Cognata | https://github.com/rod409/pp/tree/main/deeplabv3plus | fp32 |

Achieved a 99% latency of 0.655460115 and 3.825409272 seconds on an Nvidia L4 GPU at 4MP and 8MP respectively.

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the DeepLabv3+ onnx and PyTorch model checkpoints.
After downloading the datasets extract the compressed files.

In addition to Cognata, there is preprocessed data for the semantic segmentation labels. When you download the dataset all files are compressed. You can extract the entire dataset using the following commands. This will extract both the dataset and preprocessed labels.
```
cd <your path to cognata>
for f in *.tar.gz; do tar -xzvf "$f"; done
```
> [!Note]
> The instructions to process the data yourself can be found on the [source repo](https://github.com/rod409/pp/tree/main/deeplabv3plus).


## Build and run the Docker container
CPU only
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/semantic-segmentation
docker build -t deeplab_inference -f dockerfile.cpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata deeplab_inference
```

GPU enabled
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/semantic-segmentation
docker build -t deeplab_inference -f dockerfile.gpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata deeplab_inference
```
## Run the model in performance mode
Using the ONNX backend
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+.onnx --dataset-path /cognata/ --dataset cognata --image-size 1440 2560
```

Using PyTorch
```
python main.py --checkpoint /cognata/latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth --dataset-path /cognata/ --dataset cognata --image-size 1440 2560
```

## Run the model in accuracy mode and run the accuracy checker
Add the --accuracy flag to run in accuracy mode.
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+.onnx --dataset-path /cognata/ --dataset cognata --image-size 1440 2560 --accuracy
python accuracy_cognata.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --dataset-path /cognata/ --image-size 1440 2560
```

> [!Note]
> The flag --image-size needs to be the same in main.py and accuracy_cognata.py for correctness.
> Removing the --image-size flag will default to 8MP.


