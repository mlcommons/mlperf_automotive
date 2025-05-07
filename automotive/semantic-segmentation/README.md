# ABTF Semantic Segmentation Benchmark

This is the reference implementation for the ABTF semantic segmentation benchmark. The reference uses ONNX as a backend. A pytorch implementation is provided as well.

| model | accuracy | resolution | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| DeepLabv3+ | 0.8959 mIOU | 4MP | Cognata | https://github.com/rod409/pp/tree/main/deeplabv3plus | fp32 |
| DeepLabv3+ | 0.9242 mIOU | 8MP | Cognata | https://github.com/rod409/pp/tree/main/deeplabv3plus | fp32 |

Achieved a 99% latency of 0.655460115 and 3.825409272 seconds on an Nvidia L4 GPU at 4MP and 8MP respectively.

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the DeepLabv3+ onnx and PyTorch model checkpoints. You do not need the whole dataset to run the benchmark. Within mlc_cognata_dataset you can download the model checkpoints and val_seg folder. val_seg contains the preprocessed data. There are two onnx checkpoints along with one pytorch checkpoint. The onnx checkpoints are labeled deeplabv3+_8mp.onnx and deeplabv3+_dynamic.onnx. Dynamic refers to dynamic input resolutions so that will work on different image resolutions. 8mp is for 8MP images. The pytorch version is latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth that will work on different image resolutions. 

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
python main.py --backend onnx --checkpoint /cognata/deeplabv3+_8mp.onnx --dataset-path /cognata/val_seg --dataset cognata
```

Using PyTorch
```
python main.py --checkpoint /cognata/latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth --dataset-path /cognata/val_seg --dataset cognata 
```

## Run the model in accuracy mode and run the accuracy checker
Add the --accuracy flag to run in accuracy mode.
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+_8mp.onnx --dataset-path /cognata/val_seg --dataset cognata --accuracy
```
Run the accuracy checker
```
python accuracy_cognata.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --dataset-path /cognata/val_seg 
```

## Preprocessing data
If you wanted to preprocess the dataset yourself you can use preprocess.py within the docker container. You will need to download the entire cognata dataset and extract the compressed files first. Then run the preprocessing script.

In addition to Cognata, there is segmentation ground truth data for the semantic segmentation labels. When you download the dataset all files are compressed. You can extract the entire dataset using the following commands. This will extract both the dataset and preprocessed labels.
```
cd <your path to cognata>
for f in *.tar.gz; do tar -xzvf "$f"; done

Run the preprocessing script.
```
python preprocess.py --dataset-root /cognata/ --workers <num of processes> --output /cognata/val_seg
```

You can add the --image-size flag for different resolutions. In the accuracy checker you will need to include --image-size with them same dimensions used during preprocessing.
