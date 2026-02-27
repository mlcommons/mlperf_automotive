# UniAD MLPerf Automotive Benchmark

This repository contains the reference implementation for UniAD. It is based on implementations from the official UniAD implementation (https://github.com/OpenDriveLab/UniAD) and DL4AGX (https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt)

| model | accuracy | dataset | precision |
| ---- | ---- | ---- | ---- |
| UniAD (tiny) | 0.8959 L2 Avg. | NuScenes | fp32 |

## Dataset Download

Contact [MLCommons](https://docs.google.com/forms/d/e/1FAIpQLSdUsbqaGcoIAxoNVrxpnkUKT03S1GbbPcUIAP3hKOeV7BCgKQ/viewform) support for accessing the NuScenes dataset. This includes preprocessed data as well as the model checkpoint. The preprocessed data is located in nuscenes_data/preprocessed_uniad/. The model checkpoint is located in model_checkpoint_uniad

## Environment Setup

The environment, dependencies, OpenMMLab libraries, and the custom uniad_mmdet3d package are all built directly into the Docker image.


## Build the Docker Image

From the end-to-end folder of your repository, build the Docker container:

```
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/end-to-end
docker build -t uniad_mlperf -f ./docker/Dockerfile .
```
## Run the benchmark

Run the docker container. Mount the dataset and source code.
```
docker run -it --gpus 1 --shm-size=8g -v <path/to/dataset>:/workspace/end-to-end/data -v /home/rshojaei/test/mlperf_automotive/automotive/end-to-end:/workspace/end-to-end uniad_mlperf /bin/bash
```

Run the model in performance mode
```
cd /workspace/end-to-end
python main.py \
    --config ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py \
    --checkpoint <path/to/tiny_imgx0.25_e2e_ep20.pth> \
    --dataset-path <path/to/preprocessed_uniad> \
    --log-dir <path/to/store/logs>
```


Run the model in accuracy mode
```
python main.py \
    --config ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py \
    --checkpoint <path/to/tiny_imgx0.25_e2e_ep20.pth> \
    --dataset-path <path/to/preprocessed_uniad> \
    --log-dir <path/to/store/logs> \
    --accuracy
```

Run the accuracy checker
```
python accuracy.py \
    --dataset-path preprocessed_data \
    --log-file results/mlperf_log_accuracy.json
```
