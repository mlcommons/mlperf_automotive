# ABTF 3D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| BEVFormer (tiny) | 0.2694 mAP/0.3788 NDS | Nuscenes | https://github.com/rod409/BEVFormer | fp32 |

## Downloading the dataset and model checkpoints

Contact [MLCommons](https://docs.google.com/forms/d/e/1FAIpQLSdUsbqaGcoIAxoNVrxpnkUKT03S1GbbPcUIAP3hKOeV7BCgKQ/viewform) support for accessing the NuScenes dataset. This includes preprocessed data as well as the model checkpoint.

## Build and run the Docker container

```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/camera-3d-detection
docker build -t bevformer_inference -f dockerfile.cpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to nuscenes dataset>:/mlperf_automotive/automotive/camera-3d-detection/data --rm bevformer_inference
```

> [!Note]
> The library mmdetection3d has a CUDA dependency but is not required to run the models. The container uses an image with CUDA to compile the library. You can run `export CUDA_VISIBLE_DEVICES=""` in the container to only use the CPU.
> The Docker file dockerfile.cpu is for running the model in performance and accuracy mode and no longer requires mmdetection3d. The accuracy checker still has the mmdetection3d dependency and will require a separate docker build for now.


## Run the model in performance mode
```
cd /mlperf_automotive/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path ./data --checkpoint ./data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py --scene-file ./data/scene_lengths.pkl
```

## Run in accuracy model and accuracy checker
```
cd /mlperf_automotive/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path ./data --checkpoint ./data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py --scene-file ./data/scene_lengths.pkl --accuracy
```

```
cd mlperf_automotive/automotive/camera-3d-detection
docker build -t bevformer_accuracy -f dockerfile .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to nuscenes dataset>:/mlperf_automotive/automotive/camera-3d-detection/data --rm bevformer_accuracy
python accuracy_nuscenes.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --config projects/configs/bevformer/bevformer_tiny.py --nuscenes-dir ./data
```
