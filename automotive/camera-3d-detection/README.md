# ABTF 3D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| BEVFormer (tiny) | 0.2694 mAP/0.3788 NDS | Nuscenes | https://github.com/rod409/BEVFormer | fp32 |

## Downloading the dataset and model checkpoints

Contact [MLCommons](https://docs.google.com/forms/d/e/1FAIpQLSdUsbqaGcoIAxoNVrxpnkUKT03S1GbbPcUIAP3hKOeV7BCgKQ/viewform) support for accessing the NuScenes dataset. This includes preprocessed data as well as the model checkpoint.

## Build and run the Docker container

```
git clone -b v0.5abtf git@github.com:rod409/inference.git
cd inference/automotive/camera-3d-detection
docker build -t bevformer_inference -f dockerfile.gpu .
docker run -it -v ./inference:/inference -v <path to nuscenes dataset>:/inference/automotive/camera-3d-detection/data -v <path to nuscenes dataset>:/inference/automotive/camera-3d-detection/output/data --rm bevformer_inference
```

## Run the model in performance mode
```
cd /inference/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path ./output/data --checkpoint ./data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py --scene-file ./data/scene_lengths.pkl
```
The dataset being mounted twice is intentional due to some hard coded paths in the original BEVFormer code. This is the solution for now until we push a fix.

## Run in accuracy model and accuracy checker
```
cd /inference/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path ./output/data --checkpoint ./output/data/bevformer_tiny.onnx --config projects/configs/bevformer/bevformer_tiny.py --scene-file ./output/data/scene_lengths.pkl --accuracy
python accuracy_nuscenes.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --config projects/configs/bevformer/bevformer_tiny.py --nuscenes-dir ./output/data
```
