# ABTF 3D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend.

This model requires a 99.9% latency target and a 99% accuracy constraint of the reference model.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| BEVFormer (tiny) | 0.2683556 mAP/0.37884288 NDS | Nuscenes | https://github.com/rod409/BEVFormer | fp32 |

## Downloading the dataset and model checkpoints

Contact [MLCommons](https://docs.google.com/forms/d/e/1FAIpQLSdUsbqaGcoIAxoNVrxpnkUKT03S1GbbPcUIAP3hKOeV7BCgKQ/viewform) support for accessing the NuScenes dataset. This includes preprocessed data as well as the model checkpoints. The preprocessed data is located in nuscenes_data/preprocessed/val_3d.tar.gz and is used for the model during performance and accuracy mode. nuscenes_data/nuscenes_min.tar.gz contains minimal data needed to run the accuracy checker. The nuscenes folder contains the entire dataset and is not needed to run the benchmark. The model checkpoints are located in model_checkpoint_beformer. After downloading and extracting all compressed files you should have a directory structure that looks like the following.

```
├── bevformer_tiny.onnx
├── nuscenes
│   ├── maps
│   ├── nuscenes_infos_temporal_val.pkl
│   └── v1.0-trainval
├── scene_lengths.pkl
├── scene_lengths.txt
├── scene_starts.pkl
├── scene_starts.txt
└── val_3d
```


## Build and run the Docker container

```
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/camera-3d-detection
docker build -t bevformer_inference -f dockerfile.cpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to nuscenes dataset>:/nuscenes_data --rm bevformer_inference
```

## Run the model in performance mode
```
cd /mlperf_automotive/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path /nuscenes_data/val_3d  --checkpoint /nuscenes_data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py
```

## Run in accuracy mode
```
cd /mlperf_automotive/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path /nuscenes_data/val_3d --checkpoint /nuscenes_data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py --accuracy
```

> [!Note]
> If you mounted the main nuscenes directory differently than the instructions you can add the flag --nuscenes-root and specify the location.

## Run the accuracy checker
This assumes you generated the mlperf accuracy log in an output folder within the benchmark directory. Modify accordingly.
```
cd mlperf_automotive/automotive/camera-3d-detection
python accuracy_nuscenes_cpu.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --config projects/configs/bevformer/bevformer_tiny.py --nuscenes-dir /nuscenes_data/
```

## Preprocessing data
If you wanted to preprocess the dataset yourself you can use preprocess.py within the docker container.
```
python preprocess.py --dataset-root /nuscenes_data/ --workers <num workers> --output /nuscenes_data/val_3d
```
