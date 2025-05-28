# ABTF 3D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend.

This model requires a 99.9% latency target and a 99% accuracy constraint of the reference model.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| BEVFormer (tiny) | 0.2683556 mAP/0.37884288 NDS | NuScenes | https://github.com/rod409/BEVFormer | fp32 |

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site]()(TBD) for an automated way to run this benchmark across different available implementations and perform an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

**Important notes when using MLCFlow**

- Currently, the benchmark workflow is tested only for SingleStream-CPU runs.
- While not mandatory, it is recommended to follow MLCFlow commands throughout for a seamless experience with MLCFlow automation.
- If you encounter any issues with automation, please feel free to raise an issue in the [mlperf-automations](https://github.com/mlcommons/mlperf-automations/issues) repository.
- The dataset and model downloads based on the framework, as well as attaching to the docker container, will be automatically handled by MLCFlow if you are building the docker through the MLCFlow command in the [Build and run the Docker container](#build-and-run-the-docker-container) section.
- The email account that has access to the model files should be used to login when prompted by RClone for dataset and model downloads.

## Downloading the dataset and model checkpoints

Contact [MLCommons](https://docs.google.com/forms/d/e/1FAIpQLSdUsbqaGcoIAxoNVrxpnkUKT03S1GbbPcUIAP3hKOeV7BCgKQ/viewform) support for accessing the NuScenes dataset. This includes preprocessed data as well as the model checkpoints. The preprocessed data is located in nuscenes_data/preprocessed/val_3d.tar.gz and is used for the model during performance and accuracy mode. nuscenes_data/nuscenes_min.tar.gz contains minimal data needed to run the accuracy checker. The nuscenes folder contains the entire dataset and is not needed to run the benchmark. The model checkpoints are located in model_checkpoint_beformer. After downloading and extracting all compressed files, you should have a directory structure that looks like the following:

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

### Download model through MLCFlow Automation

**ONNX**
```
mlcr get,ml-model,bevformer,_mlc,_rclone,_onnx --outdirname=<path_to_download>
```

**PyTorch**
```
mlcr get,ml-model,bevformer,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
```

### Download dataset through MLCFlow Automation

**Preprocessed Validation**
```
mlcr get,preprocessed,dataset,nuscenes,_mlc,_validation --outdirname=<path_to_download>
```

**Preprocessed Calibration**
```
mlcr get,preprocessed,dataset,nuscenes,_mlc,_calibration --outdirname=<path_to_download>
```

**Unprocessed**
```
mlcr get,dataset,nuscenes,_mlc,_rclone --outdirname=<path_to_download>
```

## Build and run the Docker container

### CPU only

**Using MLCFlow Docker**

```
mlcr run-abtf-inference,reference,_v0.5,_full --model=bevformer --docker --quiet --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream
```

**Using Native approach**

```
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/camera-3d-detection
docker build -t bevformer_inference -f dockerfile.cpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to nuscenes dataset>:/nuscenes_data --rm bevformer_inference
```

## Run the model in performance mode

### Using MLCFlow run command:

```
mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=bevformer --quiet --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream 
```

### Using Native run command:
```
cd /mlperf_automotive/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path /nuscenes_data/val_3d  --checkpoint /nuscenes_data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py
```

## Run in accuracy mode

**MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_accuracy-only --model=bevformer  --quiet --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream 
```

**Native run command:**
```
cd /mlperf_automotive/automotive/camera-3d-detection
python main.py --dataset nuscenes --dataset-path /nuscenes_data/val_3d --checkpoint /nuscenes_data/bevformer_tiny.onnx --config ./projects/configs/bevformer/bevformer_tiny.py --accuracy
```

> [!Note]
> If you mounted the main NuScenes directory differently than the instructions, you can add the flag --nuscenes-root and specify the location.

## Run the accuracy checker
This assumes you generated the MLPerf accuracy log in an output folder within the benchmark directory. Modify accordingly.

### Evaluate the accuracy using MLCFlow
```bash
mlcr process,mlperf,accuracy,_nuscenes --result_dir=<Path to directory where files are generated after the benchmark run>
```

### Evaluate the accuracy by directly calling the accuracy script
```
cd mlperf_automotive/automotive/camera-3d-detection
python accuracy_nuscenes_cpu.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --config projects/configs/bevformer/bevformer_tiny.py --nuscenes-dir /nuscenes_data/
```

## Preprocessing data
If you want to preprocess the dataset yourself, you can use preprocess.py within the docker container.
```
python preprocess.py --dataset-root /nuscenes_data/ --workers <num workers> --output /nuscenes_data/val_3d
```
