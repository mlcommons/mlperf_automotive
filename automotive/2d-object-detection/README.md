# ABTF 2D Object Detection Benchmark

This is the reference implementation for the ABTF camera-based 3D object detection benchmark. The reference uses ONNX as a backend. This model requires a 99.9% latency target and a 99.9% accuracy constraint of the reference.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| SSD-ResNet50 | 0.7179 mAP | Cognata | https://github.com/mlcommons/abtf-ssd-pytorch | fp32 |

Achieved a 99% latency of 0.862741101 seconds on an Nvidia L4 GPU.

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site]()(TBD) for an automated way to run this benchmark across different available implementations and perform an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

**Important notes when using MLCFlow**

- Currently, the benchmark workflow is tested only for SingleStream runs.
- While not mandatory, it is recommended to follow MLCFlow commands throughout for a seamless experience with MLCFlow automation.
- If you encounter any issues with automation, please feel free to raise an issue in the [mlperf-automations](https://github.com/mlcommons/mlperf-automations/issues) repository.
- The dataset and model downloads based on the framework, as well as attaching to the docker container, will be automatically handled by MLCFlow if you are building the docker through the MLCFlow command in the [Build and run the Docker container](#build-and-run-the-docker-container) section.
- The email account that has access to the model files should be used to login when prompted by RClone for dataset and model downloads.
- To take a valid benchmark run, provide `--execution_mode=valid` argument. By default, the runs are executed in test mode. 
 

## Downloading the dataset and model checkpoints
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the Cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the SSD model checkpoints. You do not need the entire dataset to run the benchmark. The mlc_cognata_dataset/preprocess_2d folder contains validation and calibration data to run the benchmark. The model checkpoints are located under the mlc_cognata/dataset_model_checkpoint_ssd folder. You should have a cognata directory with the model checkpoints and the val_2d folder extracted directly within it.

```
├── cognata
│   ├── baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth
│   ├── ssd_resnet50.onnx
│   ├── val_2d
```

### Download model through MLCFlow Automation

**ONNX**
```
mlcr get,ml-model,ssd,resnet50,_mlc,_rclone,_onnx --outdirname=<path_to_download>
```

**PyTorch**
```
mlcr get,ml-model,ssd,resnet50,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
```

### Download dataset through MLCFlow Automation

**Preprocessed Validation**
```
mlcr get,preprocessed,dataset,cognata,_mlc,_2d_obj_det,_validation --outdirname=<path_to_download>
```

**Preprocessed Calibration**
```
mlcr get,preprocessed,dataset,cognata,_mlc,_2d_obj_det,_calibration --outdirname=<path_to_download>
```

**Unprocessed**
```
mlcr get,raw,dataset,cognata,_mlc,_rclone --outdirname=<path_to_download>
```


## Build and run the Docker container


### CPU only

**Using MLCFlow Docker**

```
mlcr run-abtf-inference,reference,_v0.5,_full --model=ssd --docker --quiet --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream
```

**Using Native approach**

```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.cpu .
docker run -it -v <your repo path>/mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata/ ssd_inference
```

### GPU enabled

**Using MLCFlow Docker**

```
mlcr run-abtf-inference,reference,_v0.5,_full --model=ssd --docker --quiet --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes --device=cuda --implementation=reference --framework=pytorch --scenario=SingleStream
```

**Using Native approach**
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/2d-object-detection
docker build -t ssd_inference -f dockerfile.gpu .
docker run -it -v <your repo path>/mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata/ ssd_inference
```

## Run the model in performance mode
### Using the ONNX backend

**MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=ssd  --quiet --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream 
```

- Use `--performance_sample_count` to adjust the value of performance sample count. By default, it is set to 128 in the reference implementation.

**Native run command:**
```
python main.py --backend onnx --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --checkpoint /cognata/ssd_resnet50.onnx
```

### Using PyTorch

**MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=ssd  --quiet --device=cpu --implementation=reference --framework=pytorch --scenario=SingleStream 
```

- Use `--device=gpu` to run on GPU

**Native run command:**
```
python main.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --checkpoint /cognata/baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth
```

## Run the model in accuracy mode and run the accuracy checker

**MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_accuracy-only --model=ssd  --quiet --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream 
```

- Use `--device=gpu` to run on GPU
- Use `--framework=pytorch` to use the PyTorch framework for running the implementation

**Native run command:**

Add the --accuracy flag to run in accuracy mode.

```
python main.py --backend onnx --config baseline_8MP_ss_scales_fm1_5x5_all --dataset cognata --dataset-path /cognata/val_2d --checkpoint /cognata/ssd_resnet50.onnx --accuracy
```

### Evaluate the accuracy using MLCFlow
```bash
mlcr process,mlperf,accuracy,_cognata_ssd --result_dir=<Path to directory where files are generated after the benchmark run>
```

### Evaluate the accuracy by directly calling the accuracy script

Run the accuracy checker. If you changed the output log location in accuracy mode, modify the file location.
```
python accuracy_cognata.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset-path /cognata/val_2d --mlperf-accuracy-file ./output/mlperf_log_accuracy.json
```


## Preprocessing data
If you want to preprocess the dataset yourself, you can use preprocess.py within the docker container. You will need to download the entire Cognata dataset and extract the compressed files first. Then run the preprocessing script.
```
python preprocess.py --config baseline_8MP_ss_scales_fm1_5x5_all --dataset-root /cognata/ --output /cognata/val_2d
```