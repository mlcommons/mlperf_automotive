# ABTF Semantic Segmentation Benchmark

This is the reference implementation for the ABTF semantic segmentation benchmark. The reference uses ONNX as a backend. A PyTorch implementation is provided as well.

This model requires a 99.9% latency target and a 99.9% accuracy constraint of the reference.

| model | accuracy | resolution | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- | ---- |
| DeepLabv3+ |  0.924355 mIOU | 8MP | Cognata | https://github.com/rod409/pp/tree/main/deeplabv3plus | fp32 |

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
Contact [MLCommons](https://mlcommons.org/datasets/cognata) to access the Cognata dataset. Access requires MLCommons membership and signing the EULA. The dataset download also contains the DeepLabv3+ ONNX and PyTorch model checkpoints. You do not need the entire dataset to run the benchmark. Within mlc_cognata_dataset, you can download the model checkpoints under mlc_cognata_dataset/model_checkpoint_deeplab. The preprocessed data is located under mlc_cognata_dataset/preprocessed_seg. Extract val_seg after downloading. There are two ONNX checkpoints along with one PyTorch checkpoint. The ONNX checkpoints are labeled deeplabv3+_8mp.onnx and deeplabv3+_dynamic.onnx. Dynamic refers to dynamic input resolutions so that will work on different image resolutions. 8mp is for 8MP images. The PyTorch version is latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth that will work on different image resolutions. After downloading and extracting, you should have the following folder structure:

```
├── cognata
│   ├── deeplabv3+_dyn.onnx
│   ├── deeplabv3+_8mp.onnx
│   ├── latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth
│   ├── val_seg
```

### Download model through MLCFlow Automation

**ONNX**
```
mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_onnx --outdirname=<path_to_download>
```

**PyTorch**
```
mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
```

### Download dataset through MLCFlow Automation

**Preprocessed Validation**
```
mlcr get,preprocessed,dataset,cognata,_mlc,_segmentation,_validation --outdirname=<path_to_download>
```

**Preprocessed Calibration**
```
mlcr get,preprocessed,dataset,cognata,_mlc,_segmentation,_calibration --outdirname=<path_to_download>
```

**Unprocessed**
```
mlcr get,raw,dataset,cognata,_mlc,_rclone --outdirname=<path_to_download>
```

## Build and run the Docker container

### CPU only

**Using MLCFlow Docker**

```
mlcr run-abtf-inference,reference,_v0.5,_full --model=deeplabv3plus --docker --quiet --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream
```

**Using Native approach**

```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/semantic-segmentation
docker build -t deeplab_inference -f dockerfile.cpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata deeplab_inference
```

### GPU enabled

**Using MLCFlow Docker**

```
mlcr run-abtf-inference,reference,_v0.5,_full --model=deeplabv3plus --docker --quiet --env.MLC_USE_DATASET_FROM_HOST=yes --env.MLC_USE_MODEL_FROM_HOST=yes --device=cuda --implementation=reference --framework=pytorch --scenario=SingleStream
```

**Using Native approach**
```
git clone -b v0.5abtf git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/semantic-segmentation
docker build -t deeplab_inference -f dockerfile.gpu .
docker run -it -v ./mlperf_automotive:/mlperf_automotive -v <path to cognata>:/cognata deeplab_inference
```

## Run the model in performance mode

### Using the ONNX backend

**Using MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=deeplabv3plus --quiet --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream 
```

**Using Native run command:**
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+_8mp.onnx --dataset-path /cognata/val_seg --dataset cognata
```

### Using PyTorch backend

**Using MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_find-performance --model=deeplabv3plus --quiet --device=cpu --implementation=reference --framework=pytorch --scenario=SingleStream 
```

- Use `--device=gpu` to run on GPU

**Using Native run command:**
```
python main.py --checkpoint /cognata/latest_deeplabv3plus_resnet50_cognata_os16_it100000.pth --dataset-path /cognata/val_seg --dataset cognata 
```

## Run the model in accuracy mode and run the accuracy checker

**MLCFlow run command:**

```
mlcr run-abtf-inference,reference,_v0.5,_accuracy-only --model=deeplabv3plus  --quiet --device=cpu --implementation=reference --framework=onnxruntime --scenario=SingleStream 
```

- Use `--framework=pytorch` to run using the PyTorch framework

**Native run command:**

Add the --accuracy flag to run in accuracy mode.
```
python main.py --backend onnx --checkpoint /cognata/deeplabv3+_8mp.onnx --dataset-path /cognata/val_seg --dataset cognata --accuracy
```

### Evaluate the accuracy using MLCFlow
```bash
mlcr process,mlperf,accuracy,_cognata_deeplabv3plus --result_dir=<Path to directory where files are generated after the benchmark run>
```

### Run the accuracy checker directly
```
python accuracy_cognata.py --mlperf-accuracy-file ./output/mlperf_log_accuracy.json --dataset-path /cognata/val_seg 
```

## Preprocessing data
If you want to preprocess the dataset yourself, you can use preprocess.py within the docker container. You will need to download the entire Cognata dataset and extract the compressed files first. Then run the preprocessing script.

In addition to Cognata, there is segmentation ground truth data for the semantic segmentation labels. When you download the dataset, all files are compressed. You can extract the entire dataset using the following commands. This will extract both the dataset and preprocessed labels.
```
cd <your path to cognata>
for f in *.tar.gz; do tar -xzvf "$f"; done
```
Run the preprocessing script:
```
python preprocess.py --dataset-root /cognata/ --workers <num of processes> --output /cognata/val_seg
```

You can add the --image-size flag for different resolutions. In the accuracy checker, you will need to include --image-size with the same dimensions used during preprocessing.
