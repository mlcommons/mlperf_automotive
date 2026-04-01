# MLPerf Automotive Benchmark: Llama 3.1 8B (Infotainment)

This repository implements an In-Vehicle Infotainment (IVI) benchmark using Meta Llama 3.1 8B Instruct.
It evaluates the model's performance (latency) and accuracy on the **MMLU (Massive Multitask Language Understanding)** dataset.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| Llama3.1 8B Instruct | 64.96% | MMLU | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | fp16 |


## Get Model
### MLCommons Members Download (Recommended for official submission)

You need to request for access to [MLCommons](http://llama3-1.mlcommons.org/) and you'll receive an email with the download instructions. 

**Official Model download using MLCFlow Automation**
You can download the model automatically via the below command
```
mlcr get,ml-model,llama3,_mlc,_8b,_r2-downloader --outdirname=<path to download> -j
```


### External Download (Not recommended for official submission)
+ First go to [llama3.1-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). **Please note your authentication credentials** as you may be required to provide them when cloning below.
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=meta-llama/Llama-3.1-8B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout be673f326cab4cd22ccfef76109faf68e41aa5f1
```

**External Model download using MLCFlow Automation**
You can download the model automatically through HuggingFace via the below command

```
mlcr get,ml-model,llama3,_meta-llama/Llama-3.1-8B-Instruct,_hf --outdirname=${CHECKPOINT_PATH} --hf_token=<huggingface access token> -j
```

**Note:**
Downloading llama3.1-8b model from Hugging Face will require an [**access token**](https://huggingface.co/settings/tokens) which could be generated for your account. Additionally, ensure that your account has access to the [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model. 

## Get Dataset
### MLCommons Storage Download (Recommended for official submission)

You can download the raw dataset automatically through the below command:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d mmlu/ https://automotive.mlcommons-storage.org/metadata/mmlu.uri
```

### External Download (Not recommended for official submission)
The download script will download the dataset from https://huggingface.co/datasets/cais/mmlu and preprocess it for the benchmark. It will generate the full dataset for inference and a calibration dataset.

```bash
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/llm
python download_data.py
```



## Environment Setup
The download script will download the dataset from https://huggingface.co/datasets/cais/mmlu and preprocess it for the benchmark. It will generate the full dataset for inference and a calibration dataset.

```bash
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/llm
python download_data.py
```

Build the docker image:
```bash
docker build -t mlperf-auto-llama .

docker run --gpus all -it -v <path/to/model/and/dataset>:/home/llama mlperf-auto-llama
```

## Running the benchmark

Run in performance mode

`python main.py --model_path <path/to/model> --dataset_path <path/to/dataset> --device <cpu|cuda>`

Run in accuracy mode

`python main.py --model_path <path/to/model> --dataset_path <path/to/dataset> --device <cpu|cuda> --accuracy`

Run the accuracy checker accuracy checker
`python accuracy_checker.py --dataset_file <path/to/dataset>`
