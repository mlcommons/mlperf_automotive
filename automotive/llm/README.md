# MLPerf Automotive Benchmark: Llama 3.2 3B (Infotainment)

This repository implements an In-Vehicle Infotainment (IVI) benchmark using Meta Llama 3.2 3B Instruct.
It evaluates the model's performance (latency) and accuracy on the **MMLU (Massive Multitask Language Understanding)** dataset.

| model | accuracy | dataset | model source | precision |
| ---- | ---- | ---- | ---- | ---- |
| Llama3.2 3B | 58.61% | MMLU | https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct | fp16 |


## 1. Environment Setup
The download script will download the dataset from https://huggingface.co/datasets/cais/mmlu and preprocess it for the benchmark. It will generate the full dataset for inference and a calibration dataset.

```
git clone git@github.com:mlcommons/mlperf_automotive.git
cd mlperf_automotive/automotive/llm
python download_data.py
```

Build the docker image:
```
docker build -t mlperf-auto-llama .

docker run --gpus all -it -v <path/to/model/and/dataset>:/home/llama mlperf-auto-llama
```

## Running the benchmark

Run in performance mode

`python main.py --model_path <path/to/model> --dataset_path <path/to/dataset> --device <cpu|cuda>`

Run in accuracy mode

`python main.py --model_path <path/to/model> --dataset_path <path/to/dataset> --device <cpu|cuda> --accuracy`

Run the accuracy checker accuracy checker
`python accuracy_checker.py`--dataset_file <path/to/dataset>
