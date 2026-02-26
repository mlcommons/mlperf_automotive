# MLPerf Automotive Benchmark: Llama 3.2 3B (Infotainment)

This repository implements an In-Vehicle Infotainment (IVI) benchmark using Meta Llama 3.2 3B Instruct.
It evaluates the model's performance (latency) and accuracy on a subset of the **MMLU (Massive Multitask Language Understanding)** dataset.


## 1. Environemnt Setup
Download the dataset

`python download_data.py`

Build the image:
```
docker build -t mlperf-auto-llama .

docker run --gpus all -it -v /path/to/model/and/dataset:/home/llama mlperf-auto-llama
```

## Running the benchmark

### Performance mode

`python main.py --model_path <path/to/model> --dataset_path <path/to/dataset> --device <cpu|cuda>`

### Accuracy model

`python main.py --model_path <path/to/model> --dataset_path <path/to/dataset> --device <cpu|cuda> --accuracy`

### Accuracy checker
`python accuracy_checker.py`--dataset_file <path/to/dataset>
