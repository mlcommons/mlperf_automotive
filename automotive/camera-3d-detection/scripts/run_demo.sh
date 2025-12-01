#!/bin/bash
set -e

DATA_PATH=""
MODEL_PATH=""
LOG_DIR=""

# Capture arguments
while [ $# -gt 0 ]; do
  case "$1" in
  --data_path=*) DATA_PATH="${1#*=}" ;;
  --model_path=*) MODEL_PATH="${1#*=}" ;;
  --log_dir=*) LOG_DIR="${1#*=}" ;;
  *) ;;
  esac
  shift
done

if [[ -z "$DATA_PATH" || -z "$MODEL_PATH" || -z "$LOG_DIR" ]]; then
  echo "Error: Missing arguments." >&2
  exit 1
fi

export DATAROOT="$DATA_PATH"
export MODELROOT="$MODEL_PATH"
export LOGDIR="$LOG_DIR"

echo "--- Checking for GPU Support ---"
if ! python -c "import onnxruntime; print(onnxruntime.get_available_providers())" | grep -q "CUDAExecutionProvider"; then
    echo "WARNING: CUDA Provider missing. Fixing dependencies..."
    pip uninstall -y onnxruntime onnxruntime-gpu numpy || true
    pip install onnxruntime-gpu==1.15.1
    pip install "numpy==1.21.0" --force-reinstall
else
    echo "CUDA Provider is already available."
fi

echo "--- Setting up Symlinks ---"
rm -rf /dataset /models
ln -sf "$DATAROOT" /dataset
ln -sf "$MODELROOT" /models

if [ -f "/models/bevformer_tiny.onnx" ]; then
    ONNX_FILE="/models/bevformer_tiny.onnx"
else
    ONNX_FILE=$(find -L /models/ -name "bevformer_tiny.onnx" -type f | head -n 1)
fi

if [ -z "$ONNX_FILE" ]; then
  echo "Error: bevformer_tiny.onnx not found."
  exit 1
fi

SCENE_FILE=$(find -L /dataset -name "scene_lengths.pkl" -type f | head -n 1)

if [ -z "$SCENE_FILE" ]; then
    echo "WARNING: scene_lengths.pkl not found. Attempting to fix..."
    curl -f -o /dataset/scene_lengths.pkl https://storage.googleapis.com/mlperf_training_demo/bevformer/scene_lengths.pkl || true
    
    if [ -f "/dataset/scene_lengths.pkl" ]; then
        SCENE_FILE="/dataset/scene_lengths.pkl"
    else
        echo "Generating dummy metadata..."
        SAMPLE_COUNT=$(find /dataset/val_3d -name "*.pkl" | wc -l | xargs)
        python3 -c "import pickle; f=open('/dataset/scene_lengths.pkl', 'wb'); pickle.dump([$SAMPLE_COUNT], f); f.close()"
        SCENE_FILE="/dataset/scene_lengths.pkl"
    fi
fi

NUSCENES_ROOT=$(dirname "$SCENE_FILE")

if [ -d "/dataset/val_3d" ]; then
    DATASET_Input="/dataset/val_3d"
else
    DATASET_Input="/dataset"
fi

echo "Using Model: $ONNX_FILE"
echo "Using Dataset Path: $DATASET_Input"
echo "Logs will be saved to: $LOGDIR"

if grep -q "ConstantStream" main.py; then
    echo "--- Patching main.py (Removing deprecated ConstantStream) ---"
    sed -i '/ConstantStream/d' main.py
fi

if grep -q "sum(scene_lengths), performance_sample_count" main.py; then
    echo "--- Patching main.py (Fixing GroupedQSL NumPy argument) ---"
    sed -i 's/sum(scene_lengths), performance_sample_count/np.array(scene_lengths), performance_sample_count/g' main.py
fi

if grep -q "lg.QuerySamplesComplete(response)" main.py; then
    echo "--- Patching main.py (Enabling progress dots) ---"
    sed -i "s/lg.QuerySamplesComplete(response)/lg.QuerySamplesComplete(response); print('.', end='', flush=True)/g" main.py
fi

if [ -f "backend_deploy.py" ]; then
    echo "--- Patching backend_deploy.py (Forcing CUDAExecutionProvider) ---"
    sed -i 's/ort.InferenceSession(self.checkpoint)/ort.InferenceSession(self.checkpoint, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])/g' backend_deploy.py
fi

echo "Starting ..."

python main.py \
  --dataset nuscenes \
  --dataset-path "$DATASET_Input" \
  --nuscenes-root "$NUSCENES_ROOT" \
  --checkpoint "$ONNX_FILE" \
  --config ./projects/configs/bevformer/bevformer_tiny.py \
  --output "$LOGDIR" \
  --scenario Offline \
  --count 20 \
  --performance-sample-count 20 \
  --time 60 \
  --threads 4