#!/bin/bash
set -e

# Default values
DATA_PATH="./dataset"
MODEL_PATH="./models"
RCLONE_CONFIG="./rclone.conf"

# Capture arguments
while [ $# -gt 0 ]; do
  case "$1" in
  --data_path=*)
    DATA_PATH="${1#*=}"
    ;;
  --model_path=*)
    MODEL_PATH="${1#*=}"
    ;;
  --rclone_config=*)
    RCLONE_CONFIG="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

START_DIR=$(pwd)

pip install mlc-scripts

echo "--- Preparing Dataset Directory ---"
mkdir -p "$DATA_PATH"
cd "$DATA_PATH"
echo "Downloading dataset to: $(pwd)"

echo "--- Downloading and unzipping dataset ---"
curl -O https://storage.googleapis.com/mlperf_training_demo/bevformer/can_bus.zip
unzip -o -q can_bus.zip && rm can_bus.zip

curl -O https://storage.googleapis.com/mlperf_training_demo/bevformer/nuscenes.zip
unzip -o -q nuscenes.zip && rm nuscenes.zip

curl -O https://storage.googleapis.com/mlperf_training_demo/bevformer/val_3d.zip
unzip -o -q val_3d.zip && rm val_3d.zip

echo "Dataset downloaded successfully."

cd "$START_DIR"

echo "--- Preparing Model Directory ---"
mkdir -p "$MODEL_PATH"
echo "Downloading model to: $MODEL_PATH"

mkdir -p ~/.config/rclone/
if [ -f "$RCLONE_CONFIG" ]; then
    cp "$RCLONE_CONFIG" ~/.config/rclone/rclone.conf
    export RCLONE_CONFIG=~/.config/rclone/rclone.conf
else
    echo "WARNING: Config file $RCLONE_CONFIG not found."
fi

echo "--- Patching MLC scripts ---"
mlc pull repo mlcommons@mlperf-automations
CONFIG_SCRIPT=$(find ~/MLC /root/MLC -path "*/script/get-rclone-config/run.sh" 2>/dev/null | head -n 1)

if [ -n "$CONFIG_SCRIPT" ]; then
    cat <<EOF > "$CONFIG_SCRIPT"
#!/bin/bash
exit 0
EOF
    chmod +x "$CONFIG_SCRIPT"
fi

mlcr get,ml-model,bevformer,_mlc,_rclone,_onnx --outdirname="$MODEL_PATH"

echo "--- Finalizing Model Structure ---"

DOWNLOADED_FILE=$(find "$MODEL_PATH" -name "bevformer_tiny.onnx" -type f | head -n 1)

if [ -n "$DOWNLOADED_FILE" ]; then
    TARGET_FILE="$MODEL_PATH/bevformer_tiny.onnx"

    if [ "$DOWNLOADED_FILE" != "$TARGET_FILE" ]; then
        echo "Flattening directory structure..."
        echo "Found file at: $DOWNLOADED_FILE"

        mv "$DOWNLOADED_FILE" "$MODEL_PATH/temp_bevformer.onnx"

        DIR_TO_REMOVE=$(dirname "$DOWNLOADED_FILE")

        if [[ "$DIR_TO_REMOVE" == "$MODEL_PATH"* ]]; then
             rm -rf "$DIR_TO_REMOVE"
        fi

        mv "$MODEL_PATH/temp_bevformer.onnx" "$TARGET_FILE"
        echo "Model moved to: $TARGET_FILE"
    else
        echo "Model is already in the correct location."
    fi
else
    echo "Error: Download completed but .onnx file not found!"
    exit 1
fi