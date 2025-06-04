---
hide:
  - toc
---

# Semantic Segmentation using DeepLabv3+

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Preprocessed Dataset"
    DeepLabv3+ validation run uses the Cognata dataset. The preprocessed data is located under mlc_cognata_dataset/preprocessed_seg.

    ### Get Validation Dataset
    ```
    mlcr get,preprocessed,dataset,cognata,_mlc,_segmentation,_validation --outdirname=<path_to_download>
    ```

    ### Get Calibration Dataset
    ```
    mlcr get,preprocessed,dataset,cognata,_mlc,_segmentation,_calibration --outdirname=<path_to_download>
    ```

=== "Raw Dataset"
    For preprocessing the dataset yourself, you can download the raw dataset.

    ### Get Raw Dataset
    ```
    mlcr get,raw,dataset,cognata,_mlc,_rclone --outdirname=<path_to_download>
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf DeepLabv3+ Model

=== "ONNX"

    ### ONNX
    Two variants are available:
    ```
    # 8MP optimized version
    mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_onnx --outdirname=<path_to_download>
    
    # Dynamic resolution version
    mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_onnx,_dynamic --outdirname=<path_to_download>
    ```

=== "PyTorch"

    ### PyTorch
    ```
    mlcr get,ml-model,deeplabv3-plus,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
    ``` 