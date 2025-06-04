---
hide:
  - toc
---

# 2D Object Detection using SSD-ResNet50

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Preprocessed Dataset"
    SSD validation run uses the Cognata dataset. The preprocessed data is located in mlc_cognata_dataset/preprocess_2d folder.

    ### Get Validation Dataset
    ```
    mlcr get,preprocessed,dataset,cognata,_mlc,_2d_obj_det,_validation --outdirname=<path_to_download>
    ```

    ### Get Calibration Dataset
    ```
    mlcr get,preprocessed,dataset,cognata,_mlc,_2d_obj_det,_calibration --outdirname=<path_to_download>
    ```

=== "Raw Dataset"
    For preprocessing the dataset yourself, you can download the raw dataset.

    ### Get Raw Dataset
    ```
    mlcr get,raw,dataset,cognata,_mlc,_rclone --outdirname=<path_to_download>
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf SSD Model

=== "ONNX"

    ### ONNX
    ```
    mlcr get,ml-model,ssd,resnet50,_mlc,_rclone,_onnx --outdirname=<path_to_download>
    ```

=== "PyTorch"

    ### PyTorch
    ```
    mlcr get,ml-model,ssd,resnet50,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
    ``` 