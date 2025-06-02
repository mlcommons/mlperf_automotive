---
hide:
  - toc
---

# 3D Object Detection using BEVFormer (tiny)

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Preprocessed Dataset"
    BEVFormer validation run uses the NuScenes dataset. The preprocessed data is located in nuscenes_data/preprocessed/val_3d.tar.gz.

    ### Get Validation Dataset
    ```
    mlcr get,preprocessed,dataset,nuscenes,_mlc,_validation --outdirname=<path_to_download>
    ```

    ### Get Calibration Dataset
    ```
    mlcr get,preprocessed,dataset,nuscenes,_mlc,_calibration --outdirname=<path_to_download>
    ```

=== "Raw Dataset"
    For preprocessing the dataset yourself, you can download the raw dataset.

    ### Get Raw Dataset
    ```
    mlcr get,dataset,nuscenes,_mlc,_rclone --outdirname=<path_to_download>
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf BEVFormer Model

=== "ONNX"

    ### ONNX
    ```
    mlcr get,ml-model,bevformer,_mlc,_rclone,_onnx --outdirname=<path_to_download>
    ```

=== "PyTorch"

    ### PyTorch
    ```
    mlcr get,ml-model,bevformer,_mlc,_rclone,_pytorch --outdirname=<path_to_download>
    ``` 