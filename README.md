# MLPerf™ Automotive Benchmark Suite
MLPerf Automotive is a benchmark suite for measuring how fast automotive systems can run models in a variety of deployment scenarios. 

## MLPerf Automotive v0.5
Use the r0.5 branch (```git checkout v0.5abtf```) if you want to reproduce v0.5 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| ssd-resnet50| [v0.5/2D object detection](https://github.com/mlcommons/mlperf_automotive/tree/v0.5abtf/automotive/2d-object-detection) | pytorch, onnx | Cognata |
| bevformer-tiny | [v0.5/camera-based 3D object detection](https://github.com/mlcommons/mlperf_automotive/tree/v0.5abtf/automotive/camera-3d-detection) |pytorch, onnx | NuScenes |
| DeepLabV3+ | [v0.5/semantic segmentation](https://github.com/mlcommons/mlperf_automotive/tree/v0.5abtf/automotive/semantic-segmentation) |pytorch, onnx | Cognata |
