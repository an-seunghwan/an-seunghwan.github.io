TensorFlow is tested and supported on the following 64-bit systems:
-   Ubuntu 16.04 or later

GPU support is available for Ubuntu and Windows with CUDA®-enabled cards.

TensorFlow GPU support requires an assortment of drivers and libraries. To simplify installation and avoid library conflicts, we recommend using a [TensorFlow Docker image with GPU support](https://www.tensorflow.org/install/docker) (Linux only). This setup only requires the [NVIDIA® GPU drivers](https://www.nvidia.com/drivers).

## Hardware requirements

The following GPU-enabled devices are supported:

-   NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher. See the list of  [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus).

## Software requirements

The following NVIDIA® software must be installed on your system:

-   [NVIDIA® GPU drivers](https://www.nvidia.com/drivers)  —CUDA 10.0 requires 410.x or higher.
-   [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)  —TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
-   [CUPTI](http://docs.nvidia.com/cuda/cupti/)  ships with the CUDA Toolkit.
-   [cuDNN SDK](https://developer.nvidia.com/cudnn)  (>= 7.4.1)
-   _(Optional)_  [TensorRT 5.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)  to improve latency and throughput for inference on some models.



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI3MDg0NjMxMV19
-->