## setting

- ubuntu 18.04 LTS version
- GTX 1080 Ti


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

#### Ubuntu 18.04 (CUDA 10)
```python
# Add NVIDIA package repositories  
`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb`  
`sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb`  
`sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub`  
`sudo apt-get update`  
`wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb`  
`sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb`  
`sudo apt-get update`  
  
# Install NVIDIA driver  
`sudo apt-get install --no-install-recommends nvidia-driver-418`  
# Reboot. Check that GPUs are visible using the command: nvidia-smi  
  
# Install development and runtime libraries (~4GB)  
`sudo apt-get install --no-install-recommends \ cuda-10-0  \ libcudnn7=7.6.2.24-1+cuda10.0  \ libcudnn7-dev=7.6.2.24-1+cuda10.0  
`  
  
# Install TensorRT. Requires that libcudnn7 is installed above.  
`sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0  \ libnvinfer-dev=5.1.5-1+cuda10.0  
`
```

![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ubuntu1.PNG?raw=true)


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTczMDgxMTEyLC04NDE1OTgyMTAsLTM5MT
cxMzY1XX0=
-->