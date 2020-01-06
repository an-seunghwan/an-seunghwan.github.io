## setting

- ubuntu 18.04 LTS version
- GTX 1080 Ti
- 참고: [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)

## software requirements(for GPU)
### install flow
-   Verify the system has a CUDA-capable GPU.
-   Verify the system is running a supported version of Linux.
-   Verify the system has gcc installed.
-   Verify the system has the correct kernel headers and development packages installed.
-   Download the NVIDIA CUDA Toolkit.
-   Handle conflicting installation methods.


### Ubuntu 16.04 or later (TensorFlow 2.0 CPU version)
* ubuntu 18.04 LTS version

### CUDA®-enabled cards.
* Geforce RTX 1080: Compute Capability = 6.1

### [NVIDIA® GPU drivers](https://www.nvidia.com/drivers)  —CUDA 10.0 requires 410.x or higher.
* ubuntu 18.04에 맞는 드라이버 설치 가능

### [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)  —TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)

* [https://docs.nvidia.com/cuda/archive/10.0/cuda-installation-guide-linux/index.html](https://docs.nvidia.com/cuda/archive/10.0/cuda-installation-guide-linux/index.html)

 ![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/ubuntu1.PNG?raw=true)

* For Ubuntu LTS on x86-64, both the HWE kernel (e.g. 4.13.x for 16.04.4) and the server LTS kernel (e.g. 4.4.x for 16.04) are supported in CUDA 10.0.
* kernel download: [https://kernel.ubuntu.com/~kernel-ppa/mainline/](https://kernel.ubuntu.com/~kernel-ppa/mainline/)
* GCC 7.3.0: [https://gcc.gnu.org/onlinedocs/7.3.0/](https://gcc.gnu.org/onlinedocs/7.3.0/)
* CUDA 10을 Ubuntu 18.04에 설치하는 방법이 TensorFlow 홈페이지에 소개되어 있음

### [CUPTI](http://docs.nvidia.com/cuda/cupti/)  ships with the CUDA Toolkit.
* CUPTI 1.0: [https://developer.nvidia.com/CUPTI-1_0](https://developer.nvidia.com/CUPTI-1_0)

### [cuDNN SDK](https://developer.nvidia.com/cudnn)  (>= 7.4.1)

### (Optional)_  [TensorRT 5.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)  to improve latency and throughput for inference on some models.

## hardware requirements
- NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher
	* Geforce RTX 1080: Compute Capability = 6.1
- 





<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4MjcwMDkzNjAsLTIwODQ4MTk0OTMsND
k5MjIwNDY1LC0zMDI0MDQ4OCwyMTM2MTMyMjc2LDE0NjcyODQz
MCwtODQxNTk4MjEwLC0zOTE3MTM2NV19
-->