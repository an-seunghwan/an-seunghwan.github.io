---
title: "Spyder Remote Kernel"
excerpt: "server 컴퓨터의 환경을 원격으로 사용하자!"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-03-02 21:00:00 -0000
categories: 
  - advanced
tags:
  - spyder
---
> 까먹지 않기 위해 하는 개인적인 목적의 정리 포스팅입니다.

## 0. 환경
- ubuntu 18.04
- spyder 4

## 1. spyder kernel 설치
(server와 client 둘다)
```
conda install spyder-kernels
```

## 2. remote kernel 구동
```
python -m spyder_kernels.console — matplotlib=’inline’ — ip=x.x.x.x -f=./remotemachine.json
```
위와 같은 명령어를 실행하면 마지막 줄에
```
To connect another client to this kernel, use:
    --existing kernel-27911.json
```
라고 뜬다.
```
jupyter --runtime-dir
```
를 이용해 server 컴퓨터의 해당 경로를 확인하면 `kernel-27911.json`이름의 json 파일이 있다. 이를 filezilla 등을 이용해 client 컴퓨터로 복사한다(파일 내용 수정 절대 금지.

해당 json 파일을 확인해보면 `shell_port`등의 정보가 담긴 파일임을 확인할 수 있다.

참고로 이 과정은 매번 연결시에 반복해야 한다...

## 3. remote connection
이제 client 컴퓨터의 spyder를 구동한다.
spyder kernel 우측 상단의 메뉴를 클릭하면 `connect to an existing kernel`이라는 항목이 있다. 이 항목을 클릭하면 다음과 같은 창이 뜬다.

<center><img  src="https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/existing_kernel_setting.png?raw=true
" width="600"  height="450"></center>

- `connection file`: client 컴퓨터로 복사한 `kernel-27911.json`의 경로를 입력
- `Hostname`: server 컴퓨터의 ip 주소
- `Username`: server 컴퓨터로 접속할 때 사용하는 putty id
- `Password`: server 컴퓨터로 접속할 때 사용하는 putty id의 비밀번호

연결이 성공적으로 이루어 졌다면 kernel tab의 이름이 `Username@Hostname`으로 뜨는 것을 확인할 수 있다.

## 4. 확인하기
1. server 컴퓨터의 gpu 사용 확인(tensorflow 구동 전)
	- Processes에 python이 잡히지 않는 것을 확인할 수 있다.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  On   | 00000000:03:00.0 Off |                  N/A |
|  0%   33C    P8    13W / 250W |    280MiB / 11177MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1693      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      1825      G   /usr/bin/gnome-shell                          49MiB |
|    0      2112      G   /usr/lib/xorg/Xorg                            95MiB |
|    0      2237      G   /usr/bin/gnome-shell                         112MiB |
+-----------------------------------------------------------------------------+
```

2. tensorflow gpu 확인

```python
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available(
      cuda_only=False,
      min_cuda_compute_capability=None
))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

```
2.0.0
2020-03-02 21:00:36.868012: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2593720000 Hz
2020-03-02 21:00:36.869945: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558618b03590 executing computations on platform Host. Devices:
2020-03-02 21:00:36.869980: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
2020-03-02 21:00:36.873378: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
True
2020-03-02 21:00:37.106238: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x558618b7a3e0 executing computations on platform CUDA. Devices:
2020-03-02 21:00:37.106281: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): GeForce GTX 1080 Ti, Compute Capability 6.1
2020-03-02 21:00:37.107096: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:03:00.0
2020-03-02 21:00:37.107414: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-03-02 21:00:37.109061: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-03-02 21:00:37.110305: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-03-02 21:00:37.110645: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-03-02 21:00:37.112646: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-03-02 21:00:37.114507: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-03-02 21:00:37.120095: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-02 21:00:37.121323: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-03-02 21:00:37.121380: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-03-02 21:00:37.122328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-02 21:00:37.122363: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-03-02 21:00:37.122375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-03-02 21:00:37.123644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 10213 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
```

```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 13908055485246881972
, name: "/device:XLA_CPU:0"
device_type: "XLA_CPU"
memory_limit: 17179869184
locality {
}
incarnation: 9526915761610223816
physical_device_desc: "device: XLA_CPU device"
, name: "/device:XLA_GPU:0"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 6844963740812801904
physical_device_desc: "device: XLA_GPU device"
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 10710076621
locality {
  bus_id: 1
  links {
  }
}
incarnation: 9885594885116113
physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
]
2020-03-02 21:00:47.161841: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:03:00.0
2020-03-02 21:00:47.161906: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-03-02 21:00:47.161934: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-03-02 21:00:47.161957: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-03-02 21:00:47.161979: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-03-02 21:00:47.162000: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-03-02 21:00:47.162022: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-03-02 21:00:47.162045: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-02 21:00:47.163465: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-03-02 21:00:47.163514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-02 21:00:47.163528: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-03-02 21:00:47.163539: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-03-02 21:00:47.165033: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 10213 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
```

 GeForce GTX 1080 Ti가 device로 확인됨을 알 수 있다.

3. nvidia 확인
다시 server 컴퓨터의 nvidia를 확인하면 다음과 같다.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  On   | 00000000:03:00.0 Off |                  N/A |
|  0%   34C    P8    12W / 250W |    427MiB / 11177MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1693      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      1825      G   /usr/bin/gnome-shell                          49MiB |
|    0      2112      G   /usr/lib/xorg/Xorg                            95MiB |
|    0      2237      G   /usr/bin/gnome-shell                         112MiB |
|    0     29715      C   python                                       137MiB |
+-----------------------------------------------------------------------------+
```

이전에 없던 python이 processes에 등록되어 있음을 볼 수 있다.

## 5. 활용하기

```python
tf.debugging.set_log_device_placement(True)

# 텐서를 GPU에 할당
with tf.device('/device:GPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)
```

```
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
2020-03-02 21:05:27.333477: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:03:00.0
2020-03-02 21:05:27.333551: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-03-02 21:05:27.333578: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-03-02 21:05:27.333601: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-03-02 21:05:27.333623: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-03-02 21:05:27.333645: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-03-02 21:05:27.333667: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-03-02 21:05:27.333690: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-02 21:05:27.335480: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-03-02 21:05:27.336997: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:03:00.0
2020-03-02 21:05:27.337035: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-03-02 21:05:27.337062: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-03-02 21:05:27.337085: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-03-02 21:05:27.337107: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-03-02 21:05:27.337130: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-03-02 21:05:27.337152: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-03-02 21:05:27.337175: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-02 21:05:27.338956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-03-02 21:05:27.339006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-02 21:05:27.339020: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-03-02 21:05:27.339031: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-03-02 21:05:27.340899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10213 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
2020-03-02 21:05:28.096080: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
```

가장 위의 줄들의 내용을 통해 tensor 연산이 gpu에서 이루어짐을 알 수 있다.

```
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
2020-03-02 21:05:27.333477: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
```

또한 server 컴퓨터의 nvidia를 확인해보면 python의 메모리 사용량이 증가했음을 알 수 있다.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  On   | 00000000:03:00.0 Off |                  N/A |
|  0%   35C    P8    12W / 250W |  10767MiB / 11177MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1693      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      1825      G   /usr/bin/gnome-shell                          49MiB |
|    0      2112      G   /usr/lib/xorg/Xorg                            95MiB |
|    0      2237      G   /usr/bin/gnome-shell                         112MiB |
|    0     29715      C   python                                     10485MiB |
+-----------------------------------------------------------------------------+
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbNTgxMjA5NjU3LC0xNjI3NDIwNjU5LDY5MT
E4MTU0LDEzMzczMDU4NzYsLTI1NDc3ODc3Ml19
-->