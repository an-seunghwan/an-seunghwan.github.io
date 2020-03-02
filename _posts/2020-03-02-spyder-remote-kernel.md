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

## 4. 사용하기
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

2. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTkxNDYzMDcyOCwxMzM3MzA1ODc2LC0yNT
Q3Nzg3NzJdfQ==
-->