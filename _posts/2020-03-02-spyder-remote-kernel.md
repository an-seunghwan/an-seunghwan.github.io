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


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI1NDc3ODc3Ml19
-->