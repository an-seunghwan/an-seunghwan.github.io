---
title: "Generating Sentences from a Continuous Space 2편(작성중)"
excerpt: "tensorflow로 구현해보자!"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-07-21 20:00:00 -0000
categories: 
  - VAE
  - NLP
tags:
  - tensorflow
  - keras
  - RNN
---

> [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) 논문에 대한 간단한 리뷰와 tensorflow 코드입니다. 
>  본 포스팅은 위의 내용에 대한 2편(tensorflow 코드 작성) 입니다.
>  정확한 내용과 수식들은 논문을 참조해주시기 바랍니다. 

## . 환경

```
모델명:  iMac Pro
프로세서 이름:  8-Core Intel Xeon W
프로세서 속도:  3.2 GHz
프로세서 개수:  1
총 코어 개수:  8
메모리:  32 GB
```

## . 데이터

* 네이버 영화 리뷰 데이터 'Naver sentiment movie corpus v1.0'
* 출처: [https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)

## . setting

```python
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('즉시 실행 모드:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
```

```
TensorFlow version: 2.2.0
즉시 실행 모드: True
available GPU: []
==========================================
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 7747759211262961658
, name: "/device:XLA_CPU:0"
device_type: "XLA_CPU"
memory_limit: 17179869184
locality {
}
incarnation: 10969879987829226263
physical_device_desc: "device: XLA_CPU device"
]
```

```python

```

## .데이터 전처리


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUwNzgyOTk4MSwxMzk1Mzg4OTEzLC03NT
gzNTYwNDgsNjMyOTY5NzM4XX0=
-->