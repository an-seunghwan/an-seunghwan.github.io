---
title: "가중치가 반복되는 Fully Connected Layer 만들기"
excerpt: ""
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-09-03 20:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---

## 목표
- Tensorflow의 행렬 연산은 $$h = \sigma(X W + b)$$ 의 형태로 이루어진다. ($X \in \mathbb{R}^{n \times p}, W \in \mathbb{R}^{p \times d}, b \in \mathbb{R}^d$)
- 만약, 결과값인 $h \in \mathbb{R}^d$ 벡터의 모든 원소들의 값을 동일하게 만들고 싶다면 $W$를 어떻게 설정해야할까?
- $W=[w_1, w_2, \cdots, w_d], w_i \in \mathbb{R}^p, i=1,\cdots,d$에서 $W$의 모든 열을 동일하게, 즉 $$w_i = w_j$$ 로 가중치 행렬을 설정하면 될 것이다!

## code

### python package import

```python
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
```
```
TensorFlow version: 2.4.0 
Eager Execution Mode: True 
available GPU: [] 
========================================== 
[name: "/device:CPU:0" device_type: "CPU" memory_limit: 268435456 locality { } incarnation: 1077612906216489537]
```

### custom layer

```python
class CustomLayer(K.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, 1),
                                            dtype='float32'),
                                            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(),
                                            dtype='float32'),
                                            trainable=True)
        self.w_repeated = tf.repeat(self.w, self.output_dim, axis=-1) # 가중치 벡터 반복을 통해 가중치 행렬 정의
        self.b_repeated = tf.repeat(self.b, self.output_dim)

    def call(self, x):
        h = tf.matmul(x, self.w_repeated) + self.b_repeated # h = xW + b
        h = tf.nn.relu(h) # nonlinear activation
        return h
```

- `self.w`: 가중치 행렬에서 반복되어 사용될 하나의 열벡터
- `self.w`를 변환 후의 차원인 $d$ 개수만큼 반복하여 (`tf.repeat`) 동일한 열이 반복된 가중치 행렬 `self.w_repeated`를 생성
- tensorflow 2.0의 custom layer 생성 방법은 [custom modeling](https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1))을 참고해주세요!

### argument 정의

```python
input_dim = 10
output_dim = 5

custom_layer = CustomLayer(input_dim, output_dim)
```

### 결과 확인

```python
inputs = tf.random.normal((8, input_dim))
outputs = custom_layer(inputs)
outputs
```

```
<tf.Tensor: shape=(8, 5), dtype=float32, numpy=
array([[0.10263357, 0.10263357, 0.10263357, 0.10263357, 0.10263357],
       [0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.11914004, 0.11914004, 0.11914004, 0.11914004, 0.11914004],
       [0.06578927, 0.06578927, 0.06578927, 0.06578927, 0.06578927],
       [0.24528006, 0.24528006, 0.24528006, 0.24528006, 0.24528006],
       [0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.19364165, 0.19364165, 0.19364165, 0.19364165, 0.19364165],
       [0.18903199, 0.18903199, 0.18903199, 0.18903199, 0.18903199]],
      dtype=float32)>
```

- `outputs`의 5개의 모든 값이 동일한 것을 확인할 수 있다!
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNTc4OTc5MSw1NzU4NDE2ODcsLTg2MD
g4Mzc2M119
-->