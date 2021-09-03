<<<<<<< HEAD
---
title: "가중치가 반복되는 Fully Connected Layer 만들기"
excerpt: ""
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-09-03 20:00:00 -0000
categories: 
  - python
tags:
  - tensorflow
  - keras
---

## 목표
- Tensorflow의 행렬 연산은 $\sigma(x W + b)$ 의 형태로 이루어진다.
- 

##

```python
#%%
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
#%%
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
        self.w_repeated = tf.repeat(self.w, self.output_dim, axis=-1)
        self.b_repeated = tf.repeat(self.b, self.output_dim)

    def call(self, x):
        h = tf.matmul(x, self.w_repeated) + self.b_repeated # h = xW + b
        h = tf.nn.relu(h) # nonlinear activation
        return h
#%%
input_dim = 10
output_dim = 5

custom_layer = CustomLayer(input_dim, output_dim)
#%%
inputs = tf.random.normal((64, input_dim))
outputs = custom_layer(inputs)
outputs
#%%
```
=======

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwODg3NDY2MTJdfQ==
-->
>>>>>>> 13cb57a929b93c91a3915561360da45342cb27ec
