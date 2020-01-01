---
title: "Custom modeling with Keras (1)"
excerpt: 기초적인 custom 모형 제작 방법
toc: true
toc_sticky: true

author_profile: false

date: 2019-12-29 16:30:00 -0400
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---

> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/custom_layers_and_models](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()  # For easy reset of notebook state.
```
## Dense layer
```python
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                                                  trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  # 위와 같이 shape을 적용하면 column에 더하기 아님
                                                  # 각 units에 값이 더해진다고 생각!
                                                  dtype='float32'),
                                                  trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```
```python
x = tf.ones((3,3))
linear_layer = Linear(5, 3)
y = linear_layer(x)
print(linear_layer.b)
print(y)
print(tf.matmul(x, linear_layer.w))
print(tf.matmul(x, linear_layer.w) + linear_layer.b)
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```
```
<tf.Variable 'Variable:0' shape=(5,) dtype=float32, numpy=array([0., 0., 0., 0., 0.], dtype=float32)>
tf.Tensor(
[[-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]], shape=(3, 5), dtype=float32)
tf.Tensor(
[[-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]], shape=(3, 5), dtype=float32)
tf.Tensor(
[[-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]], shape=(3, 5), dtype=float32)
 ```
 
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDE1MjA1NzUxXX0=
-->