---
title: "Layer Class in Keras"
excerpt: "Keras의 Layer Class"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-05 14:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---

Custom Model을 작성하다 보면 자연스럽게 custom layer를 작성해야 하는 순간이 많이 발생한다. 이러한 경우에 Base layer class인 `tf.keras.layers.Layer`을 subclassing해야하는데, class에 대한 개념이 잡혀있지 않으면 어떤 부분이 어떻게 작동하는지 이해하기 어려운 경우가 많다.

따라서 이번 게시글에서는 Keras의 `Layer` class를 이해할 정도의 python class에 대한 개념과 실제 예제를 통해 subclassing을 적용한 custom layer가 어떻게 구성되어 있는지 살펴보도록 하겠다.

> custom layer에 대한 자세한 내용은 게시글 [https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1)/](https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1)/)을 참고해 주세요!

## setup
```python
import tensorflow as tf
from tensorflow.keras import layers
```
가장 간단한 `Dense` layer를 예제로 사용해보자.

## example) Dense layer
```python
class Linear(layers.Layer): 
    
    def __init__(self, units=32): # self(객체 자신이 호출시 전달) 내부의 속성들을 초기화
        super(Linear, self).__init__()
        self.units = units
    
    def build(self, input_shape): # 이게 핵심!!!!
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer='random_normal',
                                 trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```
### 1. naming convention
naming convention은 CamelCase이다.
CamelCase는 합성어 명칭에서 단어들이 합쳐질 때 단어의 첫 글자를 대문자로 표기하는 방식이다.

### 2. 변수 할당
```python
linear_layer = Linear(units=12) # 변수에 할당
type(linear_layer)
```
```
__main__.Linear
```
### 3. 




> 참고: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=stable
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2NDgxMzg2NTRdfQ==
-->