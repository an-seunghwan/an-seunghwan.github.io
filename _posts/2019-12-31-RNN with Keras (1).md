---
title: "RNN with Keras (1)"
excerpt: Keras RNN의 기초

author_profile: false

date: 2019-12-31 16:45:00 -0400
categories: 
  - NLP
tags:
  - tensorflow 2.0
  - NLP
  - RNN
---
> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/rnn](https://www.tensorflow.org/guide/keras/rnn)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

RNN은 neural network의 일종으로 시계열 데이터나 언어와 같은 순서열 데이터에 효과적이다.

계획적으로, RNN layer는 `for` loop을 순서열의 시간 단위(timestep)별로 반복하고, 이때 현재까지 시간단위별로 관측한 정보들을 내부 상태로써 유지한다.

Keras RNN API는 다음과 같은 목적을 둔다.
- **쉬운 사용**: built-in `tf.keras.layers.RNN`, `tf.keras.layers.LSTM`, `tf.keras.layers.GRU` layer들을 
어려운 configuration 설정이 없이 빠르게 recurrent 모형을 적합할 수 있도록 해준다.
- **쉬운 customization**: 자신만의 RNN cell layer(`for` loop의 내부 부분)를 정의할 수 있고, 이를 일반적인 `tf.keras.layers.RNN` layer(`for` loop itself)와 함께 사용할 수 있다.

이는 최소한의 코드로 유연하게 다른 연구 아이디어의 원형을 만들 수 있도록 해준다.

### setup
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print('즉시 실행 모드:', tf.executing_eagerly())
```
```
2.0.0
즉시 실행 모드: True
```

### 간단한 모형

Keras의 built-in RNN layer
- `tf.keras.layers.SimpleRNN`: 이전 timestep에서 다음 timestep으로 정보를 넘겨주는(fed) fully-connected RNN  
- `tf.keras.layers.GRU`
- `tf.keras.layers.LSTM`

다음의 예제는 정수 순서열이 입력으로 주어지면, 이를 64차원의 벡터로 임베딩하고, `LSTM` layer로 벡터 순서열을 처리하는 `Sequential` 모형이다.

```python
model = tf.keras.Sequential()
# 단어의 개수가 1000이고, 임베딩 차원의 크기가 64인 Embedding layer를 추가한다.
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# 128개의 내부 unit을 가지는 LSTM layer를 추가한다.
model.add(layers.LSTM(128))
# 10개의 unit과 softmax 활성화 함수를 갖는 Dense layer를 추가한다.
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```
```
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_7 (Embedding)      (None, None, 64)          64000     
_________________________________________________________________
lstm_4 (LSTM)                (None, 128)               98816     
_________________________________________________________________
dense_6 (Dense)              (None, 10)                1290      
=================================================================
Total params: 164,106
Trainable params: 164,106
Non-trainable params: 0
_________________________________________________________________
```

### Output과 은닉 상태

default로써, RNN layer는 각각의 sample 별로 하나의 벡터를 갖는다. 이 벡터는 마지막 timestep에 해당하는 RNN cell의 output으로, 전체 입력 순서열의 정보를 갖고 있다. 이 경우 RNN layer의 output 차원(shape)은 `(batch_size, units)`이고, 이때 `units`는 layer를 구성하는 unit의 개수이다.

RNN layer는 또한 각각의 sample의 모든 timestep 별로 RNN cell의 output을 반환할 수 있다. 이는 `return_sequences=True`으로 설정하면 가능하다. 이러한 경우의 RNN layer의 output 차원(shape)은 `(batch_size, timesteps, units)`이다.

* RNN layer의 은닉 상태와 parameter들의 차원에 관해서는 RNN with Keras (0)을 참고해주세요! (coming soon!)

```python
model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))
# GRU layer의 output의 shape은 (batch_size, timesteps(= input_dim), 256)이다.
model.add(layers.GRU(256, return_sequences=True))
# SimpleRNN layer의 output의 shape은 (batch_size, 128)이다.
model.add(layers.SimpleRNN(128, return_sequences=False))
model.add(layers.Dense(10, activation='softmax'))
model.summary() 
```
```
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_9 (Embedding)      (None, None, 64)          64000     
_________________________________________________________________
gru_3 (GRU)                  (None, None, 256)         247296    
_________________________________________________________________
simple_rnn_11 (SimpleRNN)    (None, 128)               49280     
_________________________________________________________________
dense_8 (Dense)              (None, 10)                1290      
=================================================================
Total params: 361,866
Trainable params: 361,866
Non-trainable params: 0
_________________________________________________________________
```
아래의 간단한 예제로 위의 내용을 확인해보자.
```pyh
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDk4NTY5ODczXX0=
-->