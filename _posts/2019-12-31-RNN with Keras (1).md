---
title: "RNN with Keras (1)"
excerpt: Keras RNN의 기초
toc: true
toc_sticky: true

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

## setup
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

## 간단한 모형

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

## Output과 은닉 상태

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
```python
batch_size = 64
timesteps = 100
embedding_size = 64
# 임의의 batch embedding input을 생성
sample_embedding = np.random.normal(size=(batch_size, timesteps, embedding_size)).astype('float32')
print(layers.SimpleRNN(128, return_sequences=False)(sample_embedding).shape)
print(layers.SimpleRNN(128, return_sequences=True)(sample_embedding).shape)
```
```
(64, 128)
(64, 100, 128)
```

추가로, RNN layer는 최종 은닉 상태(state)를 반환할 수 있다. 반환된 은닉 상태는 후에 RNN layer 실행을 이어가거나, 다른 RNN을 초기화하는데 사용될 수 있다. 이러한 설정은 흔히 encoder-decoder sequence-to-sequence 모형에서 encoder의 최종 내부 은닉 상태를
decoder의 초기 상태로 사용하기위해 활용된다.

RNN layer가 내부 은닉 상태를 반환하기 위해서는, `return_state` parameter를 `True`으로 설정하면 된다. 특히 `LSTM`은 2개의 은닉 상태 tensor를 갖지만, `GRU`는 1개만 갖는다.

layer의 초기 은닉 상태를 지정하기 위해서는, layer를 호출할 때 `initial_state` 인자를 추가하면 된다. 이때 은닉 상태의 차원(shape)은 반드시 layer의 unit size와 일치해야 한다(아래의 예제 참조).

```python
encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None, ))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(encoder_input)

# output과 함께 은닉 상태를 반환
encoder_output, state_h, state_c = layers.LSTM(64,
                                               return_sequences=False,
                                               return_state=True,
                                               name='encoder')(encoder_embedded)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None, ))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(decoder_input)

# 새로운 LSTM layer의 초기 은닉 상태에 앞의 2개의 상태 값을 지정
decoder_output = layers.LSTM(64,
                             name='decoder')(decoder_embedded, initial_state=encoder_state)
output = layers.Dense(10, activation='softmax')(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output)
model.summary()
```
```
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
input_6 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
embedding_10 (Embedding)        (None, None, 64)     64000       input_5[0][0]                    
__________________________________________________________________________________________________
embedding_11 (Embedding)        (None, None, 64)     128000      input_6[0][0]                    
__________________________________________________________________________________________________
encoder (LSTM)                  [(None, 64), (None,  33024       embedding_10[0][0]               
__________________________________________________________________________________________________
decoder (LSTM)                  (None, 64)           33024       embedding_11[0][0]               
                                                                 encoder[0][1]                    
                                                                 encoder[0][2]                    
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 10)           650         decoder[0][0]                    
==================================================================================================
Total params: 258,698
Trainable params: 258,698
Non-trainable params: 0
__________________________________________________________________________________________________
```
## RNN layers and RNN cells

RNN API는 built-in RNN layer 뿐만 아니라 cell-level API 또한 제공한다. RNN layer가 입력된 batch 순서열을 모두 처리하는 것과 다르게, RNN cell은 오직 하나의 timestep만을 처리한다.

RNN cell은 RNN layer의 `for` loop 내부라고 할 수 있다. cell을 `tf.keras.layers.RNN`으로 래핑(wrapping)하면 batch 순서열을 처리할 수 있도록 해준다. (즉, `RNN(LSTMCell(10))`)

수학적으로, `RNN(LSTMCell(10))`는 `LSTM(10)`과 동일한 결과물을 제공한다. 사실, tensorflow 1.x에서는 해당하는 RNN cell을 만들고 이를 RNN layer으로 래핑(wrapping)했어야 한다. 하지만 built-in `GRU`나 `LSTM`은 CuDNN을 사용할 수 있도록 해주므로 더 나은 성능을 가질 수 있다.

built-in RNN cell
- tf.keras.layers.SimpleRNNCell: corresponds to the SimpleRNN layer.
- tf.keras.layers.GRUCell: corresponds to the GRU layer.
- tf.keras.layers.LSTMCell: corresponds to the LSTM layer.

cell과 일반적인 `tf.keras.layers.RNN` class를 같이 사용한다면 자신만의 RNN 구조를 실행하기 매우 간편하다.

### Cross-batch Statefulness

만약 매우 긴(또는 무한한) 순서열을 처리한다면, **cross-batch statefulness**를 사용하고 싶을 수도 있다.

일반적으로, RNN layer의 내부 은닉 상태는 새로운 새로운 batch가 입력될 때마다 초기화된다. (즉, 이는 모든 sample이 이전의 sample과 독립임을 가정한다). layer는 주어진 sample을 처리하는 동안에만 내부 은닉 상태를 유지할 것이다.

만약 매우 긴 순서열을 처리해야 한다면, 이를 보다 짧은 순서열로 나누고 내부 은닉 상태를 초기화 하지 않고 짧게 나누어진 순서열을 순차적으로 입력한다면 매우 유용할 것이다. 이렇게 한다면, layer는 전체 순서열의 정보를 오직 한번에 순서열의 일부분만을 보고 얻을 수 있다.

이는 layer를 구성할 때, `stateful=True`으로 설정하면 된다.

만약 `s = [t0, t1, ..., t1546, t1547]과 같은 긴 순서열을 가지고, 다음과 같이 나누었다고 생각하자 

```
s1 = [t0, t1, ..., t100]
s2 = [t101, ..., t200]
...
s16 = [t1501, ..., t1547]
```

이는 다음과 같이 처리 가능하다.
```python
lstm_layer = layers.LSTM(64, stateful=True)
for s in sub_sequences:
  output = lstm_layer(s)
```
state를 초기화하고 싶다면, `layer.reset_states()`를 사용하면 된다.

> **주의**: 이러한 설정에서는, 반드시 다음에 이어지는 batch가 이전 batch의 연속이어야 하며, 그 크기(batch size)또한 동일해야 한다.
E.g. 만약 batch가 다음과 같다면 [sequence_A_from_t0_to_t100, sequence_B_from_t0_to_t100], 다음의 batch는 [sequence_A_from_t101_to_t200, sequence_B_from_t101_to_t200] 이어야 한다.  

예제를 살펴보자

```python
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states()는 cached state를 원래의 initial_state로 초기화 한다.
# 만약 initial_state가 주어지지 않았다면, zero_state가 default로 사용된다.
lstm_layer.reset_states()
```
## Bidirectional RNNs

시계열 순서열에 대해서(e.g. text), RNN model은 앞에서부터 뒤로 처리하는 것 뿐만 아니라, 반대 방향으로도 같이 처리를 한다면 더 성능이 좋아진다. 예를 들어, 문장에서 다음 단어를 예측하는데 있어서, 이전에 오는 단어만 보는 것이 아닌 단어 주변의 문맥을 사용하는 것이 더 효과적일 수도 있다.

Keras는 이러한 양방향 RNN을 구현할 수 있도록 쉬운 API를 제공한다: `tf.keras.layers.Bidirectional`

```python
model = tf.keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(64, 
                                           return_sequences=True), 
                               input_shape=(5, 10))) # timesteps = 5, input_dim = 10
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```
```
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_2 (Bidirection (None, 5, 128)            38400     
_________________________________________________________________
bidirectional_3 (Bidirection (None, 64)                41216     
_________________________________________________________________
dense_10 (Dense)             (None, 10)                650       
=================================================================
Total params: 80,266
Trainable params: 80,266
Non-trainable params: 0
_________________________________________________________________
```

* 세부적인 Bidirectional wrapper: coming soon!


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3NjMzNjI1MCwtMzExODY4MDYzXX0=
-->