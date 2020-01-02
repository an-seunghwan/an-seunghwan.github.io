---
title: "RNN with Keras (2)"
excerpt: Keras RNN의 더 많은 특징
toc: true
toc_sticky: true

author_profile: false

date: 2019-12-31 19:40:00 -0000
categories: 
  - NLP
tags:
  - tensorflow 2.0
  - keras
  - NLP
  - RNN
---
> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/rnn](https://www.tensorflow.org/guide/keras/rnn)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

## 성능 최적화와 CuDNN kernels

Tensorflow 2.0에서는, built-in LSTM과 GRU layer는 GPU가 사용 가능할 경우 default로써 CuDNN kernel의 이점을 활용하도록
업데이트 되었다. 기존의 `keras.layers.CuDNNLSTM/CuDNNGRU` layer는 삭제되었으며,  따라서 모형을 구성할 때 해당 하드웨어에서 작동할지 걱정할 필요가 없다.

CuDNN은 특정한 가정 하에서 작동될 수 있는데, 이는 만약 built-in LSTM과 GRU의 default를 수정하면 CuDNN을 사용할 수 없다.

- `activation` 함수를 `tanh`가 아닌 다른 함수로 바꾸는 경우
- `recurrent_activation` 함수를 `sigmoid`가 아닌 다른 함수로 바꾸는 경우
- `recurrent_dropout` > 0인 경우
- `unroll`을 True으로 설정하는 경우(이는 LSTM/GRU의 내부 `tf.while_loop`을 unroll된 `for` loop으로 분해되도록 한다)
- `use_bias`가 False인 경우
- 입력 데이터가 strictly right padded 되지 않은 경우 masking을 사용하는 경우 (만약 mask가 strictly right padded data에 대응된다면, CuDNN을 여전히 사용할 수 있다. 이는 가장 흔한 경우이다.)

## CuDNN kernel 사용해보기

간단한 LSTM 모형을 통해 성능의 차이를 살펴보자. 여기서 입력 순서열로써 MNIST digits의 순서열 데이터(각각의 pixel row들을 timestep으로써 간주)를 사용할 것이고, 숫자의 label을 예측할 것이다.

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
print(tf.__version__)
print('즉시 실행 모드:', tf.executing_eagerly())
```
```
2.0.0
즉시 실행 모드: True
```
```python
batch_size = 64
# 각각의 MNIST 이미지 batch는 (batch_size, 28, 28)의 shape을 갖는다.
# 각각의 입력 순서열은 (28, 28)의 크기를 갖는다. (여기서 높이(height)은 시간의 의미를 가진다)
input_dim = 28

units = 64
output_size = 10 # 0에서 9까지의 label

# RNN model
def build_model(allow_cudnn_kernel=True):
    # CuDNN은 오직 layer 단계에서만 사용가능하며, cell 단계에서는 사용할 수 없다.
    # 이는 LSTM(units)는 CuDNN을 사용하지만,
    # RNN(LSTMCell(units))는 CuDNN 없이 실행됨을 의미한다.
    if allow_cudnn_kernel:
        # LSTM layer는 default option으로 CuDNN을 사용한다.
        lstm_layer = tf.keras.layers.LSTM(units, 
                                          input_shape=(None, input_dim))
    else:
        # CuDNN을 사용하지 않는 LSTMCell을 RNN layer로 wrapping한다.
        lstm_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units),
                                         input_shape=(None, input_dim))
    
    model = tf.keras.models.Sequential([lstm_layer,
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(output_size,
                                                              activation='softmax')])
    return model
```
## MNIST 데이터
```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]
```
## 모형 생성과 컴파일
모형의 output은 `[batch_size, 10]`의 shape을 가진다.
모형의 target은 정수 벡터이며, 각각의 정수는 0에서 9까지 이다.

```python
model = build_model(allow_cudnn_kernel=True)
# 모형의 손실 함수: sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```
```python
from datetime import datetime
start = datetime.now()
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=1)
print('걸린 시간:', (datetime.now() - start).seconds, '초')
```
```
Train on 60000 samples, validate on 10000 samples
60000/60000 [==============================] - 55s 917us/sample - loss: 0.1324 - accuracy: 0.9587 - val_loss: 0.2054 - val_accuracy: 0.9306
걸린 시간: 55 초
```
## CuDNN이 없는 모형
```python
slow_model = build_model(allow_cudnn_kernel=False)
slow_model.set_weights(model.get_weights())
slow_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='sgd',
                   metrics=['accuracy'])
start = datetime.now()
slow_model.fit(x_train, y_train,
               validation_data=(x_test, y_test),
               batch_size=batch_size,
               epochs=1)
print('걸린 시간:', (datetime.now() - start).seconds, '초')
```
```
Train on 60000 samples, validate on 10000 samples
60000/60000 [==============================] - 57s 953us/sample - loss: 0.1201 - accuracy: 0.9627 - val_loss: 0.1297 - val_accuracy: 0.9583
걸린 시간: 57 초
```
실제로는 CuDNN이 일반적인 Tensorflow kernel을 사용한 경우보다 빠르지만, 필자의 컴퓨터에는 GPU가 없어 비교를 할 수 없었다...

동일한 CuDNN-enabled 모형은 CPU만 있는 환경에서도 추론(inference) 목적을 위해 사용될 수 있다. `tf.device`는 장치의 배치를 지정한다. 다음과 같은 코드를 구성한다면 모형은 만약 GPU를 사용할 수 없다면 default로써 CPU만을 사용할 것이다.

```python
import matplotlib.pyplot as plt
with tf.device('CPU:0'):
    cpu_model = build_model(allow_cudnn_kernel=True)
    cpu_model.set_weights(model.get_weights())
    result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
    print('Predicted result is: {}, target result is: {}'.format(result.numpy(), sample_label))
    plt.imshow(sample, cmap=plt.get_cmap('gray'))
```
```
Predicted result is: [5], target result is: 5
```

![](https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/mnist1.jpg?raw=true)

## list/dict 입력 또는 중첩된 입력인 경우의 RNN
* python class에 대한 자세한 공부 후에 추가하도록 하겠습니다 (coming soon!)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk3ODAwNjY5NiwzNTk1MzAzNDcsMTUxND
U1NjU3OSwtMTgxMTI2NjEwLC01MDExMzc3MjQsLTUwMTEzNzcy
NCwtMjA1NTQ5MjQ2NF19
-->