---
title: "Train and Evaluate with Keras (1)"
excerpt: "Part 1, 1부(loss, metric, optimizer)"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-04 15:30:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---
> 이 글은 다음 문서를 참조하고 있습니다!
>[https://www.tensorflow.org/guide/keras/train_and_evaluate](https://www.tensorflow.org/guide/keras/train_and_evaluate)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.
> 비교적 간단한 내용이나 코드와 같은 경우에는 번역 없이 생략하니 꼭 원문을 확인해주시면 감사하겠습니다.

해당 게시글은 training, evaluation, 그리고 prediction(inference) 모형을 2가지 광범위한 상황에서 다룬다:
- training & validation을 위한 built-in API(`model.fit()`, `model.evaluation()`, `model.predict()`)를 사용하는 경우. 이는 **built-in training and evaluation loop 사용** 파트에서 다뤄진다.
- 즉시 실행(eager execution)과 `GradientTape`을 이용하여 처음부터 자신만의 loop를 만드는 경우. 이는 **처음부터 자신만의 training and evaluation loop 작성하기** 파트에서 다뤄진다.

일반적으로, built-in loop을 사용하든지 자신만의 모형을 작성하든지, 모형의 training & evaluation은 Keras의 모든 종류의 모형에 대해서 정확히 동일하게 작동한다- Sequential model, Functional API를 사용한 모형, model subclassing을 이용한 모형

해당 가이드는 분산 training에 대해서는 다루지 않는다.

## Setup
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
```
## Part 1. built-in training and evaluation loop 사용

built-in training loop에 데이터를 전달하는 경우에, 반드시 **Numpy arrays**(데이터의 크기가 작고 memory 크에 잘 맞는 경우)를 사용하거나 **tf.data Dataset** 객체를 사용해야 한다. 다음 몇 개의 단락에서, optimizer, losses, metrics를 사용하는 방법을 설명하기 위해 MNIST dataset을 Numpy arrays처럼 다룰 것이다.

### API overview: 첫 번째 end-to-end 예제

다음의 모형을 생각하자.

```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='prediction')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```
다음은 원래의 training data로부터 생성된 holdout set에 대한 training, validation, 그리고 test data에 대해 evaluation을 하는 전체적인 과정을 담은 전형적인 end-to-end workflow이다.

```python
# data load
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# validation을 위해 10000개의 sample을 별도로 남겨둔다
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# training의 세부 설정을 명시
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalCrossentropy()]) # list

# data를 "batch_size"를 가지는 "batches"로 잘게 나누어 모형을 훈련
# 그리고 전체 데이터를 주어진 "epochs"만큼의 회수로 반복한다
print("# training data에 모형을 적합")
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    # 각 epoch이 끝날 때마다 몇개의 validation data를 전달하여
                    # validation loss와 metrics을 monitoring한다
                    validation_data=(x_val, y_val))

# history.history 객체는 loss와 metric 값에 대한 학습 과정 동안의 기록을 가지고 있다.
print("\nhistory dict:", history.history)

# `evaluate`을 이용해 test data에 대해 모형을 평가
print("\n# test data에 대한 평가")
results = model.evaluate(x_test, y_test,
                        batch_size=128)
print('test loss, test acc: ', results)

# prediction을 생성 (확률값들 -- 마지막 layer의 output)
# 이때 새로운 데이터에 대하여 'predict'를 이용한다

print('\n# 3개의 sample에 대하여 predictions 생성')
predictions = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)
```
```python

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY0ODEzNjE5MF19
-->