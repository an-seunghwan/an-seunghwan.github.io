---
title: "Save and Serialize Models"
excerpt: "모형의 저장과 직렬화"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-10 21:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---
> 이 글은 다음 문서를 참조하고 있습니다!
>[https://www.tensorflow.org/guide/keras/save_and_serialize](https://www.tensorflow.org/guide/keras/save_and_serialize)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.
> 비교적 간단한 내용이나 코드와 같은 경우에는 번역 없이 생략하니 꼭 원문을 확인해주시면 감사하겠습니다.

## setup
```python
import tensorflow as tf
from pprint import pprint
```
## Part 1. Sequential model과 Functional model의 저장

```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
model.summary()
```
```
Model: "3_layer_mlp"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
digits (InputLayer)          [(None, 784)]             0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                50240     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
predictions (Dense)          (None, 10)                650       
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
```
위의 모형을 저장할 가중치의 값과 optimizer의 state을 만들기 위해 학습을 시킨다.
물론, 학습을 하지 않고도 저장하는 것은 가능하다.

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop())
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=1)

predictions = model.predict(x_test)
```
```
Train on 60000 samples
60000/60000 [==============================] - 4s 59us/sample - loss: 0.3110
```
### 전체 모형을 저장

Functional API를 이용한 모형을 하나의 파일로 저장할 수 있다. 모형을 적합할 때 사용한 코드에 전혀 접근하지 않고도 나중에 동일한 모형을 재생성 할 수 있다.

파일은 다음과 같은 내용을 포함한다:
- 모형의 구조
- 모형의 가중치 값(training 동안에 학습)
- 모형의 training config(`compile`에 전달된 설정)
- optimizer와 이의 state(이는 나중에 training을 멈춘 곳부터 다시 시작할 수 있도록 한다)

<!--stackedit_data:
eyJoaXN0b3J5IjpbNTE5MjYxMTI0XX0=
-->