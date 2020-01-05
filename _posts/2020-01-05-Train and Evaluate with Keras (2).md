---
title: "Train and Evaluate with Keras (2)"
excerpt: "Part 1, 2부(multi-inputs and outputs, callback, checkpoint)"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-05 17:00:00 -0000
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

## setup
지난 Part 1, 1부에서 필요한 코드를 미리 실행시켜 놓는다.
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def get_uncompiled_model():
    inputs = keras.Input(shape=(784, ), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='prediction')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
    return model
```
## Part 1. built-in training and evaluation loop 사용(continued)

### tf.data Datasets을 이용해 Training & Evaluation

사용하는 데이터가 tf.data Dataset인 경우를 생각하자. tf.data API는 빠르고 확장성이 있는 데이터 loading과 전처리를 위한 TensorFlow 2.0의 도구이다. 
* Datasets에 대한 더 자세한 내용은 추후에 tf.data에 관한 가이드에서 다루겠습니다(coming soon!).

Dataset을 `fit()`, `evaluate()`, `predict()` method를 이용해 바로 입력할 수 있다.

```python
model = get_compiled_model()

# 우선 Dataset instance를 생성한다.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

# 이미 dataset이 batch로 나누어져 있으므로, 'batch_size'인자가 필요없다.
model.fit(train_dataset, epochs=3)

print('\n# Evaluate')
model.evaluate(test_dataset)
```
```
Epoch 1/3
782/782 [==============================] - 7s 9ms/step - loss: 0.3304 - sparse_categorical_accuracy: 0.9070
Epoch 2/3
782/782 [==============================] - 4s 5ms/step - loss: 0.1511 - sparse_categorical_accuracy: 0.9552
Epoch 3/3
782/782 [==============================] - 3s 4ms/step - loss: 0.1105 - sparse_categorical_accuracy: 0.9671

# Evaluate
157/157 [==============================] - 1s 4ms/step - loss: 0.1269 - sparse_categorical_accuracy: 0.9623
Out[37]: [0.12690303274534145, 0.9623]
```
Dataset이 각 epoch가 끝날 때 마다 초기화되므로, 다음 epoch에서 재사용이 가능하다.

만약 이 Dataset으로부터 특정 개수의 batch만을 학습하고자 한다면, `steps_per_epoch` 인자를 이용하면 된다. 이는 모형이 이 Dataset을 이용해 다음 epoch로 넘어가기 전에 해당 epoch에서 몇번의 학습 step을 진행할 지를 명시한다.

만약 이를 사용한다면, dataset은 각 epoch의 마지막에서 초기화되지 않고, 다음 batch들을 단순히 계속 이용한다. dataset은 결국 모든 데이터를 다 사용하게 될 것이다(무한 loop의 dataset을 제외한다면).

```python
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# 각 epoch에서 100개의 batch만을 사용(즉, 64 * 100개의 sample을 이용)
model.fit(train_dataset.take(100), epochs=3)
```
```
Epoch 1/3
100/100 [==============================] - 2s 16ms/step - loss: 0.7716 - sparse_categorical_accuracy: 0.7997
Epoch 2/3
100/100 [==============================] - 1s 8ms/step - loss: 0.3162 - sparse_categorical_accuracy: 0.9136
Epoch 3/3
100/100 [==============================] - 1s 8ms/step - loss: 0.2439 - sparse_categorical_accuracy: 0.9312
Out[38]: <tensorflow.python.keras.callbacks.History at 0x22e30fdb3c8>
```
validation dataset에서도 유사하게 적용이 가능하다.
```python
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(train_dataset, epochs=3,
          # `validation_steps=10` argument: 
          # validation dataset의 첫 번째 10개의 batch만을 이용해 validation 과정을 진행
          validation_data=val_dataset, validation_steps=10)
```
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc5ODAyNjU2XX0=
-->