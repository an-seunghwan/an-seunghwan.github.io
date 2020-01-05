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
Epoch 1/3
782/782 [==============================] - 5s 6ms/step - loss: 0.3416 - sparse_categorical_accuracy: 0.9031 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00
Epoch 2/3
782/782 [==============================] - 3s 3ms/step - loss: 0.1576 - sparse_categorical_accuracy: 0.9522 - val_loss: 0.2568 - val_sparse_categorical_accuracy: 0.9375
Epoch 3/3
782/782 [==============================] - 3s 3ms/step - loss: 0.1144 - sparse_categorical_accuracy: 0.9656 - val_loss: 0.1659 - val_sparse_categorical_accuracy: 0.9516
Out[40]: <tensorflow.python.keras.callbacks.History at 0x22e2f44a348>
```
이때 사용이 끝난 후에 validation dataset이 초기화 된다(따라서 모든 epoch마다 동일한 sample에 대하여 validation 과정이 진행된다.)

`validation_split`인자는 Dataset 객체를 사용하는 경우에는 지원되지 않는다. 왜냐하면 이 특징은 datasets의 sample에 대한 index 기능이 필요한데, 이는 Dataset API에서는 가능하지 않다.

### 다른 형식의 input

Numpy나 Tensorflow Datasets 이외에도, Padas dataframe이나, Python generator를 이용해 
batch 훈련이 가능하지만, 이는 권장하는 방법이 아니다.

### sample과 분류 항목에 대한 가중치 부여

`fit`을 사용할 때, sample이나 분류 항목에 대해 가중치를 부여하는 것이 가능하다.
- Numpy data를 이용할 때: `sample_weight`과 `class_weight`인자를 이용
- Datasets를 이용할 때: Dataset이 `(input_batch, target_batch, sample_weight_batch)`를 반환하게 함으로써 가능

"sample weights" array는 batch에서 각 sample이 전체 loss를 계산할 때 얼마나 많은 가중치를 갖는지를 명시한 array이다. 이는 보통 불균형한 분류 문제(imbalanced classification)에서 보통 사용한다(관측되기 어려운 분류 항목에 대해 더 많은 가중치를 부여하는 아이디어).
weights의 값으로 0과 1이 사용되는 경우에, 해당 array는 loss function에 대해 *mask*로써 사용될 수 있다(전체 loss에 대한 기여도를 없애는 것).

"class weight" dict는 동일한 개념의 좀 더 구체적인 예시이다: 분류 항목의 번호와 해당 분류 항목에 속하는 sample들의 비율을 mapping하는 dict이다. 예를 들어, 분류 항목 '0'이 분류 항목 '1'보다 2배로 적게 나타난다면, `class_weight={0: 1., 1: 0.5}`와 같이 사용하면 된다.

다음은 MNIST dataset 예제에서 분류 항목 #5의 정확한 분류에 대해 더 많은 중요도를 부여하기 위해 class weights이나 sample weights을 사용하는 경우이다.

```python
import numpy as np

# class weight 사용
class_weight = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
                # Set weight "2" for class "5",
                # making this class 2x more important
                5: 2.,
                6: 1., 7: 1., 8: 1., 9: 1.}
print('Fit with class weight')
model.fit(x_train, y_train,
          class_weight=class_weight,
          batch_size=64,
          epochs=4)

# sample weight 사용
sample_weight = np.ones(shape=len(y_train, ))
sample_weight[y_train == 5] = 2.
print('\nFit with sample weight')

model = get_compiled_model()
model.fit(x_train, y_train,
          sample_weight=sample_weight,
          batch_size=64,
          epochs=4)

# dataset 사용하는 경우
sample_weight = np.ones(shape=len(y_train, ))
sample_weight[y_train == 5] = 2.

# sample_weight을 포함하는 Dataset을 생성한다
# (tuple의 반환되는 3번째 원소를 sample weight로 지정)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))

# Shuffle and slice the dataset.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model = get_compiled_model()
model.fit(train_dataset, epochs=3)
```
```
Fit with class weight
Train on 50000 samples
Epoch 1/4
50000/50000 [==============================] - 3s 60us/sample - loss: 0.1002 - sparse_categorical_accuracy: 0.9727
Epoch 2/4
50000/50000 [==============================] - 3s 57us/sample - loss: 0.0828 - sparse_categorical_accuracy: 0.9769
Epoch 3/4
50000/50000 [==============================] - 4s 73us/sample - loss: 0.0701 - sparse_categorical_accuracy: 0.9800
Epoch 4/4
50000/50000 [==============================] - 4s 76us/sample - loss: 0.0615 - sparse_categorical_accuracy: 0.9820

Fit with sample weight
Train on 50000 samples
Epoch 1/4
50000/50000 [==============================] - 5s 96us/sample - loss: 0.3671 - sparse_categorical_accuracy: 0.9021
Epoch 2/4
50000/50000 [==============================] - 2s 46us/sample - loss: 0.1639 - sparse_categorical_accuracy: 0.9548
Epoch 3/4
50000/50000 [==============================] - 2s 46us/sample - loss: 0.1181 - sparse_categorical_accuracy: 0.9675
Epoch 4/4
50000/50000 [==============================] - 2s 47us/sample - loss: 0.0949 - sparse_categorical_accuracy: 0.9734
Epoch 1/3
782/782 [==============================] - 5s 6ms/step - loss: 0.3671 - sparse_categorical_accuracy: 0.9027
Epoch 2/3
782/782 [==============================] - 3s 4ms/step - loss: 0.1703 - sparse_categorical_accuracy: 0.9521
Epoch 3/3
782/782 [==============================] - 3s 4ms/step - loss: 0.1241 - sparse_categorical_accuracy: 0.9657
Out[41]: <tensorflow.python.keras.callbacks.History at 0x22e352eed08>
```
### 다중 input과 output을 갖는 모형

다음의 모형은 `(32, 32, 3)`의 shape(`(height, width, channels)`)을 갖는 이미지 input과 `(None, 10)`의 shape(`(timesteps, features)`)을 갖는 timeseries input을 다중 input으로 입력받는 모형이다. 이 모형은 이러한 inputs로부터 2개의 output을 가진다: "score"(shape `(1,)`)과 5개의 분류 항목에 대한 확률 분포(shape `(5,)`)

```python
from tensorflow import keras
from tensorflow.keras import layers

image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, activation='softmax', name='class_output')(x)

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[score_output, class_output])
```
list 형식으로 loss function을 전달하여 각각의 output별로 별도의 loss function 지정이 가능하다.
```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),
          keras.losses.CategoricalCrossentropy()])
```
metric도 마찬가지 방법으로 적용 가능하다(metric은 항상 list 형식이므로 이중 list로 입력).
```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),
          keras.losses.CategoricalCrossentropy()],
    metrics=[[keras.metrics.MeanAbsolutePercentageError(),
              keras.metrics.MeanAbsoluteError()],
             [keras.metrics.CategoricalAccuracy()]])
```
dict를 이용해 output별로 별도의 loss와 metric을 지정 가능하다(이때 dict의 key는 반드시 model 정의에서 지정된 각 output의 `name`을 이용). 이는 output이 2개 이상일 경우 TensorFlow 2.0에서 권장되는 방식이다.

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy()},
    metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                              keras.metrics.MeanAbsoluteError()],
             'class_output': [keras.metrics.CategoricalAccuracy()]})
```
각각의 loss에 대해서 서로 다른 가중치를 부여할 수 있다(`loss_weights` 인자를 이용).
```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy()},
    metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                              keras.metrics.MeanAbsoluteError()],
             'class_output': [keras.metrics.CategoricalAccuracy()]},
    loss_weights={'score_output': 2., 'class_output': 1.})
```
특정한 output이 prediction을 위한 것이지 training을 위한 것이 아니라면, loss를 굳이 계산하지 않아도 된다.
```python
# List loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()]) # 첫 번째 output의 loss function은 None

# Or dict loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'class_output': keras.losses.CategoricalCrossentropy()})
```
`compile`과 유사하게 `fit`에서도 다중 input과 output을 지정할 수 있다: Numpy array list 혹은 Numpy array와 각 input과 output의 이름을 mapping한 dict를 이용

```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),
          keras.losses.CategoricalCrossentropy()])

# Generate dummy Numpy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets], # keras.Model의 inputs, outputs와 순서를 동일하게 맞춘다.
          batch_size=32,
          epochs=3)

# Alternatively, fit on dicts
model.fit({'img_input': img_data, 'ts_input': ts_data},
          {'score_output': score_targets, 'class_output': class_targets},
          batch_size=32,
          epochs=3)
```
Dataset을 사용하는 경우: Numpy array로 이루어진 dict로 구성된 tuple을 사용한다.
```python
train_dataset = tf.data.Dataset.from_tensor_slices(({'img_input': img_data, 'ts_input': ts_data},
                                                    {'score_output': score_targets, 'class_output': class_targets}))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model.fit(train_dataset, epoch=3)
```

### callbacks 사용하기

Keras의 callback은 training 도중에 서로 다른 시점에서 호출될 수 있는 객체이고(epoch의 시작, 또는 끝, batch의 종료 등)
다음과 같이 실행될 수 있는 특징이 있다.

- training 중에 서로 다른 시점에서 validation이 가능하다(epoch당 실행하는 built-in validation 외에)
- 규칙적인 주기 또는 특정한 정확도 기준치를 초과하는 경우에 checkpointing
- training이 안정 수준이 되어(plateauing) 보이면 learning rate을 수정한다
- training이 종료되거나 특정한 성능 기준치를 초과하면 이메일을 보내거나 메세지 알림을 보낸다.
- 기타 등등

callback은 `fit`에 list 형식으로 전달하면 된다.


<!--stackedit_data:
eyJoaXN0b3J5IjpbNTU3MTM5MTc4XX0=
-->