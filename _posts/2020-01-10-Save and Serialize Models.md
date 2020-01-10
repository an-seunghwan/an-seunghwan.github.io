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

```python
# 모형 저장
MODEL_PATH = r'C:\Users\dpelt\Desktop\Mayson\UOS_graduate\tensorflow2'
model.save(MODEL_PATH + '/model.h5')

# 위의 파일로부터 정확히 동일한 모형을 재생성
new_model = keras.models.load_model(MODEL_PATH + '/model.h5')

import numpy as np

# state가 유지되었는 지 확인한다
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
```
아무런 결과가 출력되지 않았으므로, opimizer의 state가 유지되었음을 알 수 있다: training을 멈춘 곳부터 다시 시작이 가능하다.

### SavedModel로 내보내기

전체 모형을 `SavedModel` 형식으로 저장할 수 있다. `SavedModel`은 TensroFlow 객체를 위한 독립적인 serialization format이고, TensorFlow serving과 TensorFlow implementation을 지원한다,

```python
# Export the model to a SavedModel
model.save('path_to_saved_model', save_format='tf')

# Recreate the exact same model
new_model = keras.models.load_model('path_to_saved_model')

# Check that the state is preserved
new_predictions = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

# Note that the optimizer state is preserved as well:
# you can resume training where you left off.
```
`SavedModel`은 다음을 포함한다:
- 모형의 가중치를 가지고 있는 TensorFlow checkpoint
- A `SavedModel` proto containing the underlying TensorFlow graph.

### 구조만 저장하기

때때로, 모형의 구조만에 관심이 있고, 가중치의 값이나 opimizer의 저장에는 관심이 없는 경우가 있다. 이러한 경우에, 모형의 "config"를 `get_config()` method를 이용해 불러올 수 있다. config는 training 과정에서 학습한 어떠한 정보도 없이 동일한 모형을 재생성할 수 있도록 해주는 Python dict이다.
```pytho

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTczNjE2MzkwMF19
-->