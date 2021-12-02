---
title: "Custom loss와 model.fit()을 같이 사용해보자!"
excerpt: ""
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-12-02 20:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
  - custom modeling
---

> Tensorflow 공식 홈페이지의 [https://www.tensorflow.org/tutorials/images/segmentation?hl=ko](https://www.tensorflow.org/tutorials/images/segmentation?hl=ko) 코드를 기반으로 작성하였습니다.

## Custom loss and `model.fit()`

신경망 모형의 중간 layer의 output에 penalty가 포함된 custom loss를 이용할 경우, 일반적인 `model.fit()`을 이용해서 모형을 적합하기는 쉽지 않다. 왜냐하면 `model.fit()`에 사용되는 손실 함수는 모형의 output 1개에 대한 loss 만을 계산하기 때문이다...

따라서, 이번 포스팅에서는 `.add_loss()`를 이용하여 penalty가 포함된 custom loss를 이용해 `model.fit()`을 이용해 모형을 적합하는 방법을 소개해보도록 하겠다!

### 0. import 

```python
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
```

```
TensorFlow version: 2.4.0 
Eager Execution Mode: True 
available GPU: [] 
========================================== 
[name: "/device:CPU:0" 
device_type: "CPU" 
memory_limit: 268435456 
locality { } 
incarnation: 10055035958512198800 
]
```

### 1. load MNIST dataset

```python
(x_train, y_train), (x_test, y_test) =  K.datasets.cifar10.load_data()
x_train =  x_train.astype('float32')  /  255.
x_test =  x_test.astype('float32')  /  255.
y_train_onehot =  to_categorical(y_train, num_classes=10)
y_test_onehot =  to_categorical(y_test, num_classes=10)
```

### 2. model 정의

```python
#%%
input_layer = layers.Input(x_train.shape[1:])
conv1 = layers.Conv2D(8, 4, 2, padding='same', activation='relu')
conv2 = layers.Conv2D(16, 4, 2, padding='same', activation='relu')
conv3 = layers.Conv2D(32, 4, 2, padding='same', activation='relu')
output_layer = layers.Dense(10, activation='softmax')

h1 = conv1(input_layer)
h2 = conv2(h1)
h3 = conv3(h2)
output = output_layer(layers.GlobalAveragePooling2D()(h3))

model = tf.keras.Model(inputs=input_layer, outputs=output)
model.summary()
```

### 3. penalty 추가

이제 신경망 모형의 중간 결과인 `h1`, `h2`, `h3`에 L2 norm 정규화 penalty를 추가하고 싶다고 해보자. 그러면 다음과 같이 `.add_loss()`를 이용해 원래의 loss에 penalty를 더해줄 수 있다!

```python
#%%
reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(h1), axis=[1, 2, 3]))
reg_loss += tf.reduce_mean(tf.reduce_sum(tf.square(h2), axis=[1, 2, 3]))
reg_loss += tf.reduce_mean(tf.reduce_sum(tf.square(h3), axis=[1, 2, 3]))
lambda_ = 0.1
model.add_loss(lambda_ * reg_loss)
```

### 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTgyMzcwMjA0MiwtMTY0MjE4NDk4NSwyMD
Q5NTQ3MzAzLDIwOTk5OTMwMDQsLTIwNTczMjQ0MDVdfQ==
-->