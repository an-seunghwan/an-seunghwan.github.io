---
title: "Custom modeling with Keras (1)"
excerpt: 기초적인 custom 모형 제작 방법
toc: true
toc_sticky: true

author_profile: false

date: 2019-12-29 16:30:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---

> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/custom_layers_and_models](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()  # notebook의 초기화
```

##  Layer class
** Layer는 state(weights)와 몇몇 계산을 캡슐화한다.
주로 다루게 될 주요 data structure는 `Layer`이다. Layer는 state (the layer's "weights")와 input을 output으로 변환("call", layer의 전진 학습) 과정을 캡슐화한다.

다음의 코드는 densely-connected layer이고 `w`와 `b` 변수라는 2개의 상태를 갖는다.

## Dense layer
```python
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                                  dtype='float32'),
                                                  trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                                  # 위와 같이 shape을 적용하면 column에 더하기 아님
                                                  # 각 units에 값이 더해진다고 생각!
                                                  dtype='float32'),
                                                  trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

weights `w`와 `b`는 layer의 attribute로써 자동으로 추적이 가능하다.

```python
x = tf.ones((3,3))
linear_layer = Linear(5, 3)
y = linear_layer(x)
print(linear_layer.b)
print(y)
print(tf.matmul(x, linear_layer.w))
print(tf.matmul(x, linear_layer.w) + linear_layer.b)
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```
```
<tf.Variable 'Variable:0' shape=(5,) dtype=float32, numpy=array([0., 0., 0., 0., 0.], dtype=float32)>
tf.Tensor(
[[-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]], shape=(3, 5), dtype=float32)
tf.Tensor(
[[-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]], shape=(3, 5), dtype=float32)
tf.Tensor(
[[-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]
 [-0.19611591 -0.00809563  0.1501273   0.13595319 -0.16803369]], shape=(3, 5), dtype=float32)
 ```

### `add_weight` method
```python
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(units, ),
                                 initializer='zeros',
                                 trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```
```python
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
print('trainable_weights:', linear_layer.trainable_weights)
```
```
tf.Tensor(
[[-0.10722479  0.01107529  0.1577724   0.08675833]
 [-0.10722479  0.01107529  0.1577724   0.08675833]], shape=(2, 4), dtype=float32)
trainable_weights: [<tf.Variable 'Variable:0' shape=(2, 4) dtype=float32, numpy=
array([[-0.06456654,  0.05076444,  0.13045819, -0.01007326],
       [-0.04265825, -0.03968915,  0.02731422,  0.09683159]],
      dtype=float32)>, <tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
```

## Layer는 학습이 불가능한 weights를 가질 수 있다.

학습 가능한 weights이외에, 학습 불가능한 weights를 layer에 추가할 수 있다. 이러한 weights는 학습에서 역전파 과정 동안에 고려되지 않는다. 다음은 예제 코드이다.

```python
class Compute_Sum(layers.Layer):
    def __init__(self, input_dim):
        super(Compute_Sum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim, )),
                                 trainable=False)
    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0)) # column-wise(axis=0)
        # 지금까지 입력된 값들을 total에 상수처럼 누적하여 저장
        return self.total
```
```python
x = tf.ones((2, 2))
my_sum = Compute_Sum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
```
```
[2. 2.]
[4. 4.]
```
```python
print('weights:', len(my_sum.weights))
print('non-trainable weights:', len(my_sum.non_trainable_weights))
# It's not included in the trainable weights:
print('trainable_weights:', my_sum.trainable_weights)
```

이러한 학습 불가능한 weights는 layer의 weights attribute로써 인식된다.

```
weights: 1
non-trainable weights: 1
trainable_weights: []
```

## 최고의 연습: input의 shape이 알려지기 전에 weight를 생성하기

앞의 logistic regression 예제는, `Linear` layer는 `input_dim` 인자를 받아 이를 `__init__`의 `w`와 `b` weights의 shape을 계산한다.

많은 경우에, input의 크기를 미리 알 수 없는 경우가 있고, layer를 만든 이후에 이러한 input 값이 알려지면 weights를 생성하고 싶을 수 있다.

Keras API에서는, `build(inputs_shape)` method를 이용해 다음과  같이 weights를 이후에 생성할 수 있다.

```python
class Linear(layers.Layer):
    def __init__(self, units=32):
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

`__call__` method는 첫 번째 호출이 되는 시점에 자동으로 `build`를 실행시킨다.

```python
x = tf.ones((3, 3))        
linear_layer = Linear(units=12) # 객체를 할당하는 시점에, 어떠한 input이 사용될 지 모른다.
y = linear_layer(x) # layer의 weights는 동적으로 처음으로 호출되는 시점에 생성된다.
print(y)
```
```
tf.Tensor(
[[-0.02907148  0.11085058 -0.14688957  0.07985388  0.11514413  0.09643906
  -0.04974959  0.13078684 -0.12947026  0.22152902 -0.10075628  0.10019987]
 [-0.02907148  0.11085058 -0.14688957  0.07985388  0.11514413  0.09643906
  -0.04974959  0.13078684 -0.12947026  0.22152902 -0.10075628  0.10019987]
 [-0.02907148  0.11085058 -0.14688957  0.07985388  0.11514413  0.09643906
  -0.04974959  0.13078684 -0.12947026  0.22152902 -0.10075628  0.10019987]], shape=(3, 12), dtype=float32)
```

## Layer는 재귀적으로 구성이 가능하다.

만약 어떤 Layer instance를 다른 Layer의 attribute으로 설정하면, 바깥 layer가 내부 layer의 weights를 추적한다.

이러한 sublayer는 `__init__` method 내부에 생성하면 된다(왜냐하면 sublayer는 일반적으로 `build` method를 가지므로, 바깥 layer가 생성될 때 같이 생성된다).

```python
class MLPBlock(layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)
    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)
```
```python
mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64))) # 'mlp'의 첫 번째 호출이고 이때 weights들이 생성된다.
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))
print(y)
print(mlp.linear_1.w.shape)
print(mlp.linear_2.w.shape)
print(mlp.linear_3.w.shape)
```
```
weights: 6
trainable weights: 6
tf.Tensor(
[[-0.01152304]
 [-0.01152304]
 [-0.01152304]], shape=(3, 1), dtype=float32)
(64, 32)
(32, 32)
(32, 1)
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU4NTk2ODQ4LC0xNjAyMDI4NzgwLDE2Mz
MzNzQwNTRdfQ==
-->