---
title: "Custom modeling with Keras (1)"
excerpt: 기초적인 custom 모형 제작 방법
toc: true
toc_sticky: true

author_profile: false

date: 2019-12-29 16:30:00 -0400
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

다음의 코드는 densely-connected layer이고 `w`와 `b`라는 

## The Layer class

### Layers encapsulate a state (weights) and some computation

The main data structure you'll work with is the  `Layer`. A layer encapsulates both a state (the layer's "weights") and a transformation from inputs to outputs (a "call", the layer's forward pass).

Here's a densely-connected layer. It has a state: the variables  `w`  and  `b`.

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

## `add_weight` method
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
```python
class Compute_Sum(layers.Layer):
    def __init__(self, input_dim):
        super(Compute_Sum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim, )),
                                 trainable=False)
    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0)) # column-wise
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
```
weights: 1
non-trainable weights: 1
trainable_weights: []
```
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
```python
#The __call__ method of your layer will automatically run build the first time it is called. 
#You now have a layer that's lazy and easy to use:

x = tf.ones((3, 3))        
linear_layer = Linear(units=12)  # At instantiation, we don't know on what inputs this is going to get called
y = linear_layer(x)  # The layer's weights are created dynamically the first time the layer is called
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
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
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
eyJoaXN0b3J5IjpbMTE4ODgwMDc4MCwxNjMzMzc0MDU0XX0=
-->