# Custom modeling with Keras (2)

---
title: "Custom modeling with Keras (2)"
excerpt: Custom modeling with Keras 2편

date: 2019-12-30 16:30:00 -0400
categories: 

- tensorflow 2.0

tags:

- tensorflow 2.0

---

> 이 글은 다음 문서를 참조하고 있습니다!
[https://www.tensorflow.org/guide/keras/custom_layers_and_models](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session() # 간단한 초기화 방법(노트북 환경에서)
```
* What is ```__future__```? (coming soon!)

**layer는 재귀적으로 전진 방향 전파 학습을 하는 도중 손실함수 값을 수집한다!**

layer에서 `call` method는 손실 값을 저장하는 tensor를 생성할 수 있도록 해주어, 후에 training loop을 작성할 때 사용가능하도록 해준다.
→ `self.add_loss(value)`를 사용!

```python
class ActivityRegularizationLayer(layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate
    
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
```

이렇게 생성된 손실 값은(임의의 내부 layer의 손실 함수 값을 포함하여) `layer.losses`를 이용해 불러올 수 있다. 이러한 특성은 top-level layer에서의 모든 `__call__`의 시작에 초기화 된다. 이는 `layers.losses`가 항상 마지막 전진 방향 전파 학습의 손실 값만을 저장하기 위함이다.
```python
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activitiy_reg = ActivityRegularizationLayer(1e-2)
    
    def call(self, inputs):
        return self.activitiy_reg(inputs)
```
```python
layer = OuterLayer()
'''어떠한 layer도 call되지 않았으므로 손실 값이 없다'''
assert len(layer.losses) == 0
_ = layer(tf.zeros(1, 1))
'''layer가 1번 call되었으므로 손실 값은 1개'''
assert len(layer.losses) == 1
'''layer.losses는 각각의 __call__의 시작에서 초기화'''
_ = layer(tf.zeros(1, 1))
'''마지막으로 생성된 손실 값'''
assert len(layer.losses) == 1
```
추가로, `loss` 특성은 임의의 내부 layer에서 생성된 정규화 손실 값 또한 포함한다.
```python
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    
    def call(self, inputs):
        return self.dense(inputs)
```
```python
layer = OuterLayer()
_ = layer(tf.zeros((1, 1)))
print(layer.dense.kernel)
print(tf.reduce_sum(layer.dense.kernel) ** 2)
'''
이 값은 1e-3 * sum(layer.dense.kernel ** 2)와 같다
(이 손실은 kernel_regularizer에 의해 생성)
'''
print(layer.losses)
```
```python
<tf.Variable 'outer_layer_3/dense_1/kernel:0' shape=(1, 32) dtype=float32, numpy=
array([[-0.17390427, -0.33782747,  0.00282753, -0.3532169 ,  0.34208316,
        -0.37428847, -0.05844164,  0.01640856,  0.32005012,  0.3649932 ,
         0.35369265,  0.20181292,  0.23604548, -0.2578826 ,  0.09839004,
        -0.18697263,  0.0741716 , -0.06126347,  0.4143306 , -0.16958284,
         0.08949876,  0.2845322 , -0.26741046,  0.32063776,  0.15464622,
         0.37672937,  0.3461277 ,  0.00118405, -0.15776005, -0.14735147,
        -0.3484411 , -0.25038716]], dtype=float32)>
tf.Tensor(0.7283453, shape=(), dtype=float32)
[<tf.Tensor: id=119, shape=(), dtype=float32, numpy=0.0020875973>]
```

training loop에 응용: [https://www.tensorflow.org/guide/keras/train_and_evaluate](https://www.tensorflow.org/guide/keras/train_and_evaluate) (coming soon!)

**layers들에 대해 직렬화(serialization)(optional)**

* What is serialization?(coming soon!)

만약 함수형 모형의 일부분으로써 custom layer의 직렬화가 필요하다면, `get_config`  method를 사용하면 된다.

```python
class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units, ),
                                 initializer='random_normal',
                                 trainable=True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        return {'units': self.units}
```

```python
layer = Linear(64)
config = layer.get_config()
print(config)
'''
기존의 config를 이용해 새로운 layer를 생성 가능
'''
new_layer = Linear.from_config(config)
print(new_layer.get_config())
```
```python
{'units': 64}
{'units': 64}
```
기본적인 Layer class의 `__init__`은 몇몇 주요한 인자(name or dtype)를 입력받는다. 이러한 인자들을 실제로 이용해보자!
```python
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units, ),
                                 initalizer='random_normal',
                                 trainable=True)
        
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config
```

```python
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
print(new_layer.get_config())
```
```python
{'name': 'linear_7', 'trainable': True, 'dtype': 'float32', 'units': 64}
{'name': 'linear_7', 'trainable': True, 'dtype': 'float32', 'units': 64}
```
만약 보다 더 유연하게 layer의 config로부터 deserializing을 원한다면, `from_config` 방법을 이용하여 무효화 할 수 있다.

* serialization: [https://www.tensorflow.org/guide/keras/save_and_serialize](https://www.tensorflow.org/guide/keras/save_and_serialize) (coming soon!)

**call method의 특별한 training argument**

몇몇 특정한 layer(`BatchNormalization`, `Dropout`, ...)들은 training과 inference loop에서 다르게 작동해야하는 경우가 있다.
→ 이러한 경우에는 `call` method에서 training이라는 boolean 인자를 사용하면 된다! 이렇게 하면 built-in training or inference loops(e.g. `fit`)에서 특정한 layer를 목적에 맞게 사용할 수 있다.

```python
class CustomDropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout. self).__init__(**kwargs)
        self.rate = rate
    
    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwNDcyOTYwMzUsLTE2MTg2MjQ0MDVdfQ
==
-->