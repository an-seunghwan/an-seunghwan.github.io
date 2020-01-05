---
title: "Layer Class in Keras"
excerpt: "Keras의 Layer Class"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-05 14:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---

Custom Model을 작성하다 보면 자연스럽게 custom layer를 작성해야 하는 순간이 많이 발생한다. 이러한 경우에 Base layer class인 `tf.keras.layers.Layer`을 subclassing해야하는데, class에 대한 개념이 잡혀있지 않으면 어떤 부분이 어떻게 작동하는지 이해하기 어려운 경우가 많다.

따라서 이번 게시글에서는 Keras의 `Layer` class를 이해할 정도의 python class에 대한 개념과 실제 예제를 통해 subclassing을 적용한 custom layer가 어떻게 구성되어 있는지 살펴보도록 하겠다.

> custom layer에 대한 자세한 내용은 게시글 [https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1)/](https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1)/)을 참고해 주세요!

## setup
```python
import tensorflow as tf
from tensorflow.keras import layers
```
가장 간단한 `Dense` layer를 예제로 사용해보자.

## example) Dense layer
```python
class Linear(layers.Layer): 
    
    def __init__(self, units=32): # self(객체 자신이 호출시 전달) 내부의 속성들을 초기화
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
```
### 1. naming convention
naming convention은 CamelCase이다.
CamelCase는 합성어 명칭에서 단어들이 합쳐질 때 단어의 첫 글자를 대문자로 표기하는 방식이다.

### 2. 변수 할당
```python
linear_layer = Linear(units=12) # 변수에 할당
type(linear_layer)
```
```
__main__.Linear
```
### 3. method
class내에 정의된 함수를 method라고 한다. 따라서 'Linear' class의 method는 `__init__`, `build`, `call` 3가지 이다(TensorFlow 2.0에서 권장하는 method 3가지).

* `__init__()`

Save configuration in member variables = 객체 내부의 속성들을 저장(초기화)한다는 것을 의미
```python
def __init__(self, units=32): # self(객체 자신이 호출시 전달) 내부의 속성들을 초기화
    super(Linear, self).__init__()
    self.units = units
```

* `build()`

inputs의 shape과 `dtype`이 일단 알려지면, `__call__`으로부터 단 한번 호출이 된다. 우선 `add_weight()`를 호출을 하고, 그 다음 super의 `build()`를 호출한다(이 것은 `self.build = True`으로 설정하므로, 첫 번째 `__call__`이 호출되기 전에 수동으로 `build()`를 호출하고 싶은 경우에 매우 유용하다).

특히, subclass implementer들을 위한 layer의 변수를 생성하는 method이다. **`Layer`나 `Model`에서 subclasses를 실행하는 사람들이 만약 layer instantiation과 layer call 사이에 state-creation(변수(가중치) 생성) 단계가 필요하다면 override할 수 있도록 만들어 주는 method이다.** 이 method는 일반적으로 `Layer` subclasses의 가중치를 생성하는데 쓰인다.

실제로 코드를 보면서 확인해보자.
```python
def build(self, input_shape): 
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units, ),
                             initializer='random_normal',
                             trainable=True)
```
```python
x = tf.ones((3, 3))
linear_layer = Linear(units=12)
y = linear_layer(x) 
print(y)
```


* `call()`

`build()`가 확실히 실행되고 난 뒤에 `__call__`에서 호출이 된다. 실제로 input tensor에 대해 layer에 적용되는 logic을 수행한다.
```python
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```
inputs와 가중치 `w`를 행렬곱을 한 뒤, `b`를 더해주는 logic이 적용되는 것을 볼 수 있다.

### 4. `self.`
`__init__`에서 `self.`으로 할당한 변수들은 모두 instance 속성! 따라서 units는 instance 속성이다.

### 5. Class Inheritance(클래스 상속)
* 기본 사용 방식
```
class child_class(parent_class):
    ...
    ...
```
이때 parent_class는 `tf.keras.layers.Layer`이고, child_class는 `Linear`이다. 이 `Linear`는 `tf.keras.layers.Layer`의 모든 속성과 method를 상속받으므로 Linear class 내에서 따로 정의할 필요가 없다.

* **Method overriding**

만약 parent_class의 method를 child_class에서 method를 재정의 한다면, parent_class의 method는 무시되고 child_class의 method만 실행된다.

* `super()`

`super()`를 이용하면, child_class 내에서 parent_class를 호출할 수 있다.

> 참고: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=stable
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE2MjUwMzg5LDEyNjA5NzcxNjEsMTEwNj
Q2MjI4MSw4MzM3ODUxMDUsLTIxMDYyMjg4NDVdfQ==
-->