---
title: "Custom modeling with Keras (3)"
excerpt: 모형의 적합!

author_profile: false

date: 2019-12-30 16:30:00 -0400
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
---
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()
```
일반적으로, `Layer` class를 내부 계산 블록으로, `Model` class를 학습의 대상인 외부 모형으로 정의한다. 예를 들어, ResNet50 모형에서, 여러개의 ResNet 블록을 `Layer`로 정의하고,  하나의 단일 `Model`을 이용해 전체 네트워크를 아우르는 모형을 적합한다.

`Model` class는 `Layer`와 거의 동일한 API를 갖지만, 몇가지 차이를 갖는다.
- built-in loops를 갖는다(`model.fit()`, `model.evaluate()`, `model.predict()`)
- `model.layers`를 이용해 내부 layer의 특성에 접근할 수 있다.
- saving and serialization APIs를 가진다.

쉽게 말해서, "Layer" class는 우리가 흔히 이야기하는 "layer"("convolution layer" or "recurrent layer")나 "block"("ResNet block" or "Inception block")의 의미를 갖는다. 반면에, "Model" class는 우리하 흔히 이야기하는 "모형"("deep learning model")이나 "네트워크"("deep neural network")의 의미를 갖는다.

예를 들어, 다음과 같이 `Model`에 대해 `fit()`을 이용해 training할 수 있고, `save_weights`를 이용해 저장할 수 있다.


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NjY1NTQ5MTUsLTc3MTgzMDg3MiwtMj
AyNzkwMTg1MF19
-->