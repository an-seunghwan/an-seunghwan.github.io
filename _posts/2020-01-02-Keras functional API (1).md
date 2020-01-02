---
title: "Keras functional API (1)"
excerpt: Keras functional API 1부
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-02 14:25:00 -0400
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---
> 이 글은 다음 문서를 참조하고 있습니다!
>[https://www.tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.
> 비교적 간단한 내용이나 코드와 같은 경우에는 번역 없이 생략하니 꼭 원문을 확인해주시면 감사하겠습니다.

### setup
```python
import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
```
```
2.0.0
True
```
## 도입부

모형을 만들기 위해 우리는 흔히 `keras.Sequential()`을 사용한다. Functional API는 `Sequential`보다 더 유연한 모형을 만들 수 있게 해준다: 비선형 위상과, 공유 layer, 그리고 다중 input과 output을 다룰 수 있다.

이는 딥 러닝이 DAG에 아이디어를 기반하고 있기 때문이다. Functional API는 **layer의 graph를 구축**하는 도구이다.

다음의 모형을 생각하자.
```
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: probability distribution over 10 classes)
```

이는 단순한 3개 layer의 그래프이다.

이 모형을 Functional API를 이용해 만들기 위해서는, input node 부터 만드는 것이 필요하다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDMyMjg3NjIwLC0xMzIwODQxNjk5XX0=
-->