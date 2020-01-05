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

따라서 이번 게시글에서는  

> 참고: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=stable
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk5NzA1OTUzMF19
-->