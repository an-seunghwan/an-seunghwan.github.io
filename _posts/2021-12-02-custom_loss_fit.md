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

신경망 모형의 중간 layer의 output에 penalty가 포함된 custom loss를 이용할 경우, 일반적인 `model.fit()`을 이용해서 모형을 적합하기는 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTMyOTU5NDQwOSwyMDQ5NTQ3MzAzLDIwOT
k5OTMwMDQsLTIwNTczMjQ0MDVdfQ==
-->