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

U-Net을 이용한 semantic segmentation 작업에서, U-Net의 각 skip connection마다 penalty가 추가된 loss를 사용하고 싶은 경우를 가정해 보겠다!
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA0OTU0NzMwMywyMDk5OTkzMDA0LC0yMD
U3MzI0NDA1XX0=
-->