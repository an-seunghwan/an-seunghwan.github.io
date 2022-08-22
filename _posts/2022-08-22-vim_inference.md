---
title: "Inference on Variable Imp (Non-zero Null Hypothesis)"
excerpt: "Real NVP를 tensorflow로 구현하기"
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
---

> Keras 공식 홈페이지의 [https://keras.io/examples/generative/real_nvp/](https://keras.io/examples/generative/real_nvp/)의 코드를 기반으로 작성되었습니다.
> Real NVP 논문: [DENSITY ESTIMATION USING REAL NVP](https://arxiv.org/pdf/1605.08803.pdf)

## Real NVP

Normalizing Flow를 이용한 분포 추정에서, 널리 사용되는 방법 중 하나인 Real NVP를 tensorflow와 keras로 구현해보겠다. 해당 논문에 관한 이론적인 내용은 기회가 된다면 추후 포스팅 할 예정!

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgwNDU1OTAxNF19
-->