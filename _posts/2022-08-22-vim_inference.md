---
title: "Inference on Variable Importance Measure"
excerpt: "Non-zero Null Hypothesis하에서의 통계적 추론"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2022-08-22 20:00:00 -0000
categories: 
  - VIM
tags:
  - pytorch
  - statistical inference
---

> 참조논문
> [Demystifying statistical learning based on efficient influence functions](https://arxiv.org/abs/2107.00681)
> [Semiparametric doubly robust targeted double machine learning: a review](https://arxiv.org/abs/2203.06469)
> [A General Framework for Inference on Algorithm-Agnostic Variable Importance](

## Real NVP

Normalizing Flow를 이용한 분포 추정에서, 널리 사용되는 방법 중 하나인 Real NVP를 tensorflow와 keras로 구현해보겠다. 해당 논문에 관한 이론적인 내용은 기회가 된다면 추후 포스팅 할 예정!

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxOTczNTA1MV19
-->