---
title: "Generating Sentences from a Continuous Space 2편(작성중)"
excerpt: "tensorflow로 구현해보자!"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-07-21 20:00:00 -0000
categories: 
  - VAE
  - NLP
tags:
  - tensorflow
  - keras
  - RNN
---

> [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) 논문에 대한 간단한 리뷰와 tensorflow 코드입니다. 
>  본 포스팅은 위의 내용에 대한 2편(tensorflow 코드 작성) 입니다.
>  정확한 내용과 수식들은 논문을 참조해주시기 바랍니다. 

## 1. 환경

모델명:  iMac Pro
모델 식별자:  iMacPro1,1
프로세서 이름:  8-Core Intel Xeon W
프로세서 속도:  3.2 GHz
프로세서 개수:  1
총 코어 개수:  8

메모리:  32 GB

## 2. setting

<!--stackedit_data:
eyJoaXN0b3J5IjpbNzU2MTMzNTk1LDYzMjk2OTczOF19
-->