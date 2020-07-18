---
title: "Generating Sentences from a Continuous Space(작성중)"
excerpt: "VAE + RNN"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-07-18 20:00:00 -0000
categories: 
  - VAE
  - NLP
tags:
  - tensorflow
  - keras
  - RNN
---

> [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349) 논문에 대한 간단한 리뷰와 tensorflow 코드입니다.
>  
>  정확한 내용과 수식들은 논문을 참조해주시기 바랍니다. 

## 1. 구조

((그림 첨부))

논문에서 제안하는 모형은 생각보다 매우 간단하다. 기존의 sequence to sequence의 모형에 latent space를 도입한 것이 전부이다. 

<!--stackedit_data:
eyJoaXN0b3J5IjpbMzYxMzI4NTE5LC0xODY2ODg4MzZdfQ==
-->