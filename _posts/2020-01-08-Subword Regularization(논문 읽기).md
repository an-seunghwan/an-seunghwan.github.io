---
title: "Subword Regularization"
excerpt: "Improving Neural Network Translation Models with Multiple Subword Candidates"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-08 15:00:00 -0000
categories: 
  - NLP
tags:
  - 
---

## 1. Introduction

단어 사전의 크기를 제한하는 것은 Unknown words의 개수를 증가시키고, 이는 open-vocabulary 환경의 번역을 매우 정확하지 않게 만든다. 

BPE는 문장을 unique한 subword 문장으로 encoding한다. 하지만, 하나의 문장은 동일한 단어사전이라도  다양한 subword 문장으로 표현될 수 있다. NMT의 training 과정에서, 다양한 segmentation 후보들은 모형이 간접적으로 단어들의 구성을 배우도록 하여 noise나 segmentation 오류에 robust하도록 만든다.

본 연구에서 제시하는 **subword regularization** 방법은, 다양한 subword segmentation을 사용하여 NMT 모형을 정확하고 robust하도록 만든다.

## 논문 
Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. _arXiv preprint arXiv:1804.10959_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTkyMjY5MTQ5MiwyMDk0Mjg3NTQ4XX0=
-->