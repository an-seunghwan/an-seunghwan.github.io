---
title: "Subword Regularization"
excerpt: "Improving Neural Network Translation Models with Multiple Subword Candidates"
toc: true
toc_sticky: true

author_profile: false
use_math: true

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

본 연구의 2가지 부가 기여:
* NMT 구조와 무관하게 on-the-fly data sampling을 통해 다양한 segmentation 후보들을 고려하는 방식이므로, NMT 모형의 구조 변화 없이 subword regularization을 적용할 수 있다.
* segmentation의 확률을 제공하는 언어 모형에 기반하므로, 실제 데이터의 segmentation 과정 동안에 생성된 noise와 모형이 경쟁할 수 있도록 한다.

## 2. Neural Machine Translation with multiple subword segmentations

### 2.1 NMT training with on-the-fly subword sampling

source sentence $X$, target sentence $Y$가 주어졌을 때, $\mathbf = (x$

$$
\lim_{x\to 0}{\frac{e^x-1}{2x}}
\overset{\left[\frac{0}{0}\right]}{\underset{\mathrm{H}}{=}}
\lim_{x\to 0}{\frac{e^x}{2}}={\frac{1}{2}}
$$

ddd

## 논문 
Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. _arXiv preprint arXiv:1804.10959_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NTE2NDEyNDIsLTE1NTE4MjQwODQsLT
IwNTA4NTAzMjYsLTIyNzcyODk3NiwtMjc2NjEwMzU3LDIwOTQy
ODc1NDhdfQ==
-->