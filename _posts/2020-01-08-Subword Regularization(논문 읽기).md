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

source sentence $X$, target sentence $Y$가 주어졌을 때, $\mathbf{x}=(x_1, ..., x_M)$ 과 $\mathbf{y}=(y_1, ..., y_N)$ 을 각각에 해당하는 subword sequence라고 하자(이는 subword segmenter, BPE에 의해 segment됨). NMT는 번역 확률 $P(Y \vert X)=P(\mathbf{y} \vert \mathbf{x})$ 을 target language sequence 모형으로 모델링을 하는데, 이 모형은 target history $y_{<n}$ 과 source input sequence $\mathbf{x}$ 가 주어졌을 때 target subword $y_n$의 조건부 확률을 만든다:

$$
P(\mathbf{y} \vert \mathbf{x}; \theta)=\prod_{n=1}^N P(y_n \vert \mathbf{x}, y_{<n} ; \theta)
$$

이때 $\theta$는 모형의 모수 집합이다.

subword $y_n$을 예측하기 위한 모형으로는 RNN 구조를 생각할 수 있지만, subword regularization은 이러한 구조 이외에도 적용이 가능하다.

NMT는 standard MLE를 통해 학습이 된다. 즉, parallel corpus $D$가 주어졌을 때의 log likelihood $\mathcal{L}(\theta)$를 최대화한다.

$$
D = \{ (X^{(s)}, Y^{(s)}) \}_{s=1}^{\left\vert D \right\vert}  = \{ (\mathbf{x}^{(s)}, \mathbf{y}^{(s)}) \}_{s=1}^{\left\vert D \right\vert} 
$$

$$
\theta_{MLE} = \arg \max_{\theta} \mathcal{L}(\theta)
$$

$$
where,  \mathcal{L}(\theta) = \sum_{s=1}^{\left\vert D \right\vert} log P(\mathbf{y} \vert \mathbf{x} ; \theta)
$$

source와 target 문장 $X$와 $Y$가 각각 segmentation 확률 $P(\mathbf{x} \vert X)$와 $P(\mathbf{y} \vert Y)$ 를 통해 여러 subword sequences로 segment될 수 있다고 가정한다. subword regularization에서, parameter set $\theta$를 marginalized 가능도를 이용해 최적화 된다.

$$
\mathcal{L}_{marginal} (\theta) = \sum_{s=1}^{\left\vert D \right\vert} \mathbb{E}_{\mathbf{x} \sim P(\mathbf{x} \vert X^{(s)}), \mathbf{y} \sim P(\mathbf{y} \vert Y^{(s)})} [log P(\mathbf{y} \vert \mathbf{x} ; \theta)]
$$

위 식의 exact 최적화가 문장의 길이에 따라 가능한 segmentation의 종류가 기하급수적으로 증가하므로 실현가능하지 않다. 따라서 위의 식을 $P(\mathbf{x} \vert X)$와 $P(\mathbf{y} \vert Y)$ 의 각각을 통해 sample된 유한한 $k$개의 sequence로 근사한다. 근사 식은 아래와 같다.

$$
\mathcal{L}_{marginal} (\theta) \approx \frac{1}{k^2} \sum_{s=1}^{\left\vert D \right\vert} \sum_{i=1}^k \sum_{j=1}^k log P(\mathbf{y}_j \vert \mathbf{x}_i ; \theta)
$$

이때,
$$
\mathbf{x}_i \sim P(\mathbf{x} \vert X^{(s)}), \mathbf{y}_j \sim P(\mathbf{y} \vert Y^{(s)})
$$

간단함을 위해, 본 연구에서는 $k=1$로 하였다. 
NMT의 training을 위해 일반적으로 효율성을 위해 online training을 사용하는데, 이는 $D$의 작은 subset(mini-batch) 각각에 대해 parameter $\theta$를 반복적으로 최적화하는 방식이다. 만약 충분한 반복횟수를 사용한다면, online training의 data sampling을 통한 subword sampling은 $k=1$ 이더라도 좋은 근사 결과를 갖는다. 하지만, subword sequence는 각 parameter를 update하기 위해 on-the-fly 방식으로 sample 되었다는 것을 명심해야 한다.

### 2.2 Decoding

NMT의 decoding에서는, raw source 문장 $X$만을 가진다. decoding의 똑바른 접근은 best segmentation $P(\mathbf{x} \vert X)$를 최대화하는 $\mathbf{x}^{\ast}$로부터 번역을 하는 것이다. 즉, $\mathbf{x}^{\ast} = \arg \max_{\mathbf{x}} P(\mathbf{x} \vert X)$이다. 추가적으로, 여러개의 subword segmentation을 고려하기 위해 $P(\mathbf{x} \vert X)$의 $n$-best segmentation을 사용할 수 있다. 더 구체적으로, $n$-best segmentation $(\mathbf{x}_1, ..., \mathbf{x}_n)$가 주어졌을 때 아래의 score를 최대화하는 best translation $\mathbf{y}^{\ast}$를 선택할 수 있다.

$$
score(\mathbf{x}, \mathbf{y}) = log P(\mathbf{y} \vert \mathbf{x}) / \left\vert \mathbf{y} \right\vert
^{\lambda}
$$

이때, $\left\vert \mathbf{y} \right\vert$는 $\mathbf{y}$의 subword의 개수이고, $\lambda \in \mathbb{R}^{+}$는 짧은 문장을 penalize한다.

## 3. Subword segmentations with language model

### 3.1 Byte-Pair-Encoding

BPE의 단점은 다음과 같다. BPE는 greedy와 deterministic한 symbol 교체에 기반을 두고 있으므로, 확률과 함께 다양한 segmentation을 제공하지 못한다.

### 3.2 Unigram language model

unigram language model은 다음과 같은 가정을 한다. 각각의 subword는 독립적이고 연속적으로 발생하고, subword sequence $\mathbf{x} = (x_1, ..., x_M)$은 subword의 발생 확률 $p(x_i)$의 곱으로 형성된다.

$$
P(\mathbf{x}) = \prod_{i=1}^M p(x_i),
$$

$$
\forall i, x_i \in \mathcal{V}, \sum_{x \in \mathcal{V}} p(x) = 1
$$

이때 $\mathcal{V}$는 미리 결정된 단어 사전이다. input 문장 $X$에 대해 가장 가능성 높은 segmentation $\mathbf{x}^{\ast}$은 다음과 같이 주어진다.

$$
\mathbf{x}^{\ast} = \arg \max_{\mathbf{x} \in S(X)} P(\mathbf{x})
$$

이때 $S(X)$는 input 문장 $X$로부터 생성된 segmentation의 후보 집합이다.

## 논문 
Kudo, T. (2018). Subword regularization: Improving neural network translation models with multiple subword candidates. _arXiv preprint arXiv:1804.10959_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzU2NjA3MTA4LDE4MDA1NzQ5NzEsLTE4MT
I2NTUyNDMsMTM0OTI2MTk3OSwxMjQyMjUxNTU2LDQ0MDg0NjIy
OSwtNzgwMzA4MDUyLDEyOTc5NzgzNjAsNjY4OTIyMzA4LC0xNT
UxODI0MDg0LC0yMDUwODUwMzI2LC0yMjc3Mjg5NzYsLTI3NjYx
MDM1NywyMDk0Mjg3NTQ4XX0=
-->