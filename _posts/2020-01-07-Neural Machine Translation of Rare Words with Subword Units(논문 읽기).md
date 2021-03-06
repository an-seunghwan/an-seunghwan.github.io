---
title: "Neural Machine Translation of Rare Words with Subword Units(논문 읽기)"
excerpt: "Byte Pair Encoding을 활용한 OOV 해결"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-07 15:00:00 -0000
categories: 
  - NLP
tags:
  - 논문 읽기
  - OOV
  - tokenizer
  - 전처리
---
	톺아보기 시리즈 (톺아보다: 샅샅이 틈이 있는 곳마다 모조리 더듬어 뒤지면서 살피다)

> 이 포스팅은 OOV(out-of-vocabulary) 문제의 해결과 관련한 내용, 논문, tokenizer 등을 다루는 [OOV 톺아보기 시리즈](https://an-seunghwan.github.io/oov-top-a-bogi/)의 2편입니다.

지난 포스팅 [Byte Pair Encoding (1)](https://an-seunghwan.github.io/nlp/Byte-Pair-Encoding-(1))에서 Byte Pair Encoding에 대해 간략히 소개하였습니다. 이번 글에서는 이 알고리즘이 제시된 논문을 읽고 간략히 요약하여 Byte Pair Encoding에 기반한 Neural Machine Translation에 대해 알아보도록 하겠습니다.

논문의 내용을 전부 요약하는 것이 아닌 핵심만을 다루도록 하겠습니다.

## 1. Introduction

빈도가 높지 않은(자주 출현하지 않는) 단어들에 대한 번역은 open problem이다. 특히 교착어나 합성어를 통해 단어가 생산될 수 있는 언어에 대해서, 번역 모형은 단어 수준보다 더 아래 수준의 메카니즘을 필요로 한다.

이 논문의 가장 주된 목적은 NMT network에서, 희귀 단어에 대한 back-off 모형을 필요로 하지 않는 open-vocabulary 번역 모형을 만드는 것이다.

이 논문의 2가지 주된 기여
* open-vocabulary NMT가 단어들을 subword units를 통해 encoding이 가능하다는 보였다. 
* 단어 segmentation를 위한 압축 알고리즘으로써 BPE를 사용하였다. BPE는 open vocabulary를 고정된 크기의 다양한 길이의 character sequence를 통해 표현할 수 있게 해주었다.

## 3. Subword Translation

이 논문의 가장 주된 motivation은 몇몇 단어의 번역이 그 단어들이 처음 보는 단어라도, 형태소나 음소와 같은 subword units의 번역을 기반으로한 좋은 번역기를 통해서 명백하게 가능하다는 것이다. 이러한 명백한 번역이 가능한 단어의 종류는 다음과 같다:
* 이름 
* 같은 기원을 가지는 단어
* 형태학적으로 복잡한 단어
	- 합성, 첨가 등의 방식을 통해 단어가 생성

### 3.2 Byte Pair Encoding(BPE)

BPE는 반복적으로 가장 높은 빈도수의 byte의 pair를 하나의 사용되지 않은 byte로 교체하는 데이터 압축 기술이다. 여기서는 byte의 pair를 character sequence로 생각하여 word wegmentation에 적용한다.

1. 초기 단어 사전을 구성한다. 이때 각각의 단어를 character(음절 또는 symbol)의 sequence로 구성하고 마지막에 특수 문자 '·'를 넣어준다. 이 특수문자는 번역 후에 원래의 tokenization을 복원할 수 있도록 해준다.
2. 반복적으로 모든 symbol pair의 빈도수를 세고 이를 하나의 symbol로 교체한다. 예를 들어, ('A', 'B')가 가장 높은 빈도의 pair라면 이를 'AB'로 교체한다. 즉, 이러한 merge 과정은 character n-gram으로 나타나지는 새로운 symbol을 생성한다.
	- 가장 높은 빈도의 character n-gram(또는 전체 하나의 단어)은 결국 하나의 symbol로 합쳐지므로, BPE는 shortlist가 필요 없다.
	- 최종 symbol 단어 사전의 크기는 초기 단어 사전의 크기에 merge 과정의 횟수만큼을 더한 것과 동일하다(merge 과정의 횟수는 조절할 수 있는 parameter).

효율성을 위해, 단어 범위 바깥의 pair는 고려하지 않는다. 또한 알고리즘은 각 단어당 빈도수로 matching되어 있는 dict로부터 학습된다.

2가지 방법의 BPE가 있다

1. 2개의 독립적인 encoding을 학습: source 언어의 단어 사전 + target 언어의 단어 사전
	- 장점: text와 단어 사전의 크기가 compact하다.
	- 장점: 각각의 subword unit이 각 언어의 training text에 존재한다는 강한 보장이 있다.
	- 단점: 각각의 언어에서 동일한 단어가 다른 방식으로 segmented 될 수 있어 neural model이 subword units를 mapping하기 어려워진다.
2. 2개 언어 단어 사전의 union을 이용해 encoding을 학습(**joint BPE**)
	- 장점: source와 target segmentation 사이에 일관성이 있다.
	- 실제로는 단순히 source와 target training set를 concatenate하여 학습을 진행한다.


## 논문 
Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units. _arXiv preprint arXiv:1508.07909_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTIwNDg5NzAsMTQ1OTEwNTAwOCwtNDMwOT
kyMzk3LC0xNzg1NjA5ODMzLDE5MjI2MTc4NDEsLTU0OTc3MTEs
LTQzNjUyMTI5OCwtMTA5MzM5NDc2NV19
-->