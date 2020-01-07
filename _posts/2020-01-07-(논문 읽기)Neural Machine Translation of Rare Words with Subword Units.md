---
title: "Neural Machine Translation of Rare Words with Subword Units(논문 읽기)"
excerpt: "NLP 논문 읽기"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-07 15:00:00 -0000
categories: 
  - NLP
tags:
  - 논문 읽기
---
지난 게시글 "OOV 해결하기 (1)"([https://an-seunghwan.github.io/nlp/OOV-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-(1)/](https://an-seunghwan.github.io/nlp/OOV-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-(1)/))에서 Byte Pair Encoding에 대해 간략히 소개하였다. 이번 글에서는 이 알고리즘이 제시된 논문을 읽고 간략히 요약하여 Byte Pair Encoding에 기반한 Neural Machine Translation에 대해 알아보도록 하겠습니다.

논문의 내용을 전부 요약하는 것이 아닌 핵심만을 다루도록 하겠습니다.

## 1. Introduction

빈도가 높지 않은(자주 출현하지 않는) 단어들에 대한 번역은 open problem이다. 특히 교착어나 합성어를 통해 단어가 생산될 수 있는 언어에 대해서, 번역 모형은 단어 수준보다 더 아래 수준의 메카니즘을 필요로 한다.

이 논문의 가장 주된 목적은 NMT network에서, 희귀 단어에 대한 back-off 모형을 필요로 하지 않는 open-vocabulary 번역 모형을 만드는 것이다.

이 논문의 2가지 주된 기여
* open-vocabulary NMT가 단어들을 subword units를 통해 encoding이 가능하다는 보였다. 
* 단어 segmentation를 위한 압축 알고리즘으로써 BPE를 사용하였다. BPE는 open vocabulary를 고정된 크기의 다양한 길이의 character sequence를 통해 표현할 수 있게 해주었다.


## 논문 출처
Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units. _arXiv preprint arXiv:1508.07909_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc2NjUzMDkyMF19
-->