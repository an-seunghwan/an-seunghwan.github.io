---
title: "Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs(논문 읽기)"
excerpt: "Text Data Augmentation"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-02-03 21:00:00 -0000
categories: 
  - NLP
tags:
  - text data augmentation
  - 톺아보기
---

	톺아보기 시리즈 (톺아보다: 샅샅이 틈이 있는 곳마다 모조리 더듬어 뒤지면서 살피다)

> 이 포스팅은 Text Data Augmentation 등과 관련한 내용을 다루는 [TDA 톺아보기 시리즈](https://an-seunghwan.github.io/tda-top-a-bogi/)의 1편입니다.

> [Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs](https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf) 논문에서 text data augmentation에 관해 필요한 내용들만 간단히 요약하여 정리하였습니다. 자세한 내용은 원문을 꼭 참조해주세요.

## <span style="color:#2E86C1;">2. Related Works</span>

고전적인 text augmentation 기술은 thesaurus를 이용해 동의어를 이용한 어휘 교체였다. 현재까지도 NLP에서 사용되는 data augmentation 방법은 굉장히 제한적이다. 그 이유는 다음과 같다:

1. 자연어 data는 처리하기 굉장히 까다롭다
	- text data는 상징적(symbolic), 이산적(discrete), 합성적(compositional), 희소적(sparse)하다.
	- 또한 계층적(hirerarchical), noisy, 예외가 많고 모호하다.
2. gradient descent-based learning 기술은 text와 같은 이산적 data에 바로 적용되지 않는다.
3. 현실적인 text data를 생성하기 어렵다.

## <span style="color:#2E86C1;">3. Method</span>

### 2. data augmentation을 공식화하기 위한 시도

**분포에 대한 규칙**

augmented data는 반드시 original data와 그 통계적 분포가 유사해야 한다.

**만족도에 대한 규칙**

의미론적 단계에서 data transformation은 data의 의미에 영향을 주지 않고, pattern recognition의 관점에서 "새로운 형태"를 학습할 수 있도록 기여해야 한다. 하지만 이는 사람이 반드시 새롭게 추가된 데이터가 기존의 데이터와 구분하기 어려운 정도인 "만족도"를 평가해야한다는 점을 고려해보면 매우 복잡한 문제가 된다.

하지만 만족할만한 transformation이 발견되더라도, transformed data의 의미를 찾는 inverse problem은 매우 어려운 문제이다.

**의미 불변의 규칙**

data augmentation은 반드시 의미 불변 transformation을 사용해야 한다.

** Telephone Game Rule of Thumb**

의미론적 불변을 얻기 위해서는, 연속적 또는 결합된 transformation의 횟수가 반드시 제한되어야 한다(경험론적으로 2개).

### 3. 기술 1 - "문맥적 잡음" 삽입(textual noise injection)

text에서의 연속적인 변화에 가장 가까운 것은 약한 문맥적 잡음을 삽입하는 것이다: 변화, 추가, 단어의 일부 철자 제거, 대,소문자의 변화, 구두점의 변경.

이 논문에서는 잡음 삽입을 text data augmentation으로 생각하기 꺼려한다. 왜냐하면 잡음의 추가를 일반적으로 학습의 robustness에 도움을 주지 data의 새로운 형태에 대한 인식에 기여를 하지 않기 때문이다.

## 논문 
COULOMBE, Claude. Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs. _arXiv preprint arXiv:1812.04718_, 2018.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5Mjg2MjQ0NzYsLTE1NTA2MzA3MjEsMT
QzNzI3MzM5NSwtNDExMzk1ODMzLC02MTQzMDgyMTJdfQ==
-->