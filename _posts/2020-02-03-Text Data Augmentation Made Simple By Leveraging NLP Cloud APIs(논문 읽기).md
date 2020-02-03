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

### 1. Basic Assumption

**분포에 대한 규칙**
augmented data는 반드시 original data와 그 통계적 분포가 유사해야 한다.



<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc2MzY2MjI4MCwtNjE0MzA4MjEyXX0=
-->