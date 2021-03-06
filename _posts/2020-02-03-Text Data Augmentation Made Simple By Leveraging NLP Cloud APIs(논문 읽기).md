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

### 3.2 data augmentation을 공식화하기 위한 시도

**분포에 대한 규칙**

augmented data는 반드시 original data와 그 통계적 분포가 유사해야 한다.

**만족도에 대한 규칙**

의미론적 단계에서 data transformation은 data의 의미에 영향을 주지 않고, pattern recognition의 관점에서 "새로운 형태"를 학습할 수 있도록 기여해야 한다. 하지만 이는 사람이 반드시 새롭게 추가된 데이터가 기존의 데이터와 구분하기 어려운 정도인 "만족도"를 평가해야한다는 점을 고려해보면 매우 복잡한 문제가 된다.

하지만 만족할만한 transformation이 발견되더라도, transformed data의 의미를 찾는 inverse problem은 매우 어려운 문제이다.

**의미 불변의 규칙**

data augmentation은 반드시 의미 불변 transformation을 사용해야 한다.

**Telephone Game Rule of Thumb**

의미론적 불변을 얻기 위해서는, 연속적 또는 결합된 transformation의 횟수가 반드시 제한되어야 한다(경험론적으로 2개).

### 3.4 기술 1 - "문맥적 잡음" 삽입(textual noise injection)

text에서의 연속적인 변화에 가장 가까운 것은 약한 문맥적 잡음을 삽입하는 것이다: 변화, 추가, 단어의 일부 철자 제거, 대,소문자의 변화, 구두점의 변경.

이 논문에서는 잡음 삽입을 text data augmentation으로 생각하기 꺼려한다. 왜냐하면 잡음의 추가를 일반적으로 학습의 robustness에 도움을 주지 data의 새로운 형태에 대한 인식에 기여를 하지 않기 때문이다.

### 3.4 잘못된 철자 삽입

잘못된 철자가 삽입된 text를 이용한 학습은 특정 형태의 textual 잡음에 대해 모형이 robust하도록 한다. 또한 잘못된 철자 삽입은 의미 불변 변화이다.

### 3.5 thesaurus를 이용한 단어 교체

어휘상의 교체는 주어진 단어의 동의어를 이용하게 된다. 일반적으로, 문법적 단어들에 대한 교체는 할 수 없다. 이러한 어려움 때문에, 다음과 같은 품사가 어휘 교체의 후보가 된다: adverbs, adjectives, nouns and verbs. Verb 교체는 특히 특정 verb들은 다른 인자(arguments)들을 동반하므로 어렵다. 많은 경우에, adverbs와 adjectives에 대해서만 어휘 교체를 제한한다.

어휘 교체를 위해서는, 일반적으로 hyperonyms(더 일반적 용어, 튤립 → 꽃)를 더 선호하고, hyponyms(더 구체적인 용어, 꽃 → 튤립)를 사용하지는 않는다.

hyperonym을 이용한 어휘 교체는 의미 불변 변화이다.
hyponym을 이용한 어휘 교체는 일반적으로 의미 불변 변화가 아니다.

어휘 교체의 가장 주된 어려움은 자연어의 모호성에서 온다. 단어가 여러 의미를 가지게 되면, 여러개의 다른 동의어를 갖게 된다. 따라서 적절한 동의어 집합을 선택하는 것은 중요한 문제이다. 예를 들어, 주어진 corpus에서의 출현 빈도에 기반하여 가장 자주 등장하는 의미를 선택하는 것도 하나의 방법이다. 또는 사전에서 동의어 집합의 각각의 동의어가 가지는 정의, 예시 등의 정보와 단어의 문맥 사이의 유사도를 이용해 가장 유의한 단어를 선택할 수 도 있다.

때때로, 몇몇 thesaurus는 동의어 중에 반의어를 반환하는 경우가 있다. 이러한 경우에는, 반의어 사전을 이용해 결과로 나온 동의어를 filtering하는 것이 요구된다.

### 3.6 text augmentation by paraphrase(의역)

단어나 구, 문장 수준에서의 paraphrase를 이용해 text augmentation을 진행할 수 있다. 예를 들면, "나는 나의 일을 끝냈다"는 "나는 나의 과제를 끝냈다"와 동일하다.

**완벽한 paraphrase의 정의**

이상적인 paraphrase는 의미 보존을 하고, 자연스럽게 들리면서 원문으로부터 변화해야한다.

### 3.7 정규표현식을 이용한 paraphrase 생성

표면적 변화(surface transformation) = 통사론을 무시하고 간단한 pattern matching 규칙들을 이용한 변화

예를 들면, 축약어, acronym, 기호, orthographic의 변화 등이 있다. 하지만 축약어의 사용 등이 문맥상의 정보를 바탕으로 했을 때 잘못된 해석을 하도록 하면 안된다.

### 3.8 syntax trees transformation을 이용한 paraphrases generation

### 3.9 back-tranlation을 이용한 paraphrases generation

원래의 문장 → [번역기] → 번역된 문장 → [번역기] → 원래의 언어로 다시 번역된 문장

이때, 원래의 문장과 원래의 언어로 다시 번역된 문장 사이의 유사도를 확인하여 지나치게 유사하거나 동일하면 사용하지 않고, 적절하다면 paraphrase로써 사용된다.

빠른 실행과 계산적 성능의 이유로, 이 논문에서는 유사도를 확인하는 방법으로 원래의 문장과 원래의 언어로 다시 번역된 문장 사이의 길이를 비교한다.

하지만 이 방법은 사용하는 [번역기]에 의존한다는 단점이 있다.

## 논문 
COULOMBE, Claude. Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs. _arXiv preprint arXiv:1812.04718_, 2018.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjMxMzE4MzAxLC0xNTUwNjMwNzIxLDE0Mz
cyNzMzOTUsLTQxMTM5NTgzMywtNjE0MzA4MjEyXX0=
-->