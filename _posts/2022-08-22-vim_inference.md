---
title: "Inference on Variable Importance Measure"
excerpt: "Non-zero Null Hypothesis하에서의 통계적 추론"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2022-08-22 20:00:00 -0000
categories: 
  - VIM
tags:
  - pytorch
  - statistical inference
---

> 참조논문
> [Demystifying statistical learning based on efficient influence functions](https://arxiv.org/abs/2107.00681)
> [Semiparametric doubly robust targeted double machine learning: a review](https://arxiv.org/abs/2203.06469)
> [A General Framework for Inference on Algorithm-Agnostic Variable Importance](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.2003200)

## 변수의 중요도에 대한 통계적 추론 (Inference on Variable Importance Measure (VIM))

### Introduction

- 전통적인 통계적 방법론:
	1. 모형 설정
	2. 하나 이상의 계수에 대한 추정
	3. 추정된 계수에 대한 불확실성 측정 (신뢰구간 등)

- 전통적인 통계적 방법론의 한계:
	1. 쉽게 해석 가능한 모형을 얻기 위해 지나치게 간단한 모형을 사용하여, 모형의 misspecification 문제가 발생
	2. 선택된 모형에 의존하여 모형의 계수의 의미와 정의가 결정됨

- ```<span style="color:yellow">사전 정의된 비모수 추정치</span>```: 데이터로부터 추론하고 싶은 것을 표현할 수 있는 관측된 데이터 분포의 함수 형태 (모형이 필요하지 않음)
	- Efficient Influence Function (EIF)의 통계적 성질에 의존
	- EIF의 유도가 이전에는 'dark art'로 여겨졌지만, Gateaux derivative 덕분에 간단하게 유도할 수 있게 되었다!

### 통계적 방법론의 변화

- 과거: 모형을 수립하고 검증
- 현재: 관심있는 과학적 질문과 연결될 수 있는 추정치를 선택
	- 데이터가 얻어지기 전에 분석이 가능하다!

## 추론 절차

### 1. 관심있는 추정치를 정의

- $Y$: 관심있는 결과 값 (값이 클수록 더 좋다는 것을 의미)
- $X$: 설명변수
- $A \in \{0, 1\}$: 이진 treatment
- treatment rule $f: \mathcal{X} \mapsto \{0, 1\}$
	- $X$의 값에 의존하여 $A$의 값을 결정
- 관측된 데이터의 구조: $Z := (X, A, Y) \sim P_0$ where $P_0$ is the true distribution
	- 데이터 생성 분포 $P_0$는 오직 충분히 큰 분포들의 클래스 $\mathcal{M}$에 속한다는 것만 알려져 있음


\item let parameter mapping $V^*: P \mapsto V(f_P, P)$
    \item let $\hat{P}_n \in \mathcal{M}$: an estimator of $P_0$
    \item the plug-in estimator $V^*(\hat{P}_n)$
    \item[-] generally suffers from excessive bias whenever flexible learning techniques have been used, because $V^*(P_0)$ involves local features of $P_0$
    \item[-] for example, the conditional mean or density function
    \item Therefore, we employ nonparametric `debiasing' techniques! (ex. one-step estimator)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyMzAzMjg4MzMsMTUzNzQ2MDYyMywtMT
U2MDI5MTc3NSwtMTQwNzQzMDYxMCwtMTk1MzcyMDA0MCwtNDI4
NzUxNDg5XX0=
-->