---
title: "Inference on Variable Importance Measure"
excerpt: "Non zero Null Hypothesis하에서의 통계적 추론"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-08-22 20:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---


> 참조논문
> 1. [Demystifying statistical learning based on efficient influence functions](https://arxiv.org/abs/2107.00681)
> 2. [Semiparametric doubly robust targeted double machine learning: a review](https://arxiv.org/abs/2203.06469)
> 3. [A General Framework for Inference on Algorithm-Agnostic Variable Importance](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.2003200)

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

### Step 1. 관심있는 추정치를 정의

기호
- $Y$: 관심있는 결과 값 (값이 클수록 더 좋다는 것을 의미)
- $X$: 설명변수
- $A \in \{0, 1\}$: 이진 treatment
- treatment rule $f: \mathcal{X} \mapsto \{0, 1\}$
	- $X$의 값에 의존하여 $A$의 값을 결정
- 관측된 데이터의 구조: $Z := (X, A, Y) \sim P_0$ where $P_0$ is the true distribution
	- 데이터 생성 분포 $P_0$는 오직 충분히 큰 분포들의 클래스 $\mathcal{M}$에 속한다는 것만 알려져 있음

VIM의 정의

- $s \subseteq \{1, \cdots, p\}$: 설명변수의 부분집합 (importance를 측정하고자 하는 대상)
- a rich class $\mathcal{F}$ of functions from $\mathcal{X}$ to $\{0, 1\}$
- $\mathcal{F}_s := \{f \in \mathcal{F}: f(u) = f(v) \text{\quad for all \quad} u, v \in \mathcal{X} \text{\quad satisfying \quad} u_{-s} = v_{-s}\}$: $s$에 의존하지 않는 함수들로 구성된 집합
- $u_{-s}$ denote the elements of $u$ with index not in $s$
- 예측력의 측도 (Potential Outcome Mean): $V(f, P) := \mathbb{E}_P[Y(f(X))] = \mathbb{E}_P[Q_P(f(X), X)]$ 
	- where $Q_P(a, x) := \mathbb{E}_P[Y|A=a, X=x]$ under the usual identifying assumptions
- $f_0 = \arg\max_{f \in \mathcal{F}} V(f, P_0)$: the oracle prediction function within $\mathcal{F}$ under $P_0$ relative to $V$
- $f_{0,s} = \arg\max_{f \in \mathcal{F}_s} V(f, P_0)$
- 목표: VIM $\psi_{0,s} := V(f_0, P_0) - V(f_{0,s}, P_0) \geq 0$에 대해서 통계적 추론을 하자!
	- 변수 $X_s$의 population-level 중요도: 전체 설명변수 $X$에서 $X_s$를 제외하였을 때 잃게되는 oracle 예측도의 감소량
	- 감소량이 클수록 해당 변수가 중요하다는 것을 의미함!

- parameter mapping $V^*: P \mapsto V(f_P, P)$
- $\hat{P}_n \in \mathcal{M}$: $P_0$의 추정치
- the plug-in estimator $V^*(\hat{P}_n)$
	- 일반적으로, $V^*(P_0)$이 $P_0$의 지역적 특성을 포함하므로, 유연한 학습 방법론 (머신 러닝 등)이 사용되기 때문에 bias가 매우 큼
	- 예를 들어, 조건부 평균이나 밀도 함수
- 따라서, one-step estimator와 같은 비모수 'debiasing' 방법론이 필요하다!

### Step 2. 추정치의 EIF를 계산

비모수 EIF를 구하려는 대상

- 우리가 필요한 것: parameter mapping $V^*: P \mapsto V(f_P, P)$의 EIF
- 하지만, $V^*: P \mapsto V(f_P, P)$의 정의는 $P$-optimal prediction 함수 $f_P$를 포함하므로, EIF의 유도가 매우 복잡하다...
  
- Theorem (Williamson et al. (2021))
	> Provided condition (A5) holds, if $P \mapsto V(f_0, P)$ is pathwise differentiable at $P_0$ relative to the nonparametric model $\mathcal{M}$, then so is $P \mapsto V(f_P, P)$, and the two parameters have the same EIF.

- 즉, regularity condition (A5) 하에서, 비모수 EIF의 계산에서 $f_P$를 고정된 $f_0$로 바꿔서 유도하여도 동일한 비모수 EIF를 얻을 수 있다!
- the parameter $P \mapsto V(f_0, P)$ is pathwise differentiable at a distribution $P_0$ if $Q_0(1, W) \neq Q_0(0, W)$ occurs $P_0$-almost surely
- oracle prediction function $f_0: x \mapsto I(Q_0(1, x) > Q_0(0, x))$
- oracle residual prediction function $f_{0,s}: x \mapsto I(Q_{0,s}(1, x) > Q_{0,s}(0, x))$
	- define $Q_{0,s}$ as $Q_{0,s} := \mathbb{E}_0 [Q_0(a, x) | X_{-s} = x_{-s} ] = \mathbb{E}_0 [\mathbb{E}_0 [Y | A=a, X=x]| X_{-s} = x_{-s} ] = \mathbb{E}_0 [Y | A=a, X_{-s} = x_{-s}]$

앞의 결과를 토대로 비모수 EIF를 계산

- The nonparametric EIF of $P \mapsto V(f_0, P)$ at $P_0$:

$$
\phi_0: z \mapsto \frac{I(a = f_0(x))}{\pi_0(f_0(x), x)} (y - Q_0(f_0(x), x)) + Q_0(f_0(x), x) - V(f_0, P_0) 
$$

- 여기서 propensity score는 $\pi_0(a, x) := Pr_0(A=a | X=x)$ for each $a \in \{0, 1\}$로 정의됨

### Step 3. 추정치의 EIF를 이용하여 추정량을 계산

- Regularity conditions 하에서, 비모수 debiasing 방법론 중의 하나로 $v_0 = V(f_0, P_0)$의 one-step debiased 추정치는 다음과 같이 계산되며, 비모수 효율적(nonparametric efficient)이다: 

$$
v_n = V^*(\hat{P}_n) + \frac{1}{n} \sum_{i=1}^n \phi_n(Z_i) 
$$

$$
= V^*(\hat{P}_n) + \frac{1}{n} \sum_{i=1}^n \left( \frac{I(A_i = f_n(X_i))}{\pi_n(f_n(X_i), X_i)} (Y_i - Q_n(f_n(X_i), X_i)) + Q_n(f_n(X_i), X_i) - V^*(\hat{P}_n)  \right)
$$

$$
= \frac{1}{n} \sum_{i=1}^n \left( \frac{I(A_i = f_n(X_i))}{\pi_n(f_n(X_i), X_i)} (Y_i - Q_n(f_n(X_i), X_i)) + Q_n(f_n(X_i), X_i) \right)
$$

- $Q_n, \pi_n$ are estimators of $Q_0$ and $\pi_0$, respectively, $f_n$ is defined pointwise $f_n(x) = I(Q_n(1, x) > Q_n(0, x))$
- $\phi_n(z)$ is $\phi_0(z)$ evaluated at $\hat{P}_n$

유사한 방식으로, 

- the nonparametric EIF of $P \mapsto V(f_{0, s}, P)$ at $P_0$:

$$
\phi_{0,s}: z \mapsto \frac{I(a = f_{0, s}(x))}{\pi_0(f_{0, s}(x), x)} (y - Q_0(f_{0, s}(x), x)) + Q_0(f_{0, s}(x), x) - V(f_{0,s}, P_0) 
$$

- Regularity conditions 하에서, 비모수 debiasing 방법론 중의 하나로 $v_{0, s} = V(f_{0, s}, P_0)$의 one-step debiased 추정치는 다음과 같이 계산되며, 비모수 효율적(nonparametric efficient)이다: 

$$
v_{n,s} = V(f_{n,s}, \hat{P}_n) + \frac{1}{n} \sum_{i=1}^n \phi_{n, s}(Z_i) 
$$

$$
= \frac{1}{n} \sum_{i=1}^n \left( \frac{I(A_i = f_{n, s}(X_i))}{\pi_n(f_{n, s}(X_i), X_i)} (Y_i - Q_n(f_{n, s}(X_i), X_i)) + Q_n(f_{n, s}(X_i), X_i) \right)
$$

- $f_{n, s}$ is defined pointwise $f_{n, s}(x) = I(Q_{n, s}(1, x) > Q_{n, s}(0, x))$
- $\phi_{n, s}(z)$ is $\phi_{0, s}(z)$ evaluated at $\hat{P}_n$

### Step 4. 통계적 추론

- $\psi_{n,s} = v_n - v_{n, s}$
- $\psi_{0,s} = v_0 - v_{0, s}$
- $\phi_0(z) - \phi_{0,s}(z)$ 는 $\psi_{0,s}$의 비모수 EIF

empirical process와 나머지 항들이 0으로 수렴할 수 있는 충분조건 하에서,
If
1. $\psi_{0,s} > 0$ (Non-zero Null Hypothesis, 변수의 중요도가 0이 아님) 
2. $0 < \tau_{0,s}^2 : = \int (\phi_0(z) - \phi_{0,s}(z))^2 dP_0(z) < \infty$ (유한 점근 분산)

$$
\sqrt{n}\left(\psi_{n,s} - \psi_{0,s}\right) \overset{d}{\to} N \left(0, \tau_{0,s}^2 \right)
$$

$\psi_{0,s}$의 점근적으로 유효한 95% 신뢰 구간은 

$$
\psi_{n,s} \pm 1.96 \cdot \frac{1}{\sqrt{n}} \cdot \tau_{n,s}
$$

이고, 점근 분산은 다음과 같이 추정될 수 있다:

$$
\tau_{n,s}^2 = \frac{1}{n} \sum_{i=1}^n (\phi_n(Z_i) - \phi_{n, s}(Z_i))^2
$$
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk5ODY2OTA4XX0=
-->