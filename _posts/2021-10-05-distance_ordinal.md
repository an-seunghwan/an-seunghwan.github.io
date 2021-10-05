---
title: "Distance-Based Analysis of Ordinal Data and Ordinal Time Series (논문 읽기)"
excerpt: "Ordinal data에 대해 정의되는 거리 함수를 이해하기"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-10-05 20:00:00 -0000
categories: 
  - Ordinal Data
tags:
  - 논문 읽기
---


- [Distance-Based Analysis of Ordinal Data and Ordinal Time Series](https://www.tandfonline.com/doi/pdf/10.1080/01621459.2019.1604370?needAccess=true) 논문에 대한 리뷰와 간단한 제 생각을 적은 포스팅입니다.
- 자세하고 정확한 내용은 논문을 참고해 주세요!

## Distance-Based Analysis of Ordinal Data and Ordinal Time Series

### notations

- ordered categorical range $S = \{s_0, \cdots, s_m \}$ where $s_0 \leq s_1 \leq \cdots \leq s_m$
- ordinal random variable $X \in \{s_0, \cdots, s_m \}$
- rank count variable $I \in \{ 0, \cdots, m \}$
- $X = s_I$

### distance function

- distance function $d(s_k, s_l): S \times S \mapsto [0, \infty)$
	- example: block distance 
		- $d_{o, 1}(s_k, s_l) = \vert k - l \vert$
		- distance value do not depend on the actual labeling $s_0, \cdots, s_m$

- possible properties of ordinal distances
	1. Maximization $d(s_0, s_m) = \max_{x, y \in S} d(x, y)$
	2. $d$ is said to be compatible with the ordering if 
	$$x < y < z \text{ implies that } d(x, z) > d(x, y), d(y, z)$$
	3. $d$ is said to be additive if for given $d_1, \cdots, d_m > 0$, it holds that
	$$d(s_i, s_{i+k}) = d_{i+1} + \cdots + d_{i+k} \text{ for all } i = 0, \cdots, m-1, k = 1, \cdots, m-i$$
	4. Centrosymmetry
	$$d(s_i, s_j) = d(s_{m-i}, s_{m-j}) \text{ for all } 0 \leq i < j \leq m$$

### location

$$loc = \arg\min_{x \in S} E_X[d(X, x)]$$

### dispersion

1. $$disp = E_X[d(X, loc)]$$
2. $$disp = E[d(X_1, X_2)] = \sum_{i,j=0}^m d(s_i, s_j) p_i p_j \leq d(s_0, s_m)$$ by maximization property

### asymmetry

- reflected copy $X^r = s_{m-I}$ where $P(X = s_i) = p_{i}$ and $P(X^r = s_i) = p_{m-i}$
- If $p_i = p_{m-i}$, then $X =_d X^r$: symmetry distribution

$$
\begin{align}
asym &= E[d(X, X^r)] - disp \\
&= \sum_{i,j=0}^m d(s_i, s_j) p_i (p_{m-j} - p_j) \\
&= 0 (\text{ if distribution of $X$ is symmetry})
\end{align}
$$

### skewness

$$
\begin{align}
skew &= E[d(X, s_m)] - E[d(X, s_0)] \\
&= \sum_{i,j=0}^m d(s_i, s_m) p_i - \sum_{i,j=0}^m d(s_i, s_0) p_i \\
&= \sum_{i,j=0}^m d(s_i, s_m) p_i - \sum_{i,j=0}^m d(s_{m-i}, s_m) p_i (\text{ Centrosymmetry }) \\ 
&= \sum_{i,j=0}^m d(s_i, s_m) p_i - \sum_{i,j=0}^m d(s_{i}, s_m) p_{m-i} \\ 
&= \sum_{i,j=0}^m d(s_i, s_m) p_i - \sum_{i,j=0}^m d(s_{i}, s_m) p_{i} (\text{ symmetric distrib }) \\ 
\end{align}
$$


## Comments


## Reference 
- Weiß, C. H. (2019). Distance-based analysis of ordinal data and ordinal time series. _Journal of the American Statistical Association_.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTcxODU2MDYxOSw3OTY3MjQyMzcsMjczOT
UzOTcyXX0=
-->