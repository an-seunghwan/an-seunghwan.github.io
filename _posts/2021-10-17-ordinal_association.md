---
title: "Assessing Partial Association Between Ordinal Variables: Quantification, Visualization, and Hypothesis Testing (논문 읽기)"
excerpt: "Ordinal variable 사이에 association 계산하기"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-10-17 20:00:00 -0000
categories: 
  - Ordinal Data
tags:
  - 논문 읽기
---


- [Assessing Partial Association Between Ordinal Variables: Quantification, Visualization, and Hypothesis Testing](https://www.tandfonline.com/doi/epub/10.1080/01621459.2020.1796394?needAccess=true) 논문에 대한 리뷰와 간단한 제 생각을 적은 포스팅입니다.
- 자세하고 정확한 내용은 논문을 참고해 주세요!

## Assessing Partial Association Between Ordinal Variables: Quantification, Visualization, and Hypothesis Testing

### Continuous data correlation

continuous random variable $Y_1, Y_2$ 사이의 covariate $X$를 고려한 parametric model에서의 correlation은 다음과 같은 정규분포 가정하에서 $\Sigma$를 통해 추정될 수 있다. 여기서 중요한 것은, 두 변수 사이의 correlation이 data $X$에 conditional 하다는 점이다.



$$
\begin{align}
\begin{pmatrix}  
Y_1 \\  
Y_2
\end{pmatrix} - 
\begin{pmatrix}  
X \beta_1 \\  
X \beta_2
\end{pmatrix} \sim N(0, \Sigma)
\end{align}
$$



이때, $E(Y_1 \vert X) = X\beta_1, E(Y_2 \vert X) = X \beta_2$ 는 consistent estimator.


### Ordinal data association


하지만, ordinal data와 같은 경우는 discrete이므로 위와 같은 방식(conditional on data $X$)으로 두 변수 사이의 association을 구하는 것이 어렵다. 따라서 ordinal variable $Y_1, Y_2$에 대해서 surrogate variable $S$를 고려한다. 이때 $S$는 continuous random variable이다!

latent variable $Z$를 활용하여, $Z = X\beta + \epsilon, \epsilon \sim G$, $\epsilon$은 $X$에 독립인 latent model을 구성하면, cutpoint $\alpha_1, \cdots, \alpha_{J-1}$를 이용해 다음과 같은 cumulative link model을 구축할 수 있다. 이때, $G$는 어떠한 분포도 가능하다!



$$
G^{-1}(P(Y \leq j) = \alpha_j - X\beta, j = 1,\cdots,J-1
$$



이때, $\beta, \alpha_1, \cdots, \alpha_{J-1}$는 ML 방식을 통해 추정한다. 이제 surrogate of $Y$를 다음과 같이 정의한다.


$$
S=
\begin{cases}
Z \vert -\infty < Z \leq \alpha_1,\\
Z \vert \alpha_1 < Z \leq \alpha_2, \\
\cdots \\
Z \vert \alpha_{J-1} < Z \leq \infty. \\
\end{cases}
$$



이제 $Y_1 \perp Y_2 \vert X \Longleftrightarrow S_1 \perp S_2 \vert X$ (본문 Theorem 1)임을 이용하여 앞의 continuous case와 같이 continuous residual $R_1, R_2$를 다음과 같이 정의하고,


$$
\begin{align}
\begin{pmatrix}  
R_1 \\  
R_2
\end{pmatrix} = 
\begin{pmatrix}  
S_1 \\  
S_2
\end{pmatrix} - 
\begin{pmatrix}  
E(S_1 \vert X) \\  
E(S_2 \vert X)
\end{pmatrix} 
\end{align}
$$


association function $\phi(R_1, R_2)$를 $Z$ 분포를 이용해 $S$를 sampling 하여 empirical하게 추정한다.

## Reference 
- Liu, D., Li, S., Yu, Y., & Moustaki, I. (2020). Assessing partial association between ordinal variables: quantification, visualization, and hypothesis testing. _Journal of the American Statistical Association_, 1-14.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1OTkxNTcxNDAsLTE2ODg4Mjk5NTUsLT
QyNjMyNTMxNV19
-->