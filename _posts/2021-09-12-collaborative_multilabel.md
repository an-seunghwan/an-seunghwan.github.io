---
title: "Collaborative Multilabel Classification (논문 읽기)"
excerpt: "With semi-supervised learning consistency regularization"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2021-09-12 20:00:00 -0000
categories: 
  - multilabel
tags:
  - 논문 읽기
---

- [Collaborative Multilabel Classification](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1961783) 논문에 대한 리뷰와 간단한 제 생각을 적은 포스팅입니다.
- 자세하고 정확한 내용은 논문을 참고해 주세요!

## Collaborative Multilabel Classification

- 핵심 요약: multilabel classification 문제에서, label들 간의 관계를 고려하여 문제를 해결할 수 있는 방법을 제시

### Weights

p개의 lable $y = (y_1, \cdots, y_p)$와 각 label의 embedding vector $e_1, \cdots, e_p$를 생각하자. 이때, embedding vector는 GLOVE를 통해 학습되며, 학습의 목표는 간단히


$$
\begin{aligned} 
p(y_j|y_i) = e_j^\top e_i
\end{aligned}
$$


로 표현할 수 있다.

- False negative weight

$W_{-lk} = e_l^\top e_k$  and $W_{-lk} > 0$ when $y_l, y_k$ are semantically similar.

- False positive weight

$W_{+lk} = 0$ if $l \neq k$,  and $W_{+kk} = 1/p$ 

### novel loss

- If $y_l=+1$ and $f_k(x)<0$ 

loss에 $I(y_l=+1) W_{-lk}$가 더해지는데, 먄약에 $y_l, y_k$가 semantically similar했다면, 더 큰 weight $W_{-lk}$가 novel loss에 더해지게된다. 즉, 현재 예측하려는 label $y_k$와 semantically similar한 label $y_l$가 존재한다면, $f_k(x)>0$으로 $y_k$가 존재하는 것으로 예측하도록 만든다.

- If $y_l=-1$ and $f_k(x)>0$ 

loss에 $I(y_l=-1) W_{+lk}$가 더해지는데, $W_{+lk}=0$이므로 novel loss에 영향을 주지 않게 된다. 즉, 현재 예측하려는 label $y_k$와 semantically similar한 label $y_l$가 존재하지 않는다면, label $y_k$를 어떻게 예측하는지는 중요하지 않음을 의미한다.

- If $y_k=+1$ and $f_k(x)<0$, If $y_k=-1$ and $f_k(x)>0$ 

이는 label $y_k$의 존재 여부에 대한 정확한 예측을 하도록 만드는 weight를 novel loss에 더해준다.

## objective and conditional probability

false negative weight를 다음과 같이 conditional probability를 이용해 적을 수 있다.

$W_{-lk} = p(y_k \vert y_l)$ and $\sum_{k=1}^p W_{-lk} = 1$ (row sum equals 1)

### objective


$$
\begin{aligned} 
\min \sum_{k=1}^p \vert \delta_k(y) \vert I(\delta_k(y) \cdot f_k(x) < 0)
\end{aligned}
$$


where 


$$
\begin{aligned} 
\delta_k(y) &= \sum_{l:y_l = +1} W_{-lk} - \sum_{l:y_l = -1} W_{+lk} \\
&= \sum_{l:y_l = +1} p(y_k \vert y_l) - (0 \text{	    or	 } 1/p)
\end{aligned}
$$


- $f_k(x) > 0$ and $\delta_k(y) < 0$


$$
\begin{aligned} 
\delta_k(y) = \sum_{l:y_l = +1} p(y_k \vert y_l) - 1 / p < 0
\end{aligned}
$$


이므로 $\sum_{l:y_l = +1} p(y_k \vert y_l)$ 가 작다는 것의 의미는 label $y_k$가 label $y_l=+1$가 주어진 경우에 conditional probability가 작다는 것이다. 따라서, objective를 증가시켜 label $y_k$가 존재하지 않도록 $f_k(x) < 0$로 예측하도록 한다.


- $f_k(x) < 0$ and $\delta_k(y) > 0$


$$
\begin{aligned} 
\delta_k(y) = \sum_{l:y_l = +1} p(y_k \vert y_l) - 0 > 0
\end{aligned}
$$


이므로 $\sum_{l:y_l = +1} p(y_k \vert y_l)$ 가 크다는 것의 의미는 label $y_k$가 label $y_l=+1$가 주어진 경우에 conditional probability가 크다는 것이다. 따라서, objective를 증가시켜 label $y_k$가 존재하도록 $f_k(x) > 0$로 예측하도록 한다.

## Reference 
- Zhu, Y., Shen, X., Jiang, H., & Wong, W. H. (2021). Collaborative multilabel classification. _Journal of the American Statistical Association_, (just-accepted), 1-31.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzOTAyNDUyMDksLTQyNjkyMTM4NywtMT
M5NDM0NDQzMl19
-->