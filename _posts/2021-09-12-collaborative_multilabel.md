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

- [Collaborative Multilabel Classification](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1961783) 논문에 대한 리뷰에 대한 간단한 제 생각을 적은 포스팅입니다.
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

### False negative weights

$W_{-lk} = e_l^\top e_k$ 


## Reference 
- Zhu, Y., Shen, X., Jiang, H., & Wong, W. H. (2021). Collaborative multilabel classification. _Journal of the American Statistical Association_, (just-accepted), 1-31.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTYxODEzODIxXX0=
-->