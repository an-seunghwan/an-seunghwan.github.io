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

- 전통적인 

###

- $Y$: the outcome of interest (larger values corresponds to better)
- $X$: a covariate vector
- $A \in \{0, 1\}$: binary treatment
- treatment rule $f: \mathcal{X} \mapsto \{0, 1\}$
	- assigning the value of $A$ based on $X$ can be adjudicated
- observed data structure: $Z := (X, A, Y) \sim P_0$ where $P_0$ is the true distribution
	- a data-generating distribution $P_0$ known only to belong to a rich (nonparametric) class $\mathcal{M}$ of distributions

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0MDc0MzA2MTAsLTE5NTM3MjAwNDAsLT
QyODc1MTQ4OV19
-->