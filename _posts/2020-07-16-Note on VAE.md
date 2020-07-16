---
title: "Note on VAE"
excerpt: "VAE에 대한 여러가지 사실들"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-07-16 20:00:00 -0000
categories: 
  - VAE
tags:
  - 
---
> Variational Inference, VAE 관련 여러 논문들과 블로그들을 보고 중요하다고 생각되는 수식이 아닌 아이디어 위주의 정리 포스팅입니다.
> 개인적인 정리 목적의 글임을 밝힙니다.

## 0. 참고 논문 및 블로그


## 1.  VAE

### 1. purpose
1. sampling new data $x^*$
2. measure likelihood of new $x^*$
3. __maximize data log-likelihood__: $\log{p_{\theta}(x)}$

여기서 제일 중요한 것은 3번의 data log-likelihood를 최대화하는 것이 결국 VAE의 가장 중요한 목적이라는 것이다.

### 2. latent variable model
* model
$$
p(x) = N_z(0, I)
$$
$$
p(x|z) = N_x(f(x;\phi), \sigma^2I)
$$

이때, $f(x;\phi)$는 neural network로 구성되는 non-linear 함수이고, 이는 결국 latent variable model이 non-linear latent factor model을 학습하는 것과 동일함을 의미한다.

* latent space learning

latent space에서 정의되는 latent variable $z$ 각각의 차원이 서로 __disentangled__ 되어 독립적인 factor(feature)를 학습하는 것이 목표이다.

### . ELBO


### . variational approximation
ancestral sampling

### . tug-of-war objective

### . practical coding issues with continuous output data


## . Posterior Collapse

### . Probabilistic PCA
1. distributions
2. posterior collapse
3. stability of stationary points

### . Linear VAE vs pPCA
1. model 
2. objective
3. Deep VAE


> 수정사항이나 질문은 댓글에 남겨주시면 감사하겠습니다 :)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNjIyODgzMDRdfQ==
-->