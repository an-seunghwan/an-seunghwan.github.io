---
title: "Note on VAE(작성중)"
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

* Auto-Encoding Variational Bayes: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
* Don’t Blame the ELBO! A Linear VAE Perspective on Posterior Collapse: [https://arxiv.org/pdf/1911.02469.pdf](https://arxiv.org/pdf/1911.02469.pdf)
* β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK: [https://openreview.net/pdf?id=Sy2fzU9gl](https://openreview.net/pdf?id=Sy2fzU9gl)
* [http://paulrubenstein.co.uk/variational-autoencoders-are-not-autoencoders/](http://paulrubenstein.co.uk/variational-autoencoders-are-not-autoencoders/)
* [https://www.borealisai.com/en/blog/tutorial-5-variational-auto-encoders/](https://www.borealisai.com/en/blog/tutorial-5-variational-auto-encoders/)
* [http://ruishu.io/2018/03/14/vae/](http://ruishu.io/2018/03/14/vae/)

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

### 3. ELBO

$$
\log{p_{\theta}(x)} + KL[q_{\phi}(z|x) \| p_{\theta}(z|x)] = \mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p_{\theta}(z)]
$$

RHS를 일반적으로 ELBO(Evidence Lower Bound)라고 부르는데, 이의 생긴 형태 때문에 ELBO를 objective로 사용하는 경우에 학습 목표가 reconstruction error와 KL-divergence를 최소화하는 것이라고 오해하는 경우가 많다.

앞에서도 언급했지만, VAE의 학습 목표는 data log-likelihood를 최대화하는 것임을 다시 밝힌다.

이때, true model에 대한 parameter $\theta$ 가 주어졌다고 생각해보자. 그리고 $q_{\phi}$의 모형 공간이 충분히 크다고 하면, ELBO를 최대화하는 VAE의 학습 목표를 다음과 같이 해석할 수 있다. 

(양변에 supremum을 취한다.)

$$
\sup_{\phi}{\log{p_{\theta}(x)} + KL[q_{\phi}(z|x) \| p_{\theta}(z|x)]}
$$

$$
= \log{p_{\theta}(x)} + \sup_{\phi}{KL[q_{\phi}(z|x) \| p_{\theta}(z|x)]}
$$

$$
= \log{p_{\theta}(x)} 
$$

$$
= \sup_{\phi}{\mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p_{\theta}(z)]}
$$

즉, $q_{\phi}(z \vert x)$ 의 모형 공간이 충분히 크다면, 

$$
q_{\phi}(z|x) = p_{\theta}(z|x)
$$

일 때 KL-divergence가 0이 되고 supremum을 얻을 수 있다. 따라서, ELBO를 최대화하는 학습 목표는 결국 data log-likelihood를 __achieve__ 하는 parameter $\phi$를 찾는 것이다.

### . variational approximation

$q_{\phi}(z)$ 는 실제 prior를 근사하는 분포인데, 이를 추정하기 위해서 우리는 Gaussian과 같은 simple parametric form을 사용한다. 또한, 이 approximated prior의 optimal choice는 true posterior $p_{\theta}(z \vert x)$ 가 되며, 이때 ELBO는 tight bound를 가진다.

ancestral sampling

### . practical coding issues with continuous output data


## . Posterior Collapse

### . tug-of-war objective

### . objective interpretation

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
eyJoaXN0b3J5IjpbNDgyNjY1OTI5LC0xMTYwOTM1NzMyXX0=
-->