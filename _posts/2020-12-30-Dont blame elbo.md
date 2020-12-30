---
title: "Don't Blame the ELBO! Review"
excerpt: "Note on VAE and paper review"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-12-30 20:00:00 -0000
categories: 
  - VAE
tags:
  - ELBO
---
> Variational Inference, VAE 관련 여러 논문들과 블로그들을 보고 중요하다고 생각되는 수식과 아이디어 위주의 정리 포스팅입니다.

## 0. 참고 논문 

* Auto-Encoding Variational Bayes: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
* Don’t Blame the ELBO! A Linear VAE Perspective on Posterior Collapse: [https://arxiv.org/pdf/1911.02469.pdf](https://arxiv.org/pdf/1911.02469.pdf)

## 1. purpose

일반적으로 data log-likelihood는 closed form으로 주어지지 않는다. 따라서 $\log{p_{\theta}(x)}$를 최대화하는 $\theta$를 Maximum Likelihood 방법으로 찾기는 매우 어렵다(intractable). 따라서, latent variable ${\bf z}$를 도입하여 대신 데이터의 조건부 분포를 approximate하고, marginal data log-likelihood의 lower bound (ELBO)를 계산하여 이를 최대화하는 방법으로 $\theta$를 찾는다.

## 2. latent variable model
* distribution 가정

$$
\begin{aligned}
p({\bf z}) &= N_{\bf z}({\bf z} | 0, I) \\
p({\bf x}|{\bf z}) &= N_{\bf x}({\bf x} | D({\bf z};\theta), \beta \cdot I) 
\end{aligned}
$$

이때, $D({\bf z};\phi)$ 는 neural network로 구성되는 non-linear 함수이고, 데이터의 조건부 분포에서 parameterized mean이다. 따라서, non-linear latent factor model을 학습하는 것과 동일해진다. $p({\bf x} \vert {\bf z})$ 는 observation model이라고도 불린다.

## 3. ELBO

$$
\begin{aligned}
\log{p_{\theta}(x)} &=  \mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p(z)] + KL[q_{\phi}(z|x) \| p_{\theta}(z|x)] \\
&\geq \mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p(z)] \\
&= ELBO(\phi, \theta, \beta)
\end{aligned}
$$

$q_{\phi}(z \vert x)$ 가 표현하는 의 모형 공간이 충분히 크다고 하면, $q_{\phi}(z \vert x) = p_{\theta}(z \vert x)$ 가 되도록 하여 $KL[q_{\phi}(z \vert x) \| p_{\theta}(z \vert x)]$ 를 0으로 만드는 것이 가능하다(이 때 0은 negative KL-divergence의 supremum). 따라서, ELBO를 최대화하는 VAE의 학습 목표를 다음과 같이 해석할 수 있다. 

$$
\begin{aligned}
\sup_{\theta} {\log{p_{\theta}(x)}} &= \sup_{\phi} \sup_{\theta} {\log{p_{\theta}(x)} - KL[q_{\phi}(z|x) \| p_{\theta}(z|x)]} \\
&= \sup_{\phi} \sup_{\theta} {\mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p_{\theta}(z)]} \\
&= \sup_{\phi} \sup_{\theta} ELBO(\phi, \theta, \beta)
\end{aligned} 
$$

따라서, ELBO를 최대화하는 parameter들을 찾게되면, 이는 원래의 목적인 marginal data log-likelihood를 최대화하는 parameter와 일치하게 된다.

## 4. reparametrization trick

$x$ 에 의존하는 $z$ 에 대한 approximated posterior 분포:

$$q_{\phi}(z \vert x) = N_x(\mu_{\phi}(x), diag(\sigma_{\phi}(x)))$$

$\mu_{\phi}(x)$ 와 $\sigma^2_{\phi}(x)$ 는 neural network로 구성된 non-linear 함수의 결과이다( $diag$ 는 대각행렬를 의미).

하지만, $\mu_{\phi}(x)$ 와 $\sigma^2_{\phi}(x)$ 는 $x$가 주어졌을 때 매우 자유로운 표현력을 가지고 있지만, Gaussian이라는 분포의 형태는 uni-modal이므로 표현력이 제한적이라는 단점을 갖게 된다. 또한, 만약에 실제 prior가 multi-modal인 경우에는 분포 형태의 한계로 인해 KL-divergence를 통해서 posterior와 prior가 그 구조가 유사하도록 만들기가 어렵다.

reparametrization trick은 posterior로부터 latent variable을 backpropagation이 가능하도록 sampling해주는 방법이다.

1. sampling $\epsilon \sim p(\epsilon) = N(0, I)$ 
	
2. compute forward pass network for given $x$:  $\mu_{\phi}(x)$, $\sigma^2_{\phi}(x)$

3. sampling latent variable: $z = \mu_{\phi}(x) + \sigma_{\phi}(x) \cdot \epsilon$

## 5. Simulation of ELBO and Interpretation

$i$번째 datapoint ${\bf x}_i$에 대한 negative ELBO의 simulation 수식은 다음과 같다 (이 때 $d$는 latent variable의 차원의 크기):

$$
\begin{aligned}
\frac{1}{2\beta} \cdot \| {\bf x}_i - D_{\theta}({\bf z}_i) \|_2^2 +\frac{d}{2} \log 2\pi\beta + KL(q({\bf z}|{\bf x}_i;\phi)\|p({\bf z})) 
\end{aligned}
$$

앞의 2개의 항은 observation model이 Gaussian임을 이용하여 수식으로 전개한 것이다. ${\bf z}_i$는 posterior 분포에서 sampling되었다. 이제 양변을 variance parameter인 $\beta$로 scaling하면 아래와 같다:

$$
\begin{aligned}
\frac{1}{2} \| {\bf x}_i - D_{\theta}({\bf z}_i) \|_2^2 + \beta \cdot KL(q({\bf z}|{\bf x}_i;\phi)\|p({\bf z})) + \beta \cdot \frac{d}{2} \log 2\pi\beta
\end{aligned}
$$

이를 해석하면 parameterized mean과 datapoint 사이의 MSE와 KL-divergence에 $\beta$를 곱한 것을 더한 식이된다. 따라서, $\beta$를 작은 값으로 설정하면, 기존의 KL-divergence의 영향력을 줄이기 위해 작은 weight를 곱해주는 heuristic한 접근들과 그 의미가 일치해진다.

<!--stackedit_data:
eyJoaXN0b3J5IjpbNzQ5Mzk4NDM0LDE5NzM0NzAyMTEsLTQ2MT
Q3MzI3NSwtMTAzNDM0ODU3MCw3NjAwNzYzODldfQ==
-->