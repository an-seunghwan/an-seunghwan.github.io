---
title: "Don't Blame the ELBO Review"
excerpt: "Note on VAE and paper review"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-12-30 20:00:00 -0000
categories: 
  - VAE
tags:
  - 
---
> Variational Inference, VAE 관련 여러 논문들과 블로그들을 보고 중요하다고 생각되는 수식과 아이디어 위주의 정리 포스팅입니다.

## 0. 참고 논문 

* Auto-Encoding Variational Bayes: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
* Don’t Blame the ELBO! A Linear VAE Perspective on Posterior Collapse: [https://arxiv.org/pdf/1911.02469.pdf](https://arxiv.org/pdf/1911.02469.pdf)

## 1.  VAE

### 1. purpose

일반적으로 data log-likelihood는 closed form으로 주어지지 않는다. 따라서 $\log{p_{\theta}(x)}$를 최대화하는 $\theta$를 Maximum Likelihood 방법으로 찾기는 매우 어렵다(intractable). 따라서, latent variable ${\bf z}$를 도입하여 대신 데이터의 조건부 분포를 approximate하고, marginal data log-likelihood의 lower bound (ELBO)를 계산하여 이를 최대화하는 방법으로 $\theta$를 찾는다.

### 2. latent variable model
* distribution 가정

$$
\begin{aligned}
p({\bf z}) &= N_{\bf z}({\bf z} | 0, I) \\
p({\bf x}|{\bf z}) &= N_{\bf x}({\bf x} | D({\bf z};\phi), \beta^2I) 
\end{aligned}
$$

이때, $D({\bf z};\phi)$는 neural network로 구성되는 non-linear 함수이고, 데이터의 조건부 분포에서 parameterized mean이다. 따라서, non-linear latent factor model을 학습하는 것과 동일해진다.

### 3. ELBO

$$
\begin{aligned}
\log p_{\theta}({\bf x}) &= \log p_{\theta}({\bf x}) \int q_{\phi}({\bf z}|{\bf x}) d{\bf z} \\
\end{aligned}
$$

&=& \int q_{\phi}(\bz|\bx) \mbox{log} \frac{p_{\theta}(\bx,\bz) p(\bx)}{p_{\theta}(\bx,\bz)} dz \nonumber\\
&=& \int q_{\phi}(\bz|\bx) \mbox{log} \frac{p_{\theta}(\bx|\bz) p(\bz)}{p_{\theta}(\bz|\bx)} dz \\
&=& \int \left(q_{\phi}(\bz|\bx) \mbox{log} p_{\theta}(\bx|\bz) + q_{\phi}(\bz|\bx) \mbox{log} p(\bz) - q_{\phi}(\bz|\bx) \mbox{log} p_{\theta}(\bz|\bx) \right) dz \pm \int q_{\phi}(\bz|\bx) \mbox{log} q_{\phi}(\bz|\bx) dz \nonumber\\
&=& \int q_{\phi}(\bz|\bx) \mbox{log} p_{\theta}(\bx|\bz) dz + \int q_{\phi}(\bz|\bx) \mbox{log} \frac{q_{\phi}(\bz|\bx)}{p_{\theta}(\bz|\bx)} dz - \int q_{\phi}(\bz|\bx) \mbox{log} \frac{q_{\phi}(\bz|\bx)}{p(\bz)} dz \nonumber\\
&=& \mathbb{E}_{q_{\phi}(\bz|\bx)} [\mbox{log} p_{\theta}(\bx|\bz)]- KL\left( q_{\phi}(\bz|\bx) \| p(\bz) \right) + KL\left( q_{\phi}(\bz|\bx) \| p_{\theta}(\bz|\bx) \right) \nonumber \\
&=& -\mathcal{L}(\theta, \phi ; \bx) + KL\left(q_{\phi}(\bz|\bx) \| p_{\theta}(\bz|\bx)\right) \nonumber\\
&\geq& -\mathcal{L}(\theta, \phi ; \bx), \nonumber
\end{aligned}
$$

$$\log{p_{\theta}(x)} + KL[q_{\phi}(z|x) \| p_{\theta}(z|x)] = \mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p_{\theta}(z)]$$

RHS를 일반적으로 ELBO(Evidence Lower Bound)라고 부르는데, 이의 생긴 형태 때문에 ELBO를 objective로 사용하는 경우에 학습 목표가 reconstruction error와 KL-divergence를 최소화하는 것이라고 오해하는 경우가 많다.

앞에서도 언급했지만, VAE의 학습 목표는 data log-likelihood를 최대화하는 것임을 다시 밝힌다.

이때, true model에 대한 parameter $\theta$ 가 주어졌다고 생각해보자. 그리고 $q_{\phi}$의 모형 공간이 충분히 크다고 하면, ELBO를 최대화하는 VAE의 학습 목표를 다음과 같이 해석할 수 있다. 

(양변에 supremum을 취한다.)

$$\sup_{\phi}{\log{p_{\theta}(x)} + KL[q_{\phi}(z|x) \| p_{\theta}(z|x)]}$$

$$= \log{p_{\theta}(x)} + \sup_{\phi}{KL[q_{\phi}(z|x) \| p_{\theta}(z|x)]}$$

$$= \log{p_{\theta}(x)} $$

$$= \sup_{\phi}{\mathbb{E}_{q_{\phi}(z|x)}[\log{p_{\theta}(z|x)}] - KL[q_{\phi}(z|x) \| p_{\theta}(z)]}$$

즉, $q_{\phi}(z \vert x)$ 의 모형 공간이 충분히 크다면, 

$$q_{\phi}(z|x) = p_{\theta}(z|x)$$

일 때 KL-divergence가 0이 되고 supremum을 얻을 수 있다. 따라서, ELBO를 최대화하는 학습 목표는 결국 data log-likelihood를 __achieve__ 하는 parameter $\phi$를 찾는 것이다.

### 4. variational approximation

* 

$q_{\phi}(z)$ 는 실제 prior를 근사하는 분포인데, 이를 추정하기 위해서 우리는 Gaussian과 같은 simple parametric form을 사용한다. 또한, 이 approximated posterior의 optimal choice는 true posterior $p_{\theta}(z \vert x)$ 가 되며, 이때 ELBO는 tight bound를 가진다.

$q_{\phi}(z)$를 추정하기 위해 사용하는 방법이 variational approximation 인데, $q_{\phi}(z)$ 대신 $q_{\phi}(z \vert x)$를 사용한다. 즉, $z$에 대한 approximated posterior 분포가 $x$에 의존하도록 만드는 것이다. 이를 수식으로 쓰면 다음과 같다.

$$q_{\phi}(z \vert x) = N_x(\mu_{\phi}(x), diag(\sigma^2_{\phi}(x)))$$

$\mu_{\phi}(x)$와 $\sigma^2_{\phi}(x)$는 neural network로 구성된 non-linear 함수의 결과로 $x$에 의존하는 분포가 되도록 만들어준다($diag$는 대각행렬).

* 

하지만, $\mu_{\phi}(x)$와 $\sigma^2_{\phi}(x)$는 $x$가 주어졌을 때 매우 자유로운 표현력을 가지고 있지만, Gaussian이라는 분포의 형태상 limited expression이라는 단점을 갖게 된다.

또한, approximated prior로 uni-modal Gaussian을 사용했으므로, 만약 true prior가 multi modal인 경우에는 그 성능이 저하될 수 있다.

((그림 첨부))

* reparametrization trick

	1.) $\epsilon \sim p(\epsilon) = N(0, I)$ 
	
	2.) forward pass network:  $\mu_{\phi}(x)$, $\sigma^2_{\phi}(x)$
	
	3.) sampling $z = \mu_{\phi}(x) + \sigma^2_{\phi}(x) \epsilon$

 reparametrization trick은 $N_x(\mu_{\phi}(x), diag(\sigma^2_{\phi}(x)))$ 분포에서 직접적으로 $z$를 sampling하는 것이 아니라 쉽게 $N(0, I)$로부터 난수를 생성하여 $z$를 sampling하는 방법이다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY2ODQ3MDQ4OSw3NjAwNzYzODldfQ==
-->