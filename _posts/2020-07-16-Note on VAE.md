---
title: "Note on VAE and Posterior Collapse(작성중)"
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
>
> 논문에 대한 리뷰가 아닌 개인적인 정리 목적의 글임을 밝힙니다.

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

### 4. variational approximation

* 

$q_{\phi}(z)$ 는 실제 prior를 근사하는 분포인데, 이를 추정하기 위해서 우리는 Gaussian과 같은 simple parametric form을 사용한다. 또한, 이 approximated posterior의 optimal choice는 true posterior $p_{\theta}(z \vert x)$ 가 되며, 이때 ELBO는 tight bound를 가진다.

$q_{\phi}(z)$를 추정하기 위해 사용하는 방법이 variational approximation 인데, $q_{\phi}(z)$ 대신 $q_{\phi}(z \vert x)$를 사용한다. 즉, $z$에 대한 approximated posterior 분포가 $x$에 의존하도록 만드는 것이다. 이를 수식으로 쓰면 다음과 같다.

$$
q_{\phi}(z \vert x) = N_x(\mu_{\phi}(x), diag(\sigma^2_{\phi}(x)))
$$

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

### 5. Practical coding issues with continuous output data

* Decoder(generator)는 오직 embedding means $\mu_{\phi}(x)$만을 결과값으로 반환한다. 이때 embedding variance는 결과값으로 반환하지 않고, 학습하지 않는 parameter로써 global variance를 1로 설정한다.

예를 들어 생각해보면, 일반적으로 VAE를 실험할 때 MNIST 데이터를 주로 사용하게 된다. MNIST 데이터를 사용할 때는 보통 [-1, 1]로 pixel 값을 scaling한 뒤 사용하게 된다.

따라서, MNIST의 데이터 분포를 평균이 0인 정규분포를 따른다고 가정할 때, [-1, 1] 범위의 값에 대해 분산이 1인 정규분포를 사용한다면 아주 많은 양의 noise가 $p_{\theta}(x \vert x)$의 분포로 부터 sampling할 때 더해질 것이다.

* 또한, training을 하는 과정에서 $p_{\theta}(x \vert x)$로부터 실제로 sampling을 하지 않는다. 대신에 $\mu_{\phi}(x)$를 마치 sampling된 결과처럼 생각하여 사용한다.

reconstruction error term을 살펴보자. reconstruction error의 최소화는 다음의 log 확률값을 최대화하는 것과 동일하다.

$$
\log{p_{\theta}(x|z)} = -\frac{1}{2\sigma^2_{\theta}(z)} \| x - \mu_{\theta}(z) \|_2^2 - \frac{dimension}{2} \log{2 \pi \sigma^2_{\theta}(z)}
$$ 

$$
X|Z=z \sim N(\mu_{\theta}(z), diag(\sigma^2_{\theta}(z)))
$$

(단, $x$의 분산에 대해 모든 원소가 동일한 대각행렬을 가정)

만약, decoder의 결과인 $(\mu_{\theta}(z)$가 $x$에 대해 충분한 표현력을 가지고 있어 복원을 잘 한다는 이상적인 상황을 가정하면, L2-regularization term은 거의 0에 가까울 것이다. 따라서 이러한 이상적인 경우에 log 확률값을 최소화하기 위해서는 뒤의 $\frac{dimension}{2} \log{2 \pi \sigma^2_{\theta}(z)}$를 최소화해야하므로, decoder가 이상적인 복원력을 가지고 있을 때, $z$가 주어졌을 때 $x$의 분산은 빠르게 0으로 다가갈 것이다(Lebsegue measure를 이용해 수학적으로 증명 가능).

이러한 관점에서, decoder의 표현력이 충분히 학습되었다면, 해당하는 분포의 분산이 0에 매우 가까울 것이므로 실제 sampling을 하는 것이 아니라, 대신 $(\mu_{\theta}(z)$를 이용해 모형을 training한다.

## 2. Posterior Collapse

### 1. objective interpretation

VAE의 objective function, 그 중에서도 KL-divergence term은 closed-form으로 적을 수 있다.

$$
KL[q_{\phi}(z|x) \| p_{\theta}(z)]
$$

$$
= \frac{1}{2} \left( \| \mu_{\phi}(x) - 0 \|_2^2 - dimension + tr(diag(\sigma^2_{\phi}(x))) - \log{det(diag(\sigma_{\phi}(x)))} \right)
$$

$$
= \frac{1}{2} \left( \| \mu_{\phi}(x) - 0 \|_2^2  + \sum_{i=1}^n (\sigma_{\phi}(x)_i - \log{\sigma_{\phi}(x)_i}) - dimenstion \right) 
$$

따라서, latent variable의 차원인 $dimension$은 클수록 KL-divergence가 감소하고, embedding means인 $\mu_{\phi}(x)$는 0, 그리고 embedding variance인 $\sigma_{\phi}(x)$는 1에 가까울 수록 KL-divergence가 감소한다.

따라서, 이러한 측면에서 보면 마치 approximated posterior의 분포가 $N(0, I)$에 가까워지면서, 즉 $x$에 대한 정보를 점점 잃어가는 것처럼 생각할 수 있다(become less expressive and shrinks to $N(0, I)$). 이러한 현상을 __posterior collapse__ (learned variational distribution is close to the prior)라고 부르는데, 많은 연구들이 이러한 현상의 원인으로 KL-divergence를 생각했다.

### 2. 상충되는 objective

앞에서 잠깐 언급한 것처럼, ELBO의 식은 reconstruction error와 KL-divergence의 합으로 구성되어 있는 것으로 볼 수 있다. 하지만 이 2개의 합을 최대화하는 과정에서, 각각의 term의 목적이 서로 상충되는 경우가 발생한다.

* to get better reconstructions

embedding means $\mu_{\phi}(x)$가 서로 멀리 떨어져 있고, embedding variance $\sigma_{\phi}(x)$가 0에 가까울수록, 주어진 $x$들 사이에 구분되는 정보를 가진다. 이러한 경우에 reconstruction error가 작다.

하지만,  이러한 경우에는 KL-divergence가 커지는 문제가 발생한다.

* to get small KL-divergence

embedding means $\mu_{\phi}(x)$가 0에 가깝고, embedding variance $\sigma_{\phi}(x)$가 1에 가까울수록 true prior인 정규분포의 형태에 가까워지므로 KL-divergence가 감소한다. 

하지만, 주어진 $x$들 사이에 구분되는 정보를 잃어버리므로 이러한 경우에 reconstruction error가 크다.

((그림 첨부))

> 특히 3, 4번에 대한 내용은 많이 부족합니다. 논문에 대한 자세한 리뷰가 아니고 간단한 아이디어 정리이니 자세한 수식은 본문을 참조해주세요.

### 3. Probabilistic PCA

* distributions

다음과 같은 실제 분포가 주어져있을 때, 

$$
p(z) = N(0, I)
$$

$$
p(x|z) = N(Wz+\mu, \sigma^2I)
$$

probabilistic PCA는 다음의 data 분포와 posterior 분포를 찾게된다.

$$
p(x) = N(\mu, MM^\top + \sigma^2I)
$$

$$
p(z|x) = N(M^{-1}W^\top(x-\mu), \sigma^2I)
$$

where $M = W^\top W + \sigma^2I$

* posterior collapse

MLE가 아닌 다른 stationary points는 $W_{MLE}$의 columns를 0으로 바꿈으로써 얻을 수 있다. 극단적으로, $W_{MLE}$의 모든 columns들을 0으로 바꾼다면, posterior 분포는 $p(z \vert x) = N(0, \sigma^2I)$가 되고, 이러한 경우를 posterior collapse라고 한다.

* stability of stationary points

((아직 어려워서 내용 추가를 못했습니다...))

### 4. Linear VAE vs pPCA

* model 

다음의 모형을 linear VAE라고 하고, approximated posterior는 global optimum의 경우에 pPCA의 true posterior를 정확하게 복원할 수 있다.

$$
p(x|z) = N(Wz+\mu. \sigma^2I)
$$

$$
q(z|x) = N(V(x-\mu), D)
$$

이때, $D$는 diagonal covariance matrix($\sigma^2I$)로, 모든 데이터 point에 대해 동일하게 사용된다.

* objective

$$
\log{p(x)} = \mathbb{E}_{q(z|x)}[\log{p(x|z)}] - KL[q(z|x) \| p(z)] + KL[q(z|x) \| p(z|x)]
$$

$$
\log{p(x)} = ELBO + KL[q(z|x) \| p(z|x)]
$$

global optimum인 경우에, $q(z \vert x)$가 $p(z \vert x)$를 정확하게 복원하여 KL-divergence가 0이 되어 ELBO와 marginal likelihood(of pPCA)가 tight bound를 가지게 된다. 즉, ELBO가 marginal likelihood와 정확히 일치하게 된다.



* Deep VAE


> 수정사항이나 질문은 댓글에 남겨주시면 감사하겠습니다 :)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4Njg2MTYxMzcsLTEzNDMwNDIxMzEsOD
cwMjE4Mzc5LDE1ODA3MTIxLC00NzY1NjUwMDcsOTIwODA4NDMy
LDIwOTEzMTg4MjgsNjA0MTg2ODQ1LC0xNDI3NjkzMTM4LC0xMT
YwOTM1NzMyXX0=
-->