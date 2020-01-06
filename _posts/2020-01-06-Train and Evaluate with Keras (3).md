---
title: "Train and Evaluate with Keras (3)"
excerpt: "Part 2(everything from scratch)"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-06 17:00:00 -0000
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---
> 이 글은 다음 문서를 참조하고 있습니다!
>[https://www.tensorflow.org/guide/keras/train_and_evaluate](https://www.tensorflow.org/guide/keras/train_and_evaluate)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.
> 비교적 간단한 내용이나 코드와 같은 경우에는 번역 없이 생략하니 꼭 원문을 확인해주시면 감사하겠습니다.

## Part 2: 처음부터 자신만의 training and evaluation loop 작성하기

### GradientTape을 사용하기: 첫 번째 end-to-end 예제

모형을 `GradienTape` scope 내부에서 호출하는 것은 loss 값에 대하여 layer의 훈련 가능한 가중치의 gradients를 불러올 수 있도록 한다. optimizer를 사용함으로써, 이러한 gradients를 변수의 update에 사용할 수 있다(이러한 변수는 `model.trainable_weights`를 통해 불러올 수 있다).

Part 1의 MNIST 모형을 통해 mini-batch gradient를 이용하는 custom training loop을 작성해보자
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyMjcwMTQ4MTZdfQ==
-->