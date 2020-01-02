---
title: "Keras functional API (2)"
excerpt: Keras functional API 2부
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-02 20:00:00 -0400
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
  - keras
---
> 이 글은 다음 문서를 참조하고 있습니다!
>[https://www.tensorflow.org/guide/keras/functional](https://www.tensorflow.org/guide/keras/functional)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.
> 비교적 간단한 내용이나 코드와 같은 경우에는 번역 없이 생략하니 꼭 원문을 확인해주시면 감사하겠습니다.

## 복잡한 위상의 모형 다루기

### 다중 input과 output 다루기

Functional API는 다중 input과 output을 다루기 매우 쉽다.
이는 Sequential API에서는 할 수 없다.

여기 간단한 예제를 살펴보자. 우선 순위에 따라 맞춤 발생 티켓의 순위를 매기고 이를 적절한 부서에 routing하는 시스템을 구축한다고 가정하자.

모형은 3개의 input을 갖는다:
- 티켓의 제목(text input)
- 티켓의 내용(text input)
- 사용자에 의해 추가된 임의의 tags(categorical input)

모형은 2개의 output을 갖는다:
- 0에서 1사이의 우선순위 score(scalar sigmoid output)
- 이 티켓을 다루는 부서(softmax output over the set of departsments)


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4Mzg1OTc2ODJdfQ==
-->