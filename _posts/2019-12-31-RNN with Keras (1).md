---
title: "RNN with Keras (1)"
excerpt: Keras RNN의 기초

author_profile: false

date: 2019-12-31 16:45:00 -0400
categories: 
  - NLP
tags:
  - tensorflow 2.0
  - NLP
  - RNN
---
> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/rnn](https://www.tensorflow.org/guide/keras/rnn)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

RNN은 neural network의 일종으로 시계열 데이터나 언어와 같은 순서열 데이터에 효과적이다.

계획적으로, RNN layer는 `for` loop을 순서열의 시간 단위(timestep)별로 반복하고, 이때 현재까지 시간단위별로 관측한 정보들을 내부 상태로써 유지한다.

Keras RNN API는 다음과 같은 목적을 둔다.
- **쉬운 사용**: built-in `tf.keras.layers.RNN`, `tf.keras.layers.LSTM`, `tf.keras.layers.GRU` layer들을 
어려운 configuration 설정이 없이 빠르게 recurrent 모형을 적합할 수 있도록 해준다.
- **쉬운 customization**: 자신만의 RNN cell layer(`for` loop의 내부 부분)를 정의할 수 있고,
이를 일반적인 `tf.keras.layers.RNN` layer(`for` loop itself)와 함께 사용할 수 있다.
이는 최소한의 코드로 유연하게 다른 연구 아이디어의 원형을 만들 수 있도록 해준다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTMxNTAyNjU5XX0=
-->