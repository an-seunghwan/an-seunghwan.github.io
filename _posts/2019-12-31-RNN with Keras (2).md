---
title: "RNN with Keras (2)"
excerpt: Keras RNN의 더 많은 특징

author_profile: false

date: 2019-12-31 19:40:00 -0400
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

### Performance optimization and CuDNN kernels in TensorFlow 2.0 

Tensorflow 2.0에서는, built-in LSTM과 GRU layer는 GPU가 사용 가능할 경우 default로써 CuDNN kernel의 이점을 활용하도록
업데이트 되었다.
기존의 `keras.layers.CuDNNLSTM/CuDNNGRU` layer는 삭제되었으며, 
따라서 모형을 구성할 때 해당 하드웨어에서 작동할지 걱정할 필요가 없다.

CuDNN은 특정한 가정 하에서 작동될 수 있는데, 
이는 만약 built-in LSTM과 GRU의 default를 수정하면 CuDNN을 사용할 수 없다.

- `activation' 함수를 `tanh`가 아닌 다른 함수로 바꾸는 경우
- `recurrent_activation` 함수를 `sigmoid`가 아닌 다른 함수로 바꾸는 경우
- `recurrent_dropout` > 0인 경우
- `unroll`을 True으로 설정하는 경우(이는 LSTM/GRU의 내부 `tf.while_loop`을 unroll된 `for` loop으로 분해되도록 한다)
- `use_bias`가 False인 경우
- 입력 데이터가 strictly right padded 되지 않은 경우 masking을 사용하는 경우
(만약 mask가 strictly right padded data에 대응된다면, CuDNN을 여전히 사용할 수 있다. 이는 가장 흔한 경우이다.)
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTg2MTM3MDQxXX0=
-->