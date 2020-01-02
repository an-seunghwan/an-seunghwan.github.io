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

```python
num_tags = 12  # Number of unique issue tags
# text data를 처리하는데 필요한 전체 단어의 개수
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(shape=(None, ), name='title') # 다양한 길이의 정수(단어의 인덱스) 순서열(제목)
body_input = keras.Input(shape=(None, ), name='body') # 다양한 길이의 정수(단어의 인덱스) 순서열(내용)
tags_input = keras.Input(shape=(num_tags, ), name='tags') # 이진 벡터

# 제목의 각각의 단어를 64차원의 벡터로 변환한다.
title_features = layers.Embedding(num_words, 64)(title_input)
# 내용의 각각의 단어를 64차원의 벡터로 변환한다.
body_features = layers.Embedding(num_words, 64)(body_input)

# 제목의 임베딩된 단어들의 순서열을 하나의 128차원 벡터로 변환한다.
title_features = layers.LSTM(128)(title_features)
# 내용의 임베딩된 단어들의 순서열을 하나의 32차원 벡터로 변환한다.
body_features = layers.LSTM(32)(body_features)

# 모든 feature들을 하나의 큰 벡터로 concatenation을 이용해 묶는다.
x = layers.concatenate([title_features, body_features, tags_input])

# 우선순위를 예측하기 위해 logitstic regression layer를 추가한다.
priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(x)
# 부서를 예측하기 위해 softmax 분류 layer를 추가한다.
department_pred = layers.Dense(num_departments, activation='softmax', name='department')(x)

# 우선순위와 부서를 모두 예측하는 end-to-end 모형을 인스턴스화 한다.
model = keras.Model(inputs=[title_input, body_input, tags_input],
                    outputs=[priority_pred, department_pred])
```

이 모형을 compiling할 때, 각각의 output에 대해 다른 loss를 지정할 수 있다. 전체 training loss에 대한 기여도를 조절하기 위해 각각의 loss에 대해 다른 가중치 또한 부여할 수 있다. 추가로 각각의 loss에 이름을 부여할 수 있다.

```python
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss={'priority': 'binary_crossentropy',
                    'department': 'categorical_crossentropy'},
              # 또는 loss=['binary_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1., 0.2])
```
```python
import numpy as np

# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

# 이름별로 input과 output을 지정
model.fit({'title': title_data, 'body': body_data, 'tags': tags_data}, 
          {'priority': priority_targets, 'department': dept_targets},
          # 또는 [title_data, body_data, tags_data], [priority_targets, dept_targets] 가능
          # input과 output의 형태가 tuple임을 명심
          epochs=2,
          batch_size=32)
```
```
Train on 1280 samples
Epoch 1/2
1280/1280 [==============================] - 6s 5ms/sample - loss: 1.2764 - priority_loss: 0.7001 - department_loss: 2.8816
Epoch 2/2
1280/1280 [==============================] - 5s 4ms/sample - loss: 1.2747 - priority_loss: 0.7014 - department_loss: 2.8668
Out[39]: <tensorflow.python.keras.callbacks.History at 0x1e9c1d3a248>
```
* 학습과 평가에 대해서는 추후 다른 게시글로 더 자세히 다루도록 하겠습니다! (coming soon!)

## A toy resnet model
(생략, coming soon!)

## layer의 공유

Functional API 또 다른 장점은 공유 layer가 가능하다는 것이다. 공유 layer는 동일한 모형에서 여러번 사용되는 layer instance이다.

공유 layer는 유사한 space(말하자면 유사한 단어 특징을 가지는 다른 텍스트)에서 온 input을 encode할 때 주로 사용된다. 왜냐하면 다른 input들 사이에 정보를 공유하는 것이 가능하고, 이는 적은 데이터로부터 학습이 가능하도록 한다. 만약 주어진 단어가 하나의 input에서 관측되었다면, 공유 layer를 지나는 모든 input의 처리에 도움을 줄 것이다.

Functional API에서 layer를 공유하기 위해서는, 하나의 layer를 단순히 여러번 부르면 된다. 예를 들어, 여기 `Embedding` layer는 2개의 다른 input에 걸쳐 공유된다.

```python
# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype='int32')

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype='int32')

# 2개의 input을 encode하기 위해 동일한 layer를 사용
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
```
## layer의 graph에서 node를 추출하여 재사용

우리가 다루는 Functional API의 graph의 layer는 정적인(static) 구조이므로, 접근하여 보는 것이 가능하다. 이는 우리가 예를 들면 Functional 모형을 그래프화 할 수 있는 이유이다.

이는 중간 layer(graph의 node) activations에 접근을 하여 이를 다른 곳에서 재사용이 가능함을 의미한다. 이는 feature extraction 등에서 매우 유용하다!

예제를 통해 살펴보자.

```python
from tensorflow.keras.applications import VGG19
vgg19 = VGG19()
```
다음과 같은 방법을 통해 모형의 중간의 activations를 얻을 수 있다.
```python
features_list = [layer.output for layer in vgg19.layers]
```
따라서 우리는 중간 layer의 activations를 얻는 새로운 feature-extraction model을 만들 수 있다.
```python
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype('float32')
extracted_features = feat_extraction_model(img)
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbNTI4OTM1NTg4XX0=
-->