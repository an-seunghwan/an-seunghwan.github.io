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
neural style transfer 등에서 매우 유용하다!

## 자신만의 layer를 작성하여 API 확장하기

모든 layer는 `Layer`의 subclass이고 다음을 실행한다:
- `call` method: 해당 layer에 의해 실행되는 계산을 명시한다.
- `build` method: layer의 weights를 생성한다(코드 스타일에 따라 다르며 `__init__` 내부에도 생성 가능하다).

자세한 내용은 블로그 게시글 [https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1)/](https://an-seunghwan.github.io/tensorflow%202.0/Custom-modeling-with-Keras-(1)/)을 참조해주세요.

## Functional API를 사용하는 경우

새로운 모형을 제작할 때, Functional API를 써야할까 아니면 `Model`에서 subclass를 해야할까?

일반적으로, Functional API는 higher-level, 쉽고 사용하는데 안전하고, Models의 subclass가 지원하지 않는
많은 기능들이 있다.

하지만, Modeling subclassing은 layer들의 DAG로 표현할 수 없는 모형을 제작하는데 더 많은 유연함을 준다.

### Functional API의 강점

**It is less verbose**
```python
# Functional API
inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
mlp = keras.Model(inputs, outputs)

# subclassed version
class MLP(keras.Model):

  def __init__(self, **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(10)

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

# Instantiate the model.
mlp = MLP()
# Necessary to create the model's state.
# The model doesn't have a state until it's called at least once.
_ = mlp(tf.zeros((1, 32)))
```
**모형을 정의할 때 명확하다.**

Functional API에서는 input의 명시사항(shape과 dtype)이 미리 생성되므로(via `Input`), layer를 호출할 때마다 layer가 주어진 명시사항이 가정과 일치하는지 확인하고, 일치하지 않는다면 도움이 되는 error 메세지를 보일 것이다.

이러한 보장은 Functional API로 작성한 모든 모형이 작동함을 의미한다. 모든 디버깅(수렴과 관련한 디버깅 이외에도)은 모형 제작 도중에 정적으로 발생할 것이며, 실행 시점에 발생하지 않는다. 이는 compiler의 typechecking과 유사하다.

**Functional 모형은 시각화와 세부 내용을 확인할 수 있다.**

graph로써 모형을 시각화할 수 있고, graph 중간의 node에 접근할 수 있다.

**Functional 모형은 serialized 또는 복제될 수 있다.**

Functional 모형은 하나의 코드보다는 데이터 구조이기 때문에, 쉽게 serialized되고 원래 모형의 코드에 대한 접근이 없이 모형을 동일하게 생성 가능하게 해주는 하나의 파일로 저장될 수 있다. 
* 저장과 serialization에 대해서는 추후 다른 게시글로 더 자세히 다루도록 하겠습니다! (coming soon!)

### Functional API의 단점

**동적인 구조를 지원하지 않는다**

Functional API는 layer들의 DAG로 모형을 취급한다. 이는 대부분의 딥 러닝 구조이지만, 전부는 아니다: 예를 들어, 재귀적 네트워크나 Tree RNN은 이러한 가정을 따르지 않으므로 Functional API로 구현이 불가능하다.

**때때로, 처음부터 직접 작성할 필요가 있다.**

더 많은 차이에 대해서는 를 참조한 게시글을 확인해주세요(coming soon!)

## 다른 API style들의 Mix-and-matching

Functional API, Model subclassing, Sequential Model 등을 반드시 한가지만 선택해서 사용해야 하는 것은 아니다.
tf.keras API는 서로 상호작용이 가능하고, subclassed Model/Layer의 일부분으로써
Functional Model과 Sequential Model을 사용할 수 있다.

반대로, 어떠한 subclassed Layer나 Model을 이것이 다음의 규칙을 따르는
`call` method를 실행하기만 한다면 Functional API에 포함시킬 수 있다.
- `call(self, inputs, **kwargs)`: `inputs`가 tensor나 중첩된 tensor의 구조이거나, 
`**kwargs`가 non-tensor인자 일때(non-input)
- `call(self, inputs, training=None, **kwargs)`: `training`이 학습 모드나 추론 모드을 가리키는 boolean 변수인 경우
- `call(self, inputs, mask=None, **kwargs)`: `mask`가 boolean mask tensor인 경우(RNN에서 유용)
- `call(self, inputs, training=None, mask=None, **kwargs)`: `training`과 `mask`를 동시 사용 가능
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2ODA0MzcwMzRdfQ==
-->