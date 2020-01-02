---
title: "Keras functional API (1)"
excerpt: Keras functional API 1부
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-02 14:25:00 -0400
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

## setup
```python
import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
```
```
2.0.0
True
```
## 도입부

모형을 만들기 위해 우리는 흔히 `keras.Sequential()`을 사용한다. Functional API는 `Sequential`보다 더 유연한 모형을 만들 수 있게 해준다: 비선형 위상과, 공유 layer, 그리고 다중 input과 output을 다룰 수 있다.

이는 딥 러닝이 DAG에 아이디어를 기반하고 있기 때문이다. Functional API는 **layer의 graph를 구축**하는 도구이다.

다음의 모형을 생각하자.
```
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: probability distribution over 10 classes)
```

이는 단순한 3개 layer의 그래프이다.

이 모형을 Functional API를 이용해 만들기 위해서는, input node 부터 만드는 것이 필요하다.

```python
inputs = tf.keras.Input(shape=(784, )) 
# 여기서는 단순히 data의 shape만을 지정했다: 784차원의 벡터
# batch size는 항상 생략되며, 각 sample의 shape만을 지정했다.
```
구체적으로 `keras.Input`의 shape 인자에 대한 설명을 보자.
`shape`: shape tuple (integers)는 batch size를 포함하지 않는다. 예를 들어, `shape=(32, )`는 32차원 벡터로 구성된 batch를 input으로 가짐을 의미한다. 이 tuple의 원소에는 None도 가능하다; 'None`원소는 벡터의 shape이 알려지지 않았음을 의미한다. 

```python
# 만약 어떤 이미지 input의 shape이 (32, 32, 3)이라면, 다음과 같이 코드를 작성하면 된다.
img_inputs = tf.keras.Input(shape=(32, 32, 3))
```
```python
print(inputs.shape)
print(inputs.dtype)
```
```
(None, 32)
<dtype: 'float32'>
```
이 `inputs` object에 대해 호출을 하는 layer를 이용해 graph의 새로운 node를 만들 수 있다.
```python
dense = tf.keras.layers.Dense(64, activation='relu')
x = dense(inputs) # 새로운 node
```
몇개의 layer를 더 추가하고 input과 output을 명시하므로써 `Model`을 만들어보자.
```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='img')
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
```
## 훈련, 평가, 그리고 추론

Functional API로 제작한 모형은 Sequential 모형과 동일한 방식으로 훈련, 평가, 그리고 추론이 가능하다.

## 저장과 직렬화

Functional API로 제작한 모형은 Sequential 모형과 동일한 방식으로 저장과 직렬화가 가능하다.

Functional 모형의 일반적인 저장방식은 `model.save()`을 이용하면 하나의 파일로 모형의 모든 것을 저장할 수 있다. 나중에 모형을 만들때 사용한 코드에 직접 접근하지 않고서도 이 파일을 이용해 모형을 다시 제작할 수 있다.

이 파일은 다음과 같은 내용을 저장한다:
- 모형의 구조
- 모형의 weight의 값(학습 과정 중 배우는 것)
- 모형의 학습 구성 요소(`complie`에서 사용된 요소)
- optimizer와 이의 상태(이는 학습을 도중에 멈춘 지점부터 다시 시작할 수 있도록 해준다)

```python
model.save('{}/first_model.h5'.format(MODEL_PATH))
del model
model = keras.models.load_model('{}/first_model.h5'.format(MODEL_PATH))
```
* 저장과 직렬화에 대해서는 추후 다른 게시글로 더 자세히 다루도록 하겠습니다! (coming soon!)

## 다양한 모형을 정의하기 위해 동일한 layer의 graph를 사용

Functional API에서는 layer의 graph의 input과 output을 명시하면서 모형을 만들 수 있다. 이는 하나의 layer의 graph가 여러개의 모형을 생성하는데 사용될 수 있음을 의미한다.

아래의 예제에서는, 2개의 모형을 인스턴스화 하기 위해 동일한 layer의 층을 사용했다. 이미지를 16차원 벡터로 변환하는 `encoder` 모형과, 학습을 위한 end-to-end `autoencoder` 모형이다.

```python
encoder_input = keras.Input(shape=(28, 28, 1), name='img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()
```
```
Model: "encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
img (InputLayer)             [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 26, 26, 16)        160       
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 24, 24, 32)        4640      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 6, 6, 32)          9248      
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 4, 4, 16)          4624      
_________________________________________________________________
global_max_pooling2d_2 (Glob (None, 16)                0         
=================================================================
Total params: 18,672
Trainable params: 18,672
Non-trainable params: 0
_________________________________________________________________
Model: "autoencoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
img (InputLayer)             [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 26, 26, 16)        160       
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 24, 24, 32)        4640      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 6, 6, 32)          9248      
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 4, 4, 16)          4624      
_________________________________________________________________
global_max_pooling2d_2 (Glob (None, 16)                0         
_________________________________________________________________
reshape_2 (Reshape)          (None, 4, 4, 1)           0         
_________________________________________________________________
conv2d_transpose_8 (Conv2DTr (None, 6, 6, 16)          160       
_________________________________________________________________
conv2d_transpose_9 (Conv2DTr (None, 8, 8, 32)          4640      
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 24, 24, 32)        0         
_________________________________________________________________
conv2d_transpose_10 (Conv2DT (None, 26, 26, 16)        4624      
_________________________________________________________________
conv2d_transpose_11 (Conv2DT (None, 28, 28, 1)         145       
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________
```
encoding layer와 decoding layer의 구조가 대칭이기 때문에,  output의 shape이 input의 shape과 동일하다. `Conv2D`의 역은 `Conv2DTranspose`이고, `MaxPooling2D` layer의 역은 `UpSamping2D` layer이다.

## 모든 모형은 layer처럼 호출이 가능하다.

`Input`이나 다른 layer의 output을 호출함으로써 어떤 모형이든지 layer처럼 취급이 가능하다. 모형을 호출하는 것은 단순히 모형의 구조만을 호출하는 것이 아닌, 이의 weights 또한 재사용하는 것이다.

실제 예제를 통해 살펴보자. encoder와 decoder 모형을 생성하여 이를 연결(chain)하여 하나의 autoencoder 모형을 만드는 예제이다.

```python
encoder_input = keras.Input(shape=(28, 28, 1), name='original_img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
#encoder.summary()

decoder_input = keras.Input(shape=(16,), name='encoded_img')
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

decoder = keras.Model(decoder_input, decoder_output, name='decoder')
#decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
encoded_img = encoder(autoencoder_input) # 모형 호출
decoded_img = decoder(encoded_img) # 모형 호출
# encoder의 output을 input으로 받음으로써 chain처럼 연결이 된다.
autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
autoencoder.summary()
```
위의 예제에서 볼 수 있는 것처럼, 모형은 중첩이 가능하다: 모형은 submodel을 가질 수 있다(왜냐하면 모형은 layer와 같기 때문)

모형 중첩의 가장 흔한 사용 예시는 ensembling이다. 예를 들어, 다음의 예제는 여러 개의 모형 집합을 이들의 예측치의 평균을 이용해 ensemble하는 방법을 나타낸다.

```python
def get_model():
    inputs = keras.Input(shape=(128, ))
    outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
    return keras.Model(inputs, outputs)

model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128, ))
y1 = model1(inputs)
y2 = model1(inputs)
y3 = model1(inputs)
outputs = keras.layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs,
                             outputs=outputs)
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbMzkyMTQyMTY5LC0xMzIwODQxNjk5XX0=
-->