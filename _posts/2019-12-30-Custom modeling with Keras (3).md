---
title: "Custom modeling with Keras (3)"
excerpt: 모형의 적합!
toc: true
toc_sticky: false

author_profile: false

date: 2019-12-30 16:30:00 -0400
categories: 
  - tensorflow 2.0
tags:
  - tensorflow 2.0
---

> 이 글은 다음 문서를 참조하고 있습니다!
> [https://www.tensorflow.org/guide/keras/custom_layers_and_models](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
> 
> 아직 한글로 번역이 되어있지 않은 문서가 많아 공부를 하면서 번역을 진행하고 있습니다.

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.clear_session()
```
일반적으로, **`Layer` class를 내부 계산 블록으로, `Model` class를 학습의 대상인 외부 모형으로 정의한다.** 예를 들어, ResNet50 모형에서, 여러개의 ResNet 블록을 `Layer`로 정의하고,  하나의 단일 `Model`을 이용해 전체 네트워크를 아우르는 모형을 적합한다.

`Model` class는 `Layer`와 거의 동일한 API를 갖지만, 몇가지 차이를 갖는다.
- built-in loops를 갖는다(`model.fit()`, `model.evaluate()`, `model.predict()`)
- `model.layers`를 이용해 내부 layer의 특성에 접근할 수 있다.
- saving and serialization APIs를 가진다.

쉽게 말해서, "Layer" class는 우리가 흔히 이야기하는 "layer"("convolution layer" or "recurrent layer")나 "block"("ResNet block" or "Inception block")의 의미를 갖는다. 반면에, "Model" class는 우리하 흔히 이야기하는 "모형"("deep learning model")이나 "네트워크"("deep neural network")의 의미를 갖는다.

예를 들어, 다음과 같이 `Model`에 대해 `fit()`을 이용해 training할 수 있고, `save_weights`를 이용해 저장할 수 있다.

```python
class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)

resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save_weights(filepath)
```

## end-to-end example

지금까지 배운 것들:
- `Layer`는 상태(`__init__`이나 `build`에서 생성)나 계산(in `call`)을 요약
- Layers는 다른 계산 블록에서 재귀적으로 사용될 수 있다.
- Layers는 손실 값을 생성하고 추적할 수 있다.
- 학습의 대상은 `Model`이며, `Layer`와 유사하지만 training과 serialization 도구를 포함한다.
    
An end-to-end example: MNIST digits데이터를 이용한 Variational AutoEncoder(VAE)
여기서의 VAE는 `Model`의 subclass이며, `Layer`의 subclass layer들로 중첩되어 구성된다. 또한 정규화 손실 값을 가진다(KL divergence)

### `Layer` class를 정의
```python
class Sampling(layers.Layer):
    '''(z_mean, z_log_var)를 z를 추출하기 위해 사용한다(z는 digit을 encoding하기 위한 벡터)'''
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    '''MNIST digits를 triplet (z_mean, z_log_var, z)로 mapping'''
    
    def __init__(self,
                 latent_dim=32,
                 intermediate_dim=64,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
    
    # 함수형 API와 유사한 형태
    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    '''encoding된 digit 벡터 z를 다시 읽을 수 있는 digit으로 변환'''
    
    def __init__(self,
                 original_dim,
                 intermediate_dim=64,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    
    # 함수형 API와 유사한 형태
    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)
```

### `Model` class를 정의
```python
class VariationalAutoEncoder(tf.keras.Model):
    ''''encoder와 decoder를 training을 위한 end-to-end 모형으로 합친다'''
    
    def __init__(self,
                 original_dim,
                 intermediate_dim=64,
                 latent_dim=32,
                 name='autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, 
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim,
                               intermediate_dim=intermediate_dim)
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
```

### training을 위한 setting
```python
original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 3
```

### training 1: 직접 gradient를 계산하여 적용하는 방식
```python
# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch {}'.format(epoch))
    
    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses) # Add KLD regularization loss
        
        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))
        
        loss_metric(loss)
        
        if step % 100 == 0:
            print('step {}: mean loss = {}'.format(step, loss_metric.result()))
```

```
Start of epoch 0
step 0: mean loss = 0.3582668900489807
step 100: mean loss = 0.126471608877182
step 200: mean loss = 0.09953459352254868
step 300: mean loss = 0.0894065797328949
step 400: mean loss = 0.08442515879869461
step 500: mean loss = 0.08106688410043716
step 600: mean loss = 0.07886670529842377
step 700: mean loss = 0.07723680883646011
step 800: mean loss = 0.07605761289596558
step 900: mean loss = 0.07502540946006775
Start of epoch 1
step 0: mean loss = 0.07473193109035492
step 100: mean loss = 0.07405747473239899
step 200: mean loss = 0.07356826961040497
step 300: mean loss = 0.07308384776115417
step 400: mean loss = 0.07274124771356583
step 500: mean loss = 0.07234357297420502
step 600: mean loss = 0.07204995304346085
step 700: mean loss = 0.07174880802631378
step 800: mean loss = 0.07151103764772415
step 900: mean loss = 0.07124651968479156
Start of epoch 2
step 0: mean loss = 0.07117010653018951
step 100: mean loss = 0.07099448889493942
step 200: mean loss = 0.07085386663675308
step 300: mean loss = 0.07070086896419525
step 400: mean loss = 0.0706046000123024
step 500: mean loss = 0.07044930756092072
step 600: mean loss = 0.07033801823854446
step 700: mean loss = 0.07021458446979523
step 800: mean loss = 0.07011080533266068
step 900: mean loss = 0.0699886828660965
```

### training 2: built-in training loops를 활용
```python
vae = VariationalAutoEncoder(784, 64, 32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
```
```
Train on 60000 samples
Epoch 1/3
60000/60000 [==============================] - 9s 152us/sample - loss: 0.0746
Epoch 2/3
60000/60000 [==============================] - 5s 86us/sample - loss: 0.0676
Epoch 3/3
60000/60000 [==============================] - 4s 75us/sample - loss: 0.0675
Out[51]: <tensorflow.python.keras.callbacks.History at 0x2a101fc7348>
```

* Functional API 부분은 생략
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwNDIzNzkxMjMsMzk1NzE4NTIsLTQ4ND
Y5NjYxNiwtNzcxODMwODcyLC0yMDI3OTAxODUwXX0=
-->