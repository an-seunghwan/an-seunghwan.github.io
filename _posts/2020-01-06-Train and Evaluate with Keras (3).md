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

Part 1의 MNIST 모형을 통해 mini-batch gradient를 이용하는 custom training loop을 작성해보자.

**setup**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
```
**end-to-end example**
```python
# 모형을 정의
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 3
for epoch in range(epochs): # iterate over epochs
    print('Start of epoch {}'.format(epoch))
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset): # iterate over batches
        
        # forward pass 동안에 operation들을 기록하기 위해 GradientTape를 연다.
        # 이는 자동 미분을 가능하도록 한다.
        with tf.GradientTape() as tape:
            
            # layer의 전진 학습을 실행
            logits = model(x_batch_train) # mini-batch의 logits
            
            # mini-batch의 손실 값을 계산
            loss_value = loss_fn(y_batch_train, logits)
            
        # 각각의 loss 값에 대한 학습 가능한 변수들을 자동으로 불러온다.
        grads = tape.gradient(loss_value, model.trainable_weights)
        
        # loss 값을 최소화하는 방향으로 gradient descent 방법으로 변수들을 update 한다.
        # 이는 1개의 batch(mini-batch)에 대해서 진행
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 200 == 0:
            print('Training loss (for one batch) at step {}: {}'.format(step, float(loss_value)))
            print('seen so far: {} samples'.format((step + 1) * 64))
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1OTIzMzcwNDhdfQ==
-->