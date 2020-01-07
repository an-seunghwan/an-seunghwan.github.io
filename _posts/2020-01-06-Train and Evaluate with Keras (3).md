---
title: "2020-01-06-Train and Evaluate with Keras (3)"
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
```
Start of epoch 0
Training loss (for one batch) at step 0: 2.316042423248291
seen so far: 64 samples
Training loss (for one batch) at step 200: 2.2370457649230957
seen so far: 12864 samples
Training loss (for one batch) at step 400: 2.143077850341797
seen so far: 25664 samples
Training loss (for one batch) at step 600: 2.086883544921875
seen so far: 38464 samples
Start of epoch 1
Training loss (for one batch) at step 0: 1.9219386577606201
seen so far: 64 samples
Training loss (for one batch) at step 200: 1.9164278507232666
seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.7983959913253784
seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.6359037160873413
seen so far: 38464 samples
Start of epoch 2
Training loss (for one batch) at step 0: 1.6075100898742676
seen so far: 64 samples
Training loss (for one batch) at step 200: 1.5245256423950195
seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.4214186668395996
seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.106119155883789
seen so far: 38464 samples
```

### 저수준의 metric

built-in metrics이나 custom metrics 모두 custom training loop에서 사용하는 것은 매우 쉽다. 여기 사용 방법의 흐름이 있다:
- loop의 시작에서 metric을 Instantiate
- 각 batch가 종료되고 `metric.update_state()`을 호출
- metric의 현재 값을 보여주기 위해서는 `metric.result()`를 호출
- metric의 state를 초기화할 때 `metric.reset_states()`를 호출(일반적으로 epoch의 종료 시점)

`SparseCategoricalAccuracy`를 이용한 예제를 살펴보자.

```python
# 모형을 정의
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# metric
train_acc_metric = keras.metrics.SparseCategoricalCrossentropy()
val_acc_metric = keras.metrics.SparseCategoricalCrossentropy()

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

epochs = 3
for epoch in range(epochs): # iterate over epochs
    print('Start of epoch {}'.format(epoch))
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset): # iterate over batches
        
        with tf.GradientTape() as tape:
            
            logits = model(x_batch_train) # mini-batch의 logits
            loss_value = loss_fn(y_batch_train, logits)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # update training metric
        train_acc_metric(y_batch_train, logits)

        if step % 200 == 0:
            print('Training loss (for one batch) at step {}: {}'.format(step, float(loss_value)))
            print('seen so far: {} samples'.format((step + 1) * 64))

    # 각 epoch가 종료되고 metric을 출력
    train_acc = train_acc_metric.result()
    print('Training acc over epoch: {}'.format(float(train_acc)))
    # training metric을 초기화
    train_acc_metric.reset_states()
    
    # epoch 종료 후 validation loop를 실행
    for x_batch_val, y_batch_val in val_dataset:
        val_logtis = model(x_batch_val)
        # update val metrics
        val_acc_metric(y_batch_val, val_logtis)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: {}'.format(float(val_acc)))
```
```
Start of epoch 0
Training loss (for one batch) at step 0: 2.3578922748565674
seen so far: 64 samples
Training loss (for one batch) at step 200: 2.2744617462158203
seen so far: 12864 samples
Training loss (for one batch) at step 400: 2.1668717861175537
seen so far: 25664 samples
Training loss (for one batch) at step 600: 2.078711748123169
seen so far: 38464 samples
Training acc over epoch: 2.180353879928589
Validation acc: 1.9919371604919434
Start of epoch 1
Training loss (for one batch) at step 0: 2.0067782402038574
seen so far: 64 samples
Training loss (for one batch) at step 200: 1.8647865056991577
seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.8116886615753174
seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.75192391872406
seen so far: 38464 samples
Training acc over epoch: 1.7974212169647217
Validation acc: 1.55857515335083
Start of epoch 2
Training loss (for one batch) at step 0: 1.597001314163208
seen so far: 64 samples
Training loss (for one batch) at step 200: 1.5410616397857666
seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.3295663595199585
seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.2740886211395264
seen so far: 38464 samples
Training acc over epoch: 1.3730453252792358
Validation acc: 1.1544311046600342
```

### 저수준의 추가적인 loss

지난 section들에서 layer의 regularization loss를 `call` method의 `self.add_loss(value)`를 이용해 더할 수 있음을 보았다.

일반적인 경우에, custom training loop에서 이러한 추가적인 loss들을 고려해야 하는 경우가 많다.

regularization loss를 생성하는 layer에 대한 지난 번의 예제를 살펴보자.

```python
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```
다음과 같이 모형을 호출하게 되면, forward pass 동안에 생성된 loss들이 `model.losses` 속성에 더해진다.
```python
logits = model(x_train[:64])
print(model.losses)
```
```
[<tf.Tensor: id=1287120, shape=(), dtype=float32, numpy=8.15664>]
```
loss는 모형의 `__call__`의 시작에서 초기화되므로, 단 한번의 forward pass 동안에만 생성된 loss 만을 볼 수 있다. 예를 들어, 모형이 여러번 호출되었고, `losses`를 요청하면 가장 마지막 호출에서 생성된 loss만을 볼 수 있다.
```python
logits = model(x_train[:64])
logits = model(x_train[64: 128])
logits = model(x_train[128: 192])
print(model.losses)
```
```
[<tf.Tensor: id=1287180, shape=(), dtype=float32, numpy=8.30571>]
```
이러한 추가적인 loss를 training 과정에서 고려하기 위해서는, 전체 loss에 `sum(model.losses)`를 더해주기만 하면 된다.
```python
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

        with tf.GradientTape() as tape:

            logits = model(x_batch_train) # mini-batch의 logits
            loss_value = loss_fn(y_batch_train, logits)
            
            # forward pass에서 생성된 추가적인 loss를 더해준다.
            loss_value += sum(model.losses)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if step % 200 == 0:
            print('Training loss (for one batch) at step {}: {}'.format(step, float(loss_value)))
            print('seen so far: {} samples'.format((step + 1) * 64))
```
```
Start of epoch 0
Training loss (for one batch) at step 0: 2.296448230743408
seen so far: 64 samples
Training loss (for one batch) at step 200: 2.184938669204712
seen so far: 12864 samples
Training loss (for one batch) at step 400: 2.1210806369781494
seen so far: 25664 samples
Training loss (for one batch) at step 600: 2.0428626537323
seen so far: 38464 samples
Start of epoch 1
Training loss (for one batch) at step 0: 2.008221387863159
seen so far: 64 samples
Training loss (for one batch) at step 200: 1.7805168628692627
seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.7752149105072021
seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.5893230438232422
seen so far: 38464 samples
Start of epoch 2
Training loss (for one batch) at step 0: 1.6722396612167358
seen so far: 64 samples
Training loss (for one batch) at step 200: 1.4144307374954224
seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.3347327709197998
seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.2013696432113647
seen so far: 38464 samples
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNjg2NDE0NDMsLTYyODQ1NjI2NV19
-->