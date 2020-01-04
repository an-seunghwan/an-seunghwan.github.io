---
title: "Train and Evaluate with Keras (1)"
excerpt: "Part 1, 1부(loss, metric, optimizer)"
toc: true
toc_sticky: true

author_profile: false

date: 2020-01-04 15:30:00 -0000
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

해당 게시글은 training, evaluation, 그리고 prediction(inference) 모형을 2가지 광범위한 상황에서 다룬다:
- training & validation을 위한 built-in API(`model.fit()`, `model.evaluation()`, `model.predict()`)를 사용하는 경우. 이는 **built-in training and evaluation loop 사용** 파트에서 다뤄진다.
- 즉시 실행(eager execution)과 `GradientTape`을 이용하여 처음부터 자신만의 loop를 만드는 경우. 이는 **처음부터 자신만의 training and evaluation loop 작성하기** 파트에서 다뤄진다.

일반적으로, built-in loop을 사용하든지 자신만의 모형을 작성하든지, 모형의 training & evaluation은 Keras의 모든 종류의 모형에 대해서 정확히 동일하게 작동한다- Sequential model, Functional API를 사용한 모형, model subclassing을 이용한 모형

해당 가이드는 분산 training에 대해서는 다루지 않는다.

## Setup
```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
```
## Part 1. built-in training and evaluation loop 사용

built-in training loop에 데이터를 전달하는 경우에, 반드시 **Numpy arrays**(데이터의 크기가 작고 memory 크에 잘 맞는 경우)를 사용하거나 **tf.data Dataset** 객체를 사용해야 한다. 다음 몇 개의 단락에서, optimizer, losses, metrics를 사용하는 방법을 설명하기 위해 MNIST dataset을 Numpy arrays처럼 다룰 것이다.

### API overview: 첫 번째 end-to-end 예제

다음의 모형을 생각하자.

```python
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784, ), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='prediction')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```
다음은 원래의 training data로부터 생성된 holdout set에 대한 training, validation, 그리고 test data에 대해 evaluation을 하는 전체적인 과정을 담은 전형적인 end-to-end workflow이다.

```python
# data load
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# validation을 위해 10000개의 sample을 별도로 남겨둔다
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# training의 세부 설정을 명시
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalCrossentropy()]) # list

# data를 "batch_size"를 가지는 "batches"로 잘게 나누어 모형을 훈련
# 그리고 전체 데이터를 주어진 "epochs"만큼의 회수로 반복한다
print("# training data에 모형을 적합")
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    # 각 epoch이 끝날 때마다 몇개의 validation data를 전달하여
                    # validation loss와 metrics을 monitoring한다
                    validation_data=(x_val, y_val))

# history.history 객체는 loss와 metric 값에 대한 학습 과정 동안의 기록을 가지고 있다.
print("\nhistory dict:", history.history)

# `evaluate`을 이용해 test data에 대해 모형을 평가
print("\n# test data에 대한 평가")
results = model.evaluate(x_test, y_test,
                        batch_size=128)
print('test loss, test acc: ', results)

# prediction을 생성 (확률값들 -- 마지막 layer의 output)
# 이때 새로운 데이터에 대하여 'predict'를 이용한다

print('\n# 3개의 sample에 대하여 predictions 생성')
predictions = model.predict(x_test[:3])
print('predictions shape:', predictions.shape)
```
```
# training data에 모형을 적합
Train on 50000 samples, validate on 10000 samples
Epoch 1/3
50000/50000 [==============================] - 2s 50us/sample - loss: 0.0893 - sparse_categorical_crossentropy: 0.0893 - val_loss: 0.1016 - val_sparse_categorical_crossentropy: 0.1016
Epoch 2/3
50000/50000 [==============================] - 2s 49us/sample - loss: 0.0726 - sparse_categorical_crossentropy: 0.0726 - val_loss: 0.0973 - val_sparse_categorical_crossentropy: 0.0973
Epoch 3/3
50000/50000 [==============================] - 2s 49us/sample - loss: 0.0624 - sparse_categorical_crossentropy: 0.0624 - val_loss: 0.1052 - val_sparse_categorical_crossentropy: 0.1052

history dict: {'loss': [0.0893212428727746, 0.07260821430876852, 0.06236755557090044], 'sparse_categorical_crossentropy': [0.08932127, 0.0726082, 0.062367544], 'val_loss': [0.10160406033322215, 0.09732244644667953, 0.10523151242621243], 'val_sparse_categorical_crossentropy': [0.101604074, 0.0973224, 0.105231516]}

# test data에 대한 평가
10000/1 [==============================================] - 0s 16us/sample - loss: 0.0498 - sparse_categorical_crossentropy: 0.0985
test loss, test acc:  [0.09850201628860086, 0.09850202]

# 3개의 sample에 대하여 predictions 생성
predictions shape: (3, 10)
```
### loss, metric, optimizer 명시하기

모형을 `fit`을 이용해 train하기 위해서는, loss 함수, optimizer, 그리고 monitor하기 위한 몇개의 metric을 설정해야 한다.

이러한 것들을 `compile()` method에 인자로 전달해야 한다.

```python
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalCrossentropy()]) # list
```
`metrics` 인자는 반드시 list이어야 한다 - 여러개의 metric을 가지는 것이 가능하다.

만약 모형이 여러개의 output을 가지고 있다면, 각각의 output에 대해서 서로다른 loss와 metric을 지정할 수 있고, 모형의 전체 loss에 대한 각각의 output의 기여도를 조절할 수 있다.
"**여러개의 input과 output 전달하기**" 파트에서 더 자세한 내용을 확인할 수 있다.

많은 경우는 아니지만, string identifier를 이용해 loss와 metric을 지정할 수 있다.

```python
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
```
나중에 재사용을 위해서, 모형의 정의 부분과 compile 부분을 함수에 넣자; 이 함수들을 해당 가이드에서 여러번 사용할 예정이다.

```python
def get_uncompiled_model():
    inputs = keras.Input(shape=(784, ), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='prediction')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
    return model
```
#### 여러 built-in optimizer, losses, metrics

Optimizers:

-   `SGD()`  (with or without momentum)
-   `RMSprop()`
-   `Adam()`
-   etc.

Losses:

-   `MeanSquaredError()`
-   `KLDivergence()`
-   `CosineSimilarity()`
-   etc.

Metrics:

-   `AUC()`
-   `Precision()`
-   `Recall()`
-   etc.

**custom losses**

Keras를 이용해 custom losses를 만드는 2가지 방법이 있다. 첫 번째 예시는 input으로 `y_true`와 `y_pred`를 받는 함수를 만드는 방법이다. 다음의 예제는 실제 data와 predictions 사이에 평균 거리를 계산하는 loss 함수를 만드는 것을 보여준다.

```python
def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_true - y_pred)

model.compile(optimizers=keras.optimizers.Adam(),
              loss=basic_loss_function) # 단순히 custom loss function을 인자로 넘겨준다.

model.fit(x_train, y_train,
          batch_size=64,
          epochs=3)
```
```
Train on 50000 samples
Epoch 1/3
50000/50000 [==============================] - 3s 56us/sample - loss: 4.3488
Epoch 2/3
50000/50000 [==============================] - 2s 45us/sample - loss: 4.3488
Epoch 3/3
50000/50000 [==============================] - 2s 45us/sample - loss: 4.3488
Out[42]: <tensorflow.python.keras.callbacks.History at 0x255a3728e08>
```
만약 `y_true`와 `y_pred`이외에 parameters를 받는 loss function이 필요하다면,  `tf.keras.losses.Loss` class에 subclass를 하여 다음의 두 method를 실행하면 된다.
- `__init__(self)`: loss function을 호출하는 동안에 전달해야할 parameter들을 받는다.
- `call(self, y_true, y_pred)`: targets(`y_true`)와 모형의 predictions('y_pred`)를 사용하여 모형의 loss를 계산

`__init__()`에 전달된 parameter들은 loss를 계산할 때 `call()`에서 사용될 수 있다.

다음의 예제는 `BinaryCrossEntropy`를 계산하는 `WeightedCrossEntropy` loss function을 보여준다. 이때 특정한 분류 항목의 loss나 전체 함수는 scalar에 의해 조정될 수 있다.

```python
class WeightedBinaryCrossEntropy(keras.losses.Loss):
    '''
    Args:
        pos_weight: loss function의 positive label에 영향을 주는 scalar
        weight: loss function의 전체에 영향을 주는 scalar
        from_logits: loss를 계산할 지 확률 값을 계산할 지 결정
        reduction: loss에 적용할 tf.keras.losses.Reduction의 종류
        name: loss function의 이
    '''
    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy, self).__init__(reduction=reduction,
                                                         name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if not self.from_logits:
            # 확률 값을 계산
            # Manually calculate the weighted cross entropy.
            # Formula is qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            # where z are labels, x is logits, and q is the weight.
            # Since the values passed are from sigmoid (assuming in this case)
            # sigmoid(x) will be replaced by y_pred
            
            # qz * -log(sigmoid(x)) 1e-6 is added as an epsilon to stop passing a zero into the log
            x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)
            
            # (1 - z) * -log(1 - sigmoid(x)). Epsilon is added to prevent passing a zero into the log
            x_2 = (1 - y_true) * tf.math.log(1 - y_pred + 1e-6)
            
            return tf.add(x_1, x_2) * self.weight
        
        # Use built-in function
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight
    
model.compile(optimizer=keras.optimizers.Adam(),
              loss=WeightedBinaryCrossEntropy(0.5, 2))

model.fit(x_train, y_train,
          batch_size=64,
          epochs=3)
```
```
Train on 50000 samples
Epoch 1/3
50000/50000 [==============================] - 3s 64us/sample - loss: 11.7616
Epoch 2/3
50000/50000 [==============================] - 3s 53us/sample - loss: 10.9724
Epoch 3/3
50000/50000 [==============================] - 3s 55us/sample - loss: 10.9711
Out[39]: <tensorflow.python.keras.callbacks.History at 0x255a217b248>
```
* `keras.losses.Reduction`에 대해서는 추후에 더 자세히 다루겠습니다! (coming soon!)

**custom metric**

필요한 metric이 API에 제공되지 않는다면, `Metric` class에 subclassing을 하여 만들 수 있다. 다음의 네가지 method를 실행해야 한다.

- `__init__(self)`: metric에서 사용되는 state 변수를 생성한다.
- `update_state(self, y_true, y_pred, sample_weight=None)`: target `y_true`와 모형의 predictions `y_pred`를
state 변수를 update하기 위해 사용하는 것.
- `result(self)`: 최종 결과를 계산하기 위해 state 변수를 사용하는 것.
- `reset_states(self)`: metric의 state를 재초기화 하는 것.

state update와 결과 계산은 별도로 유지된다(`update_state()`과 `result()`에서 각각). 왜냐하면 몇몇 경우에서, 결과 계산은 매우 어려울 수 있으므로, 주기적으로만 행해져야 하기 때문이다.

다음의 예제는 `CategoricalTruePositives` metric을 어떻게 실행하는지에 대한 예제이다. 이 metric은 주어진 class에서 얼마나 많은 sample이 정확하게 분류되었는지 세어주는 metric이다.

```python
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name='categorical_true_positive', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))
        
    def result(self):
        return self.true_positives
    
    def reset_states(self):
        # 각 epoch의 시작에서 metric의 state는 초기화 될 것이다.
        self.true_positives.assign(0.)
        
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[CategoricalTruePositives()])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=3)
```
```
Train on 50000 samples
Epoch 1/3
50000/50000 [==============================] - 3s 60us/sample - loss: 0.1564 - categorical_true_positive: 47680.0000
Epoch 2/3
50000/50000 [==============================] - 2s 49us/sample - loss: 0.0863 - categorical_true_positive: 48708.0000
Epoch 3/3
50000/50000 [==============================] - 2s 46us/sample - loss: 0.0734 - categorical_true_positive: 48866.0000
Out[40]: <tensorflow.python.keras.callbacks.History at 0x255a212f688>
```

**보편적인 특징에 맞지 않은 loss와 metric 다루기**

아주 대다수의 loss와 metric은 `y_true`와 `y_pred`로부터 계산될 수 있다.
그러나 전부가 그런 것은 아니다. 예를 들어, 정규화(regularization) loss는 아마 layer의 activation만을 필요로 한다(이 경우에는 target이 없다). 그리고 이 activation은 모형의 output은 아니다.

그러한 경우에, custom layer에서 `self.add_loss(loss_value)`을 `call` method 내부에서 호출할 수 있다. 여기 activity regularization을 더하는 간단한 예제가 있다(activity regularization은 모든 Keras layer에서 built-in 되어있다 -- 이 예제 layer는 단지 명확한 예제를 위한 것이다).

```python
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs # 그냥 통과하는 layer
    
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)

x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy')

# 다음의 fit에서 보여주는 loss는 regularization 요소 때문에 전보다 매우 높을 것이다.
model.fit(x_train, y_train,
          batch_size=64,
          epochs=1)
```
```
Train on 50000 samples
50000/50000 [==============================] - 3s 59us/sample - loss: 2.4772
Out[43]: <tensorflow.python.keras.callbacks.History at 0x255a3f8e208>
```
activity regularization이 없는 모형과 loss를 실제로 비교해보자.
```python
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy')

model.fit(x_train, y_train,
          batch_size=64,
          epochs=1)
```
```
Train on 50000 samples
50000/50000 [==============================] - 3s 53us/sample - loss: 0.3314
Out[44]: <tensorflow.python.keras.callbacks.History at 0x255a581ac88>
```
loss의 값이 0.3314로 더 작음을 볼 수 있다.

metric 값에 대해서도 동일하게 할 수 있다.

```python
class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # 'aggregation' 인자는 batch당 metric 값을 전체 epoch에 대해서 어떻게 모을 것인지 정의한다.
        # 여기서는 단순히 평균을 사용한다
        self.add_metric(keras.backend.std(inputs), # 표준 편차 계
                        name='std_of_activation',
                        aggregation='mean')
        return inputs # 그냥 통과하는 layer
    
inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)

# Insert std logging as a layer.
x = MetricLoggingLayer()(x)

x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train,
          batch_size=64,
          epochs=1)
```
```
Train on 50000 samples
50000/50000 [==============================] - 3s 58us/sample - loss: 0.3374 - std_of_activation: 0.9307
Out[45]: <tensorflow.python.keras.callbacks.History at 0x255a60c5d48>
```
Functional API에서도, `model.add_loss(loss_tensor)`, `model.add_metric(metric_tensor, name, aggregation)`을 이용할 수 있다.

다음의 간단한 예제를 보자.

```python
inputs = keras.Input(shape=(784,), name='digits')
x1 = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x2 = layers.Dense(64, activation='relu', name='dense_2')(x1)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)
model.add_metric(keras.backend.std(x1),
                 name='std_of_activation',
                 aggregation='mean')

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train,
          batch_size=64,
          epochs=1)
```
```
Train on 50000 samples
50000/50000 [==============================] - 4s 81us/sample - loss: 2.4877 - std_of_activation: 0.0019
Out[46]: <tensorflow.python.keras.callbacks.History at 0x255a7a12d88>
```
**자동으로 validation set을 별도로 설정하기**

첫 번째 end-to-end 예제에서, 우리는 `validation_data`라는 인자를 `(x_val, y_val)`이라는 Numpy array tuple을 전달하기 위해 사용했다. 이는 각 epoch가 종료될 때 마다 validation loss와 metric을 평가하기 위한 목적이다.

여기 다른 방법이 있다: `validation_split`인자는 자동으로 training data의 일부분을 validation을 위해 남겨둔다. 인자 값은 validation을 위해 남겨둘 데이터의 비율을 나타내며, 0과 1 사이의 값을 가져야 한다. 예를 들어, `validation_split=0.2`는 "20%의 데이터를 validation을 위해 남겨둔다"라는 의미이다.

validation이 계산되는 방식은 `fit` 호출에서 입력된 array의 마지막 x%의 sample들을 이용해 어떠한 shuffling도 하지 않고 계산한다.

Numpy data에 대해서만 `validation_split`을 사용할 수 있다.

```python
model = get_compiled_model()
model.fit(x_train, y_train,
          batch_size=64,
          validation_split=0.2,
          epochs=1,
          steps_per_epoch=1) # 각 epoch당 몇 번의 step, 즉 몇개의 batch들을 사용할 것인지 정의
                             # 따라서 batch-sized sample 묶음(chunk)들의 순서를 suffling 해주는 suffle이라는 인자는 steps_per_epoch가 None이 아닌 경우에만 의미가 있다.
```
```
Train on 40000 samples, validate on 10000 samples
   64/40000 [..............................] - ETA: 11:32 - loss: 2.3628 - sparse_categorical_accuracy: 0.1406 - val_loss: 0.0000e+00 - val_sparse_categorical_accuracy: 0.0000e+00Out[47]: <tensorflow.python.keras.callbacks.History at 0x255a8467748>
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2NzI5NjAwNzksMTExODQxNzExOCwtMT
cxOTY1NTYyMV19
-->