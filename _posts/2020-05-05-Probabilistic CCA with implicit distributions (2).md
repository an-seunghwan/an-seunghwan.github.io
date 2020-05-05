---
title: "Probabilistic CCA with implicit distributions (2)"
excerpt: "ACCA를 python 코드로 구현해보기!"
toc: true
toc_sticky: true

author_profile: false
use_math: true

date: 2020-05-05 20:00:00 -0000
categories: 
  - CCA
tags:
  - spyder
  - autoencoder
  - adversarial
  - GAN
--- 

>  본 포스팅은 [Probabilistic CCA with implicit distributions](https://arxiv.org/pdf/1907.02345.pdf)에 대한 간단한 리뷰와 python 코드를 작성한 글입니다. 정확한 내용은 반드시 원문을 참조해 주시기 바랍니다.
>  
>  1편: 논문 리뷰
>  
>  2편: python 코드

> 추가적으로 아래의 python 코드는 **mrquincle**님의 AAE(Adversarial AutoEncoder) python 코드를 참고하여 만들었습니다.  해당 github: [https://github.com/mrquincle/keras-adversarial-autoencoders/blob/master/Keras%20Adversarial%20Autoencoder%20MNIST.ipynb](https://github.com/mrquincle/keras-adversarial-autoencoders/blob/master/Keras%20Adversarial%20Autoencoder%20MNIST.ipynb) 

## 0. setting
```python
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
# from tensorflow.keras.models import model_from_json
print('TensorFlow version:', tf.__version__)
print('즉시 실행 모드:', tf.executing_eagerly())
print(tf.test.is_gpu_available(
      cuda_only=False,
      min_cuda_compute_capability=None
))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)
```
```
TensorFlow version: 2.0.0
즉시 실행 모드: True
True
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 16133062203861420028
, name: "/device:XLA_CPU:0"
device_type: "XLA_CPU"
memory_limit: 17179869184
locality {
}
incarnation: 13116411031905517496
physical_device_desc: "device: XLA_CPU device"
, name: "/device:XLA_GPU:0"
device_type: "XLA_GPU"
memory_limit: 17179869184
locality {
}
incarnation: 11981642561625068716
physical_device_desc: "device: XLA_GPU device"
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 10747743437
locality {
  bus_id: 1
  links {
  }
}
incarnation: 2742631385821467278
physical_device_desc: "device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1"
]
2020-05-05 22:27:08.949248: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:03:00.0
2020-05-05 22:27:08.949351: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-05-05 22:27:08.949389: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-05-05 22:27:08.949427: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-05-05 22:27:08.949464: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-05-05 22:27:08.949497: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-05-05 22:27:08.949535: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-05-05 22:27:08.949573: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-05 22:27:08.950662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-05-05 22:27:08.950743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 22:27:08.950760: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-05-05 22:27:08.950771: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-05-05 22:27:08.951934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 10249 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
2020-05-05 22:27:08.953243: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.721
pciBusID: 0000:03:00.0
2020-05-05 22:27:08.953285: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-05-05 22:27:08.953315: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-05-05 22:27:08.953343: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-05-05 22:27:08.953370: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-05-05 22:27:08.953397: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-05-05 22:27:08.953425: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-05-05 22:27:08.953452: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-05-05 22:27:08.954524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-05-05 22:27:08.954555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 22:27:08.954568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-05-05 22:27:08.954579: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-05-05 22:27:08.955729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 10249 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:03:00.0, compute capability: 6.1)
```
```python
import numpy as np
import matplotlib.pylab as plt 
import os
os.chdir('/home/jeon/Desktop/an/cca')
print('current directory:', os.getcwd())
```
```
current directory: /home/jeon/Desktop/an/cca
```

## 1. 데이터
tensorflow의 MNIST 데이터를 이용합니다.
```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data() 
print(train_images.shape)
img_shape = train_images.shape[1:]
x_shape = (img_shape[0], int(img_shape[1]/2))
y_shape = (img_shape[0], int(img_shape[1]/2))
xy_shape = img_shape
```
```
(60000, 28, 28)
(28, 14)
(28, 14)
```
각각의 이미지를 세로 방향으로 반으로 잘라 각각 x view, y view로 정의합니다.

## 2. 모형

잠재변수(latent variable)의 차원을 지정하는 부분입니다.
```python
latent_dim = 100
```
### 1. encoder
encoder 모형을 생성하는 함수입니다.
```python
def build_encoder():
    x = layers.Input(shape=x_shape)   
    y = layers.Input(shape=y_shape)   
    xy = layers.Input(shape=xy_shape)   
    
    hx = layers.Flatten()(x)
    hy = layers.Flatten()(y)
    hxy = layers.Flatten()(xy)
    
    hx = layers.Dense(256)(hx)
    hx = layers.LeakyReLU(alpha=0.2)(hx)
    hx = layers.Dense(256)(hx)
    hx = layers.LeakyReLU(alpha=0.2)(hx)
    
    hy = layers.Dense(256)(hy)
    hy = layers.LeakyReLU(alpha=0.2)(hy)
    hy = layers.Dense(256)(hy)
    hy = layers.LeakyReLU(alpha=0.2)(hy)
    
    hxy = layers.Dense(256)(hxy)
    hxy = layers.LeakyReLU(alpha=0.2)(hxy)
    hxy = layers.Dense(256)(hxy)
    hxy = layers.LeakyReLU(alpha=0.2)(hxy)
    
    zx = layers.Dense(latent_dim)(hx)
    zy = layers.Dense(latent_dim)(hy)
    zxy = layers.Dense(latent_dim)(hxy)
    
    return K.models.Model([x, y, xy], [zx, zy, zxy])
```
우선 행렬 이미지를 `Flatten`을 이용해 벡터로 변환한 후, `Dense` layer와 `LeakyReLU`를 이용해 functional API를 활용하여 encoder를 정의합니다. 이 때, 각각의 x, y, xy view에 대해서 별도의 layer를 지정하여 encoder를 생성합니다.

### 2. decoder
latent variable z가 주어졌을 때, 이를 이용해 원래의 이미지로 복원하는 decoder를 정의합니다. 이 때, 입력된 latent variable을 이용해 x, y 각각의 view를 생성하고, 이를 합하여 원래의 이미지로 복원합니다.
```python
def build_decoder():
    z_input = layers.Input(shape=latent_dim)
    
    zx = layers.Dense(256)(z_input)
    zx = layers.LeakyReLU(alpha=0.2)(zx)
    zx = layers.Dense(256)(zx)
    zx = layers.LeakyReLU(alpha=0.2)(zx)
    zx = layers.Dense(np.prod(x_shape), activation='tanh')(zx)
    img_x = tf.reshape(zx, [-1]+list(x_shape))
    
    zy = layers.Dense(256)(z_input)
    zy = layers.LeakyReLU(alpha=0.2)(zy)
    zy = layers.Dense(256)(zy)
    zy = layers.LeakyReLU(alpha=0.2)(zy)
    zy = layers.Dense(np.prod(y_shape), activation='tanh')(zy)
    img_y = tf.reshape(zy, [-1]+list(y_shape))
    
    img = layers.Concatenate(axis=-1)([img_x, img_y])
    
    return K.models.Model(z_input, img)
```

### 3. discriminator
encoder로부터 생성된 latent variable과 실제 prior 분포(Gaussian)에서 생성된 latent variable을 구별해내는 구별기를 정의합니다.
```python
def build_discriminator():
    z_input = layers.Input(shape=latent_dim)
    z = layers.Dense(512)(z_input)
    z = layers.LeakyReLU(alpha=0.2)(z)
    z = layers.Dense(256)(z)
    z = layers.LeakyReLU(alpha=0.2)(z)
    
    v = layers.Dense(1, activation='sigmoid')(z)
    
    return K.models.Model(z_input, v)
```
### 4. build
```python
discriminator = build_discriminator()
discriminator.compile('adam', 'binary_crossentropy', ['accuracy'])

encoder = build_encoder()
decoder = build_decoder()
```

```python
x_input = layers.Input(shape=x_shape)
y_input = layers.Input(shape=y_shape)
xy_input = layers.Input(shape=xy_shape)

encoded_x, encoded_y, encoded_xy = encoder([x_input, y_input, xy_input])
reconstructed_img_x = decoder(encoded_x)
reconstructed_img_y = decoder(encoded_y)
reconstructed_img_xy = decoder(encoded_xy)

# For the adversarial_autoencoder model we will only train the generator
# We only set trainable to false for the discriminator when it is part of the autoencoder...
discriminator.trainable = False

validity_x = discriminator(encoded_x)
validity_y = discriminator(encoded_y)
validity_xy = discriminator(encoded_xy)

acca = K.models.Model([x_input, y_input, xy_input], 
                      [reconstructed_img_x, reconstructed_img_y, reconstructed_img_xy, validity_x, validity_y, validity_xy])

# almost no weights on discriminator...
acca.compile('adam', ['mse', 'mse', 'mse', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'], 
             loss_weights=[0.3333, 0.3332, 0.3332, 0.0001, 0.0001, 0.0001])
acca.summary()
discriminator.summary()
```
```
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_8 (InputLayer)            [(None, 28, 14)]     0                                            
__________________________________________________________________________________________________
input_9 (InputLayer)            [(None, 28, 14)]     0                                            
__________________________________________________________________________________________________
input_10 (InputLayer)           [(None, 28, 28)]     0                                            
__________________________________________________________________________________________________
model_3 (Model)                 [(None, 100), (None, 676652      input_8[0][0]                    
                                                                 input_9[0][0]                    
                                                                 input_10[0][0]                   
__________________________________________________________________________________________________
model_4 (Model)                 (None, 28, 28)       384784      model_3[1][0]                    
                                                                 model_3[1][1]                    
                                                                 model_3[1][2]                    
__________________________________________________________________________________________________
model_2 (Model)                 (None, 1)            183297      model_3[1][0]                    
                                                                 model_3[1][1]                    
                                                                 model_3[1][2]                    
==================================================================================================
Total params: 1,244,733
Trainable params: 1,061,436
Non-trainable params: 183,297
__________________________________________________________________________________________________
```
```
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 100)]             0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               51712     
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 257       
=================================================================
Total params: 366,594
Trainable params: 183,297
Non-trainable params: 183,297
_________________________________________________________________
```
## 3. training
```python
EPOCHS = 10000
BATCH_SIZE = 256
sample_interval = 100

# scaling -1 to 1
train = (train_images.astype(np.float32) - 127.5) / 127.5
train.shape
x_train = train[:, :, :14]
x_train.shape
y_train = train[:, :, 14:]
y_train.shape

valid = np.ones((BATCH_SIZE, 1))
fake = np.zeros((BATCH_SIZE, 1))
```
```
(60000, 28, 28)
(60000, 28, 14)
(60000, 28, 14)
```
Gaussian 분포로부터 생성된 latent variable을 이용해 이미지를 생성해내는 함수입니다.
```python
def sample_prior(latent_dim, batch_size):
    return np.random.normal(size=(batch_size, latent_dim))

def sample_images(latent_dim, decoder, epoch):
    r,c = 5,5

    z = sample_prior(latent_dim, r*c)
    gen_imgs = decoder.predict(z)

    gen_imgs = 0.5 * gen_imgs + 0.5 # rescaling

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[i*j, :, :], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./acca_img/acca_mnist_%d.png" % epoch)
    plt.close()
```

```python
'''training'''
for epoch in range(EPOCHS):
    #=====train discriminator=====
    idx = np.random.randint(0, train.shape[0], BATCH_SIZE) # sampling random batch images -> stochasticity
    imgs_x = x_train[idx]
    imgs_y = y_train[idx]
    imgs_xy = train[idx]
    
    latent_real = np.random.normal(size=(BATCH_SIZE, latent_dim)) # TRUE sample
    latent_fake = encoder.predict([imgs_x, imgs_y, imgs_xy])
    
    d_loss_real = discriminator.train_on_batch(latent_real, valid)
    d_loss_fake = np.zeros(())
    for i in range(3):
        d_loss_fake = d_loss_fake + discriminator.train_on_batch(latent_fake[i], fake)
    d_loss = np.add(d_loss_real, d_loss_fake) / 4
    
    #=====train generator=====
    g_loss = acca.train_on_batch([imgs_x, imgs_y, imgs_xy], [imgs_xy, imgs_xy, imgs_xy] + [valid for _ in range(3)])
    
    if epoch % 500 == 0:
        # print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*np.mean(d_loss[4:]), g_loss[0], g_loss[1]))
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
    
    if epoch % sample_interval == 0:
        sample_images(latent_dim, decoder, epoch)
```
```
0 [D loss: 0.938511, acc: 50.00%] [G loss: 0.972918]
```
위와같은 결과가 반복적으로 출력되게 된다.

## 4. inference

5개의 test sample에 대해 결과를 확인한다. 이 때, x view는 test 데이터로부터 가져와서 encoder를 이용해 latent variable을 생성하여 이를 이용해 이미지를 복원한다.
```python
'''inference'''
j = 100
r = 5
given_x = test_images[j:j+r][:, :, :14]
print(given_x.shape)
latent_x, _, _ = encoder.predict([given_x, given_x, np.concatenate([given_x, given_x], axis=-1)])
print(latent_x.shape)
recon_img = decoder.predict(latent_x)

img_result = [given_x, recon_img, test_images[j:j+5]]

recon_img = 0.5 * recon_img + 0.5 # rescaling
fig, axs = plt.subplots(r, 3)
cnt = 0
for i in range(r):
    axs[i,0].imshow(img_result[0][cnt, :, :], cmap='gray')
    axs[i,1].imshow(img_result[1][cnt, :, :], cmap='gray')
    axs[i,2].imshow(img_result[2][cnt, :, :], cmap='gray')
    axs[i,0].axis('off')
    axs[i,1].axis('off')
    axs[i,2].axis('off')
    cnt += 1
# fig.savefig("./acca_img/acca_result.png")
# plt.close()
```
```
(5, 28, 14)
(5, 100)
```
<center><img  src="https://github.com/an-seunghwan/an-seunghwan.github.io/blob/master/assets/img/acca_result.png?raw=true
" width="600"  height="450"></center>
가장 왼쪽이 주어진 x view, 가운데가 x view만을 이용해 만든 전체 이미지, 가장 오른쪽이 맞추려는 대상이 되는 실제 이미지이다.

## 논문
SHI, Yaxin, et al. Probabilistic CCA with Implicit Distributions. _arXiv preprint arXiv:1907.02345_, 2019.

> 코딩이나 내용에 대한 수정사항이나 더 좋은 의견은 언제든지 환영입니다! 감사합니다.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgxNzY4MDM2MiwxMDMzMzE3MzY2LDEzMj
M4MzYzMjIsLTE5NzY3NjQyNjgsLTE3ODk1MzY2NzJdfQ==
-->