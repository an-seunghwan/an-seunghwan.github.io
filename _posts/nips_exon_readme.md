
# EXoN: EXplainable encoder Network

  

## Requirements

  

```setup

tensorflow 2.4.0

```

  

## Training

  

To follow our step-by-step tutorial implementations of our proposed VAE model EXoN in the paper, use following codes:

  

### 0. modules and source codes

  

```

.

+-- CIFAR10_assets (results from CIFAR-10 dataset experiment)

+-- MNIST_assets (results from MNIST dataset experiment)

+-- src

| +-- modules

| +-- __init__.py

| +-- MNIST.py

| +-- CIFAR10.py

| +-- exon_mnist.py

| +-- exon_mnist_path.py

| +-- latentspace_design.py

| +-- exon_cifar10_tebce_semi.py

| +-- exon_cifar10_tebce_semi_path.py

| +-- exon_cifar10_result.py

+-- README.md

+-- LICENSE

```

  

### 1. MNIST dataset

  

Manual parameter setting:

```

batch_size: mini-batch size of unlabeled dataset, and mini-batch size of labeled dataset will be defined accordingly

data_dim: dimension size of each datapoint

class_num: the number of labels

latent_dim: dimension size of latent variable

sigma: diagonal variance element of prior distribution

activation: activation function of output layer

epochs: the number of total iterations

beta_init: initial value of observation noise

tau: temperature parameter for Gumbel-Softmax method

learning_rate: learning rate of optimizer

hard: If True, Gumbel-Max trick is used at forward pass

FashionMNIST: If True, use Fashion MNIST dataset instead of MNIST dataset

```

  

Step-by-step experiemnt source code:

```train

MNIST.py: Classes and functions for MNIST dataset experiments

exon_mnist.py: run experiment for single parameter setting (Section 4.1)

exon_mnist_path.py: run experiment for grid parameters setting (Appendix A.3.1 heatmap output)

latentspace_design.py: run experiment for manual latent space design (Appendix A.3.2)

```

  

### 2. CIFAR-10 dataset

  

Manual parameter setting:

```

batch_size: mini-batch size of unlabeled dataset, mini-batch size of labeled dataset will be defined accordingly

data_dim: dimension size of each datapoint

class_num: the number of labels

latent_dim: dimension size of latent variable

sigma_label: prior distribution's diagonal variance element of label-relevant latent dimension part

sigma_noise: prior distribution's diagonal variance element of label-irrelevant latent dimension part

channel: channel size of input image

epsilon: controls maximum value of observation noise

activation: activation function of output layer

epochs: the number of total iterations

tau: temperature parameter for Gumbel-Softmax method

learning_rate: learning rate of optimizer

hard: If True, Gumbel-Max trick is used at forward pass

```

  

Step-by-step experiemnt source code:

```train

CIFAR10.py: Classes and functions for CIFAR10 dataset experiments

exon_cifar10_tebce_semi.py: run experiment for single parameter setting with weighted Gaussian observation model (Section 4.2)

exon_cifar10_tebce_semi_path.py: run experiment for grid parameters setting with weighted Gaussian observation model (Section 4.2)

exon_cifar10_result.py: get interpolation result from fitted model (Section 4.2 and Appendix A.4.1, A.4.2)

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTEwMDc1NzUwMywtOTI1MDU3NzMzXX0=
-->