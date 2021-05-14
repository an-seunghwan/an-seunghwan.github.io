# EXoN: EXplainable encoder Network


## Requirements

```setup
tensorflow 2.4.0
numpy 1.19.0
```
 
## Training and Evaluation

To follow step-by-step tutorial implementation of our proposed model in the paper, use following codes:

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
| +-- exon_cifar10_tebce.py
| +-- exon_cifar10_tebce_semi.py
| +-- exon_cifar10_result.py
+-- README.md
```

### 1. MNIST dataset

Manual parameter setting:
```
batch_size: mini-batch size of unlabeled dataset, mini-batch size of labeled dataset will be defined accordingly.
data_dim: dimension size of each datapoint
class_num: the number of labels
latent_dim: dimension size of latent space
sigma: diagonal variance element of prior distribution
epochs: the number of iterations
beta_init: initial value of observation noise
tau: temperature parameter for Gumbel-Softmax method
learning_rate: learning rate of optimizer
hard: If True, Gumbel-Max trick is used for forward pass
FashionMNIST: If True, use Fashion MNIST dataset instead of MNIST dataset
```

step-by-step experiemnt source code:
```train
MNIST.py: Classes and functions for MNIST dataset experiments
exon_mnist.py: run experiment for single parameter setting (Section 4.1)
exon_mnist_path.py: run experiment for grid parameter setting (Appendix heatmap output)
latentspace_design.py: run experiment for manual latent space design (Section 4.2)
```
  
### 2. CIFAR-10 dataset
 
Manual parameter setting:
```
batch_size: mini-batch size of unlabeled dataset, mini-batch size of labeled dataset will be defined accordingly.
data_dim: dimension size of each datapoint
class_num: the number of labels
latent_dim: dimension size of latent space
sigma_label: prior distribution's diagonal variance element of label-relevant latent dimension
sigma_noise: prior distribution's diagonal variance element of label-irrelevant latent dimension
channel: channel size of input image
epsilon: maximum value of observation noise
activation: activation function of output layer
epochs: the number of iterations
beta_init: initial value of observation noise
tau: temperature parameter for Gumbel-Softmax method
learning_rate: learning rate of optimizer
hard: If True, Gumbel-Max trick is used for forward pass
FashionMNIST: If True, use Fashion MNIST dataset instead of MNIST dataset
```

step-by-step experiemnt source code:
```train
CIFAR10.py: Classes and functions for CIFAR10 dataset experiments
exon_cifar10_tebce_semi.py: run experiment for single parameter setting with weighted Gaussian observation model (Section 4.3)
exon_cifar10_result.py: get interpolation result from fitted model (Section 4.3 and Appendix)
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTkyNTA1NzczM119
-->