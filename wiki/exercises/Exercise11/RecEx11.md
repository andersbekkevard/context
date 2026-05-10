---
title: "Module 11: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2024"
author:
  - Daesoo Lee, Kenneth Aase, Stefanie Muff, Sara Martino
  - Department of Mathematical Sciences, NTNU
date: "April 18, 2024"
---


---

## PyTorch Installation

The ISLP lab for Chapter 10 uses PyTorch with the `pytorch_lightning` wrapper, plus a few helpers shipped in `ISLP.torch`. Install them once with

```python
pip install torch torchvision torchmetrics torchinfo pytorch_lightning ISLP
```

We will use these standard imports throughout the exercises:

```python
import numpy as np, pandas as pd
from matplotlib.pyplot import subplots

import torch
from torch import nn
from torch.optim import RMSprop, Adam
from torch.utils.data import TensorDataset

from torchmetrics import MeanAbsoluteError
from torchinfo import summary

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything

from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)

seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
```


## Problem 1

### a)

Write down the equation that describes and input is related to output in this network, using general activation functions $\phi_o$, $\phi_h$ and $\phi_{h^\star}$ and bias nodes in all layers. What would you call such a network?

![Image created here <http://alexlenail.me/NN-SVG/index.html> ](nn.png){width=100%}


### b)

The following image is the illustration of an artificial neural network at Wikipedia.

* What can you say about this network architecture
* What do you think it can be used for (regression/classification)?


![Image taken from <https://commons.wikimedia.org/wiki/File:Colored_neural_network.svg>](Colorednn.png){width=50%}


### c)

What are the similarities and differences between a feedforward neural network with one hidden layer with `linear` activation and `sigmoid` output (one output) and logistic regression?


### d)

In a feedforward neural network you may have $10'000$ weights to estimate but only $1000$ observations. How is this possible?

## Problem 2

### a)

Which network architecture and activation functions does this formula correspond to?
$$ \hat{y}_1({\bf x})=\beta_{01}+\sum_{m=1}^5 \beta_{m1}\cdot \max(\alpha_{0m}+\sum_{j=1}^{10} \alpha_{jm}x_j,0)$$
How many parameters are estimated in this network?

### b)
Which network architecture and activation functions does this formula give?

$$ \hat{y}_1({\bf x})=(1+\exp(-\beta_{01}-\sum_{m=1}^5 \beta_{m1}\max(\gamma_{0m}+\sum_{l=1}^{10} \gamma_{lm}\max(\sum_{j=1}^{4}\alpha_{jl}x_j,0),0))^{-1}$$

How many parameters are estimated in this network?

### c)
In a regression setting: Consider

* A sum of non-linear functions of each covariate in Module 7.
* A sum of many non-linear functions of sums of covariates in feedforward neural networks (one hidden layer, non-linear activation in hidden layer) in Module 11.

Explain how these two ways of thinking differ? Pros and cons?



## Problem 3: Regression with Feedforward Neural Network (FNN)

The following problem involves training a feedforward neural network on the Boston Housing Prices dataset. The Boston Housing Prices dataset is a collection of housing prices data from the Boston area, containing 506 samples with 13 numerical features each. The features include factors such as crime rate, average number of rooms per dwelling, property tax rate, and more. The goal is to predict the median value of owner-occupied homes in $1000s.

In this example, we will design our feedforward neural network (FNN) architecture using a series of fully connected (dense) layers. The model will take the 13 input features and learn to map them to a single output representing the predicted median housing price. To accomplish this, the network will be trained using an appropriate loss function, such as mean squared error, and an optimization algorithm like stochastic gradient descent or Adam.


##### 1. Load and preprocess data
```python
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load
Boston = load_data('Boston')
X = Boston.drop(columns=['medv']).to_numpy().astype(np.float32)
Y = Boston['medv'].to_numpy().astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# preprocess
scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
```

### a) Fill in the missing parts in the following steps and run the model.

##### 2. Define the model
```python
class BostonModel(nn.Module):
    def __init__(self, input_size):
        super(BostonModel, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, ...))  # fill in the output size of the last layer.

    def forward(self, x):
        return torch.flatten(self.sequential(x))

boston_model = BostonModel(input_size=...)  # fill in the length of the input.
summary(boston_model,
        input_size=X_train.shape,
        col_names=['input_size', 'output_size', 'num_params'])
```

What should the output size be to be compatible with `Y`?

##### 3. Set up the training module
```python
boston_module = SimpleModule.regression(
    boston_model,
    metrics={'mae': MeanAbsoluteError()},
    optimizer=Adam(boston_model.parameters(), lr=0.001),
    loss=...  # fill in the loss function (an `nn` module).
)
```

For the loss function, choose one among `nn.BCELoss()`, `nn.CrossEntropyLoss()` and `nn.MSELoss()`.


##### 4. Train the model
```python
X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)

boston_train = TensorDataset(X_train_t, Y_train_t)
boston_test  = TensorDataset(X_test_t,  Y_test_t)

max_num_workers = rec_num_workers()
boston_dm = SimpleDataModule(boston_train,
                             boston_test,
                             batch_size=32,
                             num_workers=min(4, max_num_workers),
                             validation=boston_test)

boston_logger = CSVLogger('logs', name='Boston')
boston_trainer = Trainer(deterministic=True,
                         max_epochs=100,
                         log_every_n_steps=5,
                         logger=boston_logger,
                         callbacks=[ErrorTracker()])
boston_trainer.fit(boston_module, datamodule=boston_dm)
```


##### 5. Test
```python
test_results = boston_trainer.test(boston_module, datamodule=boston_dm)
print("Test loss (MSE):", test_results[0]['test_loss'])
print("Test mean absolute error (MAE):", test_results[0]['test_mae'])
```


##### Plot training history
```python
boston_results = pd.read_csv(boston_logger.experiment.metrics_file_path)
fig, ax = subplots(1, 1, figsize=(6, 6))
for col, color, label in [('train_mae_epoch', 'black', 'Training'),
                          ('valid_mae',       'red',   'Validation')]:
    boston_results.plot(x='epoch', y=col, label=label,
                        marker='o', color=color, ax=ax)
ax.set_xlabel('Epoch'); ax.set_ylabel('MAE')
```


##### Additional plot: predicted vs. actual
```python
boston_model.eval()
with torch.no_grad():
    predictions = boston_module(X_test_t).numpy()

fig, ax = subplots(1, 1, figsize=(6, 6))
ax.scatter(Y_test, predictions)
ax.plot([0, 55], [0, 55], color='red', linestyle='--')
ax.set_xlim(0, 55); ax.set_ylim(0, 55)
ax.set_xlabel('Actual Values'); ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs. Actual Values (Feedforward NN)')
```


### b) Fit a linear regression model and compare its performance (i.e., MSE, MAE) to that of the feedforward network.

##### Comparison to a Linear Regression Model


```python
from sklearn.linear_model import LinearRegression

# Fit a linear regression model
linear_model = ...  # fill in (instantiate and fit a linear regression on X_train, Y_train)

# Make predictions on the test set
predictions = linear_model.predict(X_test)

# Calculate the mean squared error and mean absolute error
mse = ...  # fill in (write an expression to compute MSE)
mae = ...  # fill in (write an expression to compute MAE)

print("=== [Feedforward Neural Network] ===")
print(" Test loss (MSE):", test_results[0]['test_loss'])
print(" Test mean absolute error (MAE):", test_results[0]['test_mae'])
print("====================================\n")
print("=== [Linear Regression] ===")
print(" Test loss (MSE):", mse)
print(" Test mean absolute error (MAE):", mae)
print("===========================\n")
```


```python
fig, ax = subplots(1, 1, figsize=(6, 6))
ax.scatter(Y_test, predictions)
ax.plot([0, 55], [0, 55], color='red', linestyle='--')
ax.set_xlim(0, 55); ax.set_ylim(0, 55)
ax.set_xlabel('Actual Values'); ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs. Actual Values (Linear Regression)')
```


### c) Please share your thoughts on the comparative performance of the two models based on their results.




## Problem 4: Convolutional Neural Network (CNN)


### Problem 4.1: Image Classification with CNN

The following problem involves training a Convolutional Neural Network (CNN) model on an image dataset, called CIFAR-10. The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 testing images. The 10 classes include airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The examples and description of CIFAR-10 can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).

In this example, we have designed our CNN model architecture following established models such as AlexNet [1], VGG [2], and ResNet [3]. Our design incorporates a common pattern where the channel dimension size (i.e., filters) increases while the input shape to each layer decreases across the layers (see the figure below). It is also worth noting that a convolutional layer is sometimes followed by a non-linear pooling layer which reduces the spatial dimension size. One most common pooling layer is the max pooling layer (see the figure below). Our example model follows the same protocol while keeping the model size small.

[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90.
[2] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
[3] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.


![Fig. Architecture of AlexNet. The input data has a dimension of (height x width x #channels. #channels is 3 for RGB colors. The common CNN architecture design is the increase in #channels and decrease in height-width size across the layers.)](alexnet.png)

![Fig. Max Pooling Layer: The 2x2 max pooling layer decreases the spatial dimensions by a factor of 2 in both height and width, applying the maximum operation within each local region.)](Maxpool_illustration.png)



##### 1. Load and preprocess data
```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

(cifar_train,
 cifar_test) = [CIFAR10(root='data', train=train, download=True)
                for train in [True, False]]

transform = ToTensor()
cifar_train_X = torch.stack([transform(x) for x in cifar_train.data])
cifar_test_X  = torch.stack([transform(x) for x in cifar_test.data])
cifar_train = TensorDataset(cifar_train_X,
                            torch.tensor(cifar_train.targets))
cifar_test  = TensorDataset(cifar_test_X,
                            torch.tensor(cifar_test.targets))
```

The `ToTensor()` transform rescales the 8-bit pixel values from $[0, 255]$ to $[0, 1]$ and reorders the axes so that each image has shape `(3, 32, 32)` — three RGB channels first, then the 32x32 spatial grid, which is the layout `nn.Conv2d` expects.


### a) Fill in the missing parts in the following steps and run the model.

##### 2. Define the model
```python
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))
        self.output = nn.Sequential(
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 10))

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.output(x)

cifar_model = CIFAR10Model()
summary(cifar_model,
        input_size=(64, 3, 32, 32),
        col_names=['input_size', 'output_size', 'num_params'])
```


##### 3. Set up the training module
```python
cifar_module = SimpleModule.classification(
    cifar_model,
    optimizer=Adam(cifar_model.parameters(), lr=0.001),
    loss=...  # fill in the loss function.
)
```

For the loss function, choose one among `nn.BCELoss()`, `nn.CrossEntropyLoss()` and `nn.MSELoss()`. Note that `SimpleModule.classification` expects integer class labels (not one-hot), and the corresponding loss combines a `LogSoftmax` with the negative-log-likelihood internally.


##### 4. Train the model
```python
max_num_workers = rec_num_workers()
cifar_dm = SimpleDataModule(cifar_train,
                            cifar_test,
                            validation=cifar_test,
                            num_workers=max_num_workers,
                            batch_size=64)

cifar_logger = CSVLogger('logs', name='CIFAR10')
cifar_trainer = Trainer(deterministic=True,
                        max_epochs=20,
                        logger=cifar_logger,
                        callbacks=[ErrorTracker()])
cifar_trainer.fit(cifar_module, datamodule=cifar_dm)
```


##### 5. Test
```python
test_results = cifar_trainer.test(cifar_module, datamodule=cifar_dm)
print("Test loss:", test_results[0]['test_loss'])
print("Test accuracy:", test_results[0]['test_accuracy'])
```


##### Plot training history
```python
cifar_results = pd.read_csv(cifar_logger.experiment.metrics_file_path)
fig, ax = subplots(1, 1, figsize=(6, 6))
for col, color, label in [('train_accuracy_epoch', 'black', 'Training'),
                          ('valid_accuracy',       'red',   'Validation')]:
    cifar_results.plot(x='epoch', y=col, label=label,
                       marker='o', color=color, ax=ax)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
```


##### Additional plot: confusion matrix
```python
from sklearn.metrics import confusion_matrix

cifar_model.eval()
with torch.no_grad():
    logits = cifar_model(cifar_test_X)
    predictions = logits.argmax(dim=1).numpy()
y_true = np.array([y.item() for _, y in cifar_test])
cm = confusion_matrix(y_true, predictions)
print(cm)
```


### b) Compute the misclassification error given the confusion matrix.




### Problem 4.2: Improving the test accuarcy with data augmentation techniques

Data augmentation techniques are used to artificially increase the size of the training dataset by applying various transformations to the original images. This helps to improve the robustness and generalization ability of CNN models. Data augmentation can be viewed as a regularization technique for better generalization.

Here are some commonly used data augmentation techniques for image data (the visual examples can be found [here](https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py)):

- Rotation: Rotating the image by a certain degree to generate new samples.
- Flip: Flipping the image horizontally or vertically to generate new samples.
- Zooming: Zooming into or out of the image by a certain factor to generate new samples.
- Translation: Shifting the image horizontally or vertically by a certain distance to generate new samples.
- Shearing: Tilting the image in a particular direction by a certain angle to generate new samples.
- Noise addition: Adding random noise to the image to generate new samples.
- Color jittering: Modifying the brightness, contrast, or hue of the image to generate new samples.
- Cropping: Cropping a portion of the image to generate new samples.

By applying these transformations to the original images, we can generate new samples that are different from the original ones but still share the same class label. This helps the model to learn to recognize the important features of the images, regardless of their position or orientation, and make more accurate predictions on unseen data.

In this example, we apply several simple data augmentations such as random rotations, width and height shifts, and horizontal flips.


```python
from torchvision.transforms import (Compose,
                                    RandomAffine,
                                    RandomHorizontalFlip,
                                    ToTensor)
from torch.utils.data import Dataset

# 1) Load and preprocess data (raw uint8 arrays so transforms can be applied)
(cifar_train_raw,
 cifar_test_raw) = [CIFAR10(root='data', train=train, download=True)
                    for train in [True, False]]

train_transform = Compose([
    RandomAffine(degrees=10, translate=(0.1, 0.1)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
])
test_transform = ToTensor()

class CIFAR10WithTransform(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, target = self.base[idx]
        return self.transform(img), target

cifar_train_aug = CIFAR10WithTransform(cifar_train_raw, train_transform)
cifar_test_aug  = CIFAR10WithTransform(cifar_test_raw,  test_transform)

# 2) Define the model (same as 4.1)
cifar_model_aug = CIFAR10Model()

# 3) Set up the training module
cifar_module_aug = SimpleModule.classification(
    cifar_model_aug,
    optimizer=Adam(cifar_model_aug.parameters(), lr=0.001),
    loss=nn.CrossEntropyLoss()
)

# 4) Train the model with data augmentation
batch_size = 64
cifar_dm_aug = SimpleDataModule(cifar_train_aug,
                                cifar_test_aug,
                                validation=cifar_test_aug,
                                num_workers=max_num_workers,
                                batch_size=batch_size)

cifar_logger_aug = CSVLogger('logs', name='CIFAR10_aug')
cifar_trainer_aug = Trainer(deterministic=True,
                            max_epochs=20,
                            logger=cifar_logger_aug,
                            callbacks=[ErrorTracker()])
cifar_trainer_aug.fit(cifar_module_aug, datamodule=cifar_dm_aug)
```


```python
# Test
test_results_aug = cifar_trainer_aug.test(cifar_module_aug,
                                          datamodule=cifar_dm_aug)
print("Test loss:",     test_results_aug[0]['test_loss'])
print("Test accuracy:", test_results_aug[0]['test_accuracy'])
```

```python
# Plot training history
cifar_results_aug = pd.read_csv(cifar_logger_aug.experiment.metrics_file_path)
fig, ax = subplots(1, 1, figsize=(6, 6))
for col, color, label in [('train_accuracy_epoch', 'black', 'Training'),
                          ('valid_accuracy',       'red',   'Validation')]:
    cifar_results_aug.plot(x='epoch', y=col, label=label,
                           marker='o', color=color, ax=ax)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
```

```python
# Additional plot: confusion matrix
cifar_model_aug.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in torch.utils.data.DataLoader(cifar_test_aug,
                                                        batch_size=128):
        all_preds.append(cifar_model_aug(X_batch).argmax(dim=1).numpy())
        all_true.append(y_batch.numpy())
predictions = np.concatenate(all_preds)
y_true = np.concatenate(all_true)
cm = confusion_matrix(y_true, predictions)
print(cm)
```


### a) What do you think of the effects of data augmentation given the obtained results?






## Problem 5: Univariate Time Series Classification with CNN

The following problem involves training a 1-dimensional Convolutional Neural Network (1D-CNN) model on a univariate time series dataset called Wafer, from [the UCR Time Series Classification Archive](https://www.timeseriesclassification.com/description.php?Dataset=Wafer). The Wafer dataset, formatted by R. Olszewski as part of his thesis at Carnegie Mellon University in 2001, contains inline process control measurements from various sensors during the processing of silicon wafers for semiconductor fabrication. Each sample within the dataset includes measurements from a single sensor during the processing of one wafer by one tool. The data is categorized into two classes: normal and abnormal. The Wafer dataset is a collection of 7164 time series samples, each with a length of 152. The dataset is divided into 1000 training samples and 6164 testing samples with the two different classes.

Convolutional Neural Networks (CNNs) have been proven effective not only at capturing spatial patterns in image data but also at detecting temporal patterns in time series data [4]. In this example, we will design our 1D-CNN model architecture to learn patterns and features from the univariate time series data to perform a time series classification. The model will consist of several 1D convolutional layers, followed by pooling layers to reduce dimensionality and extract relevant features and fully connected (dense) layers for classification.

[4] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017.

![Fig. Examples of the samples from the Wafer dataset with respect to different classes.)](Wafer_visual_examples.png)


##### 1. Load and preprocess data
```python
# load the Wafer dataset
train = pd.read_csv("dataset/Wafer/Wafer_TRAIN.tsv", header=None, sep="\t")
test  = pd.read_csv("dataset/Wafer/Wafer_TEST.tsv",  header=None, sep="\t")

# the first column in `train` and `test` contains label info.
# therefore we separate them into `x` and `y`.
x_train = train.iloc[:, 1:].to_numpy().astype(np.float32)
y_train = np.clip(train.iloc[:, 0].to_numpy(), 0, 1).astype(np.int64)
x_test  = test.iloc[:, 1:].to_numpy().astype(np.float32)
y_test  = np.clip(test.iloc[:, 0].to_numpy(),  0, 1).astype(np.int64)

# create a channel dimension so that `x` has dimension of (batch, channel, length)
x_train = x_train[:, np.newaxis, :]
x_test  = x_test[:,  np.newaxis, :]

# preprocess
# The provided dataset has already been preprocessed, therefore no need for it.
```

Note: PyTorch's `nn.Conv1d` expects the channel dimension *before* the length dimension, so the tensor shape is `(batch, channel, length)` rather than `(batch, length, channel)` as in the R/keras code.


### a) Fill in the missing parts in the following steps and run the model.


##### 2. Define the model
```python
class WaferModel(nn.Module):
    def __init__(self, seq_length):
        super(WaferModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=..., kernel_size=...),
            nn.ReLU(),  # fill in the activation
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=..., out_channels=..., kernel_size=...),
            nn.ReLU(),  # fill in the activation
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=..., out_channels=..., kernel_size=...),
            nn.ReLU(),  # fill in the activation
            nn.MaxPool1d(kernel_size=2))
        # compute the flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_length)
            flat_size = self.conv(dummy).flatten(1).shape[1]
        self.head = nn.Linear(flat_size, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)

wafer_model = WaferModel(seq_length=x_train.shape[2])
summary(wafer_model,
        input_size=(64, 1, x_train.shape[2]),
        col_names=['input_size', 'output_size', 'num_params'])
```

We're building a CNN model that has filter sizes of {16, 32, 64} and kernel sizes of {8, 5, 3} with the relu activation function for all convolutional layers. Fill in the blanks accordingly.


##### 3. Set up the training module
```python
wafer_module = SimpleModule.classification(
    wafer_model,
    optimizer=Adam(wafer_model.parameters(), lr=0.001),
    loss=nn.CrossEntropyLoss()
)
```

##### 4. Train the model
```python
x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train)
x_test_t  = torch.tensor(x_test)
y_test_t  = torch.tensor(y_test)

wafer_train = TensorDataset(x_train_t, y_train_t)
wafer_test  = TensorDataset(x_test_t,  y_test_t)

wafer_dm = SimpleDataModule(wafer_train,
                            wafer_test,
                            validation=wafer_test,
                            num_workers=min(4, max_num_workers),
                            batch_size=64)

wafer_logger = CSVLogger('logs', name='Wafer')
wafer_trainer = Trainer(deterministic=True,
                        max_epochs=100,
                        logger=wafer_logger,
                        callbacks=[ErrorTracker()])
wafer_trainer.fit(wafer_module, datamodule=wafer_dm)
```


```python
# Test
test_results = wafer_trainer.test(wafer_module, datamodule=wafer_dm)
print("Test loss:",     test_results[0]['test_loss'])
print("Test accuracy:", test_results[0]['test_accuracy'])
```

```python
# Plot training history
wafer_results = pd.read_csv(wafer_logger.experiment.metrics_file_path)
fig, ax = subplots(1, 1, figsize=(6, 6))
for col, color, label in [('train_accuracy_epoch', 'black', 'Training'),
                          ('valid_accuracy',       'red',   'Validation')]:
    wafer_results.plot(x='epoch', y=col, label=label,
                       marker='o', color=color, ax=ax)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
```


### b) Fit a logistic regression model and compare its performance (i.e., accuracy) to that of the CNN model.

##### Comparison to a Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# dataset (flatten back to 2D for sklearn)
x_train_flat = train.iloc[:, 1:].to_numpy()
y_train_flat = np.clip(train.iloc[:, 0].to_numpy(), 0, 1)
x_test_flat  = test.iloc[:, 1:].to_numpy()
y_test_flat  = np.clip(test.iloc[:, 0].to_numpy(),  0, 1)

# Fit a logistic regression model
logit_reg = ...  # fill in (instantiate and fit a LogisticRegression on x_train_flat, y_train_flat)

# Make predictions on the test set
predictions = (logit_reg.predict_proba(x_test_flat)[:, 1] > 0.5).astype(int)
acc_logit = accuracy_score(y_test_flat, predictions)

print("=== [1D CNN] ===")
print(" Test accuracy:", test_results[0]['test_accuracy'])
print("============================\n")
print("=== [Logistic Regression] ===")
print(" Test accuracy:", acc_logit)
print("=============================\n")
```
