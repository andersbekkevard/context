---
title: "Module 11: Solutions to Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2024"
author:
  - Daesoo Lee, Kenneth Aase, Stefanie Muff, Sara Martino
  - Department of Mathematical Sciences, NTNU
date: "April 18, 2024"
---


---

## Problem 1

### a)

It is a 4-4-4-3 feedforward neural network with an extra bias node in both the input and the two hidden layers. It can be written in the following form

$$
y_c({\bf x})=\phi_o(\beta_{0c}+\sum_{m=1}^4 \beta_{mc}z_{m})=\phi_o(\beta_{0c}+\sum_{m=1}^4 \beta_{mc}\phi_{h*}(\gamma_{0m}+\sum_{l=1}^4 \gamma_{lm}\phi_h(\alpha_{0l}+\sum_{j=1}^4 \alpha_{jl}x_{j}))).
$$


### b)

It is not clear wheter the network has 3 input nodes, or 2 input nodes plus one bias node (both would lead to the same representation). The hidden layer has 4 nodes, but no bias node, and the output layer consists of two nodes.
This can be used for regression with two responses. If we have a classifiation problem with two classes then we usually use only one output node, but is is possible to use softmax activation for two classes, but that is very uncommon. Remember that for a binary outcome, we would usually only use one output node that encodes for the probability to be in one of the two classes.


### c)

When the hidden layer has a linear activation the model is only linear in the original covariates, so adding the extra hidden layer will not add non-linearity to the model. The feedforward model may find latent structure in the data in the hidden layer. In general, however, we would then recommend to directly use logistic regression, because you then end up with a model that is easier to interpret.


### d)

This is possible because the neural network is fitted using iterative methods. But, there is not one unique solutions here, and the network will benefit greatly by adding some sort of regulariztion, like weight decay and early stopping.

## Problem 2

### a)

This is a feedforward network with 10 input nodes plus a bias node, a hidden layer with 5 nodes plus a bias node, and a single node in the output layer. The hidden layer has a ReLU activiation function, whereas the output layer has a linear activation function.

The number of the estimated parameters are $(10+1)*5+(5+1)=61$.

### b)

Feedforward network with two hidden layers. Input layer has 4 nodes and no bias term, the first hidden layer has 10 nodes and ReLU activation and a bias node, the second hidden layer has 5 nodes plus a bias node and ReLU activiation. One node in output layer with sigmoid activiation.

The number of estimated parameters are $4*10+(10+1)*5+(5+1)=101$.

### c)

In module 7 we had an additive model of non-linear function, and interactions would be added manually (i.e., explicitly). Each coefficient estimated would be rather easy to interpret.
For neural nets we know that with one hidden layer and squashing type activation we can fit any function (regression), but may need many nodes - and then the interpretation might not be so easy. Interactions are automatically handled with the non-linear function of sums.





## Problem 3

##### Imports
```python
import numpy as np, pandas as pd
from matplotlib.pyplot import subplots

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset

from torchmetrics import MeanAbsoluteError
from torchinfo import summary

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything

from ISLP import load_data
from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)
```

##### 1. Load and preprocess data
```python
Boston = load_data('Boston')
X = Boston.drop(columns=['medv']).to_numpy().astype(np.float32)
Y = Boston['medv'].to_numpy().astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

scaler = StandardScaler(with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
```


### a)

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
            nn.Linear(32, 1))

    def forward(self, x):
        return torch.flatten(self.sequential(x))

boston_model = BostonModel(input_size=X_train.shape[1])
summary(boston_model,
        input_size=X_train.shape,
        col_names=['input_size', 'output_size', 'num_params'])
```

##### 3. Set up the training module
```python
boston_module = SimpleModule.regression(
    boston_model,
    metrics={'mae': MeanAbsoluteError()},
    optimizer=Adam(boston_model.parameters(), lr=0.001),
    loss=nn.MSELoss()
)
```


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
                             batch_size=64,
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

### b)

##### Comparison to a Linear Regression Model

```python
# Fit a linear regression model
linear_model = LinearRegression().fit(X_train, Y_train)

# Make predictions on the test set
predictions = linear_model.predict(X_test)

# Calculate the mean squared error and mean absolute error
mse = np.mean((Y_test - predictions) ** 2)
mae = np.mean(np.abs(Y_test - predictions))

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


### c)

* The feedforward neural network (FNN) demonstrates superior performance compared to the linear model. However, the FNN comes with reduced interpretability and increased complexity. As a result, some may prefer the simpler and more interpretable linear model.


```python
del(boston_model, boston_module, boston_dm,
    boston_logger, boston_trainer,
    boston_train, boston_test,
    X, Y, X_train, X_test, Y_train, Y_test,
    X_train_t, X_test_t, Y_train_t, Y_test_t)
```




## Problem 4: Convolutional Neural Network (CNN)

### Problem 4.1: Image Classification with CNN


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


### a)

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
    loss=nn.CrossEntropyLoss()
)
```


##### 4. Train the model
```python
max_num_workers = rec_num_workers()
cifar_dm = SimpleDataModule(cifar_train,
                            cifar_test,
                            validation=cifar_test,
                            num_workers=max_num_workers,
                            batch_size=32)

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


### b)

The exact misclassification rate should be slightly different for each run.
A misclassification rate is calculated as (number of misclassified samples / total number of samples).




### Problem 4.2: Improving the test accuarcy with data augmentation techniques


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

### a)
1. Increased size of the training dataset: Data augmentation allows for the creation of new training examples from the existing ones, which increases the size of the training dataset. A larger dataset helps in building more robust machine learning models that are less likely to overfit to the training data.

2. Improved generalization: By augmenting the training data, the model is exposed to more diverse examples, which helps it to generalize better to new, unseen data.

3. Increased model performance: Data augmentation can improve the performance of the model by reducing overfitting, especially in cases where the original dataset is small.

4. Cost-effectiveness: Data augmentation can be a cost-effective way of creating new training data, especially when collecting new data is expensive or time-consuming.

5. Reduced bias: Data augmentation can help to reduce bias in the dataset by balancing the class distribution, which is particularly important in cases where the original dataset is imbalanced.

6. Robustness to input variations: Data augmentation can make the model more robust to input variations such as rotation, scaling, and translation, which is useful in applications such as object recognition and natural language processing.



## Problem 5: Univariate Time Series Classification with CNN


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


### a)


##### 2. Define the model
```python
class WaferModel(nn.Module):
    def __init__(self, seq_length):
        super(WaferModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
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


### b)

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
logit_reg = LogisticRegression(max_iter=1000).fit(x_train_flat, y_train_flat)

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
