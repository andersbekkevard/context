# Module 11: Deep Learning and Neural Networks

TMA4268 Statistical Learning V2024 — Sara Martino, NTNU

# Acknowledgements

* Most of the slides were created by Stefanie Muff, also some of this material was (in a modified version) created by Mette Langaas who has put a lot of effort in creating this module in its original version. Thanks to Steffi and Mette for the permission to use the material!

* Some of the figures and slides in this presentation are taken (or are inspired) from @ISL.

## Learning material for this module

* James et al (2021): An Introduction to Statistical Learning. Chapter 10.

* All the material presented on these module slides and in class.

* Videos on neural networsk and back propagation
    + [Video 1](https://www.youtube.com/watch?v=aircAruvnKk)
    + [Video 2](https://www.youtube.com/watch?v=IHZwWFHWa-w)
    + [Video 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
    + [Video 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)

**Secondary material (not compulsory):**

* Background material: Chapters 6-8 @goodfellow
<https://www.deeplearningbook.org>

See also _References and further reading_ (last slide), for further reading material.

## What will you learn?

* Translating from statistical to neural networks language
    + linear regression
    + logistic regression
    + multiclass (multinomial) regression

* Deep learning: The timeline

* Single and multilayer feed-forward networks

* Convolutional neural networks (CNNs)

* Recurrent neural networks (RNNs)

* When to use deep learning?


# Introduction: Time line

* 1950's: First neural networks (NN) in "toy form".

* 1980s: the backpropagation algorithm was rediscovered.

* 1989: (Bell Labs, Yann LeCun) used convolutional neural networks to classifying handwritten digits.

* 2000s: After the first hype, NNs were pushed aside by boosting and support vector machines in the 2000s.

* Since 2010: Revival! The emergence of _Deep learning_ as a consequence of improved computer resources, some innovations, and applications to image and video classification, and speech and text processing.

* Brain neurons -- inspiration for neural networks.

![ ](Neuron3.png){width=40%}

Image credits: By Egm4313.s12 (Prof. Loc Vu-Quoc) <https://commons.wikimedia.org/w/index.php?curid=72816083>

* Shift from statistics to computer science and machine learning, as they are highly parameterized.

* Statisticians were skeptical: "It's just a nonlinear model".

* There are several learning resources (some listed under 'further references') that you my turn to for further knowledge into deep learning.
* There is a new IT3030 [deep learning course at NTNU](https://www.ntnu.no/studier/emner/IT3030#tab=omEmnet).

![ ](DeepLearningwithR.jpeg){width=30%} ![ ](DeepLearning.jpeg){width=30%}

## AI, machine learning and statistics

\includegraphics[]{AI_ML_DL.png}

## AI

* Artificial intelligence (AI) dates back to the 1950s, and can be seen as _the effort to automate intellectual tasks normally performed by humans_ (page 4, @kerasR).

* AI was first based on hardcoded rules (like in chess programs), but turned out to be intractable for solving more complex, fuzzy problems.

## Machine learning

* With the field of _machine learning_ the shift is that a system is _trained_ rather than explicitly programmed.

* ML deals with much larger and more complex data sets than what is usually done in statistics.

* The focus in ML is oriented towards _engineering_, and ideas are proven _empirically_ rather than theoretically (which is the case in mathematical statistics).

According to @kerasR (page 19):

_Machine learning isn't mathematics or physics, [...] it's an engineering science._

## Deep learning

> Deep Learning is an algorithm which has no theoretical limitations of what it can learn; the more data you give and the more computational time you provide, the better it is.
>
> Geoffrey Hinton (Google)

* _Deep_ does not refer to a deeper understanding.

* Rather, deep referes to the _layers of representation_, for example in a neural network.

![](deep.png){width=80%}

* In 2011 neural networks with many layers were performing well on image classification tasks.

* The [_ImageNet_](http://www.image-net.org/) classification challenge (classify high resolution color images into 1k different categories after training on 1.4M images) was won by solutions with deep convolutional neural networks (CNNs). In 2011 the accuracy was 74.3%, in 2012 83.6% and in 2015 96.4%.

* Since 2012, CNNs are the general solution for computer vision tasks. Other application area: natural language processing.

Task: Check out this website that gives overview over winning solutions:

https://mlcontests.com/state-of-competitive-machine-learning-2022/#winning-solutions

The success of deep learning is dependent upon the breakthroughts in

  * _hardware_ development, expecially with faster CPUs and massively parallell graphical processing units (GPUs).
  * _datasets_ and benchmarks (internet/tech data).
  * improved  _algorithms_.

Achievements of deep learning include

  * high quality (near-human to super human) image classification,
  * speech recognition,
  * handwriting transcription,
  * autonomous driving, and more!


# Feedforward networks

* Connections are only forward in the network, but no feedback connections that sends the output of the model back into the network.

* Examples: Linear, logistic and multinomial regression with or without any _hidden layers_ (between the input and output layers).

* We may have between zero and very many hidden layers.

* Adding _hidden layers_ with _non-linear activation functions_ between the input and output layer will make nonlinear statistical models.

* The number of hidden layers is called the _depth_ of the network, and the number of nodes in a layer is called the _width_ of the layer.

![test](drawNNp3h2o3.png){width=80%}

## The single hidden layer feedforward network

The nodes are also called _neurons_.

**Notation**

1. Inputs: $p$ input layer nodes ${\boldsymbol{x}^\top} = (x_1, x_2, \ldots, x_p)$.
2. The nodes $z_m$ in the hidden layer, $m=1,\ldots, M$; as vector ${\boldsymbol z}^\top=(z_1, \ldots, z_M)$, and the hidden layer activation function $g()$.
$$
z_m({\boldsymbol x})=g(\alpha_{0m}+\sum_{j=1}^p \alpha_{jm}x_{j})
$$
where $\alpha_{jm}$ is the weight from input $j$ to hidden node $m$, and $\alpha_{0m}$ is the bias term for the $m$th hidden node.

3. The node(s) in the output layer, $c=1,\ldots C$: $y_1, y_2, \ldots, y_C$, or as vector ${\boldsymbol y}$, and output layer activation function $f()$.
$$
\hat{y}_c({\boldsymbol x})=f(\beta_{0c}+\sum_{m=1}^M \beta_{mc}z_{m}({\boldsymbol x}))
$$
where $\beta_{mc}$ is from hidden neuron $m$ to ouput node $c$, and $\beta_{0c}$ is the bias term for the $c$th output node.

4. Taken together
$$
\hat{y}_c({\boldsymbol x})=f(\beta_{0c}+\sum_{m=1}^M \beta_{mc}z_{m})=f(\beta_{0c}+\sum_{m=1}^M \beta_{mc}g(\alpha_{0m}+\sum_{j=1}^p \alpha_{jm}x_{j}))
$$

**Hands on:**

* Identify $p, M, C$ in the network figure above, and relate that to the $\hat{y}_{c}({\boldsymbol x})$ equation.

* How many parameters (the $\alpha$ and $\beta$s) need to be estimated for this network?

* What determines the values of $p$ and $C$?

* How is $M$ determined?

### Special case: linear activation function for the hidden layer

If we assume that $g(z)=z$ (linear or identity activiation):

$$
\hat{y}_c({\boldsymbol x})= f(\beta_{0c}+\sum_{m=1}^M \beta_{mc}(\alpha_{0m}+\sum_{j=1}^p \alpha_{jm}x_{j}))
$$

**Q:** Does this look like something you have seen before?

**A:**

* Principal component regression.
* Partial least squares.

## Multilayer neural networks

Alternative: networks with more than one hidden layer. A network with _many hidden layers_ is called a _deep network_.

![](fig10_4.png){width=70%}

(Fig 10.4 @ISL)

* The idea of the multilayer NN is exactly the same as for the single-layer version.

* $f_m(X)$ is a (transformation of) the linear combination of the last layer.

## Outcome encoding

* _Continuous_ and _binary_ may only have one output node ($y_i=$ observed value).

* For $C$ _categories_, we have $C$ output nodes, where we encode the output as $Y = (Y_1, Y_2, \ldots, Y_C)$ , where ${\boldsymbol y}_i=(0,0,\ldots,0,1,0,\ldots,0)$ with a value of $1$ in the $c^{th}$ element of ${\boldsymbol y}_i$ if the class is $c$. This is called _one-hot encoding_ or _dummy encoding_.

## Example: MNIST dataset

* Aim: Classification of handwritten digits.

* Categorical outcome $C=0,1,\ldots,9$.

* This data has been analysed with a single-layer feed-forward network in a previous version of the course: <https://www.math.ntnu.no/emner/TMA4268/2018v/11NN/8-neural_networks_mnist.html>.

Objective: classify the digit contained in an image (28 $\times$ 28 greyscale).

![](MNIST_dataset_example.png)

## Neural network parts

We now focus on the different elements of neural networks.

1) Output layer activation

2) Hidden layer activation

3) Network architecture

4) Loss function

5) Optimizers

## 1) Output layer activation

These choices have been guided by solutions in statistics (multiple linear regression, logistic regression, multiclass regression)

* _Linear activation_: for _continuous outcome_ (regression problems) $$f(X)=X \ .$$

* _Sigmoid activation_: for _binary outcome_ (two-class classification problems) $$f(X)=\text{Pr}(Y=1 | X ) = \frac{1}{1+\exp(-X)} = \frac{\exp(X)}{1+\exp(X)} \ .$$

* _Softmax_: for _multinomial/categorical outcome_  (multi-class classification problems)
$$
f_m(X) =  \text{Pr}(Y=m | X ) = \frac{\exp(Z_m)}{\sum_{s=1}^{C}\exp(Z_s)} \ .
$$

Note that we denote by $Z_m$ the value in the output node $m$ _before_ the output layer activation.

## 2) Hidden layer activation

(See chapter 6.3 in @goodfellow)

**Very common**:

* The **sigmoid** $g=\sigma(x)=1/(1+\exp(-x))$ (logistic) activation functions.
* The **rectified linear unit (ReLU)** $g(x)=\max(0,x)$ activation functions.

```r
library(ggplot2); library(ggpubr)
x=seq(-6,6,length=3)
y=pmax(0,x)
p=qplot(x,y,geom="line",main="ReLU")
pp=ggplot(data.frame(x=c(-6,6)), aes(x))+
  xlab(expression(x))+
  ylab(expression(mu))+
    stat_function(fun=function(x) exp(x)/(1+exp(x)), geom="line", colour="red")+
  ggtitle("Sigmoid")
ggarrange(pp,p)
```

Among all the possibilities, ReLU is nowadays the most popular one. Why?

* The function is piecewise linear, but _in total non-linear_.

* Replacing sigmoid with ReLU is reported to be one of the major changes that have improved the performance of the feedforward networks (Goodfellow et al, Section 6.6).

## Universal approximation property

* Think of the goal of a feedforward network to approximate some function $f$, mapping our input vector ${\boldsymbol x}$ to an output value ${\boldsymbol y}$.

* What type of mathematical function can a feedforward neural network with one hidden layer and linear output activation represent?

The _universal approximation theorem_ (Goodfellow et al 2016, Section 6.4.1, https://www.deeplearningbook.org) says that a feedforward network with

  * a _linear output layer_
  * at least one hidden layer with a "squashing" activation function (e.g., ReLU or sigmoid) and "enough" hidden units

can approximate any (Borel measurable) function from one finite-dimensional space (our input layer) to another (our output layer) with any desired non-zero amount of error.

## 3) Network architecture

Network architecture contains three components:

* _Width_: How many nodes are in each layer of the network?

* _Depth_: How deep is the network (how many hidden layers)?

* _Connectivity_: How are the nodes connected to each other?

Especially the connectivity depends on the problem, and here experience is important.

* We will consider _feedforward networks_, _convolutional neural networks (CNNs)_ and _recursive neural networks (RNNs)_.

## 4) Loss function ("Method")

* The choice of the loss function is closely related to the output layer activation function.

* Most popular problem types, output activation and loss functions:

| Problem | Output nodes | Output activation | Loss function |
|--------------|---------|-----------------------|---------------|
| Regression | 1 | `linear` | `mse` |
| Classification (C=2)| 1 | `sigmoid` | `binary_crossentropy` |
| Classification (C>2)| C |  `softmax` | `categorical_crossentropy` |

* Regression: Loss function _MSE_ for a given set of parameters $\boldsymbol{\theta}$:

$$R({\boldsymbol \theta}) = \sum_{i=1}^n (y_i- f(x_i))^2$$

* Classification: _Cross-entropy_

$$R({\boldsymbol \theta}) = -\sum_{i=1}^n \sum_{m=1}^C y_i \log f_m(x_i) \ ,$$

* with special case for $C=2$ (_binary cross-entropy loss_):
$$R({\boldsymbol \theta}) = -\sum_{i=1}^n  y_i \log f_m(x_i) + (1-y_i) \log (1-f_m(x_i)) \ .$$

## 5) Optimizors

Let the unknown parameters be denoted ${\boldsymbol \theta}$ (what we have previously denoted as $\alpha$s and $\beta$s), and the loss function to be minimized $R({\boldsymbol \theta})$.

* Gradient descent

* Mini-batch stochastic gradient descent (SGD) and true SGD

where the gradient is calculated via _backpropagation_.

### Gradient descent

* We minimize a _cost function_ by iteratively tweaking the parameters along the negative gradient.

![](gradient_descent.png){width=60%}

(https://github.com/SoojungHong/MachineLearning/wiki/Gradient-Descent)

* In practice this happens in a _high-dimensional_ parameters space, along the partial derivatives for each parameter.

## Finding optimal weights: Gradient descent algorithm

1. Let $t=0$ and denote the given initial values for the parameters ${\boldsymbol \theta}^{(t)}$.
2. Until finding a (local) optimum, repeat a) to e)
    a) Calculate the predictions ${\hat{y}_1({\boldsymbol x}_i)}$.
    b) Calculate the loss function $R({\boldsymbol \theta}^{(t)})$.
    c) Find the gradient (direction) in the $(p+1)$-dimensional space of the weights, and evaluate this at the current weight values $\nabla R({\boldsymbol \theta}^{(t)})={\frac{\partial R}{\partial {\boldsymbol \theta}}}({\boldsymbol \theta}^{(t)})$.
    d) Go with a given step length (_learning rate_) $\lambda$ in the direction of the negative of the gradient of the loss function to get
$${\boldsymbol \theta}^{(t+1)}={\boldsymbol \theta}^{(t)} - \lambda \nabla R({\boldsymbol \theta}^{(t)}) \ .$$
    e) Set $t=t+1$.
3. The final values of the weights in that $(p+1)$ dimensional space are our parameter estimates and your network is _trained_.

### Full vs. stochastic gradient descent (SGD)

* Note that in _full gradient descent_, the loss function is computed as a mean over all training samples:
$$
R({\boldsymbol \theta})=\frac{1}{n}\sum_{i=1}^n R({\boldsymbol x}_i, y_i) \ .
$$

* The gradient is _an average over many individual gradients_ from the training sample. You can think of this as an estimator for an expectation

$$
\nabla_{\boldsymbol \theta} R({\boldsymbol \theta})=\frac{1}{n}\sum_{i=1}^n \nabla_{\boldsymbol \theta} R({\boldsymbol x}_i, y_i) \ .
$$

* To build a network that generalizes well, it is important to have many training samples, but that would make us spend a lot of time and computer resources at calculating each gradient descent step.

### Mini-batch stochastic gradient descent (SGD)

**Crucial idea**:

The expectation can be approximated by the average gradient over just a _mini-batch_ (random sample) of the observations.

**Advantages**:

* The optimizer will converge much faster if it can rapidly compute approximate estimates of the gradient.

* Mini-batches may be processed _in parallel_, and the batch size is often a power of 2 (32 or 256).

* Small batches also bring in a _regularization effect_, due to the variability they bring to the optimization process.

**Special case**: _True SGD_ involves only _one sample_ (mini-batch size 1). $\rightarrow$ Mini-batch SGD is a compromise between SGD (one sample per iteration) and full gradient descent (full dataset per iteration)

In the 3rd video (on backpropagation) from 3Blue1Brown there is nice example of one trajectory from gradient decent and one from SGD (10:10 minutes into the video):
<https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3>

### Backpropagation algorithm

* _Backpropagation_ is a simple and inexpensive way _to calculate the gradient_.

* Computing the analytical expression for the gradient $\nabla R$ is not difficult, but _the numerical evaluation may be expensive_.

* The _chain rule_ is used to compute derivatives of functions of other functions where the derivatives are known. This is efficiently done with backpropagation.

### Backpropagation algorithm

We look at the quadratic loss function.

We want to compute the gradient of
$$
R(\theta) = \sum R_i(\theta)
$$
where

$$
\begin{aligned}
R_i(\theta) = \frac{1}{2}\left(y_i - f(\mathbf{x}_i)\right)^2 = \frac{1}{2}\left(y_i - \beta_{0} - \sum_{k = 1}^K \beta_{k} z_{ik}\right)^2 = \\
\frac{1}{2}\left(y_i - \beta_{0} - \sum_{k = 1}^K \beta_{k} g(\alpha_{k0} + \sum_{j = 1}^p\alpha_{kj}x_{ij})\right)^2
\end{aligned}
$$

wrt $\beta_{k}$ and $\alpha_{kj}$

We consider one data point at the time.

More background:

* @ISL Chapter 10.7 (this is part of the compulsory course material - study yourself).

* Mathematical details: @goodfellow Section 6.5.

* 3Blue1Brown videos: <https://www.youtube.com/watch?v=Ilg3gGewQ5U> and <https://www.youtube.com/watch?v=tIeHLnjs5U8>

## Regularization

* Often: more weights than data samples $\rightarrow$ danger for over-fitting.

* Regularization: _any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error_ (Goodfellow et al, Chapter 7).

* Remember:

  * **Module 6**: The aim of regularization was to trade _increased bias_ for _reduced variance_.  The ideas was to add a penalty to the loss function.
  * **Module 9**: Tree pruning and L1 and L2 regularization in XGBoost.

### Regularization in neural networks

* _Ridge/Lasso penalization_: In neural networks this means adding an $L_2$ or $L_1$-penalty to the loss function to _penalize large weights_ (see chapter 10.7.2 in the course book):

\begin{align*} \tilde{J}({\boldsymbol w})= R({\boldsymbol w}) + \lambda{{\boldsymbol w}^\top{\boldsymbol w}}  \\
 \tilde{J}({\boldsymbol w})= R({\boldsymbol w}) + \alpha ||{\boldsymbol w}||
\end{align*}

* _Data augmentation_: Adding "fake data" to the dataset, in order that the trained model will generalize better. For example: rotating and scaling images.

* _Label smoothing_: Motivated by the fact that the training data may contain errors in the reponses recorded, and replaced the one-hot coding for $C$ classes with $\epsilon/(C-1)$ and $1-\epsilon$ for some small $\epsilon$.

* _Early stopping_

* _Dropout_

* _SGD_ is also a form of regularization (prevents over-fitting).

### Early stopping

Based on @goodfellow, Section 7.8

* The most commonly used for of regularization.

* For a sufficiently large model with the capacity to over-fit the training data, we observe that the training error decreases steadily during training, but the error on the validation set at some point begins to increase.

* The idea: Return the parameters that (earlier) gave the best performance on the validation set, before convergence on the training data.

### Dropout

Based on @goodfellow, Section 7.12, and @kerasR 4.4.3

Dropout was developed by Geoff Hinton and his students.

* During training: randomly _dropout_ (set to zero) some outputs in a given layer at each iteration. Drop-out rates may be chosen between 0.2 and 0.5.

* During test: no dropout, but scale down the layer output values by a factor equal to the drop-out rate (since now more units are active than we had during training).

* Alternatively, the drop-out and scaling (now upscaling) can be done during training.

## Dropout

![](fig10_19.png)
Fig 10.19, @ISL

## Ways to avoid overfitting

There are **hyperparameters** when building and fitting a neural network, like the network architecture, the number of batches to run before terminating the optimization, the drop-out rate, etc.

To avoid overfit, we have some strategies:

* Reduce network size.

* Collect more observations.

* Regularization.

It is important that the hyperparameters are chosen on a validation set or by cross-validation.

However, we may run into _validation-set overfitting_: when using the validation set to decide many hyperparameters, so many that you may effectively overfit the validation set.

## How to fit those models?

* We will use both the rather simple `nnet` R package by Brian Ripley and the currently very popular `keras` package for deep learning (the `keras` package will be presented later).

* `nnet` fits _one hidden layer_ with _sigmoid activiation function_. The implementation is not gradient descent, but instead BFGS using `optim`.

* Type `?nnet()` into your R-console to see the arguments of `nnet()`.

# An example

## Boston house prices

**Objective**: To predict the median price of owner-occupied homes in a given Boston suburb in the mid-1970s using 10 input variables.

This data set is both available in the `MASS` and `keras` R package.

Read and check the data file:

```r
library(MASS)
data(Boston)
dataset <- Boston
head(dataset)
```

Preparation: Split into training and test data:

```r
set.seed(123)
tt.train <- sort(sample(1:506,404,replace=FALSE))
train_data <- dataset[tt.train,1:13]
train_targets <- dataset[tt.train,14]

test_data <- dataset[-tt.train,1:13]
test_targets <- dataset[-tt.train,14]
```

* To make the optimization easier with gradient based methods do _feature-wise normalization_.

```r
org_train=train_data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)
```

* **Note**: the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.

Just checking out one hidden layer with 5 units to get going.

```r
library(nnet)
fit5<- nnet(train_targets~., data=train_data,size=5,linout=TRUE,maxit=1000,trace=F)
```

Calculate the MSE and the mean absolute error:

```r
pred=predict(fit5,newdata=test_data,type="raw")
mean((pred[,1]-test_targets)^2)
mean(abs(pred[,1]-test_targets))
```

```r
library(NeuralNetTools)
plotnet(fit5)
```

### Boston example using `keras`

See recommended exercise.

# Neural Networks

![](fig10_4.png){width=120%}

 - Output layer activation
 - Hidden layer activation
 - Network architecture
 - Loss function
 - Optimizers

# Today

 - Two important types of NN

    - Convolutional NN
    - Recurrent NN

# Convolutional neural networks (CNNs)

* Motivated by image classification.

* Example: the CIFAR-100 dataset (https://www.cs.toronto.edu/~kriz/cifar.html): Images of 100 categories with 600 images each.

![](cifar10.png){width=40%}

(Example from the CIFAR-10 data set with only 10 classes).

# Idea of CNNs: recognize features and patterns.

* The network identifies _low-level features_ (edges, color patched etc).

* These low-level features are then combined into _higher-level features_.

![](fig10_6.png){width=50%}

# Elements of a CNN

Two types of layers:

   - _Convolution layers_
   - _Pooling layers_.

## Convolution layers

* Composed of  _filters_.

* Example:

$$\left[
\begin{matrix}
a & b & c \\
d & e & f \\
g & h & i\\
j & k & l \\
\end{matrix}
\right] \qquad \text{Convolved with } \qquad
\left[
\begin{matrix}
\alpha & \beta \\
\gamma & \delta \\
\end{matrix}\right] $$
$\rightarrow$  Convolved image:

The filter highlights regions in the image that are similar to the filter itself.

Filtering for vertical or horizontal stripes:

![](fig10_7.png){width=80%}

(Figure 10.7)

* In _image processing_ we would use predefined (fixed) filters.

* In CNNs, the idea is that the filters are _learned_.

* One filter is applied to each color (red, green, blue), so three convolutions are happening in parallel and then immediately summed up.

* In addition, we can use $K$ different filters in a convolution step. This produces 3D feature maps (of depth $K$).

* The convolved image is then also processed with the ReLU activation function.

## Pooling layers

* Idea: consense/summarize information about the image.

* _Max pool_: Use the maximum value in each $2\times 2$ block.
$$\left[
\begin{matrix}
1 & 2 & 5 & 3 \\
3 & 0 & 1 & 2 \\
2 & 1 & 3 & 4\\
1 & 1 & 2 & 0 \\
\end{matrix}
\right] \qquad \rightarrow \qquad
\left[
\begin{matrix}
3 & 5 \\
2 & 4 \\
\end{matrix}\right] $$

In a CNN, we now combine convolution and pooling steps iteratively:

![](fig10_8.png)

* The number of channels after a convolution step is the number of filters ($K$) that is used in this iteration.

* The dimension of the 2D images after a pooling step is reduced, depending on the dimension of the filter (e.g., $2\times 2$ reduces each dimension by a factor of 2).

* In the end, all the dimensions are _flattened_ (pixels become ordered in 2D).

* The output layer has a _softmax_ activation function since the aim is classification.

### Data augmentation

* Very simple idea: Make the analysis more robust by including replicated, but slightly modified pictures of the original data.

* Example:

![](fig10_9.png){width=90%}

Figure 10.9 of @ISL

### An Interactive Node-Link Visualization of Convolutional Neural Networks

See

<https://adamharley.com/nn_vis/>

### Examples

See

* Section 10.3.5 in the book.

* Examples in the recommended exercise 11.

# Recurrent neural networks (RNNs) - Motivation

What is the next word?

How to make ...

 - pizza?
 - yoga?
 - rain?

# Problems with other "vanilla" NN

 - Length of input

 - Sequenciality of input

# Recurrent neural networks (RNNs)

* Suitable for data with sequential character.

* Examples: Text documents, time series (temperature, stock prices, music, speech,...)

* The input object $X$ is a sequence, for example a sequence of $L$ words.

* In the  most simple case, the output $Y$ is a single value (continuous, binary or a category).

* More advanced RNNs are able to map sequences to sequences (_Seq2Seq_) (Google Translate uses this technique, for example) in language modeling, and much more!

![](fig10_12.png){width=80%}

Figure 10.12 in @ISL

* Observed sequence $X=\{ X_1, \ldots , X_L \}$, where each $X_l^\top=(X_{l1},\ldots, X_{lp})$ is an input vector at point $l$ in the sequence.

* If $X$ is a text, $X_l$ can for example be the one-hot encoding for word $l$ (long vector with almost only 0s).

* Sequence of hidden layers $\{ A_1, \ldots, A_L \}$, where each $A_l$ is a layer of $K$ units $A_l^\top = (A_{l1}, \ldots , A_{lK})$.

Example of one-hot encoding a text with 20 words:

![](fig10_13.png)

Figure 10.13 in @ISL

* $A_{lk}$ is determined as
\begin{equation}\label{eq:chain}
A_{lk} = g(w_{k0} + \sum_{j=1}^p w_{kj}X_{lj} + \sum_{s=1}^K u_{ks}A_{l-1,s}) \ ,
\end{equation}
with hidden layer activation function $g()$ (e.g., ReLU).

* The output is determined as
$$O_l = \beta_0 + \sum_{k=1}^K\beta_k A_{lk} \ ,$$
potentially with a sigmoid or softmax output activation for binary or categorical outcome.

* Note: The weights $\boldsymbol{W}$, $\boldsymbol{U}$ and $\boldsymbol{B}$ are the _same_ at each point in the sequence. This is called _weight sharing_.

### Fitting the weights in an RNN

* Minimize a _loss function_. In regression problems:
$$\text{Loss} = (Y- O_L)^2 \ . $$

* Only the _last observation_ is relevant. How can this be meaningful?

* Reason: each element $X_l$ contributes to $O_L$ via equation (1).

* For input sequences $(x_i,y_i)$ ($1\leq i \leq n$), we minimize $\sum_{i=1}^n (y_i - o_{iL})^2$.

* Note: all $x_i = \{ x_{i1}, \ldots, x_{iL} \}$ are _vectors_.

Why are the outputs $O_1, \ldots, O_{L-1}$ there at all?

**A**:

* They come for free (same weights $\boldsymbol{B}$).

* Sometimes, the output is a whole sequence.

### Example of an RNN: Time series forecasting

Trading statistics from New York Stock exchange:

![](fig10_14.png){width=70%}

Figure 10.14 of @ISL

**Observations:**

* Every day ($t=1,\ldots, 6051$) we measure three things, denoted as $(v_t, r_t, z_t)$.

* All three series have high _auto-correlation_.

**Aim:**

* Predict the trading volume $v_t$ on day $t$ from
  +  $v_{t-1}$, $v_{t-2}$, ...,
  +  $r_{t-1}$, $r_{t-2}$, ..., and
  +  $z_{t-1}$, $z_{t-2}$, ...

But, how do we represent this problem in terms of Figure 10.12?

The idea is to extract shorter series up to a _lag_ of length $L$:

$$X_1 = \left(
\begin{matrix}
v_{t-L}\\
r_{t-L}\\
z_{t-L}
\end{matrix}
\right), \
\quad X_2 = \left(
\begin{matrix}
v_{t-L+1}\\
r_{t-L+1}\\
z_{t-L+1}
\end{matrix}
\right), ... ,
\quad
X_L = \left(
\begin{matrix}
v_{t-1}\\
r_{t-1}\\
z_{t-1}
\end{matrix}
\right),
\quad
Y = v_t$$

* And then continue to formulate the model as indicated in Figure 10.12.

# Recurrent Neural Network

 - We presented only a simple version
 - More complex variats do exist
 - Much used in LLM

# When to use deep learning?

* We have learned about many new "fancy" and trendy methods. But is it always worth using the most advanced ones?

* Important: Try the simple methods as well. Sometimes they perform quite well.

* Advantage of, for example, simple linear regression?

### Example: The Hitters data set

* Remember from Chapter 6: Prediction of `Salary` for 263 baseball players.

We compare

* Linear model with 20 parameters.
* Lasso with CV, where 12 variables remain in the model
* A NN with one hidden layer and 64 units. The model has 1049 parameters.

![](table10_2.png)

Conclusions?

* ...
* ...

Typically, NN are attractive when the size of the training set is extremely large and when model interpretability is not a priority!

# Double descent

![](bias_variance.png){width=60%}

# Double descent

![](double_descent.png){width=60%}

Does this contraddict the bias-variance trade off?

We want to recover the function
$$
y = \sin(x)
$$
We have 20 noisy observations of the signal where
$$
y_i = \sin(x_i) +\epsilon_i, \qquad \epsilon_i\sim\mathcal{N}(0,0.3^2)
$$
We use natural splines...

![](fig10_21.png)

![](fig10-20-2.png){width=80%}

# Double descent - some consideration

 - Does not contradict the bias variance trade-off (the $x$-axis expresses the flexibility of the model)

 - For deep learning algorithms we can sometimes achieve good test error while interpolating the training error - particularly  in problems with *high signal-to-noise ratio*, such as natural image recognition and language translation

 - Most of the statistical learning methods seen  do not exhibit double descent. Regularized methods  can give great results without interpolating the data!

**To summarize: though double descent can sometimes occur in neural networks, we typically do not want to rely on this behavior.** [James et al. Ch 10.8]

# DL in Medicine

Lots of papers like those:

![](paper_1.png){width=80%}

# DL/ML in Ecology

![](paper2.png)

# References and further reading

* <https://youtu.be/aircAruvnKk> from 3BLUE1BROWN - 4 videos - using the MNIST-data set as the running example
* Look at how the hidden layer behave: <https://playground.tensorflow.org>
* @ESL,Chapter 11: Neural Networks
* @casi, Chapter 18: Neural Networks and Deep Learning
* @kerasR
* @goodfellow (used in IT3030) <https://www.deeplearningbook.org/>
* Explaining backpropagation <http://neuralnetworksanddeeplearning.com/chap2.html>
