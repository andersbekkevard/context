---
concept: feedforward-network
module: 11-nnet
lectures: [L22, L23, L24]
isl-ref: 10.1
exercises:
  - Exercise11.1a - write input/output equation for a stated network with general activation functions; identify what kind of network it is
  - Exercise11.1b - interpret an illustrated NN architecture (depth, width, regression vs classification)
  - Exercise11.1c - compare a 1-hidden-layer FNN with linear hidden + sigmoid output to logistic regression
  - Exercise11.1d - how can a 10000-weight model fit on 1000 obs (regularization answer)
  - Exercise11.2a - given $\hat y = \beta_0 + \sum \beta_m \max(\alpha_{0m} + \sum \alpha_{jm} x_j, 0)$, identify architecture (1 hidden, ReLU) and count parameters
  - Exercise11.2b - same for a 2-hidden-layer ReLU + sigmoid output network
  - Exercise11.2c - compare GAM (sums of nonlinear functions of each covariate) vs FNN (sums of nonlinear functions of *sums* of covariates)
  - Exercise11.3 - fit a Keras feedforward NN to Boston housing, compare to linear regression
related: [activation-functions, nn-parameter-count, backpropagation, gradient-descent-and-sgd, nn-loss-functions, nn-regularization, universal-approximation, convolutional-neural-network, recurrent-neural-network, logistic-regression, generalized-additive-models, principal-component-regression]
tags:
  - concept
  - module/11-nnet
aliases:
  - FNN
  - feed-forward neural network
  - dense network
  - multilayer perceptron
---

# Feedforward neural network

The prof's framing: "the workhorse" of machine learning, input → hidden(s) → output, **no loops**, weights and biases everywhere, a non-linear activation in the hidden layer making it expressive. He wishes the course had spent more time here.

## Definition (prof's framing)

> "Today we're going to talk about feed-forward networks, neural networks, which are really fun, which should have been like the bulk of the course honestly … I think it'd be way more interesting to talk more about neural networks and like kind of the interesting tricks that they figured out and kind of a statistical perspective on those." - [[L23-nnet-1]]

> "Feed-forward networks really are kind of the workhorse of a lot of machine learning. It's not everything, but it's everywhere." - [[L23-nnet-1]]

A feedforward (FNN, "dense") network is a directed *acyclic* network of neurons organized in layers: an **input layer** (the data $\mathbf x$), one or more **hidden layers** of latent units $z_m$, and an **output layer** $\hat y$. Each non-input unit computes a weighted sum of the previous layer plus a bias, then applies an [[activation-functions|activation function]].

## Notation & setup

The prof's notation (slides + L23):
- $\mathbf x \in \mathbb R^p$ inputs, $p$ = #predictors (set by the data, pixels, tokens, …)
- $z_m, m = 1,\dots,M$ hidden units, $M$ = width (a hyperparameter)
- $\hat y_c, c = 1,\dots,C$ outputs, $C$ = #output nodes (set by the problem, 1 for regression/binary, $C$ for multi-class)
- $\alpha_{jm}$: weight from input $j$ into hidden unit $m$; $\alpha_{0m}$: bias of hidden unit $m$
- $\beta_{mc}$: weight from hidden unit $m$ into output $c$; $\beta_{0c}$: bias of output $c$
- $g(\cdot)$: hidden-layer activation; $f(\cdot)$: output-layer activation (can differ)

## Formula(s) to know cold

Hidden-unit activation:

$$z_m(\mathbf x) = g\!\left(\alpha_{0m} + \sum_{j=1}^{p} \alpha_{jm} x_j\right)$$

Output:

$$\hat y_c(\mathbf x) = f\!\left(\beta_{0c} + \sum_{m=1}^{M} \beta_{mc}\, z_m(\mathbf x)\right)$$

Stacked into one nested expression:

$$\hat y_c(\mathbf x) = f\!\left(\beta_{0c} + \sum_{m=1}^{M} \beta_{mc}\, g\!\left(\alpha_{0m} + \sum_{j=1}^{p} \alpha_{jm} x_j\right)\right)$$

Parameter count (single hidden layer): see [[nn-parameter-count]]. Headline: **$(p+1)M + (M+1)C$**.

For deeper networks with hidden widths $M_1, M_2, \dots$ stack the same recipe, the prof: "you just repeat the same thing." Multilayer is a deep network.

## Insights & mental models

### Why "feedforward" matters

> "There's a direction of flow information you're not seeing it go backwards … and also you don't have connections between nodes [within a layer]. … which is nice because then the learning of the parameters is easier. If you have them connected between them then it gets quite messy." - [[L22-unsupervised-2]]

> "What's nice about this construction of the feedforward network is that you notice here there's no loops. If there are loops it'd be a much harder problem." - [[L22-unsupervised-2]]

The acyclic structure is what makes [[backpropagation]] work in a single backward sweep. Loops (RNNs, brain) are much harder. See [[recurrent-neural-network]].

### M is engineering, not theory

P and C are pinned by the problem; the engineer picks **M, depth, and connectivity**.

> "M is largely determined through work, through effort, through figuring it's a hyperparameter. … There's no theoretical reason necessarily for the number M. I mean, obviously it can't be too small, but yeah, how big should it be? Who knows? … There's a lot of engineering that goes into machine learning." - [[L23-nnet-1]]

### Linear-activation collapse

If $g(z) = z$ (and $f$ is also linear), the whole network is just linear regression:

> "Just linear regression. Just with a lot of parameters. … if you make things linear then it all kind of collapses to boring. One of the key things in this field is that you have these non-linearities that lead to complex learning." - [[L23-nnet-1]]

Identity-hidden + non-linear-output → GLM-like. Slides identify the all-linear case with PCR / PLS. The lesson: **non-linearity in the hidden layer is what gives the model its expressiveness**. See [[activation-functions]] and [[universal-approximation]].

### Compared to GAMs (Exercise11.2c, flagged in Module 7 ↔ 11 conversation)

A [[generalized-additive-models|GAM]] is a sum of *separate* non-linear functions of *each individual* covariate: $y = \sum_j f_j(x_j)$. An FNN is a sum of non-linear functions of *linear combinations* of covariates: $y = \sum_m \beta_m\, g(\alpha_m^\top \mathbf x)$. The FNN's hidden units mix predictors before nonlinearizing, that's what lets it represent interactions natively. GAMs are interpretable per-variable; FNNs aren't.

### Compared to logistic regression (Exercise11.1c)

A 1-hidden-layer FNN with **linear hidden activation + sigmoid output** collapses (because of linearity above) to one linear combination passed through a sigmoid, exactly logistic regression with a possibly different parameterization. The FNN has more parameters but is otherwise equivalent. *Replace* the hidden activation with a non-linearity (ReLU, sigmoid) and the FNN gains expressive power that vanilla logistic regression can never reach.

### Parameters AND hidden states

> "The very confusing thing with neural networks is that you don't just learn, you learn parameters, right? You have to learn the parameters of the network, but you're also inferring states, hidden states, and a lot of hidden states. So this model has an ability to express and approximate functions that you just don't have at all with just simple regression, which is very interesting and cool." - [[L23-nnet-1]]

This is the conceptual leap from regression: the $z_m$'s are themselves *learned features*, not fixed quantities.

## Exam signals

> "There will be a question on the bias-variance decomposition." (cuts across the module, see [[bias-variance-tradeoff]]) - [[L27-summary]]

> "Q3b: Neural network parameter count + forward pass: Given a feed-forward neural network with stated input/hidden/output sizes: 'How many weights total, including biases?' Then: given specific weight values, inputs, and a ReLU activation, compute the output of the neuron." - [[L27-summary]]

> "His advice for the count: **draw the network, then count the parameters**, including bias terms." - [[L27-summary]]

> "Actually, this is a good example of where the question is technically wrong … you could technically have skip connections, or connections between neurons within a layer. So here you could say, 'just to be clear, I'm assuming a feed-forward network,' even though I think everyone would." - [[L27-summary]]

→ State your assumption explicitly when answering FNN questions.

## Pitfalls

- **Forgetting biases** when counting parameters, canonical wrong answer; see [[nn-parameter-count]].
- **Linear hidden activation** silently collapses the network to linear regression, easy T/F trap (Exercise11.1c).
- **Multi-class outputs** require $C$ output nodes with one-hot targets and softmax, not a single output node. The prof: "we encode the output as $Y = (Y_1, \ldots, Y_C)$ … with a value of 1 in the $c$th element of $\mathbf y_i$ if the class is $c$. This is called one-hot encoding."
- **More weights than data** is fine *with regularization*. See Exercise11.1d. Without regularization, the prof says you're "doing it wrong" ([[L26-nnet-3]]); with regularization, even the post-interpolation regime can generalize ([[double-descent]]).
- **Standardize inputs first.** Boston housing exercise: "always, you don't want one variable to basically suck up all the variance, just like in the PCA." See [[standardization]].
- **Don't use a NN if you don't have enough data or you need interpretability.** "If you don't have a lot of data and need interpretability, probably don't use neural networks at all. Use trees." - [[L24-nnet-2]] / [[L26-nnet-3]] (Hitters Comparison).

## Scope vs ISLR

- **In scope:** anatomy (input/hidden/output, weights, biases), the nested equation, multi-layer extension, parameter counting, the linear-activation collapse, comparisons to GAM/logistic/PCR, why feedforward matters for backprop.
- **Look up in ISLR:** §10.1-10.2, pp. 399-408: single-layer + multilayer feedforward, MNIST example, output encoding (one-hot), the cross-entropy / MSE loss table.
- **Skip in ISLR (book-only, prof excluded):** advanced architectures (skip connections, intra-layer connections, explicitly called out as ambiguous-but-out, [[L27-summary]]). Vanishing/exploding gradients and weight initialization (Xavier/He), "not discussed in any depth" - [[L24-nnet-2]]. Adam optimizer internals - [[L23-nnet-1]] / [[L27-summary]]. Detailed architecture-tuning recipes from §10.7.4.

## Exercise instances

- **Exercise11.1a**: write the input/output equation for the network using general activations $\phi_o, \phi_h, \phi_{h^*}$ and biases at every layer; name the architecture (multilayer FNN).
- **Exercise11.1b**: given the Wikipedia "colored NN" image, describe architecture (input width, hidden widths, output width) and identify whether it's regression/classification.
- **Exercise11.1c**: compare 1-hidden-layer FNN with **linear hidden + sigmoid output** to logistic regression: structurally identical model, different parameterization.
- **Exercise11.1d**: explain how 10000 weights ≫ 1000 obs is workable: regularization (L1/L2, dropout, early stopping, mini-batch SGD's implicit regularization) and benign overfitting.
- **Exercise11.2a**: given the formula $\hat y = \beta_0 + \sum_{m=1}^5 \beta_m \max(\alpha_{0m} + \sum_{j=1}^{10} \alpha_{jm} x_j, 0)$: this is a 1-hidden-layer ReLU FNN with $p=10$, $M=5$, $C=1$ (linear output). Parameters: $(10+1)\cdot 5 + (5+1)\cdot 1 = 55 + 6 = 61$.
- **Exercise11.2b**: given a deeper formula with two ReLU hidden layers ($p=4, M_1=10, M_2=5$) and sigmoid output: identify architecture, count $(4+1)\cdot 10 + (10+1)\cdot 5 + (5+1)\cdot 1 = 50 + 55 + 6 = 111$.
- **Exercise11.2c**: GAM ($y = \sum_j f_j(x_j)$, separable per-feature, interpretable, no interaction) vs FNN ($y = \sum_m \beta_m g(\alpha_m^\top \mathbf x)$, mixes features inside each unit, captures interactions, opaque).
- **Exercise11.3**: Boston Housing in Keras: 13 inputs → dense(64, ReLU) → dense(32, ReLU) → dense(1, linear), MSE loss; compare test MSE/MAE to a plain linear regression; standardize first.

## How it might appear on the exam

- **Identify architecture from formula** (Exercise11.2a/b style; 2025 exam Q3b style): given the nested expression, state $p, M, C$ and the activations, then count parameters. Bias term is the trap.
- **Interpret an architecture diagram** (Exercise11.1b style): how many inputs, layers, outputs; regression or classification; what activation makes sense.
- **Compare to a classical method** (Exercise11.1c, 11.2c): "in what way is FNN-with-sigmoid-output different from logistic regression?" Answer hinges on whether the hidden activation is linear (collapse) or non-linear (genuine expressive gain).
- **Forward-pass calculation** (2025 exam Q3b.ii, re-asked verbatim): given weights, inputs, bias, ReLU activation, compute the output of one neuron. Just multiply, sum, add bias, apply ReLU, calculator question.
- **"How can 10000 parameters fit 1000 observations?"** (Exercise11.1d, Q3b candidate): regularization story; lead with [[nn-regularization]] menu and [[double-descent]].

## Related

- [[activation-functions]]: what makes the network non-linear; the nonlinearity choice is one of the prof's "design knobs"
- [[nn-parameter-count]]: exam-flagged $(p+1)M + (M+1)C$ calculation; bias trap
- [[backpropagation]]: how the FNN is trained; works only because feedforward has no loops
- [[gradient-descent-and-sgd]]: the optimizer; mini-batch SGD also acts as implicit regularization
- [[nn-loss-functions]]: MSE for regression, cross-entropy for classification, paired with output activation
- [[nn-regularization]]: "never train without regularization"; menu of L1/L2, dropout, early stopping, augmentation, label smoothing, transfer learning
- [[universal-approximation]]: FNNs with one hidden layer (and enough width) can approximate any Borel function, existence result, not recipe
- [[convolutional-neural-network]]: feedforward variant with shared local filters; backprop drops in unchanged
- [[recurrent-neural-network]]: non-feedforward extension to sequential data
- [[logistic-regression]]: what an FNN with linear-hidden + sigmoid-output collapses to
- [[generalized-additive-models]]: additive non-linear contrast
- [[principal-component-regression]]: what an all-linear FNN collapses to per the slides
- [[standardization]]: mandatory preprocessing for NNs
- [[bias-variance-tradeoff]]: the running theme; FNNs sit in the high-variance, regularization-heavy regime
- [[double-descent]]: explains why over-parameterized FNNs still generalize
