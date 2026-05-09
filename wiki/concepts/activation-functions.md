---
concept: activation-functions
module: 11-nnet
lectures: [L22, L23]
isl-ref: 10.1
exercises:
  - Exercise11.2a - given $\hat y = \beta_0 + \sum \beta_m \max(\alpha_{0m} + \sum \alpha_{jm} x_j, 0)$, identify the ReLU hidden activation
  - Exercise11.2b - same drill: identify nested ReLU + sigmoid output activations from the formula
  - Exercise11.3 - Boston Keras example uses ReLU hidden + linear output for regression
related: [feedforward-network, nn-loss-functions, logistic-regression, universal-approximation, convolutional-neural-network]
tags:
  - concept
  - module/11-nnet
aliases:
  - activation function
  - nonlinearity
  - ReLU
  - sigmoid
  - softmax
  - GELU
---

# Activation functions (sigmoid, ReLU, GELU, softmax)

The prof's framing: the **nonlinearity is what makes a NN expressive**. Without it, the whole nested machinery collapses to linear regression. The choice of activation also tracks the history of the field , sigmoid (brain-inspired, statistical mechanics), then ReLU/GELU (modern, gradient-friendly), with softmax as the multi-class output trick.

## Definition (prof's framing)

> "It could just be a sum, and then the value of this thing would be a sum, but then that would be very boring. And often people kind of attribute the importance of having some kind of nonlinearity." - [[L22-unsupervised-2]]

> "If it was linear, then the whole thing would just decay to boring." - [[L23-nnet-1]]

> "ReLU slash GELU are the popular ones nowadays. … One of the important things with ReLU was that it was non-linear and had an more expressive , wasn't just binary, it can be kind of continuously valued." - [[L23-nnet-1]]

The activation is a (typically) non-linear function $g(\cdot)$ applied **inside each neuron** to the pre-activation $\alpha_0 + \sum_j \alpha_j x_j$. Different choices for hidden vs. output layers; the output choice is determined by the task.

## Notation & setup

- $g(\cdot)$: hidden-layer activation
- $f(\cdot)$: output-layer activation
- They can be the same or different functions

For a hidden unit:
$$z_m = g\!\left(\alpha_{0m} + \sum_j \alpha_{jm} x_j\right)$$

For an output:
$$\hat y_c = f\!\left(\beta_{0c} + \sum_m \beta_{mc} z_m\right)$$

## Formula(s) to know cold

### Hidden-layer choices

**Linear (identity).** $g(z) = z$. Special case: the network collapses to linear regression (or to PCR / PLS in the slides' framing). "Just nothing." - [[L23-nnet-1]]

**Sigmoid (logistic).**
$$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Maps $\mathbb R \to (0, 1)$. The historical hidden-layer default.

**ReLU (rectified linear unit).**
$$g(z) = \max(0, z)$$
Linear above zero, flat at zero below. Piecewise linear, **non-linear overall**.

**GELU (Gaussian error linear unit).**
Smoothed ReLU using the Gaussian error function, often approximated. The corner of ReLU at zero is replaced by a smooth dip-then-rise. Mathematically written with $\Phi$ (Gaussian CDF); implemented as a numerical approximation. The prof: "I won't ask you about that specifically." - [[L23-nnet-1]]

### Output-layer choices

The slide deck table:

| Problem | Output activation | Loss |
|---|---|---|
| Regression | linear | MSE |
| Binary classification | sigmoid | binary cross-entropy |
| Multi-class | softmax | categorical cross-entropy |

**Linear output:** $f(z) = z$. For real-valued $y$.

**Sigmoid output:** $f(z) = 1 / (1 + e^{-z})$. For binary $y$, gives $P(Y = 1 \mid \mathbf x)$.

**Softmax output:** for $C$-class classification with one-hot encoded $y$,
$$\text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{k=1}^C e^{z_k}}$$
Coupled across the $C$ output nodes; outputs sum to 1; interpretable as $P(Y = c \mid \mathbf x)$.

> "You're normalizing it so sum of them equals one and you doing the exp of the thing so it will kind of push things up or down and so if you look at it you can think of it as giving you a probability." - [[L23-nnet-1]]

## Insights & mental models

### Sigmoid was historical, ReLU/GELU rule today

The shift sigmoid → ReLU is one of those small-on-paper, huge-in-practice advances:

> "A lot of the advances were kind of seemingly trivial but not so trivial. … It started as a trick and then it's like hey that really worked well. So sadly I don't remember who did it. But someone figured out that we can do that instead of Sigmoid, and that was a good idea." - [[L23-nnet-1]]

> "I would say the sigmoid function is not so common anymore, but it was very common in its day." - [[L23-nnet-1]]

Why sigmoid lasted so long: brain inspiration (binary firing) and statistical-mechanics origins (spin systems, Hopfield networks), physicists migrated into ML and brought their on/off mental model with them.

### Why ReLU's corner is "kind of bad for gradients"

The non-differentiability at zero , and the flat zero region for $z < 0$ , can stall gradients (dead neurons). GELU smooths this away while keeping the linear-on-the-right behavior. Both are still piecewise-linear-ish, **non-linear overall** , that's all the [[universal-approximation|universal approximation theorem]] requires.

### Softmax behavior: the winner

> "[Softmax] couples the outputs together so the winner takes most of the probability , for five outputs, you get something like (0,0,0,0,1), but all coupled together, right? So that the winner is the winner and that gets the most probability." - [[L23-nnet-1]]

A common variant introduces an exponent inside the exp ("temperature", not lectured) that "makes it even more extreme … push it so that one of them is going to be a winner and push the others down." - [[L23-nnet-1]]

> "This is a clever trick. Again, machine learning is a series of very clever tricks." - [[L23-nnet-1]]

### Output activation is paired with the loss

Always: output activation choice and loss function are coupled , see [[nn-loss-functions]]. Linear ↔ MSE, sigmoid ↔ binary cross-entropy, softmax ↔ categorical cross-entropy. These are the same shape as the GLM losses (logistic regression, multinomial regression) applied to the network's final output.

## Exam signals

> "Q3f , Why are nonlinear activations necessary? Only correct answer: they let the network represent complex nonlinear functions. (A linear sum of linear things is linear.)" - [[L27-summary]]

> "Q3b … given specific weight values, inputs, and a ReLU activation, compute the output of the neuron. … just multiply, sum, add bias, apply ReLU." - [[L27-summary]]

This is the canonical exam-style ReLU forward-pass calculation. ReLU = max(0, z). 2025 exam example: weights $0.5, 0, 1$; inputs $1, -1, -1$; bias $-1$ → pre-activation $-1.5$ → ReLU(-1.5) = 0.

## Pitfalls

- **Linear hidden activation collapses the entire network to linear regression**, even with many hidden layers. You gain *parameters* but lose all *expressiveness*. Easy T/F trap (Exercise11.1c).
- **Softmax is an output-layer activation only.** The prof corrected himself mid-presentation: "oh, this is in the output layer by the way, sorry I didn't differentiate." - [[L23-nnet-1]]
- **Don't confuse softmax with binary sigmoid.** Softmax is for $C \ge 2$; for $C = 2$ a single-node sigmoid output is the standard binary setup.
- **ReLU can output exactly 0** (when pre-activation is negative); easy mistake to think it's "always positive". On a calculator question, plug the pre-activation in and check the sign.
- **Forgetting the bias** when computing pre-activation: $z = \sum w_j x_j + b$, not just the dot product. (Same trap as in [[nn-parameter-count]].)
- **Activation choice for output ≠ activation choice for hidden.** They are independent design decisions and serve different purposes.

## Scope vs ISLR

- **In scope:** linear / sigmoid / ReLU / GELU / softmax definitions; *why* nonlinearity is needed; pairing of output activation with loss; ReLU forward-pass calculation; sigmoid as historical, ReLU/GELU as modern.
- **Look up in ISLR:** §10.1 (single-layer), §10.2 (multilayer + softmax), §10.3.1 (ReLU in CNNs).
- **Skip in ISLR (book-only, prof excluded):** detailed derivative properties of activations (vanishing-gradient analysis), softmax temperature, exotic activations beyond the four named here. Vanishing/exploding gradients explicitly out of scope per [[L24-nnet-2]].

## Exercise instances

- **Exercise11.2a**: formula uses $\max(\cdot, 0)$ → identify hidden activation as ReLU and output activation as linear (none). Confirms architecture is 1-hidden-layer FNN with ReLU.
- **Exercise11.2b**: formula nests $1 / (1 + \exp(-\cdot))$ around a $\max$ around a $\max$ → identify hidden activations as ReLU at each of the two hidden layers, output as sigmoid (binary classification).
- **Exercise11.3**: Boston housing in Keras: hidden ReLU, output linear (regression), MSE loss; matches the slide-deck pairing table.

## How it might appear on the exam

- **"Why are nonlinear activations necessary?"** (verbatim 2025 Q3f reformulation): a linear sum of linear things is linear; nonlinearity is what gives the network its expressive power, links to [[universal-approximation]] (a *squashing* / non-linear hidden activation is required).
- **Forward-pass calculation through ReLU** (2025 Q3b.ii style): plug numbers in, take pre-activation, apply $\max(0, \cdot)$. Bring a calculator.
- **Match output activation to problem type**: regression → linear; binary → sigmoid; multi-class → softmax. May appear as multiple-choice (T/F on activation–loss pairings).
- **"Identify the architecture from the formula"**: spot $\max(\cdot, 0)$ → ReLU; spot $1/(1+e^{-\cdot})$ → sigmoid; spot $e^{z_c}/\sum e^{z_k}$ → softmax. Standard Exercise11.2 drill.

## Related

- [[feedforward-network]]: where activations live; the nesting structure
- [[nn-loss-functions]]: output activation pairs with the loss (linear/MSE, sigmoid/BCE, softmax/CCE)
- [[logistic-regression]]: sigmoid is the same function as the logistic; softmax generalizes it to $C$ classes
- [[universal-approximation]]: requires *squashing* (non-linear) hidden activation
- [[convolutional-neural-network]]: ReLU is the default activation in conv layers
- [[backpropagation]]: derivative $g'(z)$ shows up in the backward pass; ReLU's piecewise-linear $g'$ is exactly 0 or 1
