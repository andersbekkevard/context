---
concept: nn-loss-functions
module: 11-nnet
lectures: [L23]
isl-ref: 10.1
exercises:
  - Exercise11.3  -  choose the loss function for the Boston Keras model (mean_squared_error for regression)
related: [feedforward-network, activation-functions, gradient-descent-and-sgd, backpropagation, logistic-regression, least-squares-and-mle]
tags:
  - concept
  - module/11-nnet
aliases:
  - cross-entropy
  - binary cross-entropy
  - categorical cross-entropy
  - MSE loss
---

# NN loss functions

The prof's framing: "just new names for old ideas." The loss for a NN is the same shape as a GLM loss, applied to the network's output. The loss is **paired with the output activation**: linear ↔ MSE (regression), sigmoid ↔ binary cross-entropy (binary), softmax ↔ categorical cross-entropy (multi-class).

## Definition (prof's framing)

> "These are just new names for old ideas. So this is like the same as when we looked at logistic regression. … This one is like linear regression, and then this other one is like logistic regression. Only now your inputs are not the covariates, but rather they're hidden shit. So it can do much more than logistic regression because it has the full flexibility of the network." - [[L23-nnet-1]]

The loss function $L(\boldsymbol\theta)$ measures how badly the network's predictions $\hat y$ match the targets $y$. Training picks $\boldsymbol\theta$ to minimize $L$ via [[gradient-descent-and-sgd|gradient descent]] / mini-batch SGD, with the gradient computed by [[backpropagation]].

## Notation & setup

- $\boldsymbol\theta$: all NN parameters (weights and biases)
- $\hat y_i = \hat f(\mathbf x_i; \boldsymbol\theta)$: predicted output for sample $i$ after the output activation
- $y_i$: target value (real-valued for regression, $\{0, 1\}$ for binary, one-hot $\mathbf y_i = (0, \dots, 1, \dots, 0)$ for multi-class with $C$ categories)
- $f_m(\mathbf x_i)$: predicted probability of class $m$ for multi-class output (softmax)

## Formula(s) to know cold

The slide deck table (this *is* the exam-relevant pairing):

| Problem | Output nodes | Output activation | Loss function |
|---|---|---|---|
| Regression | 1 | linear | MSE |
| Binary classification | 1 | sigmoid | binary cross-entropy |
| Multi-class ($C \ge 2$) | $C$ | softmax | categorical cross-entropy |

### Mean Squared Error (MSE), regression

$$L(\boldsymbol\theta) = \sum_{i=1}^n (y_i - \hat f(\mathbf x_i))^2$$

Used with linear output activation. Same loss as ordinary least squares, see [[least-squares-and-mle]].

### Binary cross-entropy, binary classification

$$L(\boldsymbol\theta) = -\sum_{i=1}^n \big[ y_i \log f(\mathbf x_i) + (1 - y_i) \log (1 - f(\mathbf x_i)) \big]$$

Used with sigmoid output. Same loss as the **negative log-likelihood** of a Bernoulli GLM ([[logistic-regression]]).

### Categorical cross-entropy, multi-class

$$L(\boldsymbol\theta) = -\sum_{i=1}^n \sum_{m=1}^C y_{im} \log f_m(\mathbf x_i)$$

with $\mathbf y_i$ one-hot encoded ($y_{im} = 1$ if sample $i$ is class $m$, else 0). Used with softmax output. Same loss as a multinomial GLM.

## Insights & mental models

### Same as the GLM losses, applied to the network output

The whole point: replace the linear predictor $\eta_i = \boldsymbol\beta^\top \mathbf x_i$ in a GLM with the *network's* nonlinear output, then apply the *same* GLM loss. The network's hidden machinery learns nonlinear features; the loss function is unchanged.

This is why the loss-activation pairing in the table is rigid: each pairing matches a known GLM (Gaussian / Bernoulli / multinomial) and inherits its likelihood structure. Mismatching them (e.g. softmax output with MSE loss) gives nonsensical or pathological gradients.

### Cross-entropy is negative log-likelihood

For the Bernoulli case: maximizing the log-likelihood $\sum_i [y_i \log p_i + (1 - y_i) \log(1 - p_i)]$ is equivalent to minimizing the negative, which is exactly binary cross-entropy. Same Legendre/Gauss MLE-as-LS trick the prof flagged for the mathy exam question (see [[least-squares-and-mle]]), generalized: under the relevant exponential-family assumption, MLE equals minimizing the corresponding cross-entropy / squared-error.

### "It's also inferring hidden states"

> "The very confusing thing with neural networks is that you don't just learn, you learn parameters, right? You have to learn the parameters of the network, but you're also inferring states, hidden states, and a lot of hidden states." - [[L23-nnet-1]]

The loss is *only* on the network's final output, but minimizing it implicitly fits both the parameters $\boldsymbol\theta$ and the hidden activations $z_m$ that the network discovers as useful features. This is what makes NNs strictly more expressive than the corresponding GLM.

### Loss landscape changes during training (NN-specific)

> "As you learn some of the parameters, and then estimate some of the states, you're actually changing the landscape." - [[L23-nnet-1]]

Unlike linear regression (loss is a quadratic bowl), NN losses are highly non-convex. The geometry of the local minima makes training feasible despite this, a topic for [[gradient-descent-and-sgd]].

## Exam signals

> "These are just new names for old ideas. … Only now your inputs are not the covariates, but rather they're hidden shit. So it can do much more than logistic regression because it has the full flexibility of the network." - [[L23-nnet-1]]

> The slide-deck pairing table (regression → MSE, binary → BCE, multi-class → CCE) was emphasized in [[L23-nnet-1]] and is the kind of multiple-choice candidate the prof flags as "fair game."

> "Output activation is paired with the loss.", implicit throughout the lecture; the table is the canonical "small fact" worth memorizing.

The L27 walkthrough does not feature a direct loss-function question. But the loss pairing is a likely **fill-in / match the loss to the output activation** multiple-choice, and any pseudocode question on training would expect you to invoke the right loss for the task.

## Pitfalls

- **Mismatched activation + loss.** Sigmoid + MSE works numerically but has poorer gradient behavior than sigmoid + BCE. Softmax + MSE is genuinely wrong for classification (no probabilistic interpretation). Always pair: linear/MSE, sigmoid/BCE, softmax/CCE.
- **Multi-class with softmax requires one-hot $\mathbf y$.** A single integer-valued target won't work, the loss needs $y_{im}$ to be 0/1 indicators across the $C$ output dimensions.
- **Don't double-apply softmax.** If your loss expects logits (pre-activation $z$), don't softmax twice. Frameworks have both "from logits" and "from probabilities" loss variants. Out of scope for this exam, but a real engineering trap.
- **MSE vs. RMSE vs. SSE.** MSE = mean (divide by $n$), SSE = sum (don't), RMSE = $\sqrt{MSE}$. Slides use MSE without the $1/n$ in some places; doesn't change the optimizer, only the absolute number. Also, MAE (mean absolute error, Boston exercise) is *not* the same as MSE.
- **Cross-entropy ≠ entropy.** Cross-entropy measures the divergence between two distributions (predicted vs. true). The "entropy" in the name is information-theoretic; you don't need to derive it from first principles for the exam.
- **The loss is on the output**, not on intermediate activations. Hidden states have no targets, they're learned implicitly to minimize the output loss.

## Scope vs ISLP

- **In scope:** the three losses (MSE, BCE, CCE), the pairing table with output activations, the conceptual identification with GLM losses (logistic / linear / multinomial), MLE-as-loss-minimization connection.
- **Look up in ISLP:** §10.1 (regression NN with MSE), §10.2 (multi-class with softmax + CCE), §10.7 (training).
- **Skip in ISLP (book-only / not lectured):** focal loss, ranking losses, contrastive / triplet losses, KL-divergence formalism. Boosting losses (quadratic / absolute / Huber / deviance), those live in [[boosting-loss-functions]] (Module 9), separate atom.

## Exercise instances

- **Exercise11.3** (Boston Housing), choose the loss for `compile()`: regression with continuous target → `mean_squared_error`. Among the three options listed (`binary_crossentropy`, `categorical_crossentropy`, `mean_squared_error`), MSE is the only sensible choice.
- **Exercise11.4** (CIFAR-10 CNN), multi-class image classification, 10 categories, softmax output → `categorical_crossentropy`.
- **Exercise11.5** (Wafer 1D-CNN), binary time-series classification with 2-class softmax (one-hot encoded) → `categorical_crossentropy` per the exercise (could equivalently use binary cross-entropy with sigmoid).

## How it might appear on the exam

- **Multiple-choice loss/activation pairing:** "For multi-class classification with $C = 10$, which loss + output activation combination is appropriate?", answer: softmax + categorical cross-entropy.
- **True/false:** "MSE is an appropriate loss for binary classification with sigmoid output.", False (it works numerically but BCE is the principled / standard choice).
- **Conceptual:** "Why is the cross-entropy loss for binary classification equivalent to the negative log-likelihood of a Bernoulli model?", write out the likelihood, take logs, negate; same calculation as in [[logistic-regression]] / [[least-squares-and-mle]] mathy template.
- **Pseudocode on training**: any "describe the training loop" question expects you to specify a loss function appropriate to the task.
- **As a distractor in 2025-Q3f-style questions** about NN architecture choices.

## Related

- [[feedforward-network]]: the model whose output goes into the loss
- [[activation-functions]]: output activation determines the loss pairing
- [[gradient-descent-and-sgd]]: minimizes the loss
- [[backpropagation]]: computes $\nabla L$ for the optimizer; the output-layer $\delta^{\text{out}}$ is the derivative of the loss w.r.t. $\hat y$
- [[logistic-regression]]: binary cross-entropy is its loss
- [[least-squares-and-mle]]: MSE / negative log-likelihood equivalence under Gaussian errors; the prof's mathy-question template
