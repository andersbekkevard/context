---
concept: gradient-descent-and-sgd
module: 11-nnet
lectures: [L23, L24, L26]
isl-ref: 10.7
exercises: []
related: [feedforward-network, backpropagation, nn-regularization, nn-loss-functions, ridge-regression, double-descent, regularization]
tags:
  - concept
  - module/11-nnet
aliases:
  - gradient descent
  - SGD
  - stochastic gradient descent
  - mini-batch SGD
  - mini-batch gradient descent
---

# Gradient descent, SGD, and mini-batch

The prof's framing: same idea as for linear regression, minimize a loss by stepping in the negative-gradient direction. The NN twist is **mini-batch SGD**, which both speeds training and provides **implicit L2 regularization** (it picks the minimum-norm interpolator). The prof flags the implicit regularization as one of his headline NN facts.

## Definition (prof's framing)

> "Very much the same idea as we did for linear regression. We had a function, you know, the sum of square difference, and then to find the parameters, we just took the derivative and then we move in the direction of the derivative. Only now we have like a whole bunch of layers so it takes a bit more work but it's still the same idea." - [[L23-nnet-1]]

Three flavors of gradient-descent training for NNs:

1. **Full (batch) gradient descent**, compute the gradient over *all* $N$ samples per step. Best gradient estimate, slowest per step.
2. **True SGD**, compute the gradient on *one* sample per step. Noisiest, rarely used in practice.
3. **Mini-batch SGD**, compute the gradient on a random subset of $m \ll N$ samples per step. Standard.

## Notation & setup

- $\boldsymbol\theta$, full parameter vector (all $\alpha$s and $\beta$s in an FNN)
- $L(\boldsymbol\theta)$ or $R(\boldsymbol\theta)$, loss / cost function (e.g., MSE, cross-entropy)
- $\lambda$, learning rate (a hyperparameter, the prof's notation in [[L24-nnet-2]])
- $\nabla_{\boldsymbol\theta} L$, gradient of the loss w.r.t. parameters
- $m$, mini-batch size; $N$, full training-set size
- Per-sample loss $L_i(\boldsymbol\theta)$ so $L = \frac{1}{N} \sum_i L_i$

## Formula(s) to know cold

### Update rule (the canonical recipe)

$$\boxed{\boldsymbol\theta^{(t+1)} = \boldsymbol\theta^{(t)} - \lambda\, \nabla_{\boldsymbol\theta} L(\boldsymbol\theta^{(t)})}$$

### Full-batch gradient

$$\nabla_{\boldsymbol\theta} L = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\boldsymbol\theta} L_i$$

### Mini-batch estimator

For a random mini-batch $\mathcal B \subset \{1, \dots, N\}$ with $|\mathcal B| = m$:

$$\widehat{\nabla L} = \frac{1}{m} \sum_{i \in \mathcal B} \nabla_{\boldsymbol\theta} L_i$$

This is an **unbiased estimator** of the full gradient, same expectation, larger variance.

### Algorithm (slide deck, paraphrased, the prof's recipe in L23)

1. Initialize $\boldsymbol\theta^{(0)}$.
2. Repeat until convergence (or for $T$ epochs):
   a. **Forward pass:** compute predictions and the loss.
   b. **Backward pass** ([[backpropagation]]): compute $\nabla_{\boldsymbol\theta} L$.
   c. **Update:** $\boldsymbol\theta^{(t+1)} = \boldsymbol\theta^{(t)} - \lambda\, \widehat{\nabla L}$.
3. Return the trained $\boldsymbol\theta$.

For mini-batch SGD, step (a) and (b) use only a random batch.

## Insights & mental models

### The shifting loss landscape (NN-specific)

Linear regression has a *fixed* loss landscape determined by the data; NNs do not.

> "Once you stack hidden layers and many hidden nodes, suddenly you have a very flexible model. And as you learn some of the states, as you learn some of the parameters, and then estimate some of the states, you're actually changing the landscape. So it's actually kind of interesting because in machine learning, often you can kind of dig into one [local minimum] and then end up here. It's very funny because the loss landscape will change as you move through it. You still have this notion that there's local minima, global minima, but you have more ways of escaping local minima." - [[L23-nnet-1]]

Practical consequence: NNs almost always converge to *some* local minimum, but the geometry of those minima is forgiving (modern empirical observation, the prof: "the local minima are somehow connected" - [[L24-nnet-2]]). Doesn't prove anything; explains why training works in practice despite non-convexity.

### Mini-batch SGD: two payoffs

> "The optimizer will converge much faster if it can rapidly compute approximate estimates of the gradient. … Mini-batches may be processed in parallel, and the batch size is often a power of 2 (32 or 256). Small batches also bring in a regularization effect, due to the variability they bring to the optimization process.", slides + [[L23-nnet-1]]

1. **Speed / parallelism.** Distribute mini-batches across GPUs; "with $K$ GPUs and $N=100$, hand 10 samples to each, sum results."
2. **Regularization via noise.** *Headline fact*, see below.

### Implicit L2 regularization (memorize)

> [!important] The prof's headline NN fact
> "Mini-batch stochastic gradient descent actually gives you an implicit L2 regularization, which is super weird. … Whenever you use this and you're in a problem setting where there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized." - [[L23-nnet-1]]

> "It creates a regularization effect, meaning that you still fit the data very well but you actually end up generalizing better to data that you haven't seen, which is really not obvious." - [[L24-nnet-2]]

> "It has been proven, which is nice. The math is there, but it's not super short." - [[L24-nnet-2]]

This is the punchline that connects gradient descent to the rest of the regularization picture:
- In the under-parameterized regime, mini-batch SGD just speeds things up.
- In the over-parameterized regime ($p \gg n$, NNs are here), there are *infinitely many* zero-loss interpolators. Mini-batch SGD picks the one with minimum L2 norm, a soft equivalent to ridge regression's shrinkage.

This connects directly to [[double-descent]]: the second descent past the interpolation point happens because SGD picks the **min-norm interpolator**, which generalizes well.

### Powers-of-2 batch sizes

> "Batch sizes are powers of two (32, 256, 512, 1024) because that's what you always do in machine learning because you try to make everything as hardware efficient as possible. And most hardware things happen in powers of two." - [[L23-nnet-1]]

Pure hardware convention; nothing statistical.

### Learning rate matters

> "Tuning the learning rate matters in practice: 'the network I was running over the weekend, I realized that part of my model had a learning rate that was too high. … once I shrunk it down, it performed so much better.' Too large → bouncing; small enough (e.g. 0.1 or smaller) → fine." - [[L24-nnet-2]]

In scope: $\lambda$ is a hyperparameter you have to tune. **Out of scope:** Adam, momentum, RMSProp internals, the prof: "Adam optimizer internals, out of scope" per [[scope]].

## Exam signals

> "Mini-batch stochastic gradient descent actually gives you an implicit L2 regularization. … Whenever you use this and you're in a problem setting where there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized." - [[L23-nnet-1]]

> "I can't think of an example where you'd ever want to train a neural network without a regularization. … The whole point of regularization is to get you to generalize better." - [[L26-nnet-3]]

> "We've changed the optimization that we're doing … the model is so flexible it has an infinite number of ways of fitting it and therefore it doesn't have to pick a model that fits the training data well, all of them do, it's finding the one that has the best variance." - [[L26-nnet-3]] (double-descent connection)

The optimizer story shows up as backdrop in the [[double-descent]] discussion (the mathy "minimum-norm interpolator" framing) and in the regularization menu in [[nn-regularization]]. Direct exam questions on optimizer internals are unlikely; conceptual questions about *why* mini-batch SGD helps are fair game.

## Pitfalls

- **"SGD" alone usually means *mini-batch* SGD** in modern usage. True 1-sample-per-step SGD is rare. The slides note: "True SGD: only one sample (mini-batch size 1). Mini-batch SGD is a compromise."
- **Mini-batch noise is the *good* part.** Counterintuitive: noisier gradient estimates *help* generalization. The prof: "What's not obvious from this is that actually this is a good idea to have noisy estimates of your gradients." - [[L23-nnet-1]]
- **Implicit L2 regularization is real and proved**, not folklore. Don't dismiss as hand-waving, the prof explicitly notes "it has been proven."
- **Learning rate too high → bouncing; too low → glacial.** No closed-form way to set it; pick via experimentation or learning-rate scheduling (out of scope).
- **Don't confuse epochs with iterations.** One *epoch* = one pass through all $N$ samples. With mini-batch size $m$, that's $N / m$ iterations per epoch.
- **Gradient computed via [[backpropagation]]**, not by manual differentiation per parameter. SGD describes the *update*; backprop describes how the gradient is *computed*.

## Scope vs ISLR

- **In scope:** the update rule $\boldsymbol\theta \leftarrow \boldsymbol\theta - \lambda \nabla L$, the three flavors (full / true SGD / mini-batch), why mini-batch is the standard, the speed and regularization payoffs, **implicit L2 regularization (headline fact)**, learning rate as a hyperparameter, the shifting-loss-landscape intuition.
- **Look up in ISLR:** §10.7, fitting a neural network. §10.7.2 covers regularization + SGD.
- **Skip in ISLR (book-only, prof excluded):**
  - **Adam, RMSProp, momentum** internals - [[L23-nnet-1]] / [[L27-summary]]: "Advanced optimizers, out of scope."
  - **Learning-rate schedulers, warmup, cosine annealing**: not lectured.
  - **Vanishing / exploding gradients, weight initialization (Xavier / He)** - [[L24-nnet-2]]: "not discussed in any depth." Per [[scope]] explicitly out.
  - **Convergence proofs**: not lectured; "the math is there, but it's not super short" ([[L24-nnet-2]] re implicit L2).
  - **Batch normalization**: [[scope]] NN exclusions: explicitly out.

## Exercise instances

None, no Exercise11 problem asks you to derive or compute SGD updates by hand. The Boston Keras exercise (Exercise11.3) and CIFAR-10 CNN (Exercise11.4) use Adam optimizer with `learning_rate = 0.001` and mini-batch sizes of 32 / 64 (powers of 2), implicit demonstration.

## How it might appear on the exam

- **Conceptual / multiple choice**: "Which of the following are true about mini-batch SGD?", answers along the lines of (a) faster than full GD, (b) provides implicit regularization, (c) batch size is typically a power of 2 for hardware reasons, (d) gives an unbiased gradient estimate.
- **Pseudocode question**: "Write out the gradient-descent update rule for fitting a feedforward network." Answer: $\boldsymbol\theta^{(t+1)} = \boldsymbol\theta^{(t)} - \lambda \widehat{\nabla L}$ with $\widehat{\nabla L}$ computed via backprop on a mini-batch.
- **Tie-in to double descent / regularization**: "Why does an over-parameterized NN trained with SGD generalize?", implicit L2 regularization picks the min-norm interpolator; see [[double-descent]] and [[nn-regularization]].
- **True/false trap**: "True SGD (one sample per step) is the standard NN training algorithm." → False; mini-batch is standard.
- **Combined with backprop**: "Describe the training loop of a feedforward NN.", initialize, forward pass, compute loss, backward pass (backprop), update (SGD step), repeat.

## Related

- [[backpropagation]]: *how* the gradient is computed, complementary to *how* the update is applied
- [[feedforward-network]]: what's being trained
- [[nn-loss-functions]]: what is being minimized
- [[nn-regularization]]: mini-batch SGD lives on the regularization menu, not just the optimization menu
- [[ridge-regression]]: explicit L2 regularization that mini-batch SGD implicitly mimics
- [[double-descent]]: the second-descent regime is where the implicit min-norm bias of SGD pays off
- [[regularization]]: Specials atom; SGD is the implicit-regularization example in the menu
