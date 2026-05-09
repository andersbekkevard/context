---
concept: backpropagation
module: 11-nnet
lectures: [L23, L24]
isl-ref: 10.7.1
exercises: []
related: [feedforward-network, gradient-descent-and-sgd, activation-functions, nn-loss-functions, recurrent-neural-network]
tags:
  - concept
  - module/11-nnet
aliases:
  - backprop
  - back-propagation
  - chain rule for NNs
---

# Backpropagation

The prof's framing: "just the chain rule, applied so you don't recompute things." Forward pass stores intermediates; backward pass reuses them and propagates a $\delta$ from output back to input one layer at a time. **Why feedforward matters**: backprop relies on the acyclic structure, loops break it (which is why RNNs are harder).

## Definition (prof's framing)

> "Backprop is just the chain rule, applied so you don't recompute things. Forward pass stores intermediates; backward pass reuses them and propagates a $\delta$ from output to input. Loops break this , feed-forward is loop-free, so backprop just works." - [[L24-nnet-2]] key takeaway

> "Backpropagation is a simple and inexpensive way to calculate the gradient. … The chain rule is used to compute derivatives of functions of other functions where the derivatives are known. This is efficiently done with backpropagation." - slides, [[L24-nnet-2]]

> "It's really not brain surgery." - [[L24-nnet-2]]

Backpropagation is the algorithm that computes $\nabla_{\boldsymbol\theta} L$ for a feedforward network by:
1. **Forward pass:** push input through the model, computing and **storing** every intermediate (pre-activations $v$, activations $z$, output $f(\mathbf x)$).
2. **Backward pass:** apply the chain rule from output to input, reusing the stored intermediates so each derivative costs only a constant amount of work.

It's the *gradient calculator* that feeds [[gradient-descent-and-sgd|gradient descent / SGD]].

## Notation & setup

Single-hidden-layer FNN with linear output (the prof's worked example, [[L24-nnet-2]]):

$$f(\mathbf x_i) = \beta_0 + \sum_{k=1}^K \beta_k z_{ik}, \qquad z_{ik} = g\!\left(\alpha_{k0} + \sum_{j=1}^p \alpha_{kj} x_{ij}\right)$$

with squared-error loss

$$R_i(\boldsymbol\theta) = \tfrac{1}{2} (y_i - f(\mathbf x_i))^2$$

Define the **pre-activation** of hidden unit $k$:

$$v_{ik} = \alpha_{k0} + \sum_{j=1}^p \alpha_{kj} x_{ij}, \qquad z_{ik} = g(v_{ik})$$

## Formula(s) to know cold

### Forward pass , compute and store

For each sample $i$:
- $v_{ik} = \alpha_{k0} + \sum_j \alpha_{kj} x_{ij}$ (pre-activation of hidden unit $k$)
- $z_{ik} = g(v_{ik})$ (hidden activation)
- $f(\mathbf x_i) = \beta_0 + \sum_k \beta_k z_{ik}$ (output)

Store all of $v_{ik}, z_{ik}, f(\mathbf x_i)$, they get reused.

### Backward pass , propagate $\delta$ from output to input

**Output-layer error** (the chain-rule "seed"):

$$\delta_i^{\text{out}} = \frac{\partial R_i}{\partial f(\mathbf x_i)} = -(y_i - f(\mathbf x_i))$$

**Output-layer weight gradients** (use $\delta_i^{\text{out}}$ + stored $z_{ik}$):

$$\frac{\partial R_i}{\partial \beta_k} = \delta_i^{\text{out}} \cdot z_{ik}, \qquad \frac{\partial R_i}{\partial \beta_0} = \delta_i^{\text{out}}$$

**Hidden-layer error** (chain rule one step further back):

$$\delta_{ik}^{\text{hid}} = \frac{\partial R_i}{\partial v_{ik}} = \delta_i^{\text{out}} \cdot \beta_k \cdot g'(v_{ik})$$

**Hidden-layer weight gradients:**

$$\frac{\partial R_i}{\partial \alpha_{kj}} = \delta_{ik}^{\text{hid}} \cdot x_{ij}, \qquad \frac{\partial R_i}{\partial \alpha_{k0}} = \delta_{ik}^{\text{hid}}$$

### General pattern

At any layer $L$, $\delta^{(L)}$ is a sum involving:
- $\delta^{(L+1)}$ (the next-layer error, already computed in this backward sweep)
- the next-layer weights
- $g'(\cdot)$ of the pre-activation at this layer

You only ever look one layer ahead, the recursion just propagates back, layer by layer.

## Insights & mental models

### Why this is fast (the whole point)

> "This thing we already computed. This thing is a number. This thing we already computed in the forward pass. So we don't really need to do anything. We don't need to do a whole bunch of sums and calculations. It becomes very efficient." - [[L24-nnet-2]]

Without backprop, computing the gradient for each parameter would require its own forward pass, total cost $O(\#\text{params} \cdot N)$ per epoch. With backprop, **one forward + one backward pass** computes the gradient for every parameter, total cost $O(\#\text{params})$ per sample. This is the algorithmic breakthrough that took NNs out of the second AI winter.

### Why feedforward is essential

> "If you had loops you're kind of screwed. … with loops, then you can't always just go backwards. … It is convenient that a feedforward network doesn't have those loops and you can just go backwards." - [[L24-nnet-2]]

The backward sweep depends on having a topological order. Loops mean a node's gradient depends on its own future gradient, circular. This is why [[recurrent-neural-network|RNNs]] need specialized variants (BPTT, out of scope) and why feedforward is the workhorse.

### Backprop = chain rule (no magic)

> "Backpropagation is just the chain rule, applied so you don't recompute things." - [[L24-nnet-2]]

There's nothing fundamentally new mathematically, it's classical multivariate calculus. The "discovery" was the realization that you can organize the computation so intermediate results from the forward pass get reused. Hinton et al. popularized it; the prof: "a very, very simple idea. But without it, the computers at the time were just rubbish." - [[L24-nnet-2]]

### What gets stored in the forward pass

Crucially: **everything you'll need on the way back**. Pre-activations $v$, activations $z$, output. Storing these is a memory cost (one of the practical engineering challenges of training big models, but out of scope).

### Why this enabled the second NN era

> "Backprop in particular: 'a very, very simple idea. But without it, the computers at the time were just rubbish.'" - [[L24-nnet-2]]

> "Backpropagation … another big deal because it created an objective function that you could use easily. And it was one of the reasons we got out of one of the winters." - [[L23-nnet-1]]

Hinton in the 80s. Combined with multi-layer architectures (which solved the XOR problem of single-layer perceptrons), this is the conceptual ingredient that revived NNs after Minsky's critique.

## Exam signals

> "Backprop is just the chain rule, applied so you don't recompute things." - [[L24-nnet-2]]

> "If you had loops you're kind of screwed." - [[L24-nnet-2]] (foreshadows RNN difficulty)

The prof did not flag a specific backprop derivation as a likely exam question. No exercise asks you to derive backprop. But the *concept*, what it is, why feedforward matters, how it relates to the chain rule, is fair game as a multiple-choice or short-answer question.

## Pitfalls

- **"Backprop is the optimizer."** No, backprop *computes the gradient*; gradient descent / SGD *uses* the gradient to update parameters. They're orthogonal: backprop is a gradient calculator, SGD is an update rule. See [[gradient-descent-and-sgd]].
- **Loops kill backprop.** Don't claim "backprop works on any network", RNNs need specialized BPTT (out of scope), and arbitrary cyclic networks are intractable.
- **The forward pass must store intermediates.** A common mistake in describing backprop: omitting the storage step. Without stored $v, z$, the backward pass would have to recompute them, defeating the efficiency win.
- **$g'(v)$ shows up in the hidden-layer $\delta$.** Don't drop it. For ReLU, $g'(v) = \mathbb 1[v > 0]$, exactly 0 or 1, which is what makes ReLU's gradient computation cheap.
- **Sign convention.** $\delta^{\text{out}} = -(y - f)$ for squared-error loss; the negative comes from $\frac{d}{df}\tfrac12(y-f)^2 = -(y-f)$. Watch signs on the exam.
- **One sample at a time vs. mini-batch.** The prof's worked example is per-sample; in practice you sum / average across the mini-batch before the SGD update.

## Scope vs ISLP

- **In scope:** the chain-rule story, forward-pass-stores-intermediates / backward-pass-reuses-them, the $\delta$ recursion, the formulas above for the single-hidden-layer worked example, why feedforward matters (no loops).
- **Look up in ISLP:** §10.7.1 , *Backpropagation*. Per slides: "ISLP Chapter 10.7 (this is part of the compulsory course material , study yourself)." Goodfellow §6.5 for the matrix form.
- **Skip in ISLP:** the matrix / vectorized form of backprop (notation-heavy, same content). Backpropagation through time (BPTT) for RNNs, explicitly out of scope per [[L26-nnet-3]] / [[scope]]. Vanishing/exploding-gradient analysis, out per [[L24-nnet-2]].

## Exercise instances

None, no exercise problem asks you to apply backprop by hand. It's conceptual scaffolding for understanding *how* training works, not a calculation drill.

## How it might appear on the exam

- **Multiple-choice / true-false**: "Backpropagation is the chain rule applied so intermediates from the forward pass can be reused." → True. Or: "Backpropagation is the gradient-descent update rule." → False.
- **Short answer**: "What is backpropagation, and why does it require a feedforward (not recurrent) architecture?" , chain rule + topological order; loops break the backward sweep.
- **Conceptual link**: "Why was backpropagation a key ingredient in the NN revival?" , efficient gradient computation made multi-layer networks trainable; without it, NNs were stuck at the single-layer perceptron limitation Minsky exposed.
- **Trap question**: confusion between backprop and SGD. Backprop = gradient *calculator*; SGD = parameter *updater*.
- **Pseudocode**: "Sketch the training loop of a feedforward NN." Answer includes: initialize, forward pass (store intermediates), compute loss, backward pass (compute $\delta$s, then gradients), SGD step, repeat.

## Related

- [[feedforward-network]]: the architecture backprop is designed for; the no-loops property is essential
- [[gradient-descent-and-sgd]]: the optimizer that consumes backprop's output
- [[activation-functions]]: $g'(v)$ enters the hidden-layer $\delta$; ReLU's piecewise-linear $g'$ is especially clean
- [[nn-loss-functions]]: the loss whose gradient we're computing; choice of loss seeds the output-layer $\delta^{\text{out}}$
- [[recurrent-neural-network]]: the cautionary contrast: loops make backprop hard (BPTT, out of scope)
