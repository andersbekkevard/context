---
lecture: 23
date: 2026-04-20
module: 11-nnet
title: Neural Networks 1 (Feedforward)
slides: modules/11NNet/11Nnet.md
topics:
  - feedforward-network
  - hidden-layer
  - activation-functions
  - mnist
  - universal-approximation
  - nn-loss-functions
  - gradient-descent-and-sgd
  - backpropagation
tags:
  - lecture
  - module/11-nnet
aliases:
  - Lecture 23
---

# L23 — Neural Networks 1 (Feedforward)

The prof opened Module 11 by laying out the anatomy of a [[feedforward-network]] (inputs, hidden layers with weights/biases, output), wrote out the nested equation, motivated non-linear activation functions ([[activation-functions|sigmoid]] → [[activation-functions|relu]] → GELU), introduced [[activation-functions|softmax]] for categorical outputs, stated the [[universal-approximation]] property, and ended with loss functions and optimization (gradient descent, [[gradient-descent-and-sgd|stochastic gradient descent]], [[gradient-descent-and-sgd|mini-batch]] with its implicit L2 regularization). Got through about half the deck — finishes tomorrow.

> [!note] Prof's editorial framing
> "Today we're going to talk about feed-forward networks neural networks which are really fun which should have been like the bulk of the course honestly … the PCA part was boring as was the clustering totally irrelevant … I think it'd be way more interesting to talk more about neural networks and like kind of the interesting tricks that they figured out and kind of a statistical perspective on those." Module 11 is where he wishes the course lived.

## Key takeaways

- Single-hidden-layer feedforward net = nested function $\hat y = f(\beta_0 + \sum_m \beta_m\, g(\alpha_{0m} + \sum_j \alpha_{jm} x_j))$. P, M, C are inputs / hidden units / outputs. P and C are set by the problem; **M is a hyperparameter** ("largely determined through work, through effort"). Parameter count: $(P+1)M + (M+1)C$.
- Linear activation collapses the network to linear regression — "if you make things linear then it all kind of collapses to boring." Non-linearities are what make NNs expressive.
- Output activations: linear (regression), sigmoid (binary), softmax (multi-class). Softmax = $e^{z_c} / \sum_k e^{z_k}$ — normalizes to a probability and pushes toward a "winner."
- Hidden activations: sigmoid is historical (brain on/off, spin systems). ReLU and GELU dominate now. ReLU's corner is "kind of bad for gradients"; GELU smooths it.
- Universal approximation: a feedforward net with one hidden layer of squashing units and enough width can approximate any Borel-measurable function — "but it has been proven."
- Mini-batch SGD has an **implicit L2 regularization** ("when there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized") on top of being fast. Batch sizes are powers of two for hardware reasons.

## Anatomy of a feedforward network

Components: inputs, output layer, one or more hidden layers in between. Inputs are the data. Outputs are typically also known (training targets). The arrows are weights; each layer also has bias terms. "And then you have to learn the parameters, which are represented by weights, as well as these bias terms." Then "you just train input, input, output, learn, learn, learn. It's that easy. It is actually that easy."

Feed-forward networks "really are kind of the workhorse of a lot of machine learning. It's not everything, but it's everywhere."

Notation (slides): $\boldsymbol x \in \mathbb{R}^p$ inputs, hidden nodes $z_m$ for $m=1,\dots,M$, outputs $y_c$ for $c=1,\dots,C$. Two parameter sets — $\alpha$ from input to hidden, $\beta$ from hidden to output.

$$z_m = g(\alpha_{0m} + \textstyle\sum_{j=1}^p \alpha_{jm} x_j)$$
$$\hat y_c = f(\beta_{0c} + \textstyle\sum_{m=1}^M \beta_{mc} z_m)$$

Stack them and you get one big nested equation. $g$ is the hidden-layer activation; $f$ the output activation. They can be the same or different.

### Why "feedforward"

Directed, acyclic. "If, for example, you had connections … you would have a cycle and then it would be harder. Such things exist but they are much more difficult to deal with and so it was convenient for everybody to not have any cycles and just write it as a feedforward."

> [!note] Nature has loops; we don't
> "The brain is not just feedforward. The brain has connections everywhere. They make loops. And other systems like spin systems, like magnetic systems, or liquid, right, as a liquid network. Most networks have loops. It's not very common in nature to see a network that's fully feedforward."

### Counting parameters

P is fixed by the data — "if they're images, it would be pixels, or it could be sounds or something. It could be words, right? It could be tokens for tokenized language." C is fixed by the problem — for MNIST, the 10 digit classes. M is the knob:

> [!note] M is engineering, not theory
> "M is largely determined through work, through effort, through figuring it's a hyperparameter. You have to pick it or determine it. … There's no theoretical reason necessarily for like the number M. I mean, obviously it can't be too small, but yeah, how big should it be? Who knows? … There's a lot of engineering that goes into machine learning."

Parameter count: $\alpha$ matrix is $(P+1)\times M$ (the $+1$ is the bias), $\beta$ matrix is $(M+1)\times C$. Total $(P+1)M + (M+1)C$.

### Linear-activation special case

When $g$ (and/or $f$) is the identity, "this is just linear regression. Just with a lot of parameters." The slides identify it with PCR / partial least squares. The prof treated it as a "boring" degenerate case: "There are ways to make this more interesting. You can regularize it, and then you can stack them, and you can actually get more interesting behavior and still have a lot of the properties of the linear system. But in general, this is just linear regression."

Even with $f$ non-linear and $g$ linear it's GLM-like ("more like a GLM sort of"). Either way: "if you make things linear then it all kind of collapses to boring. One of the key things in this field is that you have these non-linearities that lead to complex learning."

## Multilayer networks and the XOR moment

Going deeper was one of the things that pulled NNs out of an "AI winter" (the prof referenced last lecture's history slide). The trigger was the XOR result:

> [!note] Why depth mattered (Minsky)
> "Someone showed with — I forget his name, Mintz or something — showed that if you only have one hidden layer then you can[not] learn XOR. You can't learn this kind of simple function. … Adding layers really made it much more expressive, which, again, I don't think this was obvious to anyone. I mean, maybe it looks obvious to us because, you know, we have the we can look through the lens of like, oh, of course, because we're so used to hearing about like 10 trillion parameter networks with like 50,000 layers." (His "Mintz" = Minsky.)

Multilayer feedforward nets behave like single-layer ones — "you just repeat the same thing." Still strictly forward, no loops between any layers.

### Output types

- Continuous (real-valued) — regression.
- Binary.
- Categorical via **one-hot encoding** ("if you want to have three outcomes" → three output nodes, each a class). Combined with [[activation-functions|softmax]].

The categorical trick: "instead of having to actually model a binary variable directly, you can use … something called a softmax, which is a function. So another set of tricks that another thing that machine learning has been really good at is finding funny-looking functions that behave really well."

## Activation functions

> [!important] The prof's headline
> "ReLU slash GELU are the popular ones nowadays. … One of the important things with ReLU was that it was non-linear and had an more expressive — wasn't just binary, it can be kind of continuously valued."

### Linear

$g(x)=x$. "Just nothing." Net collapses to linear regression.

### Sigmoid

The historical default for hidden units. Two reasons people stuck with it:

1. **Brain inspiration.** Neurons fire on/off. "If the brain does it, then why not? Arguably, the brain also can have rates, but that's a different discussion."
2. **Statistical mechanics.** "In that world the neural networks actually came about as a spin system like magnetic spins and there you think about spin up and spin down so binary 0/1 or -1/+1. And so for them the very natural thing was just to think of it as binary units." Hopfield-style. Many physicists migrated into ML — "mainly because they probably couldn't get a job in statistical physics, so they went into playing with AI. I don't want to criticize them."

> [!note] Today
> "I would say the sigmoid function is not so common anymore, but it was very common in its day."

### ReLU (rectified linear unit)

Above threshold: linear. Below: zero. Heaviside-like. "If the input is above a certain number, a threshold, then it behaves linearly and before that it's just zero."

The shift from sigmoid to ReLU was one of the seemingly-trivial advances people credit for the success of NNs:

> [!note] On "trivial" advances
> "A lot of the advances were kind of seemingly trivial but not so trivial. … It started as a trick and then it's like hey that really worked well. So sadly I don't remember who did it. But someone figured out that we can do that instead of Sigmoid, and that was a good idea."

### GELU

Smooth version of ReLU. Linear growth for large input, but the corner at zero is replaced by a smooth dip-then-rise. "The corner there can be kind of bad for gradients so it's nice to have a smooth function at the same time." Mathematically written with an error function; implemented as a numerical approximation. "I won't ask you about that specifically." Probably the more common one now.

Both ReLU and GELU are piecewise linear but **non-linear overall** — that's what matters. "If it was linear, then the whole thing would just decay to boring."

### Softmax (output, classification)

This is an **output-layer** activation, not hidden — the prof corrected himself mid-presentation ("oh, this is in the output layer by the way, sorry I didn't differentiate").

$$\text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{k} e^{z_k}}$$

For a given output, take $e^{\text{input}}$ divided by the sum over all outputs in that classification group:

> [!note] What it does
> "You're normalizing it so sum of them equals one and you doing the exp of the thing so it will kind of push things up or down and so if you look at it you can think of it as giving you a probability."

A common variant introduces an exponent inside that "makes it even more extreme … push it so that it really makes it so that one of them is going to be a winner and push the other others down."

Couples the outputs together so the winner takes most of the probability — for five outputs, you get something like $(0,0,0,0,1)$, "but all coupled together, right? So that the winner is the winner and that gets the most probability."

> [!note] Prof's editorial
> "This is a clever trick. Again, machine learning is a series of very clever tricks that, yeah, they all make sense. They're not all, but they mostly make sense."

## MNIST as the workhorse dataset

Handwritten digits 0–9, 28×28 pixels. "These are the famous. I'm sure these are going to be in museums if they're not already there. … You'd be amazed how long machine learning was stuck on just trying to find these numbers — it was probably like 10 years of just trying to get these numbers better." Small accuracy bumps were publishable: "Hey we got 98% of MNIST instead of 94, look at us, we win."

> [!note] Scale shift
> "Look at the size too — imagine if the only image models you could use had 28 by 28 pixels, tiny as shit. And this was the whole field forever. Now it's all the size of the images, the size of the networks, videos, just amazing."

MNIST will be the running example for the rest of the module: image in, predicted digit out.

## Architectural choices that matter

The prof listed the design knobs ("all of these things are important"):

- **Output-layer activation function**
- **Hidden-layer activation function**
- **Architecture of the network** (width, depth, connection pattern)
- **Loss function** — what you optimize
- **Optimizer** — how you optimize ("they do this kind of cool differentiation" — backprop)
- (Also flagged but outside the scope of this list: GPU availability, enough data — "things that have also been very important.")

Architecture has three components: **width, depth, connections.** In this course the focus is feedforward, plus **CNNs** (convolutional, next lecture) and some **RNNs** (recurrent, later).

> [!note] Personal aside on his interest in NNs
> Back in industry "15 years ago" the prof wanted to study NNs and got pushback ("everyone's like, why? They're dumb, they don't do anything"). He was specifically interested in modeling fluids as a neural network rather than a continuum, "because then you can get things like turbulence and you get very complex phenomena. And everyone's like, too hard. They're right. It's hard, but, you know, useful." Useful color for his perspective: he's a physicist who came to NNs early.

## Universal approximation

Big motivation for why feedforward suffices despite nature's loops. "It turns out that a feedforward network can approximate basically any function. So it's the universal approximation property."

> [!important] Universal approximation theorem (verbatim)
> "It has been proven that if you have a linear output layer and at least one hidden layer with some kind of squashing activation function, like a ReLU or a Sigmoid or anything, like a non-linearity, and enough hidden units … then you can approximate any Borel measurable function from one finite dimensional space to another with any desired non-zero amount of error."

Three caveats:

- Original proof used "ridiculously wide" hidden layers. "You can often compensate for depth with width in some confusing way."
- "To actually understand this, you have to go into measure theory and all sorts of hard math that we don't talk about in this class."
- It's an existence result, not a recipe — says nothing about training.

Bottom line: "the surprising thing was you can do this with a feedforward network. … You don't need loops, you don't necessarily need anything else, you don't even need that many hidden layers — it needs to be big enough."

## Loss functions

Tied to the output activation. Three flavors:

- **Mean squared error** — regression: $\sum (y - \hat y)^2$. "Like a regression thing."
- **Binary cross-entropy** — binary classification.
- **Categorical cross-entropy** — multi-class, paired with softmax.

> [!note] Old wine, new bottles
> "These are just new names for old ideas. So this is like the same as when we looked at logistic regression. … This one is like linear regression, and then this other one is like logistic regression. Only now your inputs are not the covariates, but rather they're hidden shit. So it can do much more than logistic regression because it has the full flexibility of the network."

The deeper conceptual difference is that NNs aren't just fitting parameters — they're **also inferring hidden states**:

> [!important] Parameters AND states
> "The very confusing thing with neural networks is that you don't just learn — you learn parameters, right? You have to learn the parameters of the network, but you're also inferring states, hidden states, and a lot of hidden states. So this model has an ability to express and approximate functions that you just don't have at all with just simple regression, which is very interesting and cool."

## Optimization

### Gradient descent

Same idea as for linear regression — function, gradient, step in the gradient's direction, repeat. The recipe:

1. Initial values for the parameters.
2. Forward pass: compute predictions for the hidden states then the output.
3. Compute the loss.
4. Find updates to the weights via the derivative of the loss w.r.t. each parameter.
5. Update each parameter by moving in the direction of the gradient.
6. Repeat for $t$ iterations / epochs until the network stops changing.

"Very much the same idea as we did for linear regression. We had a function, you know, the sum of square difference, and then to find the parameters, we just took the derivative and then we move in the direction of the derivative. Only now we have like a whole bunch of layers so it takes a bit more work but it's still the same idea."

Gradient is computed via [[backpropagation]] — "another big deal because it created an objective function that you could use easily. And it was one of the reasons we got out of one of the winters." (Backprop covered in detail later.)

### The shifting loss landscape

A characteristic feature of NN optimization the prof flagged — and one that distinguishes them from classical regression. With linear regression on three variables, "the shape of the cost function is very determined just by the data. It's not a very flexible model." But once you stack hidden layers and many hidden nodes,

> [!important] The landscape moves as you train it
> "Suddenly you have a very flexible model. And as you learn some of the states, as you learn some of the parameters, and then estimate some of the states, you're actually changing the landscape. So it's actually kind of interesting because in machine learning, often you can kind of dig into one [local minimum] and then end up here. It's very funny because the loss landscape will change as you move through it. You still have this notion that there's local minima, global minima, but you have more ways of escaping local minima."

Recommended visualization: the 3Blue1Brown YouTube videos on loss-landscape visualization and escaping local minima.

### Stochastic gradient descent and mini-batch

Full GD: use all the data per gradient, "your best estimate of the gradient." SGD: an average over many individual gradients — "noisy stochastic ones, and then you average over them." Counterintuitive at first: "what's not obvious from this is that actually this is a good idea to have noisy estimates of your gradients."

**Mini-batch SGD** = one particular way to get the noisy gradients. "You have so many data points you don't want to compute your gradient from all of them because it would take forever. So instead you just take some of them. You take a random subset or you partition the data into 10 random subsets or some number of random subsets. And then you just compute your gradient on a part of the data and then use that to update your parameters."

True per-sample SGD ("one sample per iteration") is rare — "would just take forever. But yeah, you compromise."

Batch sizes are **powers of two** (32, 256, 512, 1024): "because that's what you always do in machine learning because you try to make everything as hardware efficient as possible. And most hardware things happen in powers of two."

> [!important] Implicit L2 regularization (worth memorizing)
> "Mini-batch stochastic gradient descent actually gives you an implicit L2 regularization, which is super weird. … Whenever you use this and you're in a problem setting where there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized."

So mini-batch buys you two things: speed (parallelizable small batches) and regularization (implicit shrinkage toward small-norm solutions). The prof's hedge on the history: maybe people invented it for the regularization, maybe just because computing the full gradient was too expensive — "but the reality is you could also have come up with it simply because it seemed like a good idea."

## Where this leaves us

Got through ~half the deck — feedforward anatomy, multilayer + XOR motivation, activations (linear/sigmoid/ReLU/GELU + softmax for output), MNIST, universal approximation, loss functions, gradient descent + SGD + mini-batch with implicit L2. Backpropagation derivation, the rest of optimization, and explicit regularization tricks continue tomorrow ("we'll probably finish this tomorrow and then have the review next week"). Next-day's session is CNNs.

> [!note] Recommended outside material (he plugged it twice)
> 3Blue1Brown YouTube videos — both for general NN visualization and specifically for the loss-landscape / escaping-local-minima animations. "Really nice video on just visualizing the loss landscape as you dig into the parameters and the states and how it will actually escape local minima just by keep going down, which is amazing." Listed as compulsory videos in the slide deck (Videos 1–4).
