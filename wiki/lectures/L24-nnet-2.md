---
lecture: 24
date: 2026-04-21
module: 11-nnet
title: Neural Networks 2 (CNNs)
slides: modules/11NNet/11Nnet.md
topics:
  - gradient-descent-and-sgd
  - backpropagation
  - regularization
  - benign-overfitting
  - nn-regularization
  - convolutional-neural-network
  - pooling
  - activation-functions
tags:
  - lecture
  - module/11-nnet
aliases:
  - Lecture 24
---

# L24: Neural Networks 2 (CNNs)

The prof finished the feed-forward training story (gradient descent → mini-batch SGD → [[backpropagation]]), spent the bulk of the second half on regularization (L1/L2, data augmentation, label smoothing, early stopping, [[nn-regularization|dropout]], transfer learning), then introduced [[convolutional-neural-network]]s: motivation from the eye, learned filters as the CNN twist on classical edge detectors, pooling, and why CNNs are "just a feed-forward network" so backprop drops in for free. RNNs and double descent are deferred to next time.

## Key takeaways

- **Bigger model + more data + bigger compute is the bitter pill.** Sutton's "annoying fact": most ML progress is just scaling. Backprop, GPUs, mini-batching all fed this.
- **Mini-batch SGD ≠ just an optimization speedup: it's also regularization.** Noise from small batches helps generalization. "It has been proven, which is nice. The math is there, but it's not super short."
- **Backprop is just the chain rule, applied so you don't recompute things.** Forward pass stores intermediates; backward pass reuses them and propagates a $\delta$ from output to input. Loops break this. Feed-forward is loop-free, so backprop just works.
- **Benign overfitting:** AI people aren't scared of overfitting the way statisticians are. You can fit training data perfectly and still generalize, provided regularization is doing enough work.
- **Dropout = randomly zero 20–50% of nodes during training, rescale the rest.** Hinton, inspired by neuroscience: no single critical neuron, redundant representations, an ensembling effect inside one network.
- **CNNs = feed-forward with shared local filters that get learned.** Motivated by the eye / Fukushima's Neocognitron; LeCun added backprop. Pooling shrinks spatial dimension while keeping peak activations. Data augmentation falls out naturally.

## Where we are: still designing the feed-forward network

Quick recap of the design choices for a feed-forward network, "the workhorse of the machine learning world." Architecture, activation functions, loss / output function (regression-like MSE for continuous output, classification loss for classes, "tailor your model to the problem at hand"), and now **optimization**.

Universal approximation theorem mentioned but not derived: "it's quite involved and doesn't really use a lot of statistics. It's more measure theory."

> [!note] The history is co-evolution
> "Advances in all of these things have really come hand in hand. … if we switch activation functions suddenly we can gain all this functionality. Then five years later they're like oh if we change the way we're optimizing — oh shit, now we can do all this new stuff."

Backprop in particular: "a very, very simple idea. But without it, the computers at the time were just rubbish." Bigger network → more functionality, and the article-the-prof-will-bring-Monday by Richard Sutton ("the bitter lesson"-ish): "all you really need to do is get your model to be bigger, be able to train or optimize a bigger model on more data. … Move it to the GPU, suddenly, whoa, you can do so much more."

## Gradient descent

Standard story: locally find the minimum of the cost function. Update rule

$$\theta^{(t+1)} = \theta^{(t)} - \lambda \nabla_\theta L$$

where $\lambda$ is the learning rate, a hyperparameter. Aside: NN cost surfaces in very high dimension have an interesting property, "the local minima are somehow connected", but he defers this to the 3Blue1Brown video on Monday.

Tuning the learning rate matters in practice: "the network I was running over the weekend, I realized that part of my model had a learning rate that was too high. … once I shrunk it down, it performed so much better." Too large → bouncing; small enough (e.g. 0.1 or smaller) → fine.

Aside-aside: someone trained a big NN with evolutionary algorithms ("bonkers"). In general nobody bothers; gradient-following dominates.

## Stochastic / mini-batch gradient descent

Full-batch GD averages a per-sample gradient over all $N$ samples:

$$\nabla L = \frac{1}{N}\sum_{i=1}^N \nabla L_i$$

That's an *expectation* of a single-point gradient, so you can use any mini-batch of $m \ll N$ samples and get an unbiased estimate. "Of course, expectations are only expectations, but still, it's generally a good idea."

Two payoffs:

1. **Distributability.** With $K$ GPUs and $N=100$, hand 10 samples to each, sum results. Parallelism is why mini-batching took off.
2. **Regularization via noise.** "It creates a regularization effect, meaning that you still fit the data very well but you actually end up generalizing better to data that you haven't seen, which is really not obvious."

> [!important] Mini-batch SGD as regularization
> "Crucial idea is that you just need a mini-batch of samples. It's faster, and you approximate estimates of the gradient instead of better estimates, and that noise is very good. And it has this regularization effect, which is not obvious. It has been proven, which is nice. The math is there, but it's not super short."

## Backpropagation

Setup: a single-hidden-layer net with linear output activation,

$$f(x_i) = \beta_0 + \sum_{k=1}^K \beta_k z_{ik}, \qquad z_{ik} = g\!\left(\alpha_{k0} + \sum_{j=1}^p \alpha_{kj} x_{ij}\right)$$

with squared-error loss $R_i = \tfrac12 (y_i - f(x_i))^2$.

Naïve recipe: take derivatives w.r.t. every parameter, update. Backprop says do that *cleverly*: store intermediates from the forward pass, reuse them in the backward pass.

### Forward pass: compute and store

Push input through the model, storing

- $v_{ik} = \alpha_{k0} + \sum_j \alpha_{kj} x_{ij}$ (pre-activation of hidden unit)
- $z_{ik} = g(v_{ik})$ (activation)
- $f(x_i) = \beta_0 + \sum_k \beta_k z_{ik}$ (output)

"It's really not brain surgery."

### Backward pass: propagate $\delta$

Define an output-layer "error":

$$\delta_i^{\text{out}} = \frac{\partial R_i}{\partial f(x_i)} = -(y_i - f(x_i))$$

Then for output-layer weights:

$$\frac{\partial R_i}{\partial \beta_k} = \delta_i^{\text{out}} \cdot z_{ik}$$

Both factors already sit in memory from the forward pass. Bias term: just $\delta_i^{\text{out}}$ ("there's no chain rule, like it's the inside one").

For the hidden layer, define

$$\delta_{ik}^{\text{hid}} = \frac{\partial R_i}{\partial v_{ik}} = \delta_i^{\text{out}} \cdot \beta_k \cdot g'(v_{ik})$$

Then

$$\frac{\partial R_i}{\partial \alpha_{kj}} = \delta_{ik}^{\text{hid}} \cdot x_{ij}$$

> [!note] Why this is fast
> "This thing we already computed. This thing is a number. This thing we already computed in the forward pass. So we don't really need to do anything. We don't need to do a whole bunch of sums and calculations. It becomes very efficient."

The general pattern: at any layer $L$, $\delta^{(L)}$ is a sum involving $\delta^{(L+1)}$, the next-layer weights, and $g'$. You only ever look one layer ahead; it just propagates back.

### Why feed-forward matters here

"If you had loops you're kind of screwed. … with loops, then you can't always just go backwards. … It is convenient that a feedforward network doesn't have those loops and you can just go backwards." (Foreshadows the RNN difficulty next time.)

## Overfitting, the AI vs. statistician view

Statisticians are paranoid about overfitting; AI people are not. The distinction is **benign overfitting**:

> [!important] Benign overfitting: the prof's framing
> "You can be in a regime where you're actually fitting all of your data really well, so your training samples are actually perfectly explained by a model — which I mean as a statistician that sounds impossible — and yet it still can generalize well."

He flags this as debate-worthy ("part of the debate if that's even possible and if it is, if that's a good idea"), and the trick that makes it possible is **regularization**.

Regularization's job: reduce *generalization* error (lower variance in the bias-variance sense), without necessarily reducing training error. "It doesn't reduce training error, but could hurt it." We accept a small training-fit penalty, sometimes none, when many models fit equally well and we just need a tiebreaker.

Earlier modules already touched regularization: Module 6 (bias-variance, ridge/lasso), Module 9 (tree pruning, bagging/forests as ensembling). Ensembling is itself a regularization mechanism.

## Forms of regularization

### L1 and L2: the classics

Same shapes as in [[ridge-regression]] / [[lasso]]: L2 is $W^\top W = \sum_{ij} w_{ij}^2$, L1 is $\sum_{ij} |w_{ij}|$. Notation aside: $\|\cdot\|$ with subscript 2 → L2, with subscript 1 → L1.

Behavioural difference (revisited from Module 6):

- L2 (bowl shape): penalizes large values heavily, near-zero values barely. "Encourages averaging more than one solution, not so sparse."
- L1 (V shape): doesn't squash large values as aggressively, but penalizes small ones more → "leads to a sparser solution."

"Surprisingly helpful. Even though they're incredibly simple looking, they actually do quite a lot."

### Data augmentation

Take a training example, copy it, perturb it, keep the same label. MNIST: take a "1", rotate / add noise / shift / flip → still a "1". Now the model is forced to be insensitive to those transformations and you've effectively grown the dataset.

"You want to make sure that how you augment the data makes sense. Gaussian noise often makes sense, or sparse noise, rotations, flips, shifts, translations, things of that sort." Typically you keep the originals in too, so training fit isn't hurt.

### Label smoothing

Classification trick: instead of hard 0/1 targets, soften them so the model doesn't get over-confident. Same flavour as augmentation: inject uncertainty so the model generalizes.

### Early stopping

Plot training error vs. epochs (decreases monotonically, levels off) and validation error on the same axes. Validation error often goes down, then back up. "Early stop" at the validation minimum: return parameters that gave the best validation performance, *before* convergence on training.

> [!note] The prof's confession
> "I always thought it felt like cheating. It's like, really? But that's a very common thing to do. Especially when you're in this regime of having a lot of data, but where you're having a model that's smaller, so it will overfit."

When does it apply most? Models with "more than the minimum number of parameters needed but less than a lot of parameters", i.e. *not* in the benign-overfitting regime.

### Dropout

Hinton's idea, motivated by neuroscience: "if someone hits you over the head, you don't just die. There's often lots of redundancy in the brain." Don't let any single neuron / pathway become critical.

Mechanics: during training, randomly **set to zero** a fraction of node outputs (good range 20–50%, "20% is very common, never use 50%"). Rescale the surviving nodes to compensate. At test time, keep the network intact.

The deeper reason this works, direct parallel to bagged trees:

> [!important] Dropout = ensembling inside the network
> "He wanted this neural network to learn multiple ways of doing things, so not be dependent on only one specific thing, but rather learn many, many solutions. … It's kind of like really learning a community, where you don't have any one critical node or person, where if that person isn't there, everyone dies. But rather, everyone is important, everyone intermixes, and can accommodate when parts of the network are screwed."

Practical bonus: almost no hyperparameters (just the dropout rate), trivial to use ("you really just in the optimizer you write `dropout=20` and then it just works"). "Having things to tweak sucks. … Dropout is one of the nice ones."

### SGD as regularization (back-reference)

Already covered above; listed here so it's clear that mini-batching is on the regularization menu, not just the optimization menu.

## Avoiding overfitting: the menu

Hyperparameters are the annoying surface area: number of layers, layer widths, $\lambda$ for L1/L2, etc. "Each one will help you change the model a little bit, and then you can evaluate it on the validation set."

Other levers:

- **Reduce network size.** Possible but: "if you really want to do that, you're probably better off using just linear regression or trees or something like that. One of the trees, like XGBoost." If you don't have a lot of data and need interpretability, "probably don't use neural networks at all. Use trees."
- **Collect more data** (obvious). Or **train on a related dataset first**, e.g. classifier of dog images: train first on all images you can find, then fine-tune on dogs.
- **Regularization**: "the key one. Because that whole thing is designed to keep it from overfitting in a bad way." He prefers to call it "avoid bad overfitting."

> [!warning] Overfitting your validation set
> "You can use the validation set so many times that actually you're overfitting to the validation set. I've seen that happen a few times where you're like, oh, let's try this, let's try that, let's try this. And the next thing you know, you've not just overfit your training data, but also your validation data."

Some people use nested train/test/validation to guard against this.

### Library aside

PyTorch is the prof's go-to: "it's written the way I would write it." Keras: "you have to define the inputs and pass flags and remember what the hell they were, and I hate that."

## Worked example: Boston suburbs (briefly)

ISL example: small dataset, train/test split, normalize (always, "you don't want one variable to basically suck up all the variance, just like in the PCA"), one hidden layer with 5 units. "Barely a neural network model, but it is technically." Prediction error not great. The prof would have used XGBoost on this dataset.

> [!note] When NOT to use a neural network
> "Generally with neural networks, you want to have enough data. And if you don't have enough data, then you do something else."

### Transfer learning (the data-hack)

"If it's image data, I would just download a good image model and then chop off their output and basically use all their training, all their layers, as a way of using the model that they've learned that works so well, and just changing the objective function at the end and only training the last bit." Personal example: animal-tracking project, only ~10 hours of data, used a pre-trained image model and re-trained the last bit on ~100 labelled frames → "boom, we had a very nice simple model."

## Convolutional neural networks

### Motivation: variable-size, scale, translation

Feed-forward nets need fixed-size inputs. Images come in many sizes, and content can be translated, scaled, deformed. You'd like a model that handles all of those.

Inspiration came from the eye / brain. **Fukushima** (Japan, still alive) wrote the **Neocognitron** model: "unfortunate that's not the name that got kept. People now call them convolutional neural networks." This is also the first well-known instance of **geometric deep learning** (CNNs as actions of symmetry groups).

**LeCun (1989)** took the Neocognitron, added backprop (which Hinton had just published), changed the name. "It's an annoying thing that you can take someone's idea and add some shit to it and call it something else, and then no one knows it was the other guy's idea. … But I've heard Yann is quite nice, so I won't insult him too much."

### The biological intuition

Your eyes dart around, sampling tiny patches at a time. Whole-scene perception is *constructed*: retina → optic nerve → thalamus all see local bits; the visual cortex assembles a scene many layers in. CNNs mimic this: sample locally, build up.

### Classical filters → learned filters

Pre-CNN: hand-design filters (Gabor filters, vertical/horizontal edge detectors) and convolve them across the image to get feature maps. CNNs keep the convolution structure but **learn** the filter weights instead of fixing them.

Cat example: a vertical-edge filter loses horizontal whiskers; a horizontal one over-emphasizes them. You don't know which features your filter has discovered, but they're discovered automatically.

### How a conv layer works

Input: an image (or activations from the previous layer). The filter is a small (square) patch of weights (these are the parameters). Convolve the filter over the input: at each spatial location, take the weighted sum of the underlying patch → one output value. Slide → next location.

You stack $K$ filters per layer → output is a 3D feature map of depth $K$. "Like having 32 neurons in a way." Filter sizes are usually powers of 2 (8, 16, 32) for compute reasons.

Activation: ReLU is the default ("probably GELU is more common now. I'm not sure"), well-suited to non-negative image values.

### Pooling

After convolution you can **pool**, typically **max pool**: take the max over each non-overlapping patch and emit one value. Shrinks spatial dimensions, preserves peaks.

Why max specifically: "for that filter that specifically is trying to look for vertical edges, maybe you don't care when there's nothing there. What you care about is, do I see a peak, and where is the peak?" Max pool keeps the peak. Other poolings exist; max is by far the most common.

### Stacked conv + pool

Alternate conv layers (with multiple filters) and pool layers. Each pool shrinks spatial extent → deeper layers cover larger receptive fields. With just these two operations, simple early features (edges) compose into larger features (eye, face) over a few layers.

> [!note] Why depth lets you "see" a face
> "If you're always just focusing on tiny little things, it's hard to imagine that you could ever get there. But if you keep this idea of filtering it and then pooling and pooling, you could imagine that somewhere in here, like maybe this layer here, you could get something that has a selectivity for an eye or a face. … because at this point it has the scale it needs to combine across different scales and say oh this is an eye, not just because it has the shape and the color, but also where it is with respect to the face."

CNNs dominated ImageNet for a long time; "newer image models are a bit different, but CNNs are still extremely common." Same idea applies wherever there are spatial / temporal dimensions: convolve over time for time series, or over text tokens for language models.

### Data augmentation, again

Trivially natural for CNNs: rotate / shift / flip the image, keep the label.

### Why CNNs are still just feed-forward

> [!important] CNN = feed-forward, so backprop drops in
> "It's just a neural network. It's basically the same idea as a feed-forward network. It's still trained, there's still no loops backwards. So really you can trivially apply backprop to this model. … That's why Yann could make so much progress so quickly."

LeCun could lean on all the prior tricks (data augmentation, regularization, dropout) for free.

## Wrap-up and what's next

Stopping here. Next time: **RNNs** (recurrent neural networks, "you can think of it as a precursor to the transformer in a way") and the prof's favorite, **double descent**. Plus the Sutton article and a 3Blue1Brown video reference.
