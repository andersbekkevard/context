---
concept: convolutional-neural-network
module: 11-nnet
lectures: [L24]
isl-ref: 10.3
exercises:
  - Exercise11.4.1  -  build a CNN for CIFAR-10 (conv → maxpool → conv → maxpool → flatten → dense → softmax)
  - Exercise11.4.1b  -  compute misclassification from a CIFAR-10 confusion matrix
  - Exercise11.5  -  1D-CNN for Wafer time-series classification, compare to logistic regression
related: [feedforward-network, activation-functions, backpropagation, nn-regularization, recurrent-neural-network]
tags:
  - concept
  - module/11-nnet
aliases:
  - CNN
  - convolutional network
  - convnet
---

# Convolutional neural network (CNN)

The prof's framing: "just a feed-forward network" with shared local **filters** and **max-pool** layers. The filters are *learned* (vs. hand-designed in classical image processing). Backprop drops in unchanged because the network is still acyclic. **Architecture details (filter math, padding, pooling variants, modern ResNet/Transformer designs) are explicitly out of scope** , high-level concept only.

## Definition (prof's framing)

> [!important] CNN = feed-forward, so backprop drops in
> "It's just a neural network. It's basically the same idea as a feed-forward network. It's still trained, there's still no loops backwards. So really you can trivially apply backprop to this model. … That's why Yann could make so much progress so quickly." - [[L24-nnet-2]]

A CNN is a [[feedforward-network]] where some layers are **convolutional** (apply a small set of learned filters across the spatial / temporal dimension) followed by **pooling** (typically max-pool, which shrinks spatial extent and keeps peaks). The standard pattern: conv → pool → conv → pool → … → flatten → dense → softmax.

## Insights & mental models

### Motivation: variable-size, scale, translation

Feedforward nets need fixed-size inputs and have no built-in spatial structure. Images come in many sizes, with content that can be translated, scaled, deformed. CNNs bake in **translation equivariance** through filter sharing.

> "Inspiration came from the eye / brain. Fukushima (Japan, still alive) wrote the **Neocognitron** model , 'unfortunate that's not the name that got kept.'" - [[L24-nnet-2]]

LeCun (1989) added backprop to the Neocognitron and renamed it CNN.

### Classical filters → learned filters

Pre-CNN: hand-design filters (Gabor, vertical / horizontal edge detectors), convolve across image. CNNs keep the convolution structure but **learn the filter weights**. You don't know which features the filter discovered, but they're discovered automatically.

### Conv layer + max-pool (the basic block)

- **Conv layer**: a small ($K \times K$) patch of weights = the filter. Slide across the input, compute weighted sum at each location → one output per position. Stack $K$ filters per layer → 3D feature map of depth $K$.
- **Activation**: typically ReLU.
- **Max-pool**: take the max over each non-overlapping patch (e.g. 2×2). Shrinks spatial dim, preserves peaks.

> "For that filter that specifically is trying to look for vertical edges, maybe you don't care when there's nothing there. What you care about is , do I see a peak, and where is the peak? Max pool keeps the peak." - [[L24-nnet-2]]

### Why depth lets you "see" a face

Stacked conv + pool: simple early features (edges) compose into larger features (eye, face) over a few layers, because each pool shrinks spatial extent and each subsequent layer covers a larger receptive field.

### Same trick for time series / text

> "Same idea applies wherever there are spatial / temporal dimensions , convolve over time for time series, or over text tokens for language models." - [[L24-nnet-2]]

This is what 1D-CNN means (Exercise11.5 , Wafer time series).

## Exam signals

> "It's just a neural network. It's basically the same idea as a feed-forward network. It's still trained, there's still no loops backwards. So really you can trivially apply backprop to this model." - [[L24-nnet-2]]

> "Architecture details out of scope" , the [[L27-summary]] / [[scope]] verdict on CNNs (no filter math, no pooling variants, no modern architectures like ResNet/Transformer).

The 2023 exam Q5 ([[L27-summary]] reference) includes "Convolutional neural networks (CNNs)" as one option among variable-selection methods (correct answer: false , CNNs are not for variable selection). So CNNs *can* show up in MC questions, but at the **conceptual** level only.

## Pitfalls

- **CNN is *not* a different beast than FNN.** It's a feedforward network with weight-sharing structure (filters are reused across positions). Backprop works the same; the loss is the same; the regularization tools are the same.
- **Pooling shrinks dimension; convolution shifts content.** Don't confuse the two.
- **Don't compute filter math on the exam.** The prof: out of scope. If asked anything about CNNs, stay at the level of "conv applies learned filters; pool shrinks; the whole thing is feed-forward."
- **Data augmentation is especially natural for CNNs** (rotate / shift / flip images) , see [[nn-regularization]].

## Scope vs ISLP

- **In scope:** the conceptual picture , CNN = feedforward + shared local filters + max-pool; learned (not designed) filters; trained by backprop; data augmentation lives here naturally.
- **Look up in ISLP:** §10.3 (whole CNN section), specifically §10.3.1 (convolution layers) and §10.3.2 (pooling layers) for the basic concepts.
- **Skip in ISLP (book-only, prof excluded):**
  - **Detailed filter math (kernel arithmetic, stride, padding)** - [[L24-nnet-2]] / [[scope]]: high-level only.
  - **Pooling variants** beyond max-pool , [[scope]]: out.
  - **Modern architectures** (AlexNet, VGG, ResNet, Inception, Transformer, attention) , out per [[L27-summary]] / [[scope]].
  - **Data augmentation pipelines** beyond the basic concept , out.
  - **Skip / residual connections** - [[L27-summary]]: explicitly out.
  - **§10.3.5 Results Using a Pretrained Classifier**: covered briefly via transfer learning in [[nn-regularization]], no detail expected.

## Exercise instances

- **Exercise11.4.1**: CIFAR-10 image classification with a CNN: conv(32 filters, 3×3, ReLU) → max-pool(2×2) → conv(64 filters, 3×3, ReLU) → max-pool(2×2) → flatten → dense(64, ReLU) → dense(10, softmax). Loss = `categorical_crossentropy`. The architecture is the canonical conv-pool-conv-pool-flatten-dense-softmax recipe.
- **Exercise11.4.1b**: given the CIFAR-10 confusion matrix, compute misclassification rate. Standard [[confusion-matrix]] question; CNN is incidental.
- **Exercise11.5**: 1D-CNN for Wafer time-series classification (152-length series, 2 classes). Shows the spatial-→-temporal generalization. Compare to logistic regression: CNN can capture local temporal patterns that logistic regression cannot.

## How it might appear on the exam

- **Multiple-choice / true-false** at the conceptual level: "CNNs are feedforward networks that use learned local filters." → True. "CNNs require recurrent connections." → False (those are RNNs).
- **Identification**: "Which method is most appropriate for image classification?" → CNN.
- **Conceptual contrast**: CNN vs. FNN ("CNNs use weight sharing across spatial positions" , see [[L24-nnet-2]] discussion of why backprop drops in unchanged).
- **Confusion-matrix interpretation** from a CNN's output (Exercise11.4.1b style) , same skill as for any classifier.

The prof flagged that detailed CNN math is out, so don't expect filter-arithmetic questions.

## Related

- [[feedforward-network]]: the parent architecture; CNN is "just" a feedforward with shared local weights
- [[activation-functions]]: ReLU is the default activation in conv layers
- [[backpropagation]]: works unchanged on CNNs because they're acyclic
- [[nn-regularization]]: data augmentation is especially natural for CNNs
- [[recurrent-neural-network]]: the other major NN extension, for sequential rather than spatial data
- [[confusion-matrix]]: Exercise11.4.1b uses CIFAR-10 confusion matrix
