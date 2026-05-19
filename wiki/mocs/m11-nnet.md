---
moc: 11-nnet
title: "M11: Neural Networks"
lectures: [L23, L24, L26]
isl-ch: 10
slides:
  - modules/11NNet/11Nnet.md
exercises:
  - exercises/Exercise11/
tags:
  - moc
  - module/11-nnet
---

# Module 11: Neural Networks

The prof's "deep learning is just nested GLMs" module. Three lectures (Apr 20, 21, 27) move from feedforward + backprop, through CNNs for images, to RNNs and double descent. Load-bearing: parameter counting, the regularization menu, and the conceptual link between SGD's implicit regularization and double descent. Architecture details are explicitly out.

## Slides
- <a href="/pdfs/m11-nnet-slides.pdf" target="_blank" rel="noopener">m11-nnet-slides.pdf</a>

## Lectures
- [[L23-nnet-1]]: feedforward architecture, activation functions, parameter counting, gradient descent / SGD, backpropagation, universal approximation
- [[L24-nnet-2]]: CNNs (convolution + max-pool), regularization menu (L1/L2, dropout, early stopping, data augmentation, transfer learning)
- [[L26-nnet-3]]: RNNs (hidden-state recurrence) and double descent / benign overfitting; "when to use deep learning"

## Concepts (atoms in this module)
- [[feedforward-network]]: input → hidden(s) → output, no loops; nested z_m = a(α₀_m + Σⱼα_jm xⱼ); the canonical NN architecture
- [[activation-functions]]: sigmoid / ReLU / GELU / softmax; nonlinearity is what stops the network from collapsing to linear regression
- [[nn-parameter-count]]: per layer, weights between layers + bias for each receiving unit; **forgetting bias is the canonical wrong answer**; prof-flagged exam-likely calculation
- [[universal-approximation]]: one wide hidden layer can approximate any Borel-measurable function; existence result, **proof out of scope** (measure theory)
- [[gradient-descent-and-sgd]]: θ ← θ − λ∇_θL; mini-batch gives unbiased gradient + **implicit L2 regularization** (min-norm interpolator); the prof's headline regularization fact
- [[backpropagation]]: chain rule reusing forward-pass intermediates; only works for feedforward / acyclic architectures (why RNNs are harder)
- [[nn-loss-functions]]: MSE for regression, binary / categorical cross-entropy for classification; same shapes as the GLM losses
- [[nn-regularization]]: never train without it; menu = L1/L2 weight decay, data augmentation, label smoothing, early stopping, dropout (20–50%), transfer learning
- [[convolutional-neural-network]]: feedforward with shared local filters + max-pool; **architecture details out of scope** (per L27)
- [[recurrent-neural-network]]: hidden state A_t = σ(b + W X_t + U A_{t−1}); weight sharing across timesteps; **LSTM/GRU/BPTT out of scope**

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; this module reframes it via [[L26-nnet-3]] double descent, past the interpolation point, test error comes back down
- [[regularization]]: first systematic treatment in module 06; this module is its richest single venue (L24 menu: L1/L2, dropout, early stopping, augmentation, transfer learning) plus the implicit-regularization story for SGD in [[L23-nnet-1]]
- [[cross-validation]]: mechanics in module 05; here used to tune NN hyperparameters (M, λ, dropout rate), early stopping on a held-out validation split is the standard idiom
- [[standardization]]: first owed by module 06; **mandatory** before fitting an NN (Exercise11.3 explicitly preprocesses Boston housing with mean/sd)
- [[double-descent]]: prof's hobbyhorse; introduced module 02, returns here in [[L24-nnet-2]] and [[L26-nnet-3]] as "this is why deep learning works"; minimum-norm interpolator framing

## Exercises
- [[../../exercises/Exercise11]]: full module sweep: write the input/output equation for a given architecture, count parameters (1- and 2-hidden-layer ReLU + sigmoid output), compare GAM vs FNN, fit a Keras feedforward to Boston housing, build a CIFAR-10 CNN with conv → maxpool → dense → softmax, apply data augmentation, 1D-CNN for Wafer time-series classification

## Out of scope (this module)
- **CNN architecture details** (filter math, padding, pooling variants, ResNet / Transformer) - high-level concept only - [[L24-nnet-2]] / [[L27-summary]]
- **RNN architecture details** (LSTM/GRU gates, BPTT) - only the high-level "hidden state propagates" idea is in scope - [[L26-nnet-3]] / [[L27-summary]]
- **Vanishing / exploding gradients, batch normalization, weight initialization (Xavier/He)** - "not discussed in any depth" - [[L23-nnet-1]] / [[L24-nnet-2]]
- **Advanced optimizers** (momentum, Adam internals) - out of scope - [[L23-nnet-1]] / [[L27-summary]]
- **Skip connections, intra-layer connections** - "you just wouldn't need to know that" - [[L27-summary]]
- **Universal approximation proof** - "to actually understand this, you have to go into measure theory and all sorts of hard math that we don't talk about in this class" - [[L23-nnet-1]]
- **History of NNs** (McCulloch & Pitts, Rosenblatt, AI winters, AlexNet) - "I'm not going to ask you a history question on the test" - [[L22-unsupervised-2]] / [[L27-summary]]
- **Shapley values, explainable-AI machinery** - brief mention only - [[L26-nnet-3]]
- **R / Python package names, Keras / PyTorch syntax** - "no language-specific coding or anything of that sort" - [[L27-summary]]

## ISLP pointer
Chapter 10: Deep Learning. Deep treatment of in-scope concepts in this module is in `wiki/book/10-deeplearning.md`. Atoms carry section-level `isl-ref:` pointers; for the full algebra of any in-scope concept, route Anders to that chapter.
