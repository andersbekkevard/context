---
concept: nn-regularization
module: 11-nnet
lectures: [L24, L26]
isl-ref: 10.7.2
exercises:
  - Exercise11.1d  -  explain how a 10000-weight NN can fit on 1000 obs (regularization is the answer)
  - Exercise11.4.2  -  apply rotation/shift/flip data augmentation to CIFAR-10 CNN; observe regularization effect
related: [feedforward-network, gradient-descent-and-sgd, ridge-regression, lasso, double-descent, regularization, convolutional-neural-network, cross-validation]
tags:
  - concept
  - module/11-nnet
aliases:
  - dropout
  - early stopping
  - data augmentation
  - label smoothing
  - transfer learning
  - weight decay
---

# Neural-network regularization

The prof's iron rule: **never train a NN without regularization.** The menu is wide, explicit (L1 / L2 weight decay, dropout) and implicit (mini-batch SGD, early stopping, data augmentation, label smoothing, transfer learning). Their job: reduce **generalization** error, not training error.

## Definition (prof's framing)

> "I can't think of an example where you'd ever want to train a neural network without a regularization. … The whole point of regularization is to get you to generalize better." - [[L26-nnet-3]]

> "Regularization: any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.", slides (Goodfellow definition), via [[L24-nnet-2]]

> "[Regularization] doesn't reduce training error, but could hurt it. We accept a small training-fit penalty, sometimes none, when many models fit equally well and we just need a tiebreaker." - [[L24-nnet-2]]

The motivating fact: NNs typically have **more weights than data samples** (10000 weights, 1000 obs is the Exercise11.1d setup). Without regularization the model memorizes noise and generalizes poorly. The Hitters comparison in [[L26-nnet-3]] makes this concrete, a 1049-param NN with no regularization performed *worse* than lasso on 263 baseball players. The prof: **"if you constructed your model this way you doing it wrong."**

## Notation & setup

- $\mathbf w$: the weight vector (just the weights, not the biases, biases are typically excluded from regularization)
- $L(\boldsymbol\theta)$: base loss (MSE / cross-entropy)
- $\lambda$ or $\alpha$, regularization-strength hyperparameter
- For dropout: $p_{\text{drop}}$, fraction of nodes zeroed per training step

## The full menu (and the rules)

### 1) L1 / L2 weight decay (the classics)

**L2 weight decay** (analogous to [[ridge-regression]]):

$$\tilde L(\mathbf w) = L(\mathbf w) + \lambda\, \mathbf w^\top \mathbf w = L(\mathbf w) + \lambda \sum_{ij} w_{ij}^2$$

**L1 weight decay** (analogous to [[lasso]]):

$$\tilde L(\mathbf w) = L(\mathbf w) + \alpha\, \|\mathbf w\|_1 = L(\mathbf w) + \alpha \sum_{ij} |w_{ij}|$$

> "Surprisingly helpful. Even though they're incredibly simple looking, they actually do quite a lot." - [[L24-nnet-2]]

Behavior (revisited from Module 6):
- **L2** (bowl shape): penalizes large weights heavily, near-zero ones barely. "Encourages averaging more than one solution, not so sparse." - [[L24-nnet-2]]
- **L1** (V shape): penalizes small weights more aggressively → "leads to a sparser solution." Sparse weights = an implicit form of variable / unit selection.

### 2) Mini-batch SGD (implicit L2)

> [!important] The prof's headline NN fact
> "Mini-batch stochastic gradient descent actually gives you an implicit L2 regularization, which is super weird. … Whenever you use this and you're in a problem setting where there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized." - [[L23-nnet-1]]

> "It has been proven, which is nice. The math is there, but it's not super short." - [[L24-nnet-2]]

See [[gradient-descent-and-sgd]] for the full story. **Even without explicit weight decay, just using SGD gives you regularization for free.**

### 3) Data augmentation

Take a training example, perturb it, keep the label.

> "Take a training example, copy it, perturb it, keep the same label. MNIST: take a '1', rotate / add noise / shift / flip → still a '1'. Now the model is forced to be insensitive to those transformations and you've effectively grown the dataset." - [[L24-nnet-2]]

Common transforms: rotation, flip, shift / translation, zoom, shear, Gaussian / sparse noise, color jitter, crop. **Match the augmentation to the data**: don't flip handwritten 6 vertically (becomes 9). Keep the originals in too.

> "You want to make sure that how you augment the data makes sense. Gaussian noise often makes sense, or sparse noise, rotations, flips, shifts, translations, things of that sort." - [[L24-nnet-2]]

Especially natural for CNNs (Exercise11.4.2): same image content, different pose → same label.

### 4) Label smoothing

Instead of one-hot targets $(0, 0, 1, 0, 0)$, use softened targets like $(\varepsilon/(C-1), \dots, 1 - \varepsilon, \dots)$ for some small $\varepsilon$.

> "Classification trick: instead of hard 0/1 targets, soften them so the model doesn't get over-confident. Same flavour as augmentation, inject uncertainty so the model generalizes." - [[L24-nnet-2]]

> "Motivated by the fact that the training data may contain errors in the responses recorded.", slides

### 5) Early stopping

Plot training error vs. epochs (decreases monotonically) and validation error on the same axes (decreases, levels off, then often climbs). **Stop at the validation minimum**, return the parameters from that epoch, not the final ones.

> "The most commonly used form of regularization.", slides

> "I always thought it felt like cheating. It's like, really? But that's a very common thing to do. Especially when you're in this regime of having a lot of data, but where you're having a model that's smaller, so it will overfit." - [[L24-nnet-2]]

When does it apply most? Models with "more than the minimum number of parameters needed but less than a lot of parameters", i.e. **not** the benign-overfitting / [[double-descent]] regime, but the classical U-shape regime.

### 6) Dropout

Hinton's idea, neuroscience-inspired ("if someone hits you over the head, you don't just die. There's often lots of redundancy in the brain.").

**Mechanics**: during training, randomly **set to zero** a fraction of node outputs in a given layer at each iteration (good range: **20–50%, typically 20%, never 50%**). Rescale the surviving nodes to compensate. **At test time, no dropout**, scale outputs by the dropout rate (or do it during training).

> "During training: randomly dropout (set to zero) some outputs in a given layer at each iteration. Drop-out rates may be chosen between 0.2 and 0.5.", slides

> "20% is very common, never use 50%." - [[L24-nnet-2]]

The deeper reason it works:

> [!important] Dropout = ensembling inside the network
> "He wanted this neural network to learn multiple ways of doing things, so not be dependent on only one specific thing, but rather learn many, many solutions. … It's kind of like really learning a community, where you don't have any one critical node or person, where if that person isn't there, everyone dies." - [[L24-nnet-2]]

Direct parallel to bagged trees ([[bagging]] / [[random-forest]]), each forward pass at training time is effectively a different sub-network; you average over them.

> "Practical bonus: almost no hyperparameters (just the dropout rate), trivial to use ('you really just in the optimizer you write `dropout=20` and then it just works'). Having things to tweak sucks. … Dropout is one of the nice ones." - [[L24-nnet-2]]

### 7) Transfer learning (the data-hack)

> "If it's image data, I would just download a good image model and then chop off their output and basically use all their training, all their layers, as a way of using the model that they've learned that works so well, and just changing the objective function at the end and only training the last bit." - [[L24-nnet-2]]

Personal example: animal-tracking project, only ~10 hours of data, used a pre-trained image model and re-trained the last bit on ~100 labelled frames → "boom, we had a very nice simple model."

When applicable: similar input domain (images, language), large labelled dataset elsewhere, your task has limited data. Effectively imports a *prior* learned from someone else's data, a regularization technique by inheritance.

## Insights & mental models

### Reduce variance, accept some bias

The bias-variance framing applies here exactly as in Module 6. Regularization trades a small increase in bias for a larger reduction in variance, **net win in test error**. See [[bias-variance-tradeoff]].

### "Avoid bad overfitting"

> "[Regularization is] the key one. Because that whole thing is designed to keep it from overfitting in a bad way. He prefers to call it 'avoid bad overfitting.'" - [[L24-nnet-2]]

The distinction matters in the [[double-descent]] regime: you *can* fit training data perfectly and still generalize ("benign overfitting"), provided regularization is doing enough work. The wrong kind of overfitting is when test error climbs without recovery; the right kind is when many perfect-fit solutions exist and SGD picks the min-norm one.

### Hyperparameters are the cost

> "Hyperparameters are the annoying surface area: number of layers, layer widths, $\lambda$ for L1/L2, etc., each one will help you change the model a little bit, and then you can evaluate it on the validation set." - [[L24-nnet-2]]

Pick via validation set or [[cross-validation]]. Beware of **overfitting your validation set** ("you can use the validation set so many times that actually you're overfitting to the validation set"); some people use nested train/test/validation splits to guard against this.

### When to NOT use a NN

> "If you don't have a lot of data and need interpretability, probably don't use neural networks at all. Use trees." - [[L24-nnet-2]]

> "Generally with neural networks, you want to have enough data. And if you don't have enough data, then you do something else." - [[L24-nnet-2]]

## Exam signals

> [!important] Verbatim "never without regularization" rule
> "I can't think of an example where you'd ever want to train a neural network without a regularization. … The whole point of regularization is to get you to generalize better." - [[L26-nnet-3]]

> "[Hitters NN with no regularization] is contrived … if you constructed your model this way you doing it wrong." - [[L26-nnet-3]]

> "Mini-batch stochastic gradient descent actually gives you an implicit L2 regularization." - [[L23-nnet-1]] (the headline implicit-regularization fact)

> "Drop-out rates may be chosen between 0.2 and 0.5. … 20% is very common, never use 50%.", slides + [[L24-nnet-2]]

> "Q3 of the 2024 exam (per slide-deck reference)": "In the context of trees, shrinkage-type regularization is performed via *tree pruning*, whereas neural networks do this via *weight penalization* (data augmentation, label smoothing, dropout, early stopping). More generally, any type of regularization has as a main goal to *reduce test error* by avoiding that the model over-fits the data.", 2024 exam.

→ Past exams have asked **multiple-choice on the menu** of NN regularization techniques. This is a high-probability 2026 exam question.

## Pitfalls

- **"Reduces training error."** No, regularization aims to reduce **generalization** (test) error and may *hurt* training error. Easy T/F trap.
- **Dropout 50% is wrong.** The prof: "20% is very common, never use 50%." - [[L24-nnet-2]].
- **Dropout active at test time.** No, dropout is a *training-time* trick. At test time the network is intact (with rescaling).
- **"L1 and L2 are the same."** No, L2 shrinks smoothly toward zero, L1 forces exact zeros (sparsity). Same as ridge vs. lasso ([[ridge-vs-lasso-geometry]]).
- **Early stopping vs. dropout, pick one.** They serve overlapping purposes; using both is fine but redundant; a single well-tuned regularizer is usually better than stacking.
- **Augmenting the wrong way.** Flipping a "6" vertically gives a "9", wrong augmentation. The transformation must preserve the label.
- **Forgetting that mini-batch SGD is itself regularization.** Don't list it only under "optimization", it lives on the regularization menu too.
- **Treating regularization as optional in the over-parameterized regime.** The prof's iron rule: *never* train without it.

## Scope vs ISLR

- **In scope:** the full menu (L1, L2, mini-batch SGD as implicit, data augmentation, label smoothing, early stopping, dropout, transfer learning), the "never train without regularization" rule, dropout rate range (20–50%), the bias-variance framing.
- **Look up in ISLR:** §10.7.2 (regularization + SGD), §10.7.3 (dropout), §10.7.4 (network tuning).
- **Skip in ISLR (book-only / not lectured):**
  - **Batch normalization**: [[scope]] NN exclusions: explicitly out.
  - **Weight initialization (Xavier / He)** - [[L24-nnet-2]] / [[scope]]: not discussed.
  - **Vanishing / exploding gradients**: [[scope]]: out.
  - **Adam / RMSProp / momentum optimizer details**: out per [[L23-nnet-1]] / [[L27-summary]].
  - **Detailed architecture-search / hyperparameter-tuning algorithms** beyond "use validation / CV."

## Exercise instances

- **Exercise11.1d**: *Why is it possible for a 10000-weight NN to fit on 1000 obs?* Direct regularization question. Answer must include: explicit (L1/L2/dropout/early stopping), implicit (mini-batch SGD), and the [[double-descent]] / minimum-norm-interpolator framing.
- **Exercise11.4.2**: *Apply data augmentation (rotation/shift/flip) to the CIFAR-10 CNN, observe the test-accuracy effect.* Concrete demonstration that augmentation helps generalization. Discuss in terms of "effectively grew the dataset" and the model becoming "insensitive to those transformations."
- **Exercise11.3** (Boston Keras), preprocessing standardizes the inputs (related but separate from regularization). The model itself (default Adam, no explicit dropout / weight decay) is what the prof would call **"contrived"**, fits with the Hitters Comparison critique in [[L26-nnet-3]].

## How it might appear on the exam

- **Multiple-choice on the menu** (2024 exam style, from [[L27-summary]] discussion): "Which of the following are forms of regularization in NN training?", answer covers L1, L2, dropout, data augmentation, label smoothing, early stopping, mini-batch SGD.
- **True/false on dropout rate**: "Dropout of 50% is the standard recommendation." → False (20% common, 50% never).
- **True/false on training error**: "Regularization reduces training error." → False (it reduces *generalization* error, may hurt training).
- **Conceptual link to bias-variance**: "Why does adding L2 regularization to a NN improve test performance?", trade increased bias for reduced variance; net win.
- **Why early stopping works**: training error falls monotonically, validation error U-shapes; pick the validation minimum.
- **Why dropout works**: ensembling inside the network, no single critical neuron, parallel to bagging / random forests.
- **Exercise11.1d-style** ("how can a 10000-param NN fit 1000 obs?"): regularization story + double descent + minimum-norm interpolator.
- **Compare regularization across modules**: tree pruning ↔ NN dropout ↔ ridge / lasso (L2/L1), the running theme captured in [[regularization]] (Specials atom).

## Related

- [[regularization]]: the Specials atom; this is the NN-specific instance of the cross-cutting theme
- [[feedforward-network]]: the model being regularized
- [[gradient-descent-and-sgd]]: mini-batch SGD's implicit L2 is on this menu
- [[ridge-regression]]: L2 weight decay is conceptually identical (applied to NN weights instead of regression coefficients)
- [[lasso]]: L1 weight decay; sparsity in NN weights
- [[ridge-vs-lasso-geometry]]: same geometric story as in module 6
- [[double-descent]]: the prof's "why this works" framing in the over-parameterized regime
- [[convolutional-neural-network]]: data augmentation is especially natural for CNNs
- [[cross-validation]]: pick $\lambda$ / dropout rate / depth via CV or validation set
- [[bias-variance-tradeoff]]: the framework for understanding why regularization helps
- [[bagging]] / [[random-forest]], dropout's conceptual cousin (ensembling for variance reduction)
