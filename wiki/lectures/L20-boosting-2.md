---
lecture: 20
date: 2026-04-07
module: 09-boosting
title: Boosting and Additive Trees 2
slides: modules/9TreeBoosting/9TreeBoosting.md
topics:
  - boosting
  - adaboost
  - gradient-boosting
  - residuals
  - basis-functions
  - steepest-descent
  - boosting-loss-functions
  - weak-learner-and-learning-rate
  - early-stopping
  - stochastic-gradient-boosting
  - bagging
  - xgboost
  - nn-regularization
  - regularization
  - bias-variance-tradeoff
tags:
  - lecture
  - module/09-boosting
aliases:
  - Lecture 20
---

# L20: Boosting and Additive Trees 2

After a long break (project week + Easter), the prof comes back to finish module 9. He re-derives the boosting algorithm from a more general angle (boosting as **forward stagewise additive modeling**, with each new tree fit to the **negative gradient of the loss**). This unifies AdaBoost, residual boosting, and any other loss into one framework. He then walks through alternative losses (absolute, Huber, binomial deviance, multinomial deviance), the four hyperparameters of GBMs (depth, min-obs-in-leaf, M, learning rate), stochastic gradient boosting via row/column subsampling, and finally [[xgboost]] (second-order gradients, parallelization, L1/L2 regularization, dropout).

## Key takeaways

- **Boosting = forward stagewise additive modeling.** Find each new tree by minimizing $L(y, f_{m-1}(x) + T(x;\Theta_m))$. Decouple into two sub-problems: find regions $R_j$, then find values $\gamma_j$.
- **The big reframe**: when you fit the residuals in regression boosting, you are estimating the **negative gradient of the squared-error loss**. Generalize: for *any* loss, fit a tree to the negative gradient. That's [[gradient-boosting]].
- **Loss is a free choice.** Quadratic (default) penalizes outliers hard and never sends things to zero; **absolute loss** is robust but sparsifies; **Huber loss** is the compromise (quadratic near 0, linear for big errors). Binary classification → **binomial deviance**; K-class → **multinomial deviance** (one tree per class per iteration).
- **Three ingredients of gradient boosting**: weak learners, a loss function, and an additive way to combine. **Four hyperparameters**: tree depth, min observations per terminal node, number of trees $M$, learning rate $\nu$.
- **Always use weak learners**: small trees, small $\nu$ (≤ 0.1). Reason: each step should be a small step in the right direction (steepest-descent analogy); we want low *variance* via averaging many weak guys, not low bias from one strong one.
- **Stochastic gradient boosting**: subsample rows (and/or columns) before each tree to encourage diversity → variance reduction (and faster).
- **XGBoost** = current state-of-the-art: 2nd-order (Newton) gradients, parallelism, approximate split search, L1/L2 regularization on leaf weights, pruning parameter, dropout. The prof: "this is the one that everyone uses."

## Recap of where we left off

Where the previous lecture stopped: we'd defined trees as algorithmic models that split the predictor space into rectangles, then assign each region a constant (mean for regression, majority vote for classification). Single trees are weak; ensembles (forests, boosting) are how we make them work. Boosting is "another approach to making multiple trees and then combining them together to make a better prediction."

The prof re-derives the regression-tree boosting algorithm one more time as warm-up:

> "Your first prediction sucks. It's this F-hat here. We're just making our first prediction is like a nothing burger. Like it just doesn't work at all. Just zeros."

Then iterate: compute residuals $r = y - \hat f$, fit a tree $\hat f^b$ with $d$ splits to $(X, r)$, update $\hat f \leftarrow \hat f + \nu \hat f^b$, update residuals, repeat $B$ times. This is Algorithm 8.2 from ISL.

> "If you had a very weak eta… like 0.00001 then you'd need to do this a whole bunch of times before you get anywhere… whereas if it was bigger then you'd need fewer."

That trade-off between $B$ and $\nu$ recurs at the end of the lecture.

## Boosting more generally

The slides now lift the view. Both AdaBoost.M1 (binary classification, exponential loss) and the residual regression boosting are special cases of a general scheme: find $f$ as a sum of basis functions

$$f(x) = \sum_{m=1}^M \beta_m\, b(x;\gamma).$$

For trees, the basis functions are $T(x;\Theta) = \sum_j \gamma_j I(x \in R_j)$.

### Two sub-problems for one tree

For a single tree we want to minimize
$$\hat\Theta = \arg\min_\Theta \sum_{j=1}^J \sum_{x_i \in R_j} L(y_i, \gamma_j).$$

The optimization splits into:

1. **Find $\gamma_j$ given $R_j$**: easy. "Could just be getting the average, or it could be the majority vote for classification."
2. **Find the regions $R_j$**: hard. Done with a greedy algorithm in module 8. But now we'll use a "smoother approximation" that gives a function we can take gradients of.

The trick: replace the hard $L$ with a smooth $\tilde L$ to find $\tilde R_j$, then go back and use the original $L$ to set $\gamma_j$.

> "We're going to find the cute way, the cute function we can minimize to find these regions, right, these R-tildes, and then we're going to use those to find the gammas."

### Boosting = a sum of trees

For boosting we want
$$f_M(x) = \sum_{m=1}^M T(x;\Theta_m),$$
and at each step we solve
$$\hat\Theta_m = \arg\min_{\Theta_m} \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + T(x_i;\Theta_m)).$$

Two known special cases:

- **Squared-error loss → fit residuals.** "It was sort of saying, oh well, every new tree should be a correction to the old one and the best correction is to use the residuals. I mean that was like a very logical thing."
- **Exponential loss for $\{-1,1\}$ binary classification → AdaBoost.** Loss: $L(y,f) = \exp(-y f(x))$. The intuition: when $y$ and $f$ have the **same sign**, $-yf$ is a big negative number, $\exp(-yf) \approx 0$ (good). When they **disagree**, $-yf$ is a big positive number, $\exp(-yf)$ blows up. "It's really going to penalize this shit out of misclassified points."

> "These two models came out… in different articles and no one really realized that it was the same thing until like five years later. So they kind of kicked themselves for not realizing it sooner. Most things are very obvious after the fact."

Both fall out of the same scheme, and the unifying answer is gradient boosting.

## Steepest descent → gradient boosting

The connecting idea: **the residuals were estimating the gradient.** If we recognize that, we can boost any loss.

### Gradient descent recap

The prof draws the level-set picture (mountain, level rings, walking up the gradient at each step). Calculus tells you the direction; you just take a step that's small enough not to overshoot:

> "The beauty of calculus is that it really points you in the direction you should go. And that's an amazing feat and it's extremely useful."

(Personal aside about taking high-school calculus three times because he switched countries, Norway to Canada, and finally getting it the third time. Stripped.)

### From gradient to function update

Take the derivative of the total loss $L(f) = \sum_i L(y_i, f(x_i))$ with respect to **the function** $f$ at each $x_i$:

$$g_{im} = \left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = f_{m-1}}.$$

Step in the direction of $-\mathbf{g}_m$: $f_m = f_{m-1} - \rho_m \mathbf{g}_m$.

But we're not free to choose any function, as $f_m$ has to be a sum of trees. So we do the closest thing: **fit a tree to the negative gradient by least squares.**

$$\tilde\Theta_m = \arg\min_\Theta \sum_{i=1}^N (-g_{im} - T(x_i;\Theta))^2.$$

> "The nice thing is that this is a function that we can minimize. It's a nice, smooth function, right? It's a squared function. So we can actually move the parameters around in a nice way."

This gives us regions $\tilde R_{jm}$. Then we plug those regions back into the **original** loss to find $\gamma_{jm}$:

$$\hat\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in \hat R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma).$$

### Algorithm 10.3 (gradient tree boosting)

1. Initialize $f_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$.
2. For $m = 1$ to $M$:
   - (a) Compute negative-gradient pseudo-residuals $r_{im} = -[\partial L / \partial f(x_i)]_{f_{m-1}}$.
   - (b) Fit a regression tree to the $r_{im}$, giving regions $R_{jm}$.
   - (c) For each region, compute $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)$.
   - (d) Update $f_m(x) = f_{m-1}(x) + \sum_j \gamma_{jm} I(x \in R_{jm})$.
3. Output $\hat f(x) = f_M(x)$.

### The squared-error sanity check

Plug in $L(y,f) = \tfrac12(y-f)^2$. Derivative w.r.t. $f$: $-(y - f)$. Negative gradient: $y - f$, **the residual.**

> "The main conceptual idea is that basically when we were boosting the stuff for regression we were using the residuals which seemed like a hack that someone figured out. It was actually estimating the gradient. And so more generally you can just say, I let R be the gradient or the negative gradient, and then fit a tree to that negative gradient."

## Other losses for regression

Now the punchline: any loss works. Just plug a different $L$ into Algorithm 10.3 and re-derive the gradient.

### Why the quadratic loss is annoying

Two complaints the prof raises against squared-error:

1. **Doesn't push small things to zero.** Around 0 it's flat (bell-curve-ish), so "barely ever does anything become zero." Many small residuals just stay small.
2. **Massively over-weights outliers.** "If you have a number that's like 100, 100 squared is really big. So it really penalizes things that are far away."

Outlier scenario he uses: sensors recording $y$ and $x$, someone hits the table, three or four $y$ values are huge. Squared loss "is going to push hard" on these, bending your model to chase the noise.

### Absolute loss

$L(y,f) = |y - f|$. Penalizes outliers linearly. Trade-off:

- **Pro**: outliers hurt much less.
- **Pro/con**: it's a V at 0, so it pushes small residuals to zero → sparse fits. Maybe good (clean) or bad (kills weak learners that boosting wants).
- **Con**: the gradient is discontinuous at 0 (jumps from −1 to 1). "It's annoying."

### Huber loss

> "What it combines is it eliminates the quadratic thing for big numbers."

$$L(y,f) = \begin{cases} (y-f)^2 & |y-f| \le \delta \\ 2\delta|y-f| - \delta^2 & \text{otherwise.} \end{cases}$$

Quadratic near 0 (smooth, no V, doesn't aggressively zero things), linear for big errors (doesn't blow up on outliers). Has a piecewise definition but stitches together smoothly if you pick $\delta$ well.

The prof on the picture (slide with three loss curves):
- Squared error: shoots up to "really nasty" values for big errors.
- Absolute value: V-shape, discontinuous gradient at 0.
- Huber (green): smooth, looks quadratic near 0, linear far out. "The part of the squared loss here where it doesn't push everything to zero."

### Loss design as art

> "A lot of this kind of algorithmic work, which is kind of where machine learning fits in, is coming up with different loss functions that have the behaviors that you want… there's a lot of optimization theory to it and in many ways a lot of art."

He muses on people like Dario Amodei not having traditional degrees, and on how "the mathematical underpinning" often comes after the empirical discovery. Stripped further but flagged.

## Losses for classification

### Binomial deviance (binary)

For $y \in \{0,1\}$:
$$L(y,f) = -I(y=1)\log p(x) - I(y=0)\log(1-p(x)),$$
with $p(x) = e^{f(x)}/(1+e^{f(x)})$, the logistic / sigmoid function. Same shape as logistic regression. Plug into Algorithm 10.3, take the gradient, you have a classifier.

### Multinomial deviance (K classes)

$$L(y,f) = -\sum_{k=1}^K I(y=k) \log p_k(x), \qquad p_k(x) = \frac{e^{f_k(x)}}{\sum_{l=1}^K e^{f_l(x)}}.$$

The numerator gives "$f_k$ is the basis for your prediction. $e^{f_k}$ is the prediction of how probable a thing is, normalized by the sum of all the predictions" (softmax).

Take the gradient w.r.t. $f_k$:
$$-g_{ikm} = I(y_i = k) - p_k(x_i).$$

> "Indicator minus the class probability prediction, which ends up sounding very similar to residuals, right? Conceptually similar, which it should be because it is related to the gradient of the whole thing."

**Important practical detail**: for $K$ classes, you build **$K$ trees in each iteration of step 2**, one per class, each fit to its own gradient $\mathbf{g}_{km}$. In step 3 the class probabilities are aggregated across the trees.

## Three ingredients, four hyperparameters

The prof recaps — these are the headline structures.

### Three ingredients of gradient boosting

1. **Weak learners**, small trees that are intentionally not strong:
   > "Weak learners sound like a weird thing to want, but if you remember, the thing we really don't want to do is overfit. So weak learners won't overfit. And also, in the gradient descent picture, we don't want to make that huge jump."
2. **A loss function** (quadratic, Huber, absolute, deviance, …).
3. **An additive way to combine them**: sum of trees with shrinkage.

### Four hyperparameters

**Tree (weak-learner) hyperparameters:**

a) **Tree depth**: how many splits per tree. Shallow trees → weak learners (good). Too shallow → does nothing. Empirically, $4 \le J \le 8$ leaves works (Elements book).

b) **Minimum observations per terminal node**: needed so each leaf has enough samples for a stable mean / majority vote estimate. "If you only have one sample, one value for that sample, then the mean of that value is that thing. You're not getting a good average." Less critical for big datasets.

**Boosting (additive-model) hyperparameters:**

c) **Number of trees $M$**: controls how far we walk the gradient. Too small → underfit; too large → "you're going to just start fitting the noise."

d) **Learning rate $\nu$** (also written $\eta$, the prof keeps switching): shrinks the contribution of each tree. Small $\nu$ → robust but needs larger $M$. **Empirical rule: $\nu \le 0.1$**.

### Tree depth: discussion the prof flagged for the exam

> "This is like something that you should be prepared to think about and discuss in like an exam setting for example: like why would we want the trees not to be too deep?"

His own answer hooks back to the steepest-descent picture and the bias-variance trade-off:

> "We want weak learners… we don't want overly precise learners. We want learners that make good progress and a step in the right direction, but not ones that are going to throw you out into the, you know, too far or overfit. We're specifically trying to reduce variance in this setting."

Same depth across all trees is the norm now. Historically people grew big and pruned, but "they realized it didn't help as much as they wanted."

### $M$ and $\nu$ are coupled

Smaller $\nu$ ⇒ larger $M$ needed. Strategy: **fix $\nu$ at something small (≈ 0.1), then choose $M$ by early stopping** on a validation set or via cross-validation.

> "You want to monitor your prediction in held-out data, and then say, oh, well, now we're done. And so then we stop. And that's often referred to as **early stopping** in an optimization problem."

Note this is **different from random forests**: with RFs you can keep adding trees almost for free. With boosting, residuals eventually become noise and additional trees overfit.

## Ames housing demo

The R code on the slides (`gbm()` from the gbm package):

```r
ames_gbm1 <- gbm(Sale_Price ~ ., data = ames_train,
                 distribution = "gaussian",   # Gaussian = SSE = quadratic loss
                 n.trees = 3000, shrinkage = 0.1,
                 interaction.depth = 3, n.minobsinnode = 10,
                 cv.folds = 10, bag.fraction = 1)
```

Diagnostic plot from `gbm.perf(ames_gbm1, method = "cv")`: training error (black) keeps dropping; CV error (green) drops then rises, the classic U.

> "If I was plotting this myself, I would make this plot, and then I would make a second version of it where I throw away all of this really big shit. Because right now, the green one goes up again, but you can't see it because going from here to here is so big that this little uptick is like, what?"

I.e. zoom in past the first 500 trees to actually see the U.

The seed remark, also flagged:

> "It's funny when you can only get the result that the people show if you set the seed equal to like 1, 2, 3 because it shouldn't matter. You should get something very similar if you don't set the seed."

## Stochastic gradient boosting

A third regularization knob, on top of $M$ and $\nu$.

> "Instead of always using all of the training data, you take a subsample of the data."

Subsample — **without** replacement, distinct from bootstrapping which is with replacement. Build each tree on a different random subset (e.g. 50% of rows). The motivation is the same as for [[random-forest]]: by letting each weak learner see different data, the ensemble has more diversity and lower variance.

> "By subsampling or resampling the data every time, then using a random subsample… you're encouraging diversity. Because again, one nice way of reducing the variance is by making the models be very different and ensembling them together."

Bonus: less data per tree = faster training.

In `gbm()`, `bag.fraction = 0.5` does this. The Ames RMSE drops slightly (≈ 22600 → 22400 in the slide demo).

> "I don't know if that really was worth all the effort, but probably for somebody. I guess I would care about the $200."

### Variants

You can subsample:
- **Rows** before each tree (stochastic GBM, friedman2002).
- **Columns** before each tree (random-forest-style on variables).
- **Columns** before each split inside a tree.

> "I'm assuming in certain situations that would be useful. I'm sure at least one person got a PhD based off of that."

## XGBoost

> "There's XGBoost. I think this is the most popular one. It's cited like a billion times, and I think this is the one that everyone uses."

All the same boosting structure, but with several engineering and statistical tricks:

### Second-order gradients

> "Instead of just taking the first-order gradients… the distance we need to go in each direction of each gradient is actually given to us by the second-order information. That gives us our Newton method."

XGBoost uses a Taylor-series expansion to second order, keeping both gradient (direction) and Hessian (step size). Same algorithm structure, second-order $\tilde L$.

### Other tricks

- **Parallelization**: different CPU cores search different parts of the data when finding splits.
- **Approximate split search**: doesn't enumerate every possible split.
- **Regularization on leaves**:
  $$\Omega(f) = \gamma|T| + \tfrac12 \lambda\|w\|^2 \quad (\text{L2}), \qquad \Omega(f) = \gamma|T| + \alpha\|w\| \quad (\text{L1}).$$
  $|T|$ is the number of leaves (pruning), $w$ are leaf weights. Same idea as ridge / lasso, only here applied to the tree predictions. "All we needed was a cost function, and then we could apply our algorithm."
- **Pruning parameter $\gamma$**: controls tree depth via penalty.
- **Learning rate $\nu$**: same as before.
- **Dropout**: randomly drop some of the existing trees during training.

### Dropout: Hinton's idea

> "Dropout is a trick that came from Geoffrey Hinton, inspired by the brain. The idea is that you remove part of your model at different iteration steps so that the model doesn't become sensitive to one specific thing."

Brain analogy: get hit on the head, lose some neurons, but you can still think, because the brain has redundancy and no single neuron carries a memory. Dropout forces the same redundancy in the model. In boosting, this means later trees can take over what early trees did, so no early tree dominates.

### Result

On the Ames demo, XGBoost RMSE ≈ $20{,}000, meaningfully better than vanilla GBM (~$22,400).

> "You can see why people like it. But the key is the same as we talked about. It's all based on these weak learner trees. And then how do you get a bunch of trees such that they correct each other, such that they don't overfit, such that they combine to give you not only a model with a low bias, but also with a low variance. It's that low variance that really gets you down there."

## Hyperparameter tuning: a warning

Many hyperparameters → tuning is slow:

> "Tuning models sucks. That's what it's sort of saying here. It's good to practice it because it's a bit annoying."

He defers to the Boehmke & Greenwell HOML chapter for full tuning runs and to the upcoming exercise session.

## Wrap-up

Mention of more recent variants (LightGBM: leaf-wise growth + sampling by gradient size; CatBoost: categorical-feature handling), out of scope, only flagged. Not in the book, not on the test.

> "We stop here right at the end on slide 62."

Closing prompt to the class:

> "I recommend playing with these ideas. It's so easy to just kind of make your own. So you can go in and sort of say, oh, what if I want to change the way it creates the trees?"
