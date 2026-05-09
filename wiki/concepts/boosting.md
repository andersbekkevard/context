---
concept: boosting
module: 09-boosting
lectures: [L19, L20, L21, L27]
isl-ref: 8.2.3
exercises:
  - Exercise9.1  -  what is an ensemble method; main conceptual differences between bagging and boosting
  - Exercise9.2  -  explain L(.), f_m(.), M, J_m in the gradient tree boosting algorithm
related: [adaboost, gradient-boosting, weak-learner-and-learning-rate, stochastic-gradient-boosting, xgboost, boosting-loss-functions, partial-dependence-plots, bagging, random-forest, bias-variance-tradeoff, regularization, cross-validation]
tags:
  - concept
  - module/09-boosting
aliases:
  - tree boosting
  - forward stagewise additive modeling
  - sequential ensembling
---

# Boosting (forward stagewise additive modeling)

The prof's umbrella for the third tree-ensemble flavor: instead of growing many independent trees in parallel and averaging (bagging / RF , the variance-reduction angle), grow many small trees **sequentially**, each one a small correction to the previous fit. The framing he wants you to walk into the exam with is "weak learners + a loss + an additive way to combine them" , the pseudocode is explicitly *not* on the test.

## Definition (prof's framing)

> "Trees are actually grown sequentially. You build one predictor, and then using that predictor you build a refinement predictor, so you're always trying to improve over the last one… a simple version, then a simple version to correct the simple version, then a simple version to correct the simple version of the simple version." - [[L19-boosting-1]]

The clean general formulation he gave a week later:

> "We are looking for expansions that take the form $f(x) = \sum_{m=1}^M \beta_m b(x;\gamma)$… for trees, the basis functions are $T(x;\Theta)$." - [[L20-boosting-2]]

So boosting = **forward stagewise additive modeling**: add one tree (basis function) at a time, each chosen to most reduce a chosen loss on top of what's already there. The final model is a sum of $M$ trees:
$$f_M(x) = \sum_{m=1}^M T(x;\Theta_m).$$

## Notation & setup

- $f_m(x)$: the boosted model after $m$ trees (the "current ensemble"); $f_0 \equiv$ a constant baseline (often 0 or $\bar y$).
- $T(x;\Theta_m)$: the $m$-th tree, with parameters $\Theta_m = \{R_j, \gamma_j\}_{j=1}^{J_m}$ (regions + their constants).
- $J_m$: number of terminal nodes (leaves) in tree $m$. Same across all $m$ in modern usage; $4 \le J \le 8$ typical.
- $L(\cdot,\cdot)$: the loss function (squared error, exponential, deviance, Huber, …).
- $M$: total number of boosting iterations / trees.
- $\nu$ (or $\eta$ , prof switches) , the learning rate / shrinkage. See [[weak-learner-and-learning-rate]].

The prof uses $b$ and $m$ interchangeably for the iteration index; $\nu$ and $\eta$ interchangeably for the learning rate.

## Formula(s) to know cold

The single optimization at each stage:
$$\hat\Theta_m = \arg\min_{\Theta_m} \sum_{i=1}^N L\big(y_i,\; f_{m-1}(x_i) + T(x_i;\Theta_m)\big).$$

The decomposition the prof emphasized , find regions, then find leaf values:

1. **Hard sub-problem:** find regions $R_{jm}$ , done via a smooth proxy (the gradient , see [[gradient-boosting]]).
2. **Easy sub-problem:** given $R_{jm}$, set $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)$ , usually the leaf mean (regression) or majority vote (classification).

The two known special cases that fall out of this scheme:

- **Squared-error loss → fit residuals** (the original boosting-of-regression-trees recipe).
- **Exponential loss for $y \in \{-1,+1\}$ → AdaBoost** (see [[adaboost]]).

> "These two models came out… in different articles and no one really realized that it was the same thing until like five years later." - [[L20-boosting-2]]

## Insights & mental models

- **Boosting attacks bias, not variance.** This is the pivotal contrast with bagging / RF, which start with high-variance/low-bias trees and average them down. Boosting starts with high-bias/low-variance **weak learners** and combines many to reduce bias:
  > "Rather than performing variance reduction on complex trees, can we decrease the bias by only using simple trees?" - [[L19-boosting-1]]
- **Sequential, not parallel.** The number of trees $M$ is a real tuning parameter. With bagging/RF you "just use enough"; with boosting, too many trees overfits because residuals eventually become noise.
  > "In bagging and random forests you just pick a whole bunch of them. In this case, the number of models is going to be smaller." - [[L19-boosting-1]]
- **The "weak learners that build on each other" intuition.**
  > "It's very counterintuitive , instead of finding the perfect model ever to fit your data, you find a whole bunch of shitty models that build on each other so they're really diverse and different, and then average them. And then they work well. Super weird, but it does work." - [[L19-boosting-1]]
- **Three ingredients** that pin down any gradient-boosting variant:
  1. Weak learners (small trees).
  2. A loss function (any differentiable; see [[boosting-loss-functions]]).
  3. An additive way to combine them (sum + shrinkage).
- **Boosting + trees wins at Kaggle.** Especially for tabular / small-to-medium data. The prof's headline pitch:
  > "If you look on Kaggle… boosting tends to be higher performing than even the neural networks." - [[L19-boosting-1]]
  Caveat: he doesn't recommend boosting for very-large-data problems ("I don't think you'd want to build a large language model with trees").

## Exam signals

> "I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is." - [[L27-summary]]

> "We need to specify the number of trees, the depth, and the shrinkage. How do you determine good values?" - [[L27-summary]]

(Cross-validation is the answer the prof wanted.)

> "If it was covered either in the slides or in the exercises, then I would say fair game." - [[L27-summary]]

> "Boosted regression trees allow for $p>n$." , TMA4268 2024 exam (true/false fact about boosting versus NNs)

## Pitfalls

- **Conflating bagging with boosting.** Bagging trees are independent (parallel, averaged); boosting trees are sequential (each fits the residual / gradient of the previous fit). Same building block, fundamentally different ensembling logic.
- **"Just use lots of trees."** Wrong for boosting. With bagging/RF, $B$ "just has to be enough"; with boosting, $M$ overfits when too large , the prof is explicit on this distinction.
- **Memorizing pseudocode.** The prof said he won't ask you to recite Algorithm 8.2 / 10.3 line by line , know the *idea* and the *hyperparameters*, not the exact steps.
- **Treating boosting as variance reduction.** Boosting attacks bias (with weak learners that individually have low variance anyway). Variance reduction comes back in via [[stochastic-gradient-boosting]] and [[xgboost]] regularization, not the core algorithm.
- **Quoting the wrong number of hyperparameters.** Plain GBM has **4** (depth, min-obs-per-node, $M$, $\nu$). XGBoost adds more (L1/L2 leaf penalties, pruning $\gamma$, dropout). The exercise drills this.

## Scope vs ISLP

- **In scope:** the concept of boosting (sequential weak learners, forward stagewise, sum-of-trees model), bagging-vs-boosting contrast, the **three ingredients** + **four hyperparameters**, why $M$ matters here but not in RF, "use CV / early stopping to tune", AdaBoost as historical predecessor, gradient boosting as the unifying framework, partial dependence plots for interpretability.
- **Look up in ISLP:** §8.2.3 (concise treatment) and Algorithm 8.2 , pp. 343–348. Elements of Statistical Learning ch. 10 has the deep general treatment with the forward-stagewise framing; you don't need it for the exam.
- **Skip in ISLP (book-only, prof excluded):**
  - **Detailed boosting pseudocode line by line** - [[L27-summary]]: "you should know what tree boosting is" but not memorize pseudocode.
  - **CatBoost / LightGBM internals** - [[L20-boosting-2]] / [[L21-unsupervised-1]]: name-checked only.
  - **BART (Bayesian additive regression trees)**: book §8.2.4 not covered in lectures.

## Exercise instances

- Exercise9.1: define an ensemble method; spell out the main conceptual differences between bagging and boosting (parallel vs. sequential, independent vs. residual-driven, variance reduction vs. bias reduction, $B$ "use enough" vs. $M$ tuned).
- Exercise9.2: explain $L(.)$, $f_m(.)$, $M$, $J_m$ in Algorithm 10.3 , what each symbol means, what changing it does, how to choose it.

## How it might appear on the exam

- **Multiple choice / true-false**: "boosting trees are grown in parallel" (false), "boosting can overfit if $M$ is too large" (true), "boosting works for $p > n$" (true , as in 2024 exam Q1).
- **Short conceptual**: explain in 1–2 sentences how boosting differs from bagging, and why the number of trees is a tuning parameter for boosting but not for RF.
- **Hyperparameter explanation (à la 2025 Q6c)**: "we need to specify the number of trees, the depth, and the shrinkage. How would you determine good values?" Expected answer: cross-validation (or early stopping on a validation set), with $\nu$ fixed small.
- **Output interpretation**: given a `gbm.perf()`-style train/CV-error curve, identify the optimal $M$ (CV minimum), explain why training error keeps falling while CV rises.
- **Method-comparison reasoning**: "boosting got lower test MSE than linear regression on this dataset; why?" Expected answer: nonlinearities and interactions captured by additive trees that linear regression cannot represent.

## Related

- [[adaboost]]: the original (binary classification, exponential loss); special case of gradient boosting.
- [[gradient-boosting]]: the general algorithm; "boosting on residuals = gradient descent in function space."
- [[weak-learner-and-learning-rate]]: the two boosting hyperparameters that aren't $M$ or depth.
- [[stochastic-gradient-boosting]]: row/column subsampling on top of vanilla GBM.
- [[xgboost]]: the modern engineering-tuned variant.
- [[boosting-loss-functions]]: quadratic, absolute, Huber, deviance , what's available and why you'd pick each.
- [[partial-dependence-plots]]: how you claw interpretability back from a tree ensemble.
- [[bagging]]: the parallel-averaging ensembling baseline boosting is contrasted against.
- [[random-forest]]: bagging + decorrelation; the variance-reduction sibling.
- [[bias-variance-tradeoff]]: boosting's whole point is bias reduction via many weak learners; tuning $M$ trades bias for variance.
- [[regularization]]: $\nu$, early stopping, subsampling, leaf penalties are all boosting-specific regularizers.
- [[cross-validation]]: the tool for picking $M$ and the other hyperparameters.
