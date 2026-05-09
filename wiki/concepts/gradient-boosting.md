---
concept: gradient-boosting
module: 09-boosting
lectures: [L19, L20, L21, L27]
isl-ref: 8.2.3
exercises:
  - Exercise9.2 - explain L(.), f_m(.), M, J_m in the gradient tree boosting algorithm
  - Exercise9.4a - fit a basic gbm() on simulated genomic data with cv.folds=10
  - Exercise9.4b - stochastic GBMs via h2o, hyperparameter grid search
  - Exercise9.4c - XGBoost via xgb.cv on the same data
  - Exercise9.4d - find the best model and fit it on full training data
related: [boosting, adaboost, weak-learner-and-learning-rate, stochastic-gradient-boosting, xgboost, boosting-loss-functions, partial-dependence-plots, regression-tree, bias-variance-tradeoff, regularization, cross-validation]
tags:
  - concept
  - module/09-boosting
aliases:
  - GBM
  - gradient boosted machine
  - gradient tree boosting
  - GBT
---

# Gradient boosting

The general boosting algorithm: at each step, fit a tree to the **negative gradient** of the loss with respect to the current ensemble's predictions, then add a shrunken version. The prof's punchline, fitting residuals in the squared-error case is **literally** taking a gradient-descent step in function space, and once you see this, the whole framework generalizes to any differentiable loss.

## Definition (prof's framing)

The compact verbal definition the prof landed on:

> "By updating according to the residuals, what really we were doing is actually fitting the gradient of the function that we're optimizing. We're estimating the direction we have to go in. We're updating the gradient that will get us into a better direction." - [[L19-boosting-1]]

The rephrasing one lecture later:

> "The main conceptual idea is that basically when we were boosting the stuff for regression we were using the residuals which seemed like a hack that someone figured out, it was actually estimating the gradient. And so more generally you can just say, I let R be the gradient or the negative gradient, and then fit a tree to that negative gradient." - [[L20-boosting-2]]

So gradient boosting = forward-stagewise additive modeling where each new tree is fit by least squares to the negative-gradient pseudo-residuals of any chosen loss.

## Notation & setup

- $L(y_i, f(x_i))$, the loss. Free choice (squared, absolute, Huber, deviance, exponential, …), see [[boosting-loss-functions]].
- $f_m(x)$, the boosted model after $m$ trees. $f_0 = \arg\min_\gamma \sum_i L(y_i, \gamma)$, a constant baseline.
- $g_{im}$, gradient of the loss at observation $i$ with respect to the function value $f(x_i)$, evaluated at the current ensemble:
  $$g_{im} = \left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f = f_{m-1}}.$$
- $r_{im} = -g_{im}$, the **negative-gradient pseudo-residuals**.
- $T(x;\Theta_m)$, the tree fit at step $m$, with regions $R_{jm}$ and leaf values $\gamma_{jm}$.
- $J_m$, number of terminal nodes in tree $m$ (typically 4–8; same across $m$).
- $\nu$, learning rate / shrinkage. See [[weak-learner-and-learning-rate]].

## Formula(s) to know cold

### The simple regression-tree algorithm (ISL Algorithm 8.2)

Concrete special case the prof teaches first ([[L19-boosting-1]]):

Initialize $\hat f(x) = 0$, $r_i = y_i$. For $b = 1, \dots, B$:

1. Fit a tree $\hat f^b(x)$ with $d$ splits to $(X, r)$.
2. Update $\hat f(x) \leftarrow \hat f(x) + \nu \,\hat f^b(x)$.
3. Update residuals $r_i \leftarrow r_i - \nu\, \hat f^b(x_i)$.

Output $\hat f(x) = \sum_b \nu \,\hat f^b(x)$.

### The general gradient tree boosting algorithm (Elements Algorithm 10.3)

The version that lifts the idea to any loss ([[L20-boosting-2]]):

1. Initialize $f_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$.
2. For $m = 1, \dots, M$:
   - (a) Compute $r_{im} = -[\partial L(y_i, f(x_i)) / \partial f(x_i)]_{f_{m-1}}$ for $i = 1, \dots, N$.
   - (b) **Fit a regression tree to the $r_{im}$ by least squares**, giving regions $R_{jm}$ for $j = 1, \dots, J_m$.
   - (c) For each region, compute $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, f_{m-1}(x_i) + \gamma)$.
   - (d) Update $f_m(x) = f_{m-1}(x) + \sum_j \gamma_{jm}\, \mathbb{1}(x \in R_{jm})$.
3. Output $\hat f(x) = f_M(x)$.

### The squared-error sanity check

Plug $L(y, f) = \tfrac12 (y - f)^2$ into 2(a):
$$-\frac{\partial L}{\partial f(x_i)} = y_i - f(x_i) \;=\; r_i \quad\text{(the residual)}.$$

So Algorithm 10.3 with squared-error loss reduces *exactly* to the regression-tree algorithm: fitting residuals = fitting the gradient.

## Insights & mental models

- **Residuals = gradient.** This is the prof's headline "ah-ha" of the module. Once you see it, every loss becomes pluggable, and that's why "gradient boosting" is the umbrella name for the family.
- **Steepest descent in function space.**
  > "The beauty of calculus is that it really points you in the direction you should go." - [[L20-boosting-2]]
  Gradient descent in parameter space says "move θ in the direction of $-\nabla_\theta L$." Gradient boosting in function space says "move $f$ in the direction of $-g$, but project that direction onto the space of trees by fitting a tree to it via least squares."
- **Two sub-problems decoupled.** Regions are found via a smooth proxy (least-squares fit to $-g$); leaf values are found via the *original* loss $L$. The prof: "we're going to find the cute function we can minimize to find these regions, right, these R-tildes, and then we're going to use those to find the gammas." ([[L20-boosting-2]])
- **Three ingredients** (the prof's recap):
  1. Weak learners (small trees).
  2. A loss function (free choice, must be differentiable).
  3. An additive way to combine them (sum + shrinkage).
- **Four hyperparameters** (the canonical exam-flagged structure):
  1. Tree depth (or number of leaves $J$), typically $4 \le J \le 8$.
  2. Minimum observations per terminal node, less critical for big data.
  3. Number of trees $M$, overfits if too large; tune by CV / early stopping.
  4. Learning rate $\nu$, typically $\le 0.1$. See [[weak-learner-and-learning-rate]].
- **$M$ and $\nu$ are coupled.** Smaller $\nu$ ⇒ larger $M$ needed. Standard strategy: fix $\nu$ small (≈ 0.1), then choose $M$ by early stopping on a held-out validation set.
- **Boosting reduces bias; variance reduction comes from** [[stochastic-gradient-boosting|subsampling]] **and** [[xgboost|XGBoost-style regularization]].

## Worked example, Boston housing (slide deck)

`gbm(medv~., distribution="gaussian", n.trees=5000, interaction.depth=4)`. Trains both with default $\nu = 0.001$ and with $\nu = 0.2$; "doesn't make a big difference" on this dataset. Test MSE comparable to RF / better than single trees / better than bagging.

## Worked example, Ames housing (slide deck)

`gbm(Sale_Price ~ ., distribution="gaussian", n.trees=3000, shrinkage=0.1, interaction.depth=3, n.minobsinnode=10, cv.folds=10)`. Diagnostic via `gbm.perf(..., method="cv")`: training error monotonically drops, CV error shows the U, drops sharply, hits a minimum, slowly creeps up. Pick $M$ = `which.min(cv.error)`. Switching `bag.fraction = 0.5` (stochastic GBM) trims the RMSE slightly.

## Exam signals

> "I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is." - [[L27-summary]]

> "We need to specify the number of trees, the depth, and the shrinkage. How do you determine good values?" - [[L27-summary]]

(Cross-validation is the answer he wanted on Q6c of the 2025 walk-through.)

> "It's probably because there's some non-linear interactions that weren't captured in the linear regression model." - [[L27-summary]]

(The kind of one-sentence interpretation the prof wanted when contrasting boosting test MSE with linear-regression test MSE on Boston.)

## Pitfalls

- **Confusing the regression-tree special case with the general algorithm.** The "fit residuals" recipe is just $L = \tfrac12(y-f)^2$ plugged into Algorithm 10.3. The general algorithm fits the **gradient**, which equals the residual only for squared error.
- **Treating $M$ as "use enough" the way you would for RF.** $M$ in boosting is a *real* tuning parameter, too large overfits.
- **Dropping the learning rate.** Without $\nu < 1$ each tree contributes its full predictions, the ensemble jumps to a near-perfect training fit immediately, no diversity, terrible generalization. The prof: "each step should be a small step in the right direction."
- **Picking $\nu$ and $M$ independently.** They're coupled; smaller $\nu$ requires larger $M$. The standard recipe: fix $\nu$ (≈ 0.1), tune $M$.
- **Mistakenly thinking gradient boosting only works for regression.** The same framework works for classification, just plug binomial / multinomial deviance into Algorithm 10.3 (see [[boosting-loss-functions]]).
- **Forgetting that for $K$-class classification you fit $K$ trees per round** (one per class, see [[boosting-loss-functions]]).

## Scope vs ISLP

- **In scope:** the conceptual chain residuals → gradient → general gradient boosting; the squared-error sanity check; the four hyperparameters and how $M$/$\nu$ are coupled; CV / early stopping for tuning; the contrast with bagging / RF; the practical demos (Boston, Ames); generalization to any differentiable loss; partial dependence plots for interpretation.
- **Look up in ISLP:** §8.2.3 (book's Algorithm 8.2, squared-error special case), pp. 343–347. For the general Algorithm 10.3 and the gradient/loss derivations, see Elements ch. 10 (Anders does not need this).
- **Skip in ISLP (book-only, prof excluded):**
  - **Detailed boosting pseudocode line by line** - [[L27-summary]]: "you should know what tree boosting is" but not the steps verbatim.
  - **BART** (book §8.2.4), never lectured.
  - **Heavy formal derivations of why steepest descent in function space works**: the prof gives the intuition, no proofs.

## Exercise instances

- Exercise9.2, explain the meaning of $L(.)$, $f_m(.)$, $M$, $J_m$ in the gradient tree boosting algorithm; what changing each does; how to choose them.
- Exercise9.4a, fit `gbm(y ~ ., distribution="gaussian", n.trees=3, shrinkage=0.1, interaction.depth=7, n.minobsinnode=10, cv.folds=10)` on simulated genomic data (the prompt also asks how to make it less computationally intensive, answer: subsample rows and/or columns).
- Exercise9.4b, see [[stochastic-gradient-boosting]] for the h2o stochastic-GBM grid search.
- Exercise9.4c, see [[xgboost]] for the `xgb.cv` setup.
- Exercise9.4d, find the best hyperparameter combination, then fit on the full training data with no CV.

## How it might appear on the exam

- **Conceptual short-answer (à la 2025 Q6c)**: "we need to specify $M$, depth, and $\nu$, how would you determine good values?" Expected answer: cross-validation (or early stopping), with $\nu$ fixed small and $M$ then chosen at the CV minimum.
- **Output interpretation**: given a `gbm.perf()`-style training-error-vs-CV-error plot, identify the optimal $M$ at the CV minimum, explain the U-shape (training error monotonically falls, CV error rises again past the minimum because residuals start to be noise).
- **Method comparison**: given a table of test MSEs (linear regression, GAM, boosting), say which performs best and why (boosting wins when there are non-linear interactions; per the 2025 Q6c walk-through).
- **Conceptual T/F**: "fitting trees to residuals is gradient descent in function space for squared-error loss" (true), "gradient boosting works only for squared-error loss" (false), "in $K$-class classification gradient boosting fits one tree per round" (false, $K$ trees per round).
- **Mathy-ish**: derive the negative gradient for squared-error loss and observe it equals the residual; or compute the gradient for binomial deviance and observe it equals "indicator − probability."

## Related

- [[boosting]]: the parent concept (forward stagewise additive modeling).
- [[adaboost]]: the historical predecessor; falls out as gradient boosting under exponential loss.
- [[weak-learner-and-learning-rate]]: the two boosting hyperparameters that aren't $M$ or depth.
- [[stochastic-gradient-boosting]]: adds row/column subsampling for variance reduction.
- [[xgboost]]: the engineering-tuned modern variant (second-order gradients, parallelism, leaf regularization, dropout).
- [[boosting-loss-functions]]: the menu of losses you can drop into Algorithm 10.3.
- [[partial-dependence-plots]]: the standard interpretability tool for the resulting tree ensemble.
- [[regression-tree]]: the building block (the per-iteration weak learner).
- [[bias-variance-tradeoff]]: boosting attacks bias by aggregating many low-variance weak learners; tuning $M$ trades bias for variance.
- [[regularization]]: $\nu$, early stopping, subsampling are all regularizers; XGBoost adds L1/L2 leaf penalties.
- [[cross-validation]]: the standard way to pick $M$.
