---
concept: xgboost
module: 09-boosting
lectures: [L20, L21]
isl-ref: 8.2.3
exercises:
  - Exercise9.4c - XGBoost via xgb.cv on simulated genomic data
  - Exercise9.4d - find the best XGBoost model and fit it on full training data
related: [boosting, gradient-boosting, weak-learner-and-learning-rate, stochastic-gradient-boosting, ridge-regression, lasso, regularization, nn-regularization]
tags:
  - concept
  - module/09-boosting
aliases:
  - eXtreme Gradient Boosting
  - XGB
---

# XGBoost

The state-of-the-art tree-boosting toolbox. Same gradient-boosting skeleton as `gbm()`, but with engineering and statistical tricks layered on top: **second-order gradients** (Newton, not just gradient), **parallelism**, **approximate split search**, **L1/L2 regularization on leaf weights**, a **pruning parameter $\gamma$**, and **dropout**. The prof's verdict: "this is the one that everyone uses." Internals are explicitly out of scope per [[L20-boosting-2]] / [[L27-summary]]; the exam-relevant content is the conceptual menu of additions and what they do.

## Definition (prof's framing)

> "There's XGBoost. I think this is the most popular one. It's cited like a billion times, and I think this is the one that everyone uses." - [[L20-boosting-2]]

> "It combines many tricks into a very efficient toolbox: parallelization, smart approximations, shrinkage, sub-sampling, etc." - slide deck (modules/9TreeBoosting/)

## What XGBoost adds on top of vanilla GBM

A flat list, exam-relevant level:

1. **Second-order gradients (Newton's method).** Vanilla gradient boosting uses a 1st-order Taylor expansion (gradient only); XGBoost uses 2nd-order (gradient + Hessian).
   > "Instead of just taking the first-order gradients… the distance we need to go in each direction of each gradient is actually given to us by the second-order information. That gives us our Newton method." - [[L20-boosting-2]]
   Effect: more accurate steps, fewer trees needed.
2. **Parallelization.** Tree-building (specifically the split-search) is parallelized across CPU cores. Different cores search different parts of the data.
3. **Approximate split search.** Don't enumerate every possible split point; bin features and search the bins. Speed-up at minimal accuracy cost.
4. **L1/L2 regularization on leaf weights**: a tree-boosting analog of [[ridge-regression|ridge]] and [[lasso]], applied to the leaf predictions $w$:
   $$\Omega(f) = \gamma|T| + \tfrac12 \lambda\|w\|^2 \quad (\text{L2}), \qquad \Omega(f) = \gamma|T| + \alpha\|w\| \quad (\text{L1}),$$
   where $|T|$ is the number of leaves, $w$ is the leaf-value vector.
5. **Pruning parameter $\gamma$.** Penalty per leaf, large $\gamma$ → smaller trees. Equivalent in spirit to [[cost-complexity-pruning|cost-complexity pruning]] for individual trees, applied here per boosting iteration.
6. **Dropout** ([[L21-unsupervised-1]], Hinton's idea). Randomly drop trees during training so no single tree carries all the weight.
7. **Same row/column subsampling** as [[stochastic-gradient-boosting]]: `subsample`, `colsample_bytree`.
8. **Same learning rate $\nu$** as [[weak-learner-and-learning-rate|vanilla GBM]]: `eta` parameter.

## Insights & mental models

- **Same algorithm, more engineering.** XGBoost doesn't change the gradient-boosting story; it just makes it faster and adds extra regularizers. If you understand [[gradient-boosting]] + [[stochastic-gradient-boosting]] + [[weak-learner-and-learning-rate]], you understand 90% of XGBoost.
- **Newton instead of gradient descent.** Second-order info means each step uses both the slope and the curvature of the loss → larger sensible step sizes → fewer total iterations needed. The prof's analogy was vanilla GBM = first-order gradient descent, XGBoost = Newton-Raphson.
- **L1/L2 on leaves is just ridge/lasso applied to a different parameter vector.**
  > "All we needed was a cost function, and then we could apply our algorithm." - [[L20-boosting-2]]
  Same penalty story from module 6, only here the parameters are leaf weights instead of regression coefficients.
- **Dropout, brain-redundancy analogy.**
  > "Dropout is a trick that came from Geoffrey Hinton, inspired by the brain. The idea is that you remove part of your model at different iteration steps so that the model doesn't become sensitive to one specific thing." - [[L20-boosting-2]]
  In boosting, removing a tree forces later trees to "fill the void" → no single early tree dominates the ensemble.
- **The Kaggle workhorse.** The slides flag XGBoost as "the basis for many competition-winning approaches." Tabular / small-to-medium / structured data.

## Worked example: Ames housing (slide deck)

```r
ames_xgb <- xgboost(
  data = X, label = Y,
  nrounds = 6000,
  objective = "reg:squarederror",
  early_stopping_rounds = 50,
  params = list(
    eta = 0.1,                  # learning rate (= nu)
    lambda = 0.01,              # L2 regularization
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,            # row subsample
    colsample_bytree = 0.5,     # column subsample
    nthread = 12
  ),
  verbose = 0
)
```

Resulting RMSE ≈ $20{,}000, beats vanilla GBM (~$22,400) by a meaningful margin on the same dataset.

> "You can see why people like it. But the key is the same as we talked about. It's all based on these weak learner trees… It's that low variance that really gets you down there." - [[L20-boosting-2]]

## Exam signals

> "I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is." - [[L27-summary]]

The prof never said "XGBoost-internals will be on the exam" and explicitly listed XGBoost / LightGBM internals among the things excluded:

> "Stochastic gradient boosting / XGBoost / LightGBM internals: mentioned as Kaggle winners, not derived." - `docs/scope.md` (paraphrasing the [[L20-boosting-2]] verbatim)

So: know **what XGBoost adds on top of vanilla GBM** at the conceptual level, but don't expect to derive 2nd-order Taylor expansions or write Newton-step pseudocode.

## Pitfalls

- **Treating XGBoost as a different algorithm.** It's the same forward-stagewise additive modeling, just with extra knobs and faster engineering. The conceptual story doesn't change.
- **Confusing the regularizers.** XGBoost has *many*: eta (learning rate), lambda / alpha (L2/L1 on leaf weights), gamma (per-leaf penalty / pruning), subsample (row), colsample_bytree (column), max_depth, min_child_weight, dropout. Don't mix them up; each does a different thing.
- **Forgetting that hyperparameters interact.** As the prof said:
  > "XGBoost is probably the first model in the course where we have a number of hyperparameters with weird interactions (regularization vs. pruning vs. learning rate)." - paraphrased from [[L21-unsupervised-1]]
  So tuning is genuinely hard, grid search or random search with early stopping is the practical recipe.

## Scope vs ISLR

- **In scope:** the conceptual menu of what XGBoost adds: second-order gradients, parallelization, L1/L2 leaf regularization, pruning $\gamma$, dropout, same subsampling/learning-rate machinery as vanilla GBM. Plus "this is the one that wins Kaggle."
- **Look up in ISLR:** ISLR doesn't really cover XGBoost; §8.2.3 talks generic boosting only. The slides reference Boehmke & Greenwell's HOML chapter for the deep treatment (https://bradleyboehmke.github.io/HOML/gbm.html); Anders does **not** need this for the exam.
- **Skip in ISLR (book-only / explicitly out of scope):**
  - **Second-order Taylor-expansion derivation**: out per the prof.
  - **Approximate split-search algorithms (histogram-based, weighted quantile sketch)**: out.
  - **Detailed regularization derivations**: concept yes, derivations no.
  - **CatBoost / LightGBM**: name-checked only.

## Exercise instances

- Exercise9.4c: `xgb.cv()` on the simulated genomic data with `eta = 0.05`, `max_depth = 3`, `min_child_weight = 3`, `subsample = 0.2`, `colsample_bytree = 0.1`, `nrounds = 6000`, `early_stopping_rounds = 50`, `nfold = 5`. Report `min(xgb$evaluation_log$test_rmse_mean)`.
- Exercise9.4d: sweep XGBoost hyperparameters (the prompt says "expand this code to perform a search of the hyperparameter space, similar to b)"), find the best CV-RMSE configuration, refit on the full training set with no CV.

## How it might appear on the exam

- **Multiple choice / true-false**: "XGBoost uses second-order gradients" (true), "XGBoost cannot do row subsampling" (false), "XGBoost adds L1 and L2 regularization on leaf weights" (true).
- **Conceptual short-answer**: "name two ways XGBoost differs from vanilla `gbm()`." Expected: second-order gradients, parallelization, leaf-level L1/L2 regularization, dropout, etc.
- **Method comparison**: given test MSEs from `gbm()` and `xgboost()`, which is typically lower and why (more regularizers + Newton steps → typically lower variance ensemble).
- **Conceptual link-back**: "where does the L2 regularization on leaf weights come from?" (Same idea as [[ridge-regression|ridge]] applied to leaf values instead of regression coefficients.)

## Related

- [[boosting]] — the parent concept.
- [[gradient-boosting]] — the algorithm XGBoost is the engineering-tuned version of.
- [[weak-learner-and-learning-rate]] — XGBoost still uses small trees + small $\nu$.
- [[stochastic-gradient-boosting]] — XGBoost natively exposes both row- and column-subsampling.
- [[ridge-regression]] / [[lasso]] — the L2 / L1 leaf-weight penalties are these in another costume.
- [[regularization]] — XGBoost is the densest concentration of regularizers in the course.
- [[nn-regularization]] — dropout originally Hinton's idea for neural networks; XGBoost imports it.
