---
concept: stochastic-gradient-boosting
module: 09-boosting
lectures: [L20]
isl-ref: 8.2.3
exercises:
  - Exercise9.4b - h2o stochastic GBM with row/column subsampling, hyperparameter grid search
related: [boosting, gradient-boosting, weak-learner-and-learning-rate, xgboost, bagging, random-forest, regularization, bias-variance-tradeoff]
tags:
  - concept
  - module/09-boosting
aliases:
  - stochastic GBM
  - SGB
  - subsampled boosting
  - row-subsampling boosting
---

# Stochastic gradient boosting

A third regularization knob (on top of small trees and small $\nu$): subsample rows (and/or columns) of the training data **before each tree** is fit. Friedman 2002. Same idea as bagging / random forests applied inside boosting, diversity across trees lowers variance. Faster too, since each tree sees less data.

## Definition (prof's framing)

> "Instead of always using all of the training data, you take a subsample of the data… By subsampling or resampling the data every time, then using a random subsample… you're encouraging diversity. Because again, one nice way of reducing the variance is by making the models be very different and ensembling them together." - [[L20-boosting-2]]

Distinct from bagging in one detail the prof flagged: subsampling here is **without replacement**, not bootstrapping.

## Notation & setup

- **`bag.fraction`** (the gbm-package name): fraction of rows used per tree. Default `bag.fraction = 0.5`. The slide deck explicitly sets `bag.fraction = 1` for the deterministic baseline and `bag.fraction = 0.5` for the stochastic version.
- **`sample_rate`** (h2o name): same row-subsample fraction.
- **`col_sample_rate`** / **`colsample_bytree`**: column-subsample fraction (analog of random-forest's $m$); applied either before each tree or before each split.
- **Sub-fraction $\eta$** (Friedman 2002 notation): proportion of $N$ used per tree; "typical value $\eta = N/2$, but can be much smaller when $N$ is large" per the slides.

## Variants

The slides list three subsample patterns; all are in scope:

1. **Subsample rows** before creating each tree (vanilla stochastic GBM, what `gbm()` exposes via `bag.fraction`).
2. **Subsample columns** before creating each tree.
3. **Subsample columns** before considering each split inside a tree (the random-forest-style decorrelation; XGBoost exposes both column subsampling modes).

All three are available in `h2o` and `xgboost`; `gbm` only exposes #1.

## Insights & mental models

- **Bagging logic, applied iteration-by-iteration.** Bagging averages over $B$ models each fit on a different bootstrap sample. Stochastic GBM does the same thing inside the boosting loop, each new tree sees a different subsample, so the trees are less correlated, ensemble variance drops.
- **A regularizer alongside small trees and small $\nu$.** The prof groups it as "regularization strategy 3" (slide 41), making the menu:
  1. Number of trees $M$: early stopping (regularization 1).
  2. Learning rate $\nu$: shrinkage (regularization 2).
  3. Stochastic subsampling: bag.fraction (regularization 3).
  4. (XGBoost adds) L1/L2 leaf penalties + dropout (regularizations 4–5).
- **Free speed-up.** Each tree sees less data → faster fit. Especially useful when $N$ is large.
- **The actual gains are often modest.** On the Ames demo:
  > "I don't know if that really was worth all the effort, but probably for somebody. I guess I would care about the $200." - [[L20-boosting-2]]
  RMSE dropped from ≈ 22600 to ≈ 22400, small but measurable.

## Worked example: Ames housing (slide deck)

```r
ames_gbm2 <- gbm(Sale_Price ~ ., data = ames_train,
                 distribution = "gaussian",
                 n.trees = 3000, shrinkage = 0.1,
                 interaction.depth = 3, n.minobsinnode = 10,
                 cv.folds = 10,
                 bag.fraction = 0.5)   # ← the only change
```

CV-RMSE drops from $22{,}600 (deterministic) to $22{,}400 (stochastic). Marginal but free.

## Exam signals

The prof did not flag stochastic GBM as having a dedicated exam question, but it is in the slides + lectures + exercises (so in scope per the rule), and "row/column subsampling reduces variance" is exactly the kind of conceptual T/F or short-answer the exam likes.

> "If it was covered either in the slides or in the exercises, then I would say fair game." - [[L27-summary]]

## Pitfalls

- **Confusing the two distinct subsample dimensions.** Rows = bag.fraction (which observations the tree sees). Columns = colsample_bytree / col_sample_rate (which predictors the tree can split on).
- **Calling it "boosting + bootstrapping."** Subsampling here is **without replacement**, not bootstrapping. Friedman 2002's distinction.
- **Setting `bag.fraction = 1` and expecting variance reduction.** With `bag.fraction = 1`, every tree sees all the data → no diversity from this knob → you've turned off this regularizer. (The slide deck's "regular" `gbm()` baseline has `bag.fraction = 1`.)

## Scope vs ISLR

- **In scope:** the concept (subsample rows / columns before each tree → diversity → variance reduction), the bag-fraction parameter, the three variant subsample patterns, the bagging-logic-applied-inside-boosting framing.
- **Look up in ISLR:** §8.2.3 doesn't really cover stochastic GBM as a separate variant; the slides reference Boehmke & Greenwell's HOML chapter (https://bradleyboehmke.github.io/HOML/gbm.html) for the deeper treatment. Anders does **not** need this for the exam.
- **Skip in ISLR (book-only / out of scope):**
  - **Friedman 2002 derivations** of why subsampling specifically helps: out of scope per [[L20-boosting-2]].
  - **Subsampling-fraction tuning with grids**: exercise material, not exam material.
  - **LightGBM's gradient-based row sampling**: name-checked only ([[L20-boosting-2]] / [[L21-unsupervised-1]]); out.

## Exercise instances

- Exercise9.4b: explain what the h2o stochastic-GBM grid search does. Key elements:
  - `sample_rate = c(0.2, 0.5)`: row subsample fractions to grid over.
  - `col_sample_rate = 0.1`: column subsample per split.
  - `col_sample_rate_per_tree = 0.1`: column subsample per tree.
  - `learn_rate = 0.05`, `max_depth = c(3, 5, 7)`, `ntrees = 10000`, `min_rows = 10`.
  - Random-discrete grid search over the combinations, with early stopping by MSE.

## How it might appear on the exam

- **True/false**: "stochastic gradient boosting reduces variance by subsampling rows before each tree" (true), "stochastic gradient boosting bootstraps the training data with replacement" (false, without replacement), "subsampling columns inside boosting is the same idea as random forest's $m$" (true).
- **Conceptual short-answer**: "name three regularization strategies in gradient boosting." Expected: small $\nu$, early stopping on $M$, stochastic subsampling (and optionally L1/L2 leaf penalties from XGBoost).
- **Method comparison**: given two GBM test errors (one deterministic, one stochastic), explain why the stochastic one is typically lower (variance reduction via per-tree diversity).

## Related

- [[boosting]]: the parent concept; stochastic GBM is one of the regularization knobs.
- [[gradient-boosting]]: the algorithm stochastic GBM modifies (just resample inputs to step 2(b)).
- [[weak-learner-and-learning-rate]]: the *other* two regularization knobs (small trees, small $\nu$).
- [[xgboost]]: exposes both row- and column-subsampling natively.
- [[bagging]]: the original "subsample → diversity" idea, applied at the ensemble level rather than per-iteration.
- [[random-forest]]: same column-subsample logic ($m < p$ predictors per split); SGB just applies it per-tree-or-per-split inside boosting.
- [[regularization]]: subsampling is just one more regularizer in the menu.
- [[bias-variance-tradeoff]]: subsampling-induced diversity reduces ensemble variance.
