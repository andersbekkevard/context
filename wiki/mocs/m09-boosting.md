---
moc: 09-boosting
title: Boosting and Additive Trees
lectures: [L19, L20, L21]
isl-ch: 8
slides:
  - modules/9TreeBoosting/9TreeBoosting.md
exercises:
  - exercises/Exercise9/
tags:
  - moc
  - module/09-boosting
---

# Module 09: Boosting and Additive Trees

The third tree-ensemble flavor: grow many small trees **sequentially**, each one a small correction to the previous fit. The prof's load-bearing framing is "weak learners + a loss + an additive way to combine them"; the [[boosting]] / [[gradient-boosting]] pseudocode is explicitly **not** on the test, but the conceptual menu (AdaBoost, gradient boosting, $\nu$, stochastic GBM, XGBoost extras, PDPs) is.

## Lectures
- [[L19-boosting-1]]: wraps trees (bagging / OOB / RF / variable importance), then opens boosting with AdaBoost and the sequential / weak-learner intuition (Mar 17)
- [[L20-boosting-2]]: gradient boosting as steepest descent in function space, the loss menu, $\nu$, stochastic GBM, XGBoost extras (Apr 7)
- [[L21-unsupervised-1]]: opens with the XGBoost wrap and [[partial-dependence-plots|PDPs]] before pivoting to PCA (Apr 13)

## Concepts (atoms in this module)
- [[boosting]]: sequential weak learners; sum-of-trees model $f_M(x) = \sum_m T(x;\Theta_m)$; concept-only mechanics ("you don't need to memorize the pseudocode")
- [[adaboost]]: re-weight misclassified points each round, $\alpha$-weighted vote; turns out to be gradient boosting under exponential loss
- [[gradient-boosting]]: fit each new tree to the **negative gradient** of the loss; for squared-error this is just the residuals; algorithm 10.3 generalizes to any differentiable loss
- [[weak-learner-and-learning-rate]]: shallow trees + small $\nu \le 0.1$; the two "force everything to be small" knobs that keep boosting from overfitting; smaller $\nu$ → larger $M$
- [[stochastic-gradient-boosting]]: subsample rows (without replacement) and/or columns before each tree → diversity → variance reduction
- [[xgboost]]: 2nd-order (Newton) gradients + parallelism + L1/L2 leaf-weight regularization + dropout; "the one everyone uses"; internals out of scope
- [[boosting-loss-functions]]: any differentiable loss plugs into Algorithm 10.3 (quadratic / absolute / Huber / binomial & multinomial deviance / exponential)
- [[partial-dependence-plots]]: average $\hat f(x_j, X_{-j})$ over the data with $x_j$ fixed → claw back interpretability from tree ensembles

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; [[L20-boosting-2]] justifies weak-learner depth as variance control ("we're specifically trying to reduce variance in this setting")
- [[regularization]]: boosting stacks **multiple** regularizers: shrinkage $\nu$, stochastic subsampling, early stopping, plus XGBoost's L1/L2 leaf weights and pruning $\gamma$ ([[L20-boosting-2]])
- [[cross-validation]]: picks $M$ via early stopping on a validation curve; `gbm()`'s `cv.folds` and `xgb.cv` are the canonical hooks ([[L20-boosting-2]])

## Exercises
- [[../../exercises/Exercise9]]: Q1 ensembles + bagging vs boosting; Q2 explain $L(\cdot)$, $f_m(\cdot)$, $M$, $J_m$ in gradient tree boosting; Q3 interpret $\nu$; Q4 fit `gbm()` / stochastic GBM / XGBoost on simulated genomic data and tune

## Out of scope (this module)
- **Detailed boosting pseudocode (line-by-line)** - *"I won't have you memorize the names of the R functions, of course, but you should know what tree boosting is."* - [[L20-boosting-2]] / [[L27-summary]]
- **Stochastic gradient boosting / XGBoost / LightGBM internals** - mentioned as Kaggle winners, not derived - [[L20-boosting-2]]
- **CatBoost, LightGBM as separate algorithms** - name-checked only - [[L21-unsupervised-1]]
- **R / Python package names, function syntax, executable code** - *"no language, no memorizing package names, no language-specific coding"* - [[L27-summary]]

## ISLR pointer
Chapter 8 (§8.2.3 Boosting, §8.2.5 partial dependence plots, `book/08-trees.md`). Atoms carry section-level `isl-ref:` pointers; for the full forward-stagewise / Algorithm 10.3 derivation route Anders to the relevant ISLR section.
