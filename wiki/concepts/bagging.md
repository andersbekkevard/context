---
concept: bagging
module: 05-resample
lectures: [L11, L18, L19]
isl-ref: 8.2.1
exercises:
  - Exercise8.2d — bagging with 500 trees on Carseats, compute test MSE, importance()
  - Exercise8.3f — bagging on spam with B=500
related: [bootstrap, out-of-bag-error, random-forest, regression-tree, classification-tree, variable-importance]
tags:
  - concept
  - module/05-resample
aliases:
  - bootstrap aggregation
  - bootstrap aggregating
---

# Bagging (bootstrap aggregating)

The prof's preview of module 8 from inside module 5: same bootstrap trick, used not for uncertainty but **to build a better model.** Fit $B$ models on $B$ bootstrap samples, average them. Variance shrinks toward $\sigma^2/B$ if the models were IID — they're not (shared data → correlated), so the floor is $\rho\sigma^2$ for pairwise correlation $\rho$. The fix for the floor is [[random-forest|random forests]].

## Definition (prof's framing)

**B**ootstrap **AGG**regat**ING**. Draw $B$ bootstrap samples from the training data, fit a model on each, average the predictions (regression) or majority-vote (classification).

> "It's as simple as it sounds." — [[L19-boosting-1]]

> "You bootstrap your data, you fit a model, and then you average across all those models, and then magically that's a better model." — [[L11-resample-2]]

## Notation & setup

- Training data $(x_1, y_1), \dots, (x_n, y_n)$.
- For $b = 1, \dots, B$:
  1. Draw a bootstrap sample of size $n$ with replacement → $(x^*_{1,b}, y^*_{1,b}), \dots, (x^*_{n,b}, y^*_{n,b})$.
  2. Fit the model on it → $\hat f^{*b}$.
- Aggregate predictions:
  - **Regression:** $\hat f_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat f^{*b}(x)$
  - **Classification:** majority vote across $\{\hat f^{*b}(x)\}$, or average of class-probability estimates and pick argmax.
- $B$ is "use enough" — typically a few hundred. Not a tuned hyperparameter for bagging / RF (contrast with boosting).

## Formula(s) to know cold

**Bagged predictor (regression):**

$$\boxed{\hat f_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat f^{*b}(x)}$$

**Variance of an average of correlated predictions** (the slide deck and L19 derivation):

$$\boxed{\operatorname{Var}\!\Big(\tfrac{1}{B}\textstyle\sum_b \hat f^{*b}(x)\Big) = \rho \sigma^2 + \tfrac{1 - \rho}{B}\sigma^2}$$

where $\sigma^2 = \operatorname{Var}(\hat f^{*b}(x))$ and $\rho$ is the pairwise correlation between two bootstrap-trained predictors. If $\rho = 0$ (IID samples), this collapses to the textbook $\sigma^2/B$. If $\rho > 0$, the first term is a **floor that doesn't shrink with $B$** — motivates random forests, which decorrelate trees.

## Insights & mental models

- **The variance argument:** if the $B$ samples were genuinely independent, taking their average would reduce variance by $1/B$ — same as $\text{Var}(\bar X) = \sigma^2/n$. Bootstrap samples *aren't* independent (all drawn from the same data), so the actual reduction is less: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. *"It's not as good of a reduction as if it were independently sampled data, but it's still pretty good."* — [[L11-resample-2]]
- **Bagging shines for high-variance, low-bias models** — particularly trees ([[regression-tree]] / [[classification-tree]]). A single tree is *"data set sensitive"* (high variance); bagging trees fixes that.
- **The "infinite-IID-data" intuition:** in an ideal world with infinite IID datasets, if each model has variance $\sigma^2$, then the average across $B$ independent fits has variance $\sigma^2/B$. *"That's the motivation for ensembling. We don't have infinite data, so we bootstrap to fake it."* — [[L19-boosting-1]]
- **Bias reduction (less commonly mentioned):** the prof flagged that bagging can also *reduce bias*, not just variance — *"by doing that, even though you're always just resampling the same data, you can actually remove bias from your model"* — [[L11-resample-2]] — though he was clear the main and easiest argument is the variance one.
- **The "implicit bagging" aside:** very large-parameter single models (the regime where the bias-variance curve goes back *down* on the far right) end up implicitly bagging. *"It's super weird, but it's interesting."* — [[L11-resample-2]]. Reappears in [[double-descent]].
- **The compute-is-cheap framing:** the prof's running thread through module 5 + 8. *"In the age of compute… It's really not a problem to refit the model many times because hardware is cheap. I like it because it's super easy. And it just takes more compute, and compute's cheap, so who cares?"* — [[L11-resample-2]]
- **OOB error comes free.** Each bootstrap sample uses ~63.2% of the data; the ~36.8% **out-of-bag** observations form a per-tree validation set. Aggregate across trees → free test-set estimate. See [[out-of-bag-error]].
- **Interpretability is the cost.** A single tree is readable; an ensemble of 500 isn't. Recovered partially by [[variable-importance]] plots.

## Worked examples (from L18 / L19)

- **Head injury data, bagging with 500 trees.** Test misclassification *increased slightly* compared to the single-tree baseline (~0.15 vs. ~0.14). The prof noted: *"It didn't perform as well as we hoped. And considering that we made 500 trees, it's maybe a little disappointing. But it at least gets the idea across — we'll get to better versions."* — [[L19-boosting-1]] / [[L18-trees-2]]. Shows variance reduction is *possible*, not guaranteed for any given dataset.
- **Boston housing, regression.** Single tree: test MSE = 35. **Bagging** (`mtry = 13`, all predictors at every split): test MSE = 23.67. **Random forest** ($m = p/3$): test MSE = 18.9. Visible bagging improvement; random forest improves further by decorrelating.

## When bagging fails to help

Per the head-injury example: you can fit 500 bagged trees and not see a clear improvement. Why? **All 500 trees are highly correlated** — they tend to use the same dominant predictor for the root split. The $\rho\sigma^2$ floor dominates. **Random forests** address this by giving each split only a random subset of $m$ predictors, breaking the dominance of strong predictors and decorrelating trees.

## Exam signals

> "It's as simple as it sounds." — [[L19-boosting-1]]

> "If [the samples] were IID then we would have these really nice properties that the variance of our model would actually be reduced by the number of models." — [[L18-trees-2]]

> "You bootstrap your data, you fit a model, and then you average across all those models, and then magically that's a better model." — [[L11-resample-2]]

The variance formula $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ — the prof derived it on the board in L19. **Highly likely to appear** as either a T/F (does adding more trees always help?) or output-interpretation (compare MSE for 100 vs 1000 bagged trees).

## Pitfalls

- **Confusing bagging with boosting.** Bagging fits $B$ trees in **parallel** on **independent bootstrap samples**, then averages. Boosting fits $B$ trees **sequentially**, each correcting the previous one. Different goals (variance reduction vs bias reduction).
- **Thinking $B$ is a tuning parameter for bagging / RF.** It's "use enough" — pick a big number, no CV needed. (Contrast with boosting, where $B$ matters and overfits if too large.)
- **Forgetting the variance floor.** No matter how big you make $B$, the variance of a bagged predictor doesn't go below $\rho\sigma^2$. Solution: decorrelate the trees → random forests.
- **Bagging a low-variance model.** Bagging linear regression doesn't help much — the model is already low variance, so the average doesn't reduce error. Bagging is for high-variance learners (trees, KNN with small K, deep nets in some regimes).
- **Forgetting interpretability cost.** Single trees are interpretable; bagged ensembles aren't. Use [[variable-importance]] plots to recover some of it.

## Scope vs ISLR

- **In scope:** the algorithm, the variance formula and its floor, the bias-reduction footnote, the OOB connection, the compute-is-cheap framing, the worked examples, the interpretability cost, the connection to random forests as the variance-floor fix.
- **Look up in ISLR:** §8.2.1 (pp. 343–345) for the algorithm. Module 5 also previews bagging in §5 and §8.2 of the slides; full treatment in module 8.
- **Skip in ISLR (book-only, prof excluded):** detailed convergence analysis of $B \to \infty$, BART (Bayesian Additive Regression Trees, §8.2.4) — the prof skipped BART entirely.

## Exercise instances

- **Exercise8.2d** — apply bagging with 500 trees on the Carseats regression problem; compute test MSE; use `importance()` to identify the most influential predictors.
- **Exercise8.3f** — bagging on the spam classification problem with $B = 500$; compute misclassification rate.

(Bagging is also referenced in the L11 preview but no module-5 exercise drills it directly — the formal exercises live in module 8.)

## How it might appear on the exam

- **Pseudocode / equation-writing:** "Describe the bagging algorithm and write down the bagged predictor formula" → the recipe + $\hat f_{\text{bag}}(x) = \frac{1}{B} \sum_b \hat f^{*b}(x)$.
- **Conceptual / T/F:**
  - "Bagging always reduces test error" → false (depends on whether the model is high-variance and whether bootstrap samples produce diverse fits).
  - "Bagging fits $B$ trees in parallel; boosting fits them sequentially" → true.
  - "Increasing $B$ in bagging will eventually drive variance to zero" → false; floor is $\rho\sigma^2$.
- **Output interpretation:** given test MSE for single tree, bagging, and random forest, explain why RF > bagging > single tree (variance reduction + decorrelation).
- **Method comparison:** compare bagging and random forests; explain how RF improves on bagging (variance floor argument).
- **Variance-floor derivation:** the $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ formula. Could be expected as a partial derivation: start from $\text{Var}(\frac{1}{B}\sum X_i)$, expand, separate diagonal from off-diagonal covariance terms.

## Related

- [[bootstrap]] — the underlying resampling trick (uncertainty quantification version)
- [[out-of-bag-error]] — bagging's free test-set estimate
- [[random-forest]] — the decorrelation upgrade that lifts the variance floor
- [[regression-tree]], [[classification-tree]] — the high-variance base learners bagging targets
- [[variable-importance]] — how to recover interpretability from a bagged ensemble
