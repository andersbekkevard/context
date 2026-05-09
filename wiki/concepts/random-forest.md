---
concept: random-forest
module: 08-trees
lectures: [L18, L19, L27]
isl-ref: 8.2.2
exercises:
  - Exercise8.2e  -  randomForest with mtry=3 (≈ p/3) and ntree=500 on Carseats; importance plot; effect of m on error
  - Exercise8.2f  -  sweep ntree, plot test MSE vs ntree for bagging vs RF
  - Exercise8.3g  -  random forest on spam with mtry=√57 ≈ 8 predictors per split, importance(), test misclassification
related: [bagging, regression-tree, classification-tree, variable-importance, out-of-bag-error, bias-variance-tradeoff]
tags:
  - concept
  - module/08-trees
aliases:
  - RF
  - Breiman random forests
  - decorrelated bagging
---

# Random forest

[[bagging]] + a small tweak that **decorrelates** the trees: at each split, only a random subset of $m$ predictors is considered. The variance-reduction floor of bagging ($\rho\sigma^2$) drops because $\rho$ drops. Defaults: $m = \sqrt{p}$ for classification, $m = p/3$ for regression.

## Definition (prof's framing)

> "Random forests provide an improvement over bagged trees by a small tweak that decorrelates the trees. As in bagging, we build a number of decision trees on bootstrapped training samples. But each time a split in a tree is considered, a random selection of $m$ predictors is chosen as split candidates from the full set of $p$ predictors. The split is allowed to use only one of those $m$ predictors.", slide deck

> "If you want to have a lot of opinions, don't clone the same person 50 times, they're just going to say the same thing. Figure out a way to add diversity into your pool." - [[L19-boosting-1]]

## Notation & setup

- $B$: number of trees in the ensemble (a use-enough number, *not* a tuning parameter).
- $p$: total number of predictors.
- $m$: number of predictors randomly available at each split. **A fresh subset is drawn at each split**, not once per tree.
- $\hat f^{*b}(x)$: prediction of the $b$-th tree.
- Final prediction: $\hat f_{\text{rf}}(x) = \frac{1}{B}\sum_b \hat f^{*b}(x)$ for regression; majority vote for classification.

**Bagging is the special case $m = p$**, every predictor available at every split.

## Formula(s) to know cold

**The correlated-trees variance derivation**, the prof did this on the board in [[L19-boosting-1]] and it's the canonical "why RF beats bagging" argument:

For $B$ predictors with common variance $\sigma^2$ and pairwise correlation $\rho$ (so $\text{Cov}(X_i, X_j) = \rho\sigma^2$ for $i \neq j$):

$$
\text{Var}\!\left(\frac{1}{B}\sum_{i=1}^B X_i\right)
= \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2
= \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

> [!important] The point of the derivation
> "This thing does not shrink with respect to $B$. No matter how many times we make different trees, if they're all correlated the same way, then there's a portion that just never improves." - [[L19-boosting-1]]

So variance reduction has a **floor at $\rho\sigma^2$**. To reduce that, you have to reduce $\rho$, i.e. **decorrelate the trees**. RF does this by restricting each split to a random subset of $m$ predictors; in some fraction of trees the strong predictor isn't even on the menu, so the tree starts with a different split and diverges thereafter.

**Default $m$ values** (memorize):
- **Classification**: $m \approx \sqrt{p}$
- **Regression**: $m = p/3$

(Smaller $m$ is preferred when predictors are highly correlated.)

## Insights & mental models

- **Why bagged trees are highly correlated.** If there's a strong predictor (e.g. `GCS.15` in the brain-injury data), every bootstrap sample picks it for the root split → every tree starts identically → $\rho$ is large → bagging delivers little variance reduction beyond what a single tree gives. Random forests force diversity at the algorithmic level, not just at the data-resampling level.
- **The decorrelation is per-split, not per-tree.** A fresh subset of $m$ predictors is drawn at every internal node of every tree. So even within one tree, different splits see different predictor pools, maximizes diversity.
- **$B$ is "use enough", not a tuning parameter.** The prof was emphatic on this, repeated across [[L18-trees-2]] and [[L19-boosting-1]]:

  > "It just has to be enough… as long as it's enough of them, you're fine. You don't typically estimate this, you don't typically run cross-validation, you can use the OOB error if you want, but really, it's just use enough of them." - [[L19-boosting-1]]

  The slide deck shows ISLP Figure 8.10: error vs $B$ saturates well before $B = 500$ in the gene-expression example. Picking too many trees never causes overfitting (variance just keeps shrinking); pick $B$ large enough that the OOB error has settled.

- **$m$ *is* a tuning parameter.** The hyperparameter that actually matters. The slide deck and lectures show $m = \sqrt p$ beating $m = p$ and $m = p/2$ on the gene-expression data, concrete evidence that decorrelation pays.
- **Use OOB error, not CV, for hyperparameter checks.** The [[out-of-bag-error|OOB]] sample (~1/3 of training points not in any given bootstrap sample) gives a free, validated test error per tree → no separate test set required. *"Increasing $B$ will not change the goodness of fit measure. To find out which number $B$ is sufficient, we do not need to run cross-validation, but can again use the OOB error."*, slide deck.
- **Boston worked example (regression).** The prof's canonical demo (slide deck + [[L19-boosting-1]]):
  - Single regression tree → test MSE = 35.
  - Bagging ($m = 13$) → test MSE = 23.67.
  - **Random forest ($m = p/3 \approx 4$) → test MSE = 18.9.**
  - Each step a clear improvement. RF wins.
- **Brain-injury / spam / Carseats classification examples.** Bagging often barely beats or even slightly *underperforms* the single tree on these (variance reduction is real but obscured by other noise). RF then beats both, *"this is currently our winner"* - [[L19-boosting-1]].
- **Bagging ⊂ RF.** If you set `mtry = p` you've recovered bagging, useful because the same `randomForest()` function does both.

## Hyperparameter cheat sheet

| Parameter | Role | Tune? | Typical |
|---|---|---|---|
| `ntree` ($B$) | Number of trees | No, "use enough" | 500–1000 |
| `mtry` ($m$) | Predictors per split | Yes (the only real one) | $\sqrt{p}$ classif / $p/3$ regr |
| (Tree depth / `nodesize`) | How deep each tree grows | Usually leave deep, default is fine | RF trees grown unpruned |

**Important:** unlike boosting, the trees in an RF are typically grown **unpruned**, variance reduction comes from averaging, not from individual-tree regularization.

## Exam signals

> "I would generally go with randomization one because it makes more sense, but both seem reasonable." - [[L19-boosting-1]] (re: variable importance, but characteristic, see [[variable-importance]])

> "It just has to be enough … you don't typically estimate this, you don't typically run cross-validation." - [[L19-boosting-1]]

> "We're adding more diversity to our trees so that it gets better." - [[L19-boosting-1]]

The 2025 exam Q7 reformulation in [[L27-summary]]:
> "What tree-based method would you use, and justify the parameters." → e.g. random forest, $m \approx \sqrt p$, $B$ "large enough", exactly the textbook answer the prof wants.

> "Different solutions are possible, here we describe a random forest. … Here we have to choose `mtry`, which should be ca $\sqrt p$, with $p = 22$ … so we can use 4, perhaps 5. The number of trees is not a tuning parameter, but the students should mention that it should be chosen 'large enough'.", exam-key in 2025 paper.

The same justification recurs in 2023 exam keys: *"ntrees is not a tuning parameter, so the students should not optimize it."*

## Pitfalls

- **Don't tune $B$.** The cardinal sin per the prof; old exam keys deduct 1P for "I chose `ntree=500` from the lecture defaults" without saying that $B$ is not a tuning parameter and you set it large enough that OOB error has settled.
- **`mtry` defaults are not interchangeable.** $\sqrt p$ for classification, $p/3$ for regression. Mixing them up loses easy points.
- **A fresh subset per split, not per tree.** Common confusion: students sometimes describe RF as "each tree only sees $m$ predictors", wrong. Each *split* sees a fresh $m$.
- **RF doesn't always beat bagging by a huge margin.** It dominates when there's a strong predictor; the gain shrinks when predictors are roughly equal.
- **OOB error is dependent on bootstrap structure**: there's a "strange dependency on the test error from your real error on your test error on how you sampled" ([[L18-trees-2]]), but it works in practice and the prof endorses it for tree ensembles.
- **Trees are unpruned in RF.** Don't prune individual trees, the ensemble averaging is the regularizer.

## Scope vs ISLP

- **In scope:** The decorrelation trick (random subset of $m$ predictors per split); default $m$ for classification vs regression; bagging as the $m = p$ special case; $B$ is not a tuning parameter; OOB error as the test-set substitute; the correlated-trees variance formula and what it means; reading [[variable-importance]] plots from RF output.
- **Look up in ISLP:** §8.2.2 (random forests), §8.2.1 (bagging + OOB), Figure 8.10 (gene-expression $m$-comparison), Algorithm 8.2 in the boxes around here.
- **Skip in ISLP (book-only, prof excluded):** Detailed pseudocode for `randomForest` internals; extra-trees / extremely randomized trees; theoretical proofs of consistency.

## Exercise instances

- Exercise8.2e, `randomForest(Sales ~ ., ..., mtry = 3, ntree = 500, importance = TRUE)` on Carseats; compute test MSE; `importance()` and `varImpPlot()`; compare to bagging.
- Exercise8.2f, Sweep `ntree` from 1 to 500, plot test MSE vs ntree for both bagging (`mtry = 10`) and RF (`mtry = 3`); show the curve flattens, RF typically below bagging.
- Exercise8.3g, Random forest on `spam` with `mtry = round(sqrt(57)) ≈ 8`; check `varImpPlot()` (e.g. `charExclamation`, `remove`, `charDollar` come out top); compute test misclassification.

## How it might appear on the exam

- **Justify-your-choice question** (the most common pattern, per old exam keys): "Use a tree-based method on this dataset. Justify your hyperparameters." → name RF, give $m \approx \sqrt p$ or $p/3$ with reason, mention $B$ is not tuned but should be "large enough", possibly check via OOB error.
- **Conceptual short answer:** "Why are random forests better than bagging?" → write down the $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ formula; explain the floor at $\rho\sigma^2$; explain that decorrelation reduces $\rho$.
- **MC / T/F:** "$B$ in random forests is a tuning parameter to optimize via CV" → False. "Increasing $B$ will lead to overfitting" → False. "RF reduces variance more than bagging when predictors are correlated" → True.
- **Read variable importance:** given a `varImpPlot` from an RF, identify the most important predictors and interpret in context. See [[variable-importance]].
- **Compute test error** from a confusion matrix produced by an RF, same arithmetic as for any classifier.
- **Trap:** Mixing up $m = \sqrt p$ vs $m = p/3$. Memorize both; tag each to the right task type.
- **Direction-of-effect trap:** "Smaller $m$ → less correlation between trees → lower variance" is the right direction; some students get it backwards because "less information per split sounds bad."

## Related

- [[bagging]]: the parent algorithm; RF = bagging + per-split predictor restriction. Bagging is RF with $m = p$.
- [[regression-tree]] / [[classification-tree]], the base learners. Both work as RF base trees.
- [[out-of-bag-error]]: the variance-free test-error estimate; the *de facto* validation set for RF.
- [[variable-importance]]: recover "which predictors matter" from an RF; impurity-based and randomization-based flavors. The prof prefers randomization.
- [[bias-variance-tradeoff]]: RF lives entirely on the variance-reduction side: low-bias deep trees, ensembled to crush variance.
- [[boosting]]: the alternative tree-ensemble strategy: sequential, bias-reduction, $B$ is a real tuning parameter.
- [[bootstrap]]: the underlying resampling mechanism.
