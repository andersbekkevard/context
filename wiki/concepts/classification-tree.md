---
concept: classification-tree
module: 08-trees
lectures: [L18]
isl-ref: 8.1.2
exercises:
  - Exercise8.3c  -  fit tree() on the spam dataset, summary, plot, count terminal nodes
  - Exercise8.3d  -  predict on test data, compute misclassification rate
related: [regression-tree, cost-complexity-pruning, random-forest, variable-importance, confusion-matrix, bagging, cross-validation, bias-variance-tradeoff]
tags:
  - concept
  - module/08-trees
aliases:
  - CART classification
  - Gini split
  - cross-entropy split
---

# Classification tree

Same CART skeleton as the [[regression-tree]] , recursive binary splitting, build out, prune back , but with two changes: the prediction rule and the split criterion. The prof's load-bearing prof-flagged asymmetry: **train splits on Gini or cross-entropy, but prune on misclassification.**

## Definition (prof's framing)

> "Now allow for $K \ge 2$ number of classes for the response. Building a decision tree in this setting is similar to building a regression tree for a quantitative response, but there are two main differences: the prediction and the splitting criterion." , slide deck

**Prediction**: in region $R_j$, either (a) predict the **majority class** of training observations in that region, or (b) report the estimated class proportions $\hat p_{jk}$ as soft predictions.

**Splitting criterion**: minimize an **impurity measure** ([[gini-index|Gini]] or cross-entropy / deviance), not RSS.

## Notation & setup

- $K$: number of classes ($K = 2$ for binary).
- $R_j$: terminal region $j$, with $N_j$ training observations.
- $\hat p_{jk}$: proportion of class $k$ training observations in region $R_j$:
  $$\hat p_{jk} = \frac{1}{N_j}\sum_{i: x_i \in R_j} \mathbf{1}(y_i = k) = \frac{n_{jk}}{N_j}$$
- Hat = estimate, $k$ = class index, $j$ (or $m$) = region, $i$ = observation.

## Formula(s) to know cold

**Prediction (majority vote):**
$$\hat y(x) = \arg\max_k \hat p_{jk} \quad \text{for } x \in R_j$$

**Three impurity measures** for region $R_j$:

- **Misclassification error**: $E = 1 - \max_k \hat p_{jk}$
- **Gini index**: $G = \sum_{k=1}^K \hat p_{jk}(1 - \hat p_{jk})$
- **Cross-entropy / deviance**: $D = -\sum_{k=1}^K \hat p_{jk} \log \hat p_{jk}$

(R `tree`'s `split="deviance"` ≡ cross-entropy; `split="gini"` ≡ Gini index. The "deviance" reported is $-2\log L = -2\sum_k n_{jk}\log\hat p_{jk}$, a scaled cross-entropy.)

All three are zero on a pure node ($\hat p_{jk} = 0$ or $1$ for all $k$); Gini and cross-entropy are larger than misclassification on impure-but-classified-the-same nodes, which is why they're preferred for splitting.

## Insights & mental models

### The asymmetry: train Gini/entropy, prune misclassification

> [!important] The prof's prof-flagged trick
> "This is actually an interesting note that you build out the tree using deviance or Gini index but then you actually prune it using misclassification criteria … because that's ultimately what you care about. So it seems weird that you would not train on misclassification error but it turns out to be better to train on the Gini index or the deviance and then prune with misclassification." - [[L18-trees-2]]

In R: grow with `tree(..., split="deviance")` or `split="gini"`, then `cv.tree(..., FUN = prune.misclass)` for the pruning step.

### Why not split on misclassification?

Two reasons (the prof flagged the impurity-sensitivity argument as the load-bearing one):

1. **Impurity sensitivity.** Misclassification "isn't very sensitive to the right thing." The slide deck's two-class example: parent with (400, 400) splits two ways, both giving 25% misclassification, but one splits into (100,300) + (300,100) and the other into (200,400) + (200,0). Both have the same misclassification, but the second produces a **pure node** , Gini and entropy reward this; misclassification can't see the difference. *"They penalize having impure probabilities or impure classifications."* - [[L18-trees-2]]
2. **Differentiability.** Gini and cross-entropy are smooth functions of the $\hat p_{jk}$'s; misclassification has a kink at $\hat p = 0.5$. Useful for gradient-style optimizers. The prof flagged this as weaker: *"the reality is now actually [misclassification] would be also pretty easy to make differentiable. Like … we could just do a soft max."* - [[L18-trees-2]]

### "Useless-looking" splits that classify both children the same

A subtle observation also visible in ISLR Figure 8.6: sometimes a split puts class A on both children , the classification doesn't change, but the impurity drops. The example the prof worked through: parent (70A, 30B, Gini ≈ 0.42) → left (40A, 5B, Gini ≈ 0.20) + right (30A, 25B, Gini ≈ 0.50). Both children predict A, but the left one predicts A with much higher confidence. *"It put kind of like the shitty part of the parent node in one side and the more confident version on the left side."* - [[L18-trees-2]]

This is also why early stopping on misclassification breaks: the split looks "useless" by the misclassification criterion but enables a great later split. See [[cost-complexity-pruning]].

### Brain-injury / spam canonical examples

Slide deck and exercises both use a brain-injury classification (1321 patients, 10 categorical predictors like `amnesia`, `bskullf`, `GCS.15`, `vomit`, `age`). The Gini-grown tree is "very bushy"; the cross-entropy-grown tree slightly cleaner. Test misclassification: 0.14 (deviance) vs 0.16 (Gini) , *"slightly worse but not very different."*

Exercise 8.3 uses the `spam` email dataset (4601 emails, 57 numeric predictors describing word frequency / capitalization / punctuation; binary `spam` vs `nonspam`). After `tree()` + summary + plot you read off the terminal-node count; then test misclassification rate.

## Exam signals

> "This is actually an interesting note that you build out the tree using deviance or Gini index but then you actually prune it using misclassification criteria … because that's ultimately what you care about." - [[L18-trees-2]]

> "The misclassification error is not sufficiently sensitive for tree growing." , slide deck

> "The Gini index and Entropy are measure of impurity. They are more sensitive to changes in the node probabilities." , slide deck

The 2025 exam Q7 reformulation in [[L27-summary]]: "If I gave you the [[confusion-matrix]], you could estimate sensitivity and specificity from it" , fair-game for any classifier including a classification tree.

## Pitfalls

- **Train criterion vs. prune criterion.** Don't say "we use misclassification to grow the tree." We don't , we grow on Gini or cross-entropy because misclassification is too coarse, then prune on misclassification because that's what we care about for prediction.
- **Both children predicting the same class is fine.** It's not a bug , the split improved purity even though the labels didn't change.
- **`split="deviance"` is cross-entropy, not Gaussian deviance.** The slide deck spells this out: $\text{deviance} = -2\sum_k n_{jk}\log\hat p_{jk}$, a scaled cross-entropy.
- **K-class generalization.** Both Gini and cross-entropy generalize to $K > 2$ by summing over all classes. Don't restrict the formula to binary.
- **Categorical predictors with many levels** are dangerous: $q$ unordered levels → $2^{q-1} - 1$ possible partitions , high variance in tree structure. The slide deck recommends ordering levels by class proportion (binary case) to convert to ordered, then split as continuous. *"Try to avoid predictors with very many levels!"* , slide deck.
- **Missing values:** drop them, mean-impute, or use **surrogate splits** (next-best split using another variable). Trees handle missing data better than most methods. (Surrogate splits: noted, not exam-detailed.)

## Scope vs ISLR

- **In scope:** Gini and cross-entropy formulas; majority-vote prediction; the "train Gini/entropy, prune misclassification" asymmetry; why misclassification fails as split criterion; reading a fitted classification tree; computing test misclassification from a [[confusion-matrix]]; Algorithm 8.1 with impurity replacing RSS.
- **Look up in ISLR:** §8.1.2 (classification trees, Heart-data example, Figure 8.6).
- **Skip in ISLR (book-only, prof excluded):** Surrogate-split internals; detailed handling of multi-level categorical predictors beyond the "convert to ordered" trick.

## Exercise instances

- Exercise8.3c: Fit `tree(type ~ ., spam, subset=train)`, run `summary()`, plot, count terminal nodes.
- Exercise8.3d: `predict(...,type="class")`, build the confusion matrix on the test split, compute misclassification rate.

(Exercises 8.3e–g extend to pruning, bagging, and random forests , see [[cost-complexity-pruning]], [[bagging]], [[random-forest]].)

## How it might appear on the exam

- **MC / T/F:** "Gini is differentiable, misclassification is not" → True. "We use misclassification to choose the splits" → False (we use Gini or cross-entropy).
- **Short conceptual:** "Why is misclassification a poor splitting criterion?" → impurity insensitivity; reproduce the (100,300) vs (200,0) example.
- **Read a tree:** given a small classification tree on a categorical/binary predictor set, predict the class for a new observation by tracing the path.
- **Compute Gini / cross-entropy:** small region with $\hat p$ given, compute $G$ or $D$ by hand. Calculator-friendly.
- **Confusion matrix from prediction table:** test misclassification = off-diagonal / total.
- **Trap:** Conflating "node makes a prediction" with "node is pure" , a node can predict class A while still containing many class-B observations.

## Related

- [[regression-tree]]: same algorithm, RSS instead of impurity, region mean instead of majority vote.
- [[cost-complexity-pruning]]: the back-prune step; for classification, switch the cost function to misclassification.
- [[confusion-matrix]]: how to read errors and downstream metrics ([[sensitivity-specificity]], [[roc-auc]]) off a classification tree's predictions.
- [[bagging]]: average $B$ trees (majority vote), reduce variance.
- [[random-forest]]: bagging + per-split predictor restriction; default $m \approx \sqrt p$ for classification.
- [[variable-importance]]: recover "which predictors matter" from an ensemble of classification trees.
- [[bias-variance-tradeoff]]: same story as for regression trees: low bias, high variance, ensembles attack variance.
- [[cross-validation]]: `cv.tree(..., FUN=prune.misclass)`, default 10-fold.
