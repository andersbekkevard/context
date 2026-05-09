---
concept: variable-importance
module: 08-trees
lectures: [L18, L19, L27]
isl-ref: 8.2.1
exercises:
  - Exercise8.2d - importance() output for bagged Carseats; identify top predictors (Price, ShelveLoc)
  - Exercise8.2e - importance() and varImpPlot() for the Carseats random forest
  - Exercise8.3g - importance() and varImpPlot() for spam random forest (charExclamation, remove, charDollar top)
related: [bagging, random-forest, regression-tree, classification-tree, out-of-bag-error]
tags:
  - concept
  - module/08-trees
aliases:
  - importance plot
  - varImpPlot
  - permutation importance
  - mean decrease accuracy
---

# Variable importance

A way to claw back interpretability after you've ensembled a single tree into 500. Two flavors: **impurity-based** (mean decrease in Gini / RSS) and **randomization-based** (permute one predictor on OOB and measure performance drop). The prof prefers the randomization version *"because it makes more sense."*

## Definition (prof's framing)

> "Drawback of bagging: It becomes difficult to interpret the results, because instead of having just one tree, which is very easy to read, you have like 10, right? And then you often want to bring it back to what the variables actually are. Ideally consolidate the trees into one, but that's not always so easy." - [[L18-trees-2]]

So instead, sort predictors by their cumulative contribution across the ensemble. Two definitions of "contribution," both produced by R's `importance()` and visualized by `varImpPlot()`.

## The two flavors

### Flavor 1: node-impurity-based (mean decrease in Gini / RSS)

For each predictor $X_j$:

1. At every split that uses $X_j$, record the decrease in node impurity (RSS for regression, [[gini-index|Gini]] for classification).
2. Sum across all splits, average over the $B$ trees.

**Regression:** importance = total decrease in RSS due to splits on $X_j$, averaged over trees.

**Classification:** importance = total decrease in Gini due to splits on $X_j$, averaged over trees ("MeanDecreaseGini" in `randomForest` output).

R: `varImpPlot(..., type = 2)` or `importance(..., type = 2)`.

### Flavor 2: randomization-based (permutation importance, uses OOB)

For each predictor $X_j$, **using the [[out-of-bag-error|OOB]] sample**:

1. Run OOB observations through the ensemble; record performance (MSE / misclassification / accuracy).
2. **Permute** the values of $X_j$ across the OOB observations (whole column shuffled), keeping all other predictors and the trained trees fixed.
3. Re-run on the corrupted OOB; record the performance drop.
4. Average the difference across trees, optionally normalize by the SD of the differences.

> "If $X_j$ is important, permuting the observations will decrease the performance a lot. If it doesn't matter, then it won't matter." - [[L19-boosting-1]]

R: `varImpPlot(..., type = 1)` or `importance(..., type = 1)`. Reported as **MeanDecreaseAccuracy** for classification (or %IncMSE for regression).

## The prof's preference

> [!important] Prof's exam-relevant take
> "I don't have a strong preference here. I think I would generally go with randomization one because it makes more sense. But both seem reasonable." - [[L19-boosting-1]]

The reason for the preference: the impurity-based version is biased toward predictors with many possible split points (continuous predictors and high-cardinality categoricals tend to look artificially important because they have more chances to reduce impurity). Permutation importance avoids this, it directly tests whether knowing the predictor's value helps prediction. The prof didn't go that deep into the mechanism; the *"makes more sense"* line is what he flagged.

Both are fair to use on the exam. Saying "I prefer permutation importance because it doesn't depend on which split criterion was used and isn't biased toward many-level predictors" is a good answer.

## When the two disagree

Slide deck and [[L19-boosting-1]] showed both side-by-side on multiple datasets:

- **Auto / `mtcars` (regression).** Both methods identify `wt` (weight) and `hp` (horsepower) as the dominant predictors. Order of the top two is preserved between methods. Magnitudes differ.
- **Brain-injury (classification).** Top-4 the same in both rankings (e.g. `GCS.15`, `bskullf`, `age`); within those four the order shuffles. Bottom predictor (`skull` something) is the same in both.

> "They do measure slightly different things, so you don't expect them to be the same. … They do capture different aspects of what's being measured." - [[L19-boosting-1]]

In practice: top predictors tend to agree, the middle of the rankings is where the methods diverge. A discrepancy is not a bug.

## Insights & mental models

- **Variable importance is interpretation, not selection.** Don't use it to drop predictors before retraining, the rankings are conditional on the model that's already fit. For variable selection, use [[lasso]] or wrap a CV around forward stepwise.
- **It's an "explainable AI" device.** *"This would fall into the category of like explainable AI in the sense that you have a model that you don't quite understand. Each tree is understandable. The combination of 500 trees is not."* - [[L18-trees-2]]
- **Both flavors generalize to boosting**, with boosting also having its own native importance via `summary(gbm.fit)` (that's the 2025 exam Q6c "look at variable importances" question, see [[L27-summary]]).
- **Standardized magnitudes vs ranks.** The plot shows the relative importance, often normalized to the maximum. Comparing across different models (or across two flavors of importance) is unreliable in absolute terms, read the *ranking*, not the numbers.
- **Why use it.** Bagging / RF / boosting destroy single-tree interpretability. Variable importance is the consolation prize: you get *which* predictors mattered without seeing *how* they're combined. (For "how", you'd need [[partial-dependence-plots]], covered in module 9.)

## Exam signals

> "I would generally go with randomization one because it makes more sense, but both seem reasonable." - [[L19-boosting-1]]

> "Each tree is understandable. The combination of 500 trees is not." - [[L18-trees-2]]

[[L27-summary]]'s 2025 Q6c walkthrough of boosting:
> "Influence/importance plots are fair game, know what they mean and where they come from, but you won't compute them."

The 2023 exam asked students to identify "which three variables are most important to predict default, according to an importance measure based on node purity?", direct question on impurity-based importance.

## Pitfalls

- **Don't conflate the two flavors.** "Mean decrease in Gini" ≠ "Mean decrease in accuracy." Different definitions, different output columns in `randomForest`.
- **Don't use importance for variable selection.** Removing the bottom predictors and refitting is a common mistake; you might be removing predictors that interact with kept ones, and importance ranks reflect the *fitted* model.
- **Impurity-based importance is biased toward continuous / many-level predictors.** Don't be surprised when a categorical with many levels shows up high, could be inflation, not signal. Permutation importance avoids this issue.
- **Importance ≠ effect direction.** A high-importance predictor could be positively or negatively associated with the response, and could enter through interactions. Importance only says "this matters"; for direction-of-effect see [[partial-dependence-plots]].
- **"Mean decrease accuracy" is for classification; "%IncMSE" is its regression analogue.** Both are randomization-based.
- **Reporting:** the bigger the bar, the more important. Don't reverse the axis.

## Scope vs ISLR

- **In scope:** Both flavors (impurity-based and randomization-based); the OOB-based permutation procedure; the prof's preference for randomization; reading a `varImpPlot` and identifying top / bottom predictors; using importance plots for tree ensembles (bagging, RF, and boosting).
- **Look up in ISLR:** §8.2.1 ("Variable Importance Measures" subsection); Figure 8.9 (Heart data importance plot).
- **Skip in ISLR (book-only, prof excluded):** SHAP / Shapley values for tree ensembles ([[L26-nnet-3]] mentions in passing as part of explainable AI; not exam material). Conditional importance / impurity-correction methods (research literature; not lectures or exercises).

## Exercise instances

- Exercise8.2d: `bag.Carseats <- randomForest(Sales ~ ., ..., mtry = 10, ntree = 500, importance = TRUE)`; `importance(bag.Carseats)`, `varImpPlot()`. Identify `Price` and `ShelveLoc` as top predictors.
- Exercise8.2e: `rf.Carseats` with `mtry = 3`; same pipeline; same two predictors come out top.
- Exercise8.3g: `rf.spam` with `mtry = round(sqrt(57)) ≈ 8`; `varImpPlot(rf.spam)` shows `charExclamation`, `remove`, `charDollar` at the top. *"This is as expected as these variables are used in the top splits in the classification trees we have seen so far."*

(Exercise 8.2d also bridges to [[bagging]]; 8.2e/g to [[random-forest]].)

## How it might appear on the exam

- **Read a variable-importance plot:** given a `varImpPlot` from an RF or boosting model, identify the top-3 predictors and interpret them in the data context.
- **MC / definition:** "Permutation-based variable importance is computed using which sample?" → OOB. "Mean decrease in Gini measures..." → total Gini reduction across all splits using that predictor, averaged over trees.
- **Conceptual short answer:** "Why does bagging hurt interpretability, and what device do we use to recover it?" → ensembling destroys single-tree readability; variable importance plots show predictor ranking.
- **Compare flavors:** "What's the difference between impurity-based and permutation-based importance? Which would you prefer?" → reproduce the prof's *"makes more sense"* line; mention permutation isn't biased toward many-level predictors.
- **Trap:** Using variable importance to drop predictors and refit. Defensible answer: no, variable importance is for interpretation, not selection.
- **Conceptual interpretation gotcha:** A predictor with high importance could enter only through interactions. Don't claim it's individually correlated with the response.

## Related

- [[bagging]]: the first ensemble method that motivated importance plots (lose single-tree interpretability → recover it via importance).
- [[random-forest]]: the canonical home of `importance()` / `varImpPlot()` in R.
- [[regression-tree]] / [[classification-tree]]: the base learners; the impurity-decrease the importance metric averages over.
- [[out-of-bag-error]]: what the permutation procedure runs on.
- [[partial-dependence-plots]]: the *direction-and-shape* sister to variable importance; together they recover most of what a single tree gave you. (PDP lives in module 9.)
- [[classification-tree|Gini index]]: the impurity for classification importance (no separate atom; covered inside the classification-tree atom above).
- [[boosting]]: same `varImpPlot` machinery applies, with the twist that boosting also produces `summary(gbm.fit)` influence values.
