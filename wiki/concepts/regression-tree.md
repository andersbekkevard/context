---
concept: regression-tree
module: 08-trees
lectures: [L17, L18, L19]
isl-ref: 8.1.1
exercises:
  - Exercise8.1a  -  explain the regression-tree fitting algorithm and what differs for a classification tree
  - Exercise8.2b  -  fit a regression tree to Carseats Sales, plot, interpret, compute test MSE
related: [classification-tree, cost-complexity-pruning, random-forest, variable-importance, bagging, cross-validation, bias-variance-tradeoff]
tags:
  - concept
  - module/08-trees
aliases:
  - CART regression
  - decision tree (regression)
  - recursive binary splitting
---

# Regression tree

The prof's first **algorithmic model** of the course (Breiman's two-cultures framing): chunk the predictor space into rectangles via greedy recursive binary splitting, predict the region mean. Captures interactions naturally, it's the whole point of moving past additive models.

## Definition (prof's framing)

> "We're going to chunk up our X's with rectangles such that the mean of the points within that rectangle predicts the activity well." - [[L17-trees-1]]

Two-step recipe (slide deck and ISLP Algorithm 8.1):

1. Divide the predictor space into $J$ non-overlapping regions $R_1, \ldots, R_J$.
2. For every observation falling in $R_j$, predict the **mean of the training $y$'s** in that region.

The algorithm is **CART** (Classification And Regression Trees, Breiman 1984). Splits are **greedy**, **binary**, and **on one variable at a time**.

> "This is our first example of a real algorithm for building up a model, which as opposed to the classic statistical approach of, oh, I'm going to assume that this variable is this distribution and I'm going to make a model of it that has this specific form, it said, no, we're going to make an algorithm. We're going to think of it like a computer program." - [[L18-trees-2]]

## Notation & setup

- $J$: number of terminal nodes (leaves / regions), denoted $|T|$ in the cost-complexity expression.
- $R_m$: region (rectangle) corresponding to terminal node $m$.
- $\hat y_{R_m}$: prediction in region $m$ = mean of training $y_i$ for $i \in R_m$.
- For a candidate split at variable $j$, threshold $s$:
  $R_1(j,s) = \{x : x_j < s\}$, $R_2(j,s) = \{x : x_j \ge s\}$.

## Formula(s) to know cold

**Splitting criterion at one node (greedy choice of $(j, s)$):**

$$
\min_{j, s}\Bigg[\sum_{i: x_i \in R_1(j,s)} (y_i - \hat y_{R_1})^2 \;+\; \sum_{i: x_i \in R_2(j,s)} (y_i - \hat y_{R_2})^2\Bigg]
$$

**sum** of squared deviations, not mean. The prof was emphatic:

> "We're summing up the squared difference between each point and the average of all the shit in its region. … I'm not taking any mean values here. If we did, we would have a different algorithm, and one that wouldn't behave very well." - [[L17-trees-1]]

**Total training loss across all regions** (the $Q(T)$ that goes into cost-complexity later):

$$
\text{RSS}(T) = \sum_{m=1}^{|T|} \sum_{i \in R_m} (y_i - \hat y_{R_m})^2
$$

## Insights & mental models

- **Greedy = locally optimal, globally suboptimal but tractable.** Exhaustive optimal partitioning is **NP-complete** (mentioned in [[L17-trees-1]] as background, *not* on the exam). At each step we pick the best split given everything done so far, and we never revise: *"trees don't go and change their mind and reconnect."* - [[L17-trees-1]]
- **One variable per split.** Same flavor as forward stepwise selection (Module 6), and easy to draw, easy to read. A tree path *automatically encodes interactions*: "I want to be those guys who are older than X *and* hit more than Y." - [[L17-trees-1]]
- **Trees as upside-down representations.** Root at top, internal nodes = splits, leaves = regions. Drawing scales to high $p$, region-rectangle plots only work in 2–3 dimensions.
- **Trees ≠ additive models.** When a region is split, the cut **does not extend** to the rest of the predictor space. If it did, you'd be back to the additive grid of GAMs. - [[L17-trees-1]]
- **Build out, then prune.** A maximal tree is high-variance and overfits. Don't try to stop early, see [[cost-complexity-pruning]] for why. The prof's headline: *"You're definitely doing the equivalent of overfitting where we put too many parameters in our model."* - [[L18-trees-2]]
- **Trees take the bias-reduction angle of the [[bias-variance-tradeoff|bias-variance]] story.** A deep tree is low-bias / very high variance; pruning trades bias for variance. Bagging/RF sit on top to attack the residual variance.
- **Hitters / ozone are the canonical worked examples.** Hitters: salary as a function of `years` and `hits`, interaction (years matters more if hits are also high). Ozone: tree splits on `temperature` three times then `wind` once; `radiation` is dropped entirely.

## Stopping criteria (R `tree` package)

Three knobs the slides showcase, used as **lax** stopping (you grow well past the eventual size, then prune):

- `mincut`: minimum observations in a child node for a split to be allowed.
- `minsize`: smallest node that's even considered for splitting.
- `mindev`: minimum proportional decrease in deviance (RSS for regression). Default 0.01 of the root deviance.

> "More importantly, this isn't how we just decide our final model. Typically, this is more criteria to just stop splitting. We're then going to find another way of determining our tree." - [[L17-trees-1]]

That "another way" is [[cost-complexity-pruning]].

## Tree vs. linear model

- **Linear regression**: $f(X) = \beta_0 + \sum_j X_j \beta_j$, one global functional form, smooth boundaries.
- **Regression tree**: $f(X) = \sum_{m=1}^{M} c_m \cdot \mathbf{1}(X \in R_m)$, piecewise constant on axis-aligned boxes.

Picture (ISLP Figure 8.7 / slides): a linear true boundary kills the tree (it has to step-function it), a rectangular true boundary kills linear regression. Each shines where its inductive bias matches.

## Exam signals

> "Trees take a different angle from regression with splines/interactions: instead of treating years and hits as independent variables that you then find some function of, instead, you kind of say oh we could just break this stuff up and then call this one region call that another region and then give a parameter to every region. And that gives us a very powerful model that now has these combinations of variables." - [[L18-trees-2]]

> "Sometimes you actually do a seemingly worthless split, but then it's followed by a very good one. … these interactions are not super obvious." - [[L18-trees-2]]

> "We can't try every possible combination. It's just too much. It's impossible." - [[L18-trees-2]]

The 2024 exam (3d, 1P) gave students `argmin_{R_1,R_2}[ Σ(y_i − ŷ_{R_1})² + Σ(y_i − ŷ_{R_2})² ]` and asked which method this is, answer: regression tree. The 2024 exam also asked students to **read off a prediction from a fitted tree** given a row of covariates.

## Pitfalls

- **Mean vs sum.** The split criterion is the **sum** of squared residuals, not the mean. Computing means changes the algorithm and breaks it.
- **The split lines don't extend across the whole space.** Easy to draw the wrong rectangle decomposition by mistakenly extending a horizontal cut leftward through a region that wasn't split that way.
- **Greedy ≠ globally optimal.** Don't claim a CART tree minimizes RSS over all partitions, it doesn't. It minimizes RSS at each step.
- **Number of regions $J$.** When asked "what is $J$?" given a tree, count *terminal nodes* (leaves), not internal nodes.
- **Prediction is the region mean.** Not the median, not the conditional expectation under some model, just the empirical mean of training $y$'s in that leaf.

## Scope vs ISLP

- **In scope:** Recursive binary splitting algorithm; the RSS split criterion; the "build out then prune" pipeline; reading a fitted tree (region rectangles + tree diagram both); interaction-capturing intuition; greedy nature; CART name.
- **Look up in ISLP:** §8.1.1 (regression trees); §8.1.4 (advantages/disadvantages); Algorithm 8.1 box.
- **Skip in ISLP (book-only, prof excluded):** Detailed NP-completeness proofs / computational-complexity discussion ([[L17-trees-1]]: noted only in passing). Bayesian Additive Regression Trees (BART, ISLP §8.2.4, never covered in lectures).

## Exercise instances

- Exercise8.1a, Provide a detailed explanation of the algorithm used to fit a regression tree; state what differs for a classification tree (= split criterion + prediction rule, see [[classification-tree]]).
- Exercise8.2b, Fit `tree(Sales ~ ., ...)` on Carseats train set, plot, interpret splits (`ShelveLoc`, `Price`, `Age`, `Advertising`), compute test MSE.

(See [[cost-complexity-pruning]] for Exercise8.2c on CV-based pruning, and [[random-forest]] / [[bagging]] for Exercises 8.2d–f.)

## How it might appear on the exam

- **MC / fill-in:** Match an `argmin` formula to "regression tree" (2024 Q3d style).
- **Read a fitted tree:** Given a tree diagram and a row of covariates, walk the splits and report the leaf value (2024 Q2d style, "predict 3243 bikes").
- **Interpretation:** Given a fitted tree on, e.g., Carseats or Boston, identify the most important split, describe the regions a high-value/low-value observation falls into.
- **Conceptual short answer:** "How does a regression tree differ from linear regression? When would you prefer one over the other?", pull from §8.1.3 / Trees-vs-linear discussion.
- **Trap:** Drawing a region partition where a split-line crosses through both children of a higher split, trees can't do that.
- **Concept-of-greedy:** "Why is the algorithm called greedy?" → at each step we pick the locally best split and never revise; not globally optimal.

## Related

- [[classification-tree]]: same algorithm, different prediction (majority class) and split criterion (Gini / cross-entropy).
- [[cost-complexity-pruning]]: the back-prune step that actually picks tree size.
- [[bagging]]: average $B$ trees fit on bootstrap samples to attack the variance.
- [[random-forest]]: bagging plus per-split predictor restriction → decorrelated trees.
- [[variable-importance]]: how to recover "which predictors matter" once you've ensembled away the interpretable single tree.
- [[bias-variance-tradeoff]]: trees: low bias, very high variance. The whole module is about the variance-reduction fixes.
- [[cross-validation]]: used to choose the pruning parameter $\alpha$ (= equivalently, the tree size).
- [[generalized-additive-models]]: what a tree is *not*: GAMs are additive, trees are interactive.
