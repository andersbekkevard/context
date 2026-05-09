---
concept: cost-complexity-pruning
module: 08-trees
lectures: [L17, L18]
isl-ref: 8.1.1
exercises:
  - Exercise8.2c — cv.tree() on a Carseats regression tree, choose best size, compare pruned vs unpruned MSE
  - Exercise8.3e — cv.tree() with prune.misclass on the spam classification tree, choose smallest tree at min misclassification
related: [regression-tree, classification-tree, cross-validation, regularization, lasso, bias-variance-tradeoff, one-standard-error-rule]
tags:
  - concept
  - module/08-trees
aliases:
  - weakest-link pruning
  - tree pruning
  - tree size selection
---

# Cost-complexity pruning

The right way to choose tree size: **grow the tree fully (overfit), then prune backwards** under a complexity penalty $\alpha |T|$. Pick $\alpha$ by [[cross-validation]]. Prof's headline rule: **early stopping is wrong** — bad-looking splits enable good ones later, so you have to grow first and prune second.

## Definition (prof's framing)

> "Better idea: grow a very large tree $T_0$, and then prune it back in order to obtain a subtree." — slide deck

> [!important] Why early stopping fails (prof verbatim)
> "Sometimes you actually do a seemingly worthless split, but then it's followed by a very good one. Because again, sometimes you need to cut one way and then cut the other way until you really get the benefit, right? Because these interactions are not super obvious." — [[L18-trees-2]]

So: build out fully (lax stopping criteria), then **cost-complexity prune** to a sequence of nested subtrees indexed by $\alpha$, then choose $\alpha$ by **K-fold CV**.

## Notation & setup

- $T_0$ — fully grown (maximal) tree.
- $T \subset T_0$ — a candidate subtree.
- $|T|$ — number of terminal nodes (leaves) in $T$. **Cardinality, not absolute value** — the prof flagged the slide notation explicitly.
- $\alpha \ge 0$ — the complexity penalty (the equivalent of $\lambda$ in [[ridge-regression]]/[[lasso]]).
- $R_m$ — region for terminal node $m$; $\hat y_{R_m}$ — prediction in $R_m$ (mean for regression, majority class for classification).
- $Q(T)$ — base loss (RSS for regression; cross-entropy / Gini / **misclassification** for classification).

## Formula(s) to know cold

**The cost-complexity criterion:**

$$
C_\alpha(T) = \underbrace{\sum_{m=1}^{|T|}\sum_{i \in R_m}(y_i - \hat y_{R_m})^2}_{Q(T) \text{ — training RSS}} \; + \; \alpha\, |T|
$$

For regression, $Q(T) = \text{RSS}$. For classification, $Q(T)$ is misclassification (used at the prune step — see [[classification-tree]]).

- $\alpha = 0$: minimum is $T_0$ itself (no penalty → keep the maximal tree).
- $\alpha \to \infty$: minimum is the root-only tree (one leaf).
- Intermediate $\alpha$: the penalty buys you fewer leaves at the cost of higher RSS.

## How the prune sequence is generated — weakest-link pruning

The slide deck and Algorithm 8.1 don't dwell on this; the prof did:

> "You build out the tree as much as possible and then you prune the tree again back up to … you can go all the way up to the root node. It doesn't matter so much. You first build it out and then you go backwards, removing them to minimize that cost complexity thing." — [[L18-trees-2]]

Algorithmically: at each step, collapse the internal node whose removal **increases the training RSS the least** (the "weakest link"). This generates a **nested sequence** of subtrees:

$$
T_0 \supset T_1 \supset T_2 \supset \cdots \supset T_{\text{root}}
$$

> "Let's say when you first did the exhaustive one you ended up with 100 regions. As you go backwards you have one that has 99, then 98, then 97 … all the way to one. And then what that gives you is 100 different models." — [[L17-trees-1]]

These $T_k$ are the optimal solutions of the penalized criterion at increasing values of $\alpha$. The slide notes the sequence is **hierarchical / nested** — once a leaf is collapsed, it stays collapsed.

## Choosing $\alpha$ via K-fold CV (Algorithm 8.1)

1. Use recursive binary splitting to grow $T_0$ on training data.
2. Apply weakest-link pruning to $T_0$ to obtain the sequence $T_0 \supset T_1 \supset \ldots$ as a function of $\alpha$.
3. Use $K$-fold [[cross-validation]] to choose $\alpha$. For each fold $k = 1, \ldots, K$:
   - Repeat steps 1 and 2 on all but the $k$-th fold.
   - Evaluate held-out MSE (regression) / misclassification (classification) on the $k$-th fold, as a function of $\alpha$ (equivalently, tree size).
   - Average over folds.
4. Pick the $\alpha$ minimizing the CV error and return the corresponding subtree from step 2.

In R: `cv.tree(tree.mod)` for regression, `cv.tree(tree.mod, FUN = prune.misclass)` for classification (default 10-fold). The plot is **CV deviance vs tree size** (the slide notes you display by size, not by $\alpha$, because the relationship is nicer to read).

## CV plot interpretation

The classic shape:
- Tree size 1: just the global mean / majority class — high error.
- As you prune *up* from $T_0$, error first **decreases** (overfitting reduced).
- Hits a minimum at some size (5 leaves on the ozone, 6 on the Boston, 7 on the brain-injury).
- Continues into the over-pruned regime, error rises again.

> "As they prune nodes away, it didn't make the model worse and held out error. It made it better because it overfit less. And then eventually it found a minimum in this deviance. And then it got worse again." — [[L18-trees-2]]

If multiple sizes have similar CV error, pick the **smallest** — same flavor as the [[one-standard-error-rule]]. *"We picked 7 because that was the smallest model with that level of misclassification error."* — [[L18-trees-2]]

## For classification trees: prune on misclassification, not Gini/entropy

Even though the tree was *grown* using Gini or cross-entropy (because misclassification is too coarse a split criterion — see [[classification-tree]]), the *prune step* uses misclassification. R: `prune.misclass`, used inside `cv.tree(..., FUN=prune.misclass)`.

> "It seems weird that you would not train on misclassification error but it turns out to be better to train on the Gini index or the deviance and then prune with misclassification." — [[L18-trees-2]]

This asymmetry is one of the prof's flagged quirks of the trees module. Worth memorizing the verbatim quote.

## Insights & mental models

- **The penalty term is exactly the lasso/ridge form.** $C_\alpha(T) = Q(T) + \alpha |T|$ has the same shape as $\text{RSS} + \lambda \|\beta\|_p$ — see [[regularization]]. The slide deck calls this out: *"Equation 8.4 is reminiscent of the lasso (6.7)."* The "$L_0$ norm" framing (count nonzero leaves) is what the prof flagged as the precise meaning of $|T|$: *"that's the zeroth norm."* — [[L18-trees-2]]
- **Why "build out then prune" instead of "stop early."** The slide deck calls early stopping "too short-sighted." The mechanism: greedy splits that look bad locally can enable big improvements two splits down. You can only see those improvements if you let the tree grow.
- **Bias-variance angle.** Pruning trades variance for bias — fewer leaves → fewer parameters → less data-set sensitivity, but the in-region predictions are coarser. CV picks the sweet spot. Same shape as $\lambda$ in ridge/lasso. See [[bias-variance-tradeoff]].
- **The sequence is nested.** You don't search over arbitrary subtrees of $T_0$; you only ever see the at-most-$|T_0|$ subtrees in the weakest-link sequence. This makes step 2 of Algorithm 8.1 cheap.
- **Plot tree size, not $\alpha$.** The relationship between $\alpha$ and tree size is a step function — reading the plot in tree-size space makes the CV minimum easy to spot. (Slide deck: "*we can plot the CV error as a function of tree size (instead of $\alpha$ — why?)*".)

## Exam signals

> "Sometimes you actually do a seemingly worthless split, but then it's followed by a very good one." — [[L18-trees-2]]

> "It seems weird that you would not train on misclassification error but it turns out to be better to train on the Gini index or the deviance and then prune with misclassification." — [[L18-trees-2]]

> "Equivalent to saying, I want to find the shortest tree that gives me a good cost." — [[L18-trees-2]]

The prof spent significant lecture time on this on both L17 and L18 — algorithmic detail (build out then prune, nested sequence) plus the conceptual hook (early stopping fails, equivalence to lasso penalty). All exam-relevant in his "concept matters, exact algorithm doesn't" frame.

## Pitfalls

- **Don't conflate stopping criteria with pruning.** Stopping criteria (`mincut`, `minsize`, `mindev`) just control when the *grow* phase halts — they're not the model-selection knob. The model-selection knob is $\alpha$, chosen by CV.
- **Early stopping ≠ cost-complexity pruning.** Stopping early because a split's RSS reduction is small is exactly the failure mode the prof warned about. Don't propose this on the exam.
- **$|T|$ is the leaf count, not the number of edges or depth.** Shows up in the formula and in CV plots ("tree size").
- **For classification, the cost in $C_\alpha$ is misclassification, not Gini/cross-entropy.** Easy to write the wrong $Q(T)$.
- **CV picks $\alpha$ on the training data.** You then refit the chosen $\alpha$ on the *full* training data, not on a held-out fold. Same logic as CV-tuned ridge / lasso.
- **`cv.tree`'s `dev` field reports deviance** for regression and **number of misclassifications** when you pass `FUN = prune.misclass`. Don't get tripped up reading the output.

## Scope vs ISLR

- **In scope:** The cost-complexity formula $C_\alpha(T) = Q(T) + \alpha|T|$; the build-out-then-prune pipeline (Algorithm 8.1); CV-based $\alpha$ selection; the early-stopping-is-wrong rule; the train-Gini-prune-misclass asymmetry for classification; reading a CV-error-vs-tree-size plot.
- **Look up in ISLR:** §8.1.1 (Tree Pruning subsection), Algorithm 8.1 box, Figures 8.4–8.5 (Hitters worked example).
- **Skip in ISLR (book-only, prof excluded):** Detailed formal proof that the weakest-link sequence is the same as the $\alpha$-optimal sequence (the slide deck links a Bo Lindqvist note for those who want it; not exam material). Bayesian-prior interpretations of $\alpha$.

## Exercise instances

- Exercise8.2c — `cv.tree()` on the Carseats regression tree, find best size via `which.min(cv.Carseats$dev)`, prune with `prune.tree(tree.mod, best=...)`, compare pruned test MSE to unpruned.
- Exercise8.3e — `cv.tree(spam.tree, FUN = prune.misclass)`, plot misclassifications vs size, pick smallest size at the minimum, prune with `prune.misclass(...)`, compute test misclassification on the pruned tree.

## How it might appear on the exam

- **Conceptual short answer:** "Why do we grow the tree fully and then prune, rather than stopping early?" → reproduce the prof's quote about useless-looking splits enabling good ones.
- **MC / T/F:** "Cost-complexity pruning uses an L0-style penalty on the number of leaves" → True. "Increasing $\alpha$ leads to a larger tree" → False.
- **Read a CV plot:** given an error-vs-tree-size curve, identify the optimal tree size (the minimum, or the smallest tree within 1 SE).
- **Pseudocode:** "Write pseudocode for choosing the optimal pruning level $\alpha$." → Algorithm 8.1 in pseudocode form; CV inner loop, evaluate sequence on each fold, average, pick min.
- **Trap:** Saying you'd use Gini at the *prune* step in classification. The prof flagged the asymmetry — train Gini/entropy, prune misclassification.
- **Compare to other regularizers:** "How is cost-complexity pruning analogous to the lasso?" → both impose a penalty $\alpha \cdot \text{complexity}$ on the training loss; both use CV to pick the penalty.

## Related

- [[regression-tree]] — the grow phase; cost-complexity is the prune phase.
- [[classification-tree]] — same prune machinery, with misclassification as the cost.
- [[cross-validation]] — the canonical way to pick $\alpha$ here; $K = 10$ is the `cv.tree` default.
- [[regularization]] — pruning is the tree analogue of ridge/lasso: $\alpha|T|$ is an L0 leaf penalty.
- [[lasso]] / [[ridge-regression]] — same penalty-on-complexity idea; the slide deck explicitly draws the analogy.
- [[bias-variance-tradeoff]] — pruning is a bias-variance knob, with CV picking the sweet spot.
- [[one-standard-error-rule]] — sister heuristic for picking the simplest model within 1 SE of the CV minimum.
