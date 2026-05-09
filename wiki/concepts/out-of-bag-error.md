---
concept: out-of-bag-error
module: 05-resample
lectures: [L18, L19]
isl-ref: 8.2.1
exercises:
  - Exercise8.1d  -  explain OOB sample, what fraction of observations are OOB
related: [bootstrap, bagging, random-forest, cross-validation]
tags:
  - concept
  - module/05-resample
aliases:
  - OOB error
  - OOB
  - out-of-bag
---

# Out-of-bag (OOB) error

The "free test set" that comes with every bootstrap sample. ~37% of observations are not drawn into a given bootstrap sample, those are **out-of-bag** for that tree, and serve as that tree's validation set. Aggregate across trees → an honest test-error estimate **with no separate test set required.**

## Definition (prof's framing)

For each bootstrap sample of size $n$ (drawn with replacement from $n$ original observations), some originals appear multiple times and others don't appear at all. The roughly 1/3 left out are the **out-of-bag (OOB) observations** for that bootstrap. Predict on them with the tree fit on that bootstrap → per-tree validation error. Aggregate across all $B$ trees → OOB error estimate.

> "It's approximately B divided by 3." - [[L19-boosting-1]] (the prof's quick verbal, meaning ~1/3 are out-of-bag for any given tree)

> "About 1/3 of the observations are not used to fit a particular bagged tree, and serve as a built-in test set for that tree. No dedicated test set required.", paraphrase of the slide / [[L18-trees-2]]

## Notation & setup

- $n$ original observations.
- For bootstrap sample $b$, define $\text{InBag}_b \subseteq \{1, \dots, n\}$, the original indices drawn (with multiplicity) into bootstrap $b$.
- $\text{OOB}_b = \{1, \dots, n\} \setminus \text{InBag}_b$: the indices not drawn.
- For each $i \in \{1, \dots, n\}$, collect the trees $\{b : i \in \text{OOB}_b\}$ that didn't see it; their average prediction is the OOB prediction $\hat y_i^{\text{OOB}}$.
- OOB error = average loss between $y_i$ and $\hat y_i^{\text{OOB}}$ across all $i$.

## Formula(s) to know cold

**Probability obs $i$ is OOB for a single bootstrap sample** (Exercise 5.4 → Exercise 8.1d):

$$P(\text{obs } i \text{ NOT drawn in } n \text{ draws}) = (1 - 1/n)^n \xrightarrow{n \to \infty} \frac{1}{e} \approx 0.368$$

So **~36.8% of observations are OOB** for any given tree, and **~63.2% are in-bag** ($1 - 1/e \approx 0.632$).

**OOB prediction for observation $i$:**

$$\hat y_i^{\text{OOB}} = \text{avg / vote of } \big\{ \hat f^{*b}(x_i) : i \in \text{OOB}_b \big\}$$

(Average for regression, majority vote for classification.)

**OOB error:**

$$\text{OOB-MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat y_i^{\text{OOB}})^2 \quad \text{(regression)}$$

$$\text{OOB-Err} = \frac{1}{n} \sum_{i=1}^n \mathbb{1}(y_i \neq \hat y_i^{\text{OOB}}) \quad \text{(classification)}$$

## Insights & mental models

- **The result $1 - 1/e$ is exam-flagged.** Hand-calculation pattern: "Show that for large $n$, $(1 - 1/n)^n \to 1/e$ and conclude that ~37% of observations are OOB." Direct port of Exercise 5.4 to module 8.
- **OOB error ≈ leave-one-out CV error** in the limit of large $B$, for each obs $i$, the OOB prediction averages over the trees that didn't see $i$, which is the same in spirit as "leave $i$ out, fit on the rest." With many bootstrap samples, this converges to a CV-like estimate.
- **No separate test set needed.** The big practical win of bagging / RF: you don't have to set aside a held-out test set, and you don't have to do a separate k-fold CV pass. OOB gives you the assessment estimate as a byproduct of training.
- **Cheap, but not perfectly honest.** *"You have this strange dependency on the test error from your real error on your test error on how you sampled."* - [[L18-trees-2]]. The OOB error is computed from the same bootstrap samples used to train, so there's some structural dependency. In practice it's a very good test-error proxy, especially with large $B$.
- **Used for variable importance too.** The randomization-based variable-importance flavor (permute predictor $j$ on OOB samples, measure performance drop) is exactly OOB error with one column scrambled. See [[variable-importance]].
- **The two-thirds / one-third split is approximate.** $(1 - 1/n)^n$ converges to $1/e \approx 0.368$ from above. For $n = 10$: $\approx 0.349$. For $n = 100$: $\approx 0.366$. For $n \to \infty$: $0.368$. Convergence is fast and from above.

## How it's used in practice

1. **Replaces a separate test set / k-fold CV** for bagged ensembles and random forests. Most random-forest implementations report OOB error automatically.
2. **Choosing $B$:** plot OOB error vs $B$ and pick $B$ where the curve flattens. (Even though $B$ isn't a "real" tuning parameter, this confirms you've used "enough" trees.)
3. **Variable importance:** the randomization-based version uses OOB samples to measure the drop in performance when each predictor is permuted.

## Exam signals

> "It's approximately B divided by 3." - [[L19-boosting-1]]

> "Each bootstrap sample uses ~2/3 of the data; the remaining ~1/3 ('out of bag') gives an honest validation set per tree." - [[L19-boosting-1]] paraphrase

The hint in Exercise 8.1d is explicit: *"The result from RecEx5-Problem 4c can be used."*, i.e., the $1 - 1/e \approx 0.632$ derivation from module 5 is reused. That cross-reference is itself a signal, the prof expects Anders to chain the two results.

## Pitfalls

- **Confusing 0.632 (in-bag) with 0.368 (OOB).** The prof's "B divided by 3" line refers to the OOB fraction (1/3, give or take). The in-bag fraction is 2/3.
- **Forgetting it's an *aggregate* over trees that didn't see obs $i$.** Each individual tree only votes on observations it didn't train on; you collect those votes per observation and aggregate.
- **Treating OOB as identical to k-fold CV.** They're closely related, OOB is roughly equivalent to LOOCV in the limit, but technically different (each obs is OOB for a random subset of trees, not held out exactly once).
- **Using OOB on small $B$.** With $B = 50$, some observations may have very few OOB predictions to average; the per-observation prediction is noisy. With $B = 500+$, this is rarely an issue.
- **Forgetting OOB only works for bagging-family methods.** Boosting fits trees sequentially on a single training set (re-weighted in AdaBoost or on residuals in gradient boosting); there's no per-tree OOB concept.

## Scope vs ISLP

- **In scope:** the $1 - 1/e \approx 0.632 / 0.368$ probability, the role of OOB as a free test-set estimate, the connection to variable importance (randomization flavor).
- **Look up in ISLP:** §8.2.1 (pp. 345), "Out-of-Bag Error Estimation" subsection. Brief, the prof's treatment is similar in depth.
- **Skip in ISLP (book-only, prof excluded):** formal proof that OOB ~ LOOCV, deeper convergence analysis.

## Exercise instances

- **Exercise8.1d**: explain OOB sample; compute what fraction of observations are OOB. The hint explicitly says to reuse the [[bootstrap]] derivation from Exercise 5.4 (the $1 - 1/e$ result). This is the **textbook hand-calculation** for OOB.

(Other module-8 problems use OOB implicitly via `randomForest()` output, but Exercise 8.1d is the only one that asks for the conceptual derivation.)

## How it might appear on the exam

- **Hand calculation:** "What fraction of observations are out-of-bag for a given tree in a bagging procedure with $n$ large?" → derive $(1 - 1/n)^n \to 1/e$, conclude ~37% OOB. Direct port of Exercise 8.1d.
- **Conceptual / T/F:**
  - "OOB error replaces the need for a separate test set in bagging / RF" → true (with mild caveats).
  - "OOB error is computed for each tree using observations not in its bootstrap sample" → true.
  - "Boosting models report an OOB error" → false; OOB is bagging-family only.
  - "OOB error is exactly equivalent to LOOCV" → false (closely related, not identical).
- **Use-case justification:** "Why does bagging / RF not need a separate test set?" → because each observation is OOB for ~1/3 of the trees, providing per-observation held-out predictions.
- **Connect to variable importance:** the randomization-based importance permutes a predictor in the OOB samples, link the two concepts.

## Related

- [[bootstrap]]: supplies the $1 - 1/e \approx 0.632$ result
- [[bagging]]: the procedure OOB is built on
- [[random-forest]]: the natural home of OOB error in practice
- [[cross-validation]]: the alternative test-error estimate; OOB is the bagging-family substitute
