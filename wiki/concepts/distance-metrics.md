---
concept: distance-metrics
module: 10-unsuper
lectures: [L22]
isl-ref: 12.4.2
exercises: []
related: [k-means-clustering, hierarchical-clustering, knn-classification, knn-regression, standardization, curse-of-dimensionality]
tags:
  - concept
  - module/10-unsuper
aliases:
  - dissimilarity measure
  - distance measure
---

# Distance metrics for clustering

The headline hyperparameter of any clustering or KNN algorithm , and one most people forget to think about. Different distances measure *different things*: Euclidean groups infrequent shoppers (small magnitudes look similar), correlation groups people with similar preferences (similar shapes look similar). Pick the metric based on what you want the algorithm to be sensitive to.

## Definition (prof's framing)

> "Euclidean is the most common dissimilarity measure or distance measure but there's many others like a correlation one." - [[L22-unsupervised-2]]

A **dissimilarity** $d(x_i, x_{i'})$ is a non-negative function on pairs of observations measuring how unlike they are. It need not satisfy the triangle inequality (the formal "metric" requirement) , clustering only needs an ordering of pairs, so plenty of useful dissimilarities aren't proper metrics. The key property: $d(x_i, x_{i'})$ small ⇔ "$i$ and $i'$ should be in the same cluster."

The choice of $d$ is *the* design choice in clustering. Together with the linkage rule (for [[hierarchical-clustering|hierarchical]]) or with $K$ (for [[k-means-clustering|K-means]]), $d$ determines the answer.

## Notation & setup

- $x_i, x_{i'} \in \mathbb{R}^p$: two observations.
- $d(x_i, x_{i'}) \ge 0$: the dissimilarity.
- For most algorithms in this course, observations are first **standardized** (z-scored) so that no single column dominates by virtue of units. See [[standardization]].

## The relevant distances in this course

| Name | Formula | When to use |
|---|---|---|
| **Euclidean** | $\sqrt{\sum_j (x_{ij} - x_{i'j})^2}$ | Default. Sensitive to magnitudes. The standard for K-means and KNN. |
| **Squared Euclidean** | $\sum_j (x_{ij} - x_{i'j})^2$ | What K-means actually minimizes , square root is unnecessary, "won't change who wins or how they win." - [[L22-unsupervised-2]] |
| **Correlation distance** | $1 - \rho(x_i, x_{i'})$ where $\rho$ is Pearson correlation between the two profiles | When *shapes* matter, not magnitudes. Shopper preferences. Range $[0, 2]$ (perfectly correlated = 0, anticorrelated = 2). |
| **Manhattan ($L^1$)** | $\sum_j |x_{ij} - x_{i'j}|$ | Less sensitive to outliers than Euclidean; sometimes preferred in high-$p$. Name-checked only. |
| **Cosine distance** | $1 - \frac{x_i \cdot x_{i'}}{\|x_i\| \|x_{i'}\|}$ | Standard for text data and embeddings (length-invariant). Name-checked. |
| **Wasserstein** (a.k.a. Earth Mover's) | min cost to "move" one distribution into the other | The prof's "favorite because it sounds cool"; not in the curriculum. |

> "Euclidean distances often will suck in high-dimensional spaces because of the curse of dimensionality makes all the points close together, whereas other distances are going to be less prone to that." - [[L22-unsupervised-2]]

## Euclidean vs correlation , the prof's load-bearing example

**Euclidean** measures absolute pairwise differences across all features. Two shoppers who both buy *very few* items have tiny entries everywhere → small Euclidean distance → grouped together regardless of *what* they buy.

**Correlation distance** ($1 - \rho$) measures whether the *shapes* of the profiles match. Two shoppers who bought the same items in the same proportions but at totally different overall volumes have correlation ≈ 1 → distance ≈ 0 → grouped together.

> "Euclidean distance might group together infrequent shoppers, people just don't shop a lot, whereas the correlation distance would find people who have similar preferences." - [[L22-unsupervised-2]]

The book (ISL §12.4.2, Figure 12.15) makes this precise: three observations with measurements on 20 variables. Observations 1 and 3 have similar values everywhere → small Euclidean distance. But they're weakly correlated → large correlation-based distance. Observations 1 and 2 have quite different magnitudes → large Euclidean distance, but they're highly correlated → small correlation distance. Same data, opposite cluster assignments depending on the metric.

**Pick the metric to match the question.** "Are these two shoppers similar in volume?" → Euclidean. "Are these two shoppers similar in taste?" → correlation.

## Insights & mental models

**The metric is a hyperparameter equal in importance to $K$ or the linkage.** Slide deck shows that the same data, same algorithm, different metric → different clusters. There is no universal "best" distance.

**Curse of dimensionality kills Euclidean.** In high-$p$ Euclidean space, all pairwise distances tend toward equality , "nearest neighbor" stops meaning anything. Manhattan and correlation are less affected. PCA / NN-feature-extractor before clustering also helps. See [[curse-of-dimensionality]].

**Correlation isn't really a distance.** It's a similarity. The slide deck flags: "Note: Correlation is actually a similarity measure, not a distance measure. Implication?" The fix: use $1 - \rho$ (or $1 - |\rho|$ if signed direction doesn't matter). Don't plug raw $\rho$ into a clustering algorithm expecting a distance.

**Standardize first** for any magnitude-sensitive metric (Euclidean, Manhattan). Otherwise the largest-unit variable dominates. The grocery example: caviar in kg (small numbers) vs bread in kg (bigger numbers) , without standardization, Euclidean clusters by bread consumption regardless of caviar pattern. Correlation distance is naturally scale-invariant per row, but you may still want to standardize columns for interpretability. See [[standardization]].

**Squared Euclidean = Euclidean for ranking purposes.** K-means uses squared because it makes the algebra cleaner (the centroid step then exactly minimizes the cluster's contribution); the resulting partition is identical to what you'd get from plain Euclidean.

**Distance metric → algorithm.** Both [[k-means-clustering|K-means]] and [[hierarchical-clustering|hierarchical]] are abstracted by their distance , swap Euclidean for correlation and you get a correlation-based version of each. KNN ([[knn-classification]], [[knn-regression]]) is the same way.

## Returns across the course

- **Module 4 / KNN ([[knn-classification]])**: Euclidean distance is the default for nearest neighbors; the curse of dimensionality kills it in high-$p$.
- **Module 10 / clustering**: the headline atom for "what dissimilarity?" decisions in K-means and hierarchical.

## Exam signals

> "Euclidean distance might group together infrequent shoppers, people just don't shop a lot, whereas the correlation distance would find people who have similar preferences." - [[L22-unsupervised-2]]

> "Pick the metric based on what you want it to be sensitive to." - [[L22-unsupervised-2]] (paraphrase of the prof's repeated framing)

> "There's no single right answer, which is both annoying and a benefit if you want to look at it that way." - [[L22-unsupervised-2]]

## Pitfalls

- **Defaulting to Euclidean without thinking.** Euclidean is the right answer often, but not always. The shopper example is the canonical "Euclidean wrong, correlation right" trap.
- **Using raw correlation as a distance.** Use $1 - \rho$ (or $1 - |\rho|$).
- **Forgetting to standardize before any magnitude-sensitive metric.** Mixed units → meaningless distances.
- **Curse-of-dimensionality blindness.** In high $p$, all Euclidean distances bunch up , clustering output is pure noise. Dim-reduce first or switch metric.
- **Treating "distance" formally** when comparing methods. Some "distances" (correlation) aren't true metrics (no triangle inequality). Doesn't matter for clustering, but flag it if asked.

## Scope vs ISLR

- **In scope:** Euclidean and correlation distances; the *shopper-Euclidean-vs-correlation* example as the canonical "metric matters" demonstration; the rule "pick the metric based on what you want sensitivity to"; standardization mandate; curse-of-dimensionality caveat for Euclidean.
- **Look up in ISLR:** §12.4.2 "Choice of dissimilarity measure" , Figure 12.15 makes the Euclidean-vs-correlation distinction visual; §12.4.2 also covers when to standardize and the socks-vs-computers example for clustering preprocessing.
- **Skip in ISLR:**
  - **Wasserstein, Manhattan, cosine derivations**: name-checked in [[L22-unsupervised-2]] only; out per `docs/scope.md`.
  - **Mahalanobis distance** (the metric defined by $\Sigma^{-1}$) , book mentions in passing for some methods; not on the prof's curriculum.
  - **Triangle-inequality / formal-metric theory**: not relevant to clustering's needs.

## Exercise instances

(No standalone exercise targets distance choice , the metric question lives inside the K-means and hierarchical exercises, where the default is Euclidean and the alternatives are mentioned but not drilled.)

## How it might appear on the exam

- **Conceptual MC / T-F:**
  - "Two observations with high Euclidean distance always have low correlation." → **False** (counterexample: low-magnitude points → small Euclidean, can have any correlation).
  - "Correlation-based distance ignores the magnitudes of the observations." → **True**.
  - "Euclidean distance is invariant to the scaling of individual variables." → **False** (this is why we standardize).
- **Scenario question (the shopper example):** "An online retailer wants to cluster shoppers by preference, not volume. Which dissimilarity should they use?" → **Correlation distance** ($1 - \rho$). Justify with the shopper example.
- **Reasoning about high-$p$:** "Why might Euclidean distance be a poor choice when $p$ is very large?" → curse of dimensionality, all pairs bunch up.
- **Method-design question:** "If you want hierarchical clustering to be insensitive to overall observation magnitudes, which combination of preprocessing and distance would you choose?" → standardize columns, use correlation distance.

## Related

- [[k-means-clustering]]: uses (squared) Euclidean by default; metric is swappable.
- [[hierarchical-clustering]]: distance choice combines with linkage choice to determine the dendrogram.
- [[standardization]]: preprocessing step that makes magnitude-sensitive distances meaningful across mixed-unit features.
- [[knn-classification]] / [[knn-regression]] , same metric question in the supervised setting.
- [[curse-of-dimensionality]]: why Euclidean breaks down in high $p$.
