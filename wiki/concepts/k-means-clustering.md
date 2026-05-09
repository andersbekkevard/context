---
concept: k-means-clustering
module: 10-unsuper
lectures: [L22]
isl-ref: 12.4.1
exercises:
  - Exercise10.2 - show the k-means algorithm decreases the objective at each step
  - Exercise10.3 - perform k-means on the NYT stories dataset
related: [hierarchical-clustering, distance-metrics, standardization, dimensionality-reduction, principal-component-analysis]
tags:
  - concept
  - module/10-unsuper
aliases:
  - K-means
  - K means
---

# K-means clustering

The simplest partitioning algorithm, pick $K$, iterate centroid → assign → centroid until convergence. Minimizes within-cluster sum of squared Euclidean distances; converges to a **local minimum** (random initialization "can really screw you", the prof's headline pitfall). $K$ is a hyperparameter you must pick by hand or by downstream interpretability.

## Definition (prof's framing)

> "K-means clustering is definitely a very standard approach. It's kind of arguably the easiest one, so it's a good one to start with … Occam's razor is good. Keep the big guns for when you need them." - [[L22-unsupervised-2]]

Partition the $n$ observations into $K$ disjoint, exhaustive clusters $C_1, \ldots, C_K$, sets of indices satisfying:

$$\bigcup_{k=1}^K C_k = \{1, \ldots, n\}, \qquad C_k \cap C_{k'} = \emptyset \text{ for } k \neq k'.$$

> "It's a cool way of saying each point only goes to one cluster and they have to go to a cluster." - [[L22-unsupervised-2]]

A "good" clustering minimizes total **within-cluster variation**, defined as the sum of pairwise squared Euclidean distances within each cluster, normalized by cluster size.

## Notation & setup

- $C_k \subseteq \{1, \ldots, n\}$: index set of cluster $k$. $|C_k|$ = its size.
- $x_i \in \mathbb{R}^p$: observation $i$. Standardized first, see [[standardization]].
- $\bar{x}_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i$: cluster centroid (mean vector, length $p$).
- $K$: number of clusters. Hyperparameter.
- "Distance" is Euclidean unless specified, the algorithm generalizes if you swap in another metric (see [[distance-metrics]]).

## Formula(s) to know cold

The K-means objective (book Eq. 12.17, the same one that drops in Exercise 10.2):

$$
\boxed{\;\min_{C_1, \ldots, C_K} \sum_{k=1}^K \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2\;}
$$

The within-cluster pairwise-sum identity (book Eq. 12.18, the trick behind the Exercise-10.2 monotonicity proof):

$$
\frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2
\;=\; 2 \sum_{i \in C_k} \sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2.
$$

So the pairwise-distance objective is equivalent to the **sum of squared distances from each point to its cluster mean** (up to the factor of 2). This is the form that makes the proof of monotonicity work, both algorithm steps minimize the same quantity in the right variables.

## The algorithm

> [!important] Algorithm (per ISL 12.2 / [[L22-unsupervised-2]])
> 1. **Random init**: assign each observation a number $1, \ldots, K$ at random.
> 2. **Iterate** until cluster assignments stop changing:
>    - **(a) Centroid step**: for each cluster, compute the centroid $\bar{x}_k$ (the mean of its current members).
>    - **(b) Assign step**: reassign each observation to the cluster whose centroid is closest in Euclidean distance.
> 3. Stop when no assignments change in a full pass.

> "It's not clear that this will lead to a fixed point just from looking at it, but it does … we're not going to prove it, but it does, at least in practice." - [[L22-unsupervised-2]]

The slide-deck example showed K=3 converging in ~3 iterations from a random start. Squared rather than plain Euclidean because "there's no reason to take the square root, it just an extra step that won't get you anything … it won't change the goal or won't change who wins or how they win." - [[L22-unsupervised-2]]

### Why monotone (Exercise 10.2)

Use identity 12.18 to rewrite the objective as $2 \sum_k \sum_{i \in C_k} \|x_i - \bar{x}_k\|^2$. Then:
- **Step (a)** picks $\bar{x}_k$ as the cluster mean, provably the minimizer of $\sum_{i \in C_k} \|x_i - c\|^2$ over $c$ (calculus: derivative zero at the mean). So the centroid update can only decrease the objective (or leave it alone).
- **Step (b)** reassigns each point to its closest centroid, by definition this can only decrease (or leave unchanged) each point's contribution.

Both steps decrease (weakly) the same nonnegative quantity → must converge (only finitely many partitions exist). It converges to a local minimum, **not necessarily the global one**.

## Insights & mental models

**Brute force is hopeless.** $K^n$ partitions: with $n=1000$ and $K=10$ that's $10^{1000}$. The algorithm trades global optimality for tractability.

**Random init is the headline pitfall.** "It actually rather, especially in certain data sets, this random initialization can really screw you. … you might get a very different solution, depending on the initialization. … This is often why people don't like K-means, because of this local minima problem." - [[L22-unsupervised-2]] **Standard fix**: run K-means many times (R's `nstart = 20` or higher; the NYT exercise uses `nstart = 20`), pick the run with the lowest objective. Slide-deck demo showed six K=3 runs landing on three different local optima, only two of six found the best.

**Ensemble fix (the prof's neat aside).** Across many random-init runs, count how often each pair of points is co-clustered. Use those frequencies as a new dissimilarity, then re-cluster on that. "Actually that works quite well but it's kind of a funny way of doing things." - [[L22-unsupervised-2]]

**$K$ is a hyperparameter, there's no universal CV.** Pure unsupervised, no held-out $Y$ to score against. "Hyperparameters are death." - [[L22-unsupervised-2]] Pick $K$ from domain knowledge, downstream interpretability ("we can only study 4 things at a time"), or ad hoc heuristics like the elbow on within-cluster sum of squares vs $K$. The prof's animal-behavior example: he picks the $K$ that produces a number of behavior types small enough to interpret biologically (4–7), even if the data could justify more.

**Standardize first**, Euclidean distance is sensitive to units. If one variable is in nanograms and another in centimeters, the nanogram column dominates. See [[standardization]].

**$K = n$ is the trivial degenerate solution**, every point is its own cluster, objective is 0, you've learned nothing.

**Dependence on the distance metric.** Swap Euclidean for any other metric and you get a different algorithm. The prof name-checks Wasserstein ("because it sounds cool"), Manhattan, cosine. Euclidean is sensitive to the curse of dimensionality, "Euclidean distances often will suck in high-dimensional spaces because of the curse of dimensionality makes all the points close together, whereas other distances are going to be less prone to that." See [[distance-metrics]].

**No nesting between $K$ and $K+1$.** K-means with $K=4$ and $K=5$ can produce totally unrelated partitions, there's no guarantee that the $K=5$ solution is "the $K=4$ solution with one cluster split." This is a key contrast with [[hierarchical-clustering]], where the dendrogram makes the partitions nested by construction.

## Exam signals

> "It actually rather, especially in certain data sets, this random initialization can really screw you. … This is often why people don't like K-means, because of this local minima problem." - [[L22-unsupervised-2]]

> "Hyperparameters are death." - [[L22-unsupervised-2]]

> "Each point only goes to one cluster and they have to go to a cluster." - [[L22-unsupervised-2]]

> "Also k-means is fair game with similar hand-computation style." - [[L27-summary]] (hierarchical was the worked Q5, but he flagged k-means hand computations as also fair game)

## Pitfalls

- **Random initialization → local minimum.** Always set `nstart` ≥ 20. Reporting a single run is a methodological red flag.
- **Forgetting to standardize.** Euclidean distance over mixed-unit columns is meaningless.
- **Picking $K$ without justification.** State *why*: domain knowledge, elbow on within-SS, downstream interpretability. Don't just say "K=3 because."
- **Confusing K-means with hierarchical.** K-means produces *one partition for one $K$*; hierarchical produces all partitions simultaneously via the dendrogram.
- **Asking for nested K-means partitions across $K$.** Doesn't exist by construction.
- **Squared-vs-plain Euclidean confusion.** The objective uses *squared* distances. "There's no reason to take the square root." Don't introduce extra square roots that aren't in the formula.
- **Curse of dimensionality.** In high $p$, all pairwise Euclidean distances tend toward equality → K-means can't tell points apart. Dimension-reduce first (PCA, NN feature extractor) or switch to a less-cursed distance.

## Scope vs ISLR

- **In scope:** the partition definition, the within-cluster-variation objective, the iterative algorithm, the local-minimum-and-rerun fix, the no-nesting property vs hierarchical, the standardization mandate, the proof-style monotonicity argument (Exercise 10.2).
- **Look up in ISLR:** §12.4.1, the canonical treatment, including the (12.17) ↔ (12.18) identity that powers the monotonicity proof, and Figures 12.7–12.9 (algorithm progression + local-optima demo).
- **Skip in ISLR:**
  - **K-means++ initialization** - [[L22-unsupervised-2]]: name-checked as a smarter init scheme, not derived. Out of scope per `docs/scope.md`.
  - **Soft / mixture-model K-means (EM, mixtures of Gaussians)**: book §12.4.3 mentions "mixture models are an attractive approach"; deferred to ESL, never lectured. Skip.
  - **Gap statistic** for choosing $K$, name-checked only.

## Exercise instances

- **Exercise 10.2**: Show that the K-means algorithm decreases the objective at each step. Use identity (12.18) to rewrite the pairwise-distance objective as a sum-of-squared-deviations-from-the-mean form, then argue (a) the centroid step minimizes that form (mean is the minimizer of squared deviations), (b) the assign step can only improve a point's contribution. Both steps weakly decrease the same nonnegative quantity → convergence. **The book's solution is on p. 517 (2nd ed.) around Eq. 12.18.**
- **Exercise 10.3**: Perform k-means on NYT stories. Run `kmeans(nyt_data[, -1], 2, nstart = 20)` (note `nstart = 20`, the local-minima fix). Plot the cluster assignments in PC1/PC2 space (using PCA from Exercise 10.1 as a visualization tool) and compare to the true labels (art / music). The exercise drives home: K-means recovers the music/art split *without* ever seeing the labels, a genuine unsupervised win.

## How it might appear on the exam

- **Hand computation (small $K$, small $n$):** given 4–6 points in 2D and $K=2$, show one or two iterations of the K-means algorithm starting from a given random assignment. Compute centroids, reassign, recompute. Shows you understand the loop. Per [[L27-summary]]: "k-means is fair game with similar hand-computation style" to the hierarchical Q5.
- **Conceptual MC / T-F:**
  - "K-means is guaranteed to find the global minimum of its objective." → **False** (local minimum only).
  - "Different random initializations can yield different K-means clusterings." → **True**.
  - "K-means clusters are nested across $K$." → **False** (this is hierarchical).
  - "The K-means objective is non-increasing across iterations." → **True**.
- **Output interpretation:** given a printout of `kmeans` output (cluster sizes, within-cluster SS, between-cluster SS), interpret. The "between/total" ratio is a rough fit-quality measure.
- **Algorithm explanation:** "describe the K-means algorithm" or "explain why K-means converges" (recite the monotonicity argument).
- **Method comparison:** K-means vs hierarchical, when to prefer each? K-means: known $K$, want a flat partition, large $n$. Hierarchical: don't know $K$, want a dendrogram, want nested structure (e.g. taxonomic / behavioral hierarchies). See [[hierarchical-clustering]].

## Related

- [[hierarchical-clustering]]: the natural counterpoint: nested clusters, dendrogram, no $K$ to pick.
- [[distance-metrics]]: Euclidean is the default; swap in another and the algorithm changes.
- [[standardization]]: mandatory before clustering on any mixed-unit data.
- [[dimensionality-reduction]]: common preprocessing for clustering high-dim data; also the visualization tool (project clusters into PC1/PC2).
- [[principal-component-analysis]]: used in Exercise 10.3 to *visualize* the K-means result in PC space.
