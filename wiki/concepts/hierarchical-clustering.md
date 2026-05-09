---
concept: hierarchical-clustering
module: 10-unsuper
lectures: [L22, L27]
isl-ref: 12.4.2
exercises:
  - Exercise10.4 — perform hierarchical clustering on the NYT stories dataset (complete + single + average linkage)
related: [k-means-clustering, distance-metrics, standardization, dimensionality-reduction]
tags:
  - concept
  - module/10-unsuper
aliases:
  - agglomerative clustering
  - dendrogram
  - linkage
---

# Hierarchical (agglomerative) clustering and linkage

The "no $K$ required" alternative to [[k-means-clustering|K-means]]. Build a tree (the **dendrogram**) bottom-up by repeatedly merging the two closest clusters; cut the tree at any height to read off any number of clusters. Two design choices replace K-means's "pick $K$": the **dissimilarity measure** between points (Euclidean? correlation?) and the **linkage** that extends point-to-point dissimilarity to set-to-set dissimilarity (complete, single, average). The prof flagged "do this by hand on a 4×4 distance matrix" as a canonical exam question — **−1 point per mistake**.

## Definition (prof's framing)

> "Agglomerative. If you know the word, like to glom is to like stick on stuff. So agglomerative is like sticky, sticky, sticky." — [[L22-unsupervised-2]]

Start with every observation as its own cluster ($n$ clusters total). Iteratively find the two closest clusters and **fuse** them, recording the fusion height = the inter-cluster dissimilarity at that moment. Continue until one cluster remains. The resulting tree is the **dendrogram** — and you cut it at any horizontal height to read off the partition at that resolution.

> "It's a nested set of clustering. You can think of it like that, it has, you know, first everyone is together and then it would split it and then split those splits." — [[L22-unsupervised-2]]

(The prof's phrasing is divisive-flavored, but the algorithm is *agglomerative* — leaves merging up, not the root splitting down. Same dendrogram either direction.)

## Notation & setup

- $n$ observations, $p$ features. Standardize first — see [[standardization]].
- $d(x_i, x_{i'})$: a chosen point-to-point **dissimilarity** (Euclidean, $1-\rho$, Manhattan, cosine, …). See [[distance-metrics]].
- $D(A, B)$: the **set-to-set dissimilarity** induced by a linkage rule (see table below).
- Dendrogram axes: x-axis is just observation labels (arbitrary horizontal order — meaningless); y-axis is the **fusion height** = the dissimilarity at which two clusters were merged.

## The algorithm

> [!important] Algorithm 12.3 (per ISL / [[L22-unsupervised-2]])
> 1. Compute all $\binom{n}{2}$ pairwise dissimilarities. Treat each observation as its own cluster ($n$ clusters total).
> 2. **For** $i = n, n-1, \ldots, 2$:
>    - **(a)** Find the pair of clusters with smallest inter-cluster dissimilarity $D(\cdot, \cdot)$. Fuse them. Record the fusion height = that dissimilarity.
>    - **(b)** Recompute the inter-cluster dissimilarities between the new cluster and all remaining clusters.
> 3. Continue until one cluster remains. Plot the dendrogram.

## Linkage — the set-to-set dissimilarity

The catch: at step 2(a) onward, you're measuring distance between **sets** of points, not points. The linkage rule fixes this:

| Linkage  | $D(A, B) =$                                          | Notes |
|----------|------------------------------------------------------|-------|
| Complete | $\max_{i \in A, j \in B} d(x_i, x_j)$                | Most pessimistic; balanced clusters. |
| Single   | $\min_{i \in A, j \in B} d(x_i, x_j)$                | Most optimistic; can produce "trailing chains." |
| Average  | $\frac{1}{|A||B|} \sum_{i \in A, j \in B} d(x_i, x_j)$ | Compromise; balanced clusters. Most popular. |
| Centroid | $d(\bar{x}_A, \bar{x}_B)$                            | Used in genomics; can produce *inversions* (a fusion at lower height than its children). Not on the prof's short list. |

> "I don't think single linkage is particularly common. I can't think of any case where I've actually used it. I'm sure someone does, but I don't know who or when." — [[L22-unsupervised-2]]

> "Average and complete tend to yield more balanced clusters." — [[L22-unsupervised-2]]

The prof name-checks median linkage too; centroid linkage is in the book but not in his lecture. In R: `hclust(d, method = "complete" / "single" / "average")`.

## Hand-computation procedure (the −1-point exam question)

This is *the* canonical Module 10 exam question. From [[L27-summary]] §Q5 walkthrough on a 4×4 dissimilarity matrix:

> [!warning] Negative-point convention
> "Negative one point for each mistake … I won't give negative points, don't worry." — [[L27-summary]]

The prof grades hand-dendrograms strictly because the algorithm is mechanical and partial credit is hard to give: fix the matrix, fix the linkage, the answer is unique. Mistakes propagate.

**Recipe (complete linkage, the typical exam variant):**

1. **Smallest off-diagonal entry → first fusion.** Find the smallest $d_{ij}$ in the matrix; fuse $i$ and $j$ at that height.
2. **Recompute** the dissimilarities between the new cluster $\{i,j\}$ and every remaining point $k$. Under **complete** linkage: $D(\{i,j\}, k) = \max(d_{ik}, d_{jk})$. Under **single**: take the min. Under **average**: take the mean.
3. **Repeat** on the new (smaller) matrix: find the next-smallest entry, fuse, recompute.
4. Continue until one cluster.
5. **Draw the dendrogram**: x-axis labels = the observation indices in fusion order; y-axis = fusion heights. Each fusion is a horizontal bar at the fusion height, connecting the two children's vertical lines.

**Worked example sketch from L27** (slide-deck Exercise-2-from-the-book matrix):
- Input: $C = \begin{pmatrix} 0 & .3 & .4 & .7 \\ .3 & 0 & .5 & .8 \\ .4 & .5 & 0 & .45 \\ .7 & .8 & .45 & 0 \end{pmatrix}$.
- Smallest = $C_{12} = 0.3$ → fuse $\{1, 2\}$ at height 0.3.
- New 3×3 (complete linkage): $D(\{1,2\}, 3) = \max(.4, .5) = .5$; $D(\{1,2\}, 4) = \max(.7, .8) = .8$; $D(3, 4) = .45$.
- Smallest = $.45$ → fuse $\{3, 4\}$ at height 0.45.
- Final fusion: $D(\{1,2\}, \{3,4\}) = \max(.4, .5, .7, .8) = .8$ → fuse at height 0.8.
- Cut at height 0.6 (anywhere between 0.45 and 0.8) → **two clusters**: $\{1, 2\}$ and $\{3, 4\}$.

(Single linkage on the same matrix gives different intermediate fusion heights but here the same final partition. The exercise asks to do both and compare cuts at $K = 2$.)

## Insights & mental models

**Cuts give nested partitions.** Cutting higher → fewer clusters. Cutting lower → more clusters. Two cuts at different heights produce partitions where the finer one refines the coarser one. **Contrast K-means**: the $K=4$ and $K=5$ K-means solutions can be totally unrelated. "By construction, this approach does [give nesting]." — [[L22-unsupervised-2]]

**Dendrogram horizontal layout is meaningless.**
> "Don't assume, for example, that 9 and 2 are somehow closer together for some reason. Don't interpret this [dendrogram horizontal layout] as being as distances between things. Yeah, it's just a visualization." — [[L22-unsupervised-2]]

There are $2^{n-1}$ valid horizontal orderings of the same dendrogram (you can swap children at every fusion). Only the **vertical fusion height** carries information. Two leaves that are far apart on the x-axis but fuse at low height *are* similar.

**Hierarchical can be wrong for non-hierarchical data.** If your data has no hierarchical structure (the prof's example: people split by gender into 2 groups but by nationality into 3 groups — the partitions don't nest), hierarchical clustering gives suboptimal results and K-means may do better. "If you're trying to cluster data that doesn't necessarily have a hierarchical structure to it, then probably bad." — [[L22-unsupervised-2]]

**Average and complete linkage = balanced trees.** Single linkage tends to produce **chaining** — long, snaking clusters that grow one observation at a time because the min-distance keeps finding a near neighbor. The prof prefers average / complete for "more balanced clusters."

**The dendrogram is one object you can cut anywhere.** That's the whole point of going hierarchical: one fit, all $K$'s available simultaneously. Useful when you want to compare partitions across resolutions.

**Distance metric and linkage are independent choices.** $4 \text{ linkages} \times \text{many distances} = $ many clusterings. Different choices give visibly different dendrograms (slide deck illustrates).

## Exam signals

> [!important] Negative-point convention
> "Negative one point for each mistake … I won't give negative points, don't worry." — [[L27-summary]]

> "Don't assume, for example, that 9 and 2 are somehow closer together for some reason. Don't interpret this [dendrogram horizontal layout] as being as distances between things." — [[L22-unsupervised-2]]

> "Average and complete tend to yield more balanced clusters." — [[L22-unsupervised-2]]

> "I don't think single linkage is particularly common." — [[L22-unsupervised-2]]

> [Q5 worked walkthrough, complete linkage on a 4×4 matrix, with `−1 per mistake` rubric.] — [[L27-summary]]

## Pitfalls

- **Misreading horizontal proximity as similarity.** Big trap. *Only* fusion height (vertical) tells you similarity.
- **Forgetting to recompute** the dissimilarity matrix at each step. Each fusion shrinks the matrix by one row/column; the new entries depend on the linkage rule.
- **Mixing up linkage formulas under pressure.** Complete = max, single = min, average = mean. The recipe step that breaks most often: under complete linkage, when computing $D(\{i,j\}, k)$, the prof's worked example takes $\max(d_{ik}, d_{jk})$ — a common mistake is to take the min or the mean.
- **Forgetting to standardize.** Same as K-means: mixed units → meaningless distances.
- **Choosing single linkage by default**, then getting chained / unbalanced clusters. Default to complete or average.
- **Cutting at an unjustified height** in a worked-by-hand exam answer. State *why* you cut where you cut (gap in fusion heights, target $K$, etc.).
- **Centroid-linkage inversions.** A fusion can occur *below* the height of one of its children — the dendrogram looks visually broken. Avoid centroid linkage unless you have a domain reason.
- **Hierarchical on non-hierarchical data.** If the true groupings don't nest, K-means may beat hierarchical for any given $K$.

## Scope vs ISLR

- **In scope:** the agglomerative algorithm, the linkage menu (complete / single / average — centroid mentioned but not the prof's focus), the dendrogram-reading rules (height = info, x-axis = arbitrary), the nested-cuts property, when hierarchical is wrong (non-nested data), the standardization mandate, and the **hand-computation procedure** (the L27-flagged exam question).
- **Look up in ISLR:** §12.4.2 — full algorithm + Table 12.3 (linkage definitions) + Figures 12.10–12.16 (dendrogram interpretation, linkage comparison, scaling effects, correlation vs Euclidean shopper example).
- **Skip in ISLR:**
  - **Ward linkage formula** — name-checked in [[L22-unsupervised-2]] only; out per `docs/scope.md`.
  - **Centroid-linkage inversion mathematics** — book mentions, prof doesn't dwell.
  - **Divisive (top-down) clustering** — book mentions briefly; prof doesn't cover.
  - **Cluster-validity statistics, gap statistic, silhouette scores** — name-checked at most.

## Exercise instances

- **Exercise 10.4** — Perform hierarchical clustering on NYT stories. Use `hclust(dist(nyt_data[, -1]), method = "complete" / "single" / "average")`. Plot all three dendrograms; cut at $K = 2$ via `cutree`; compare the resulting clusters to the true labels (art / music) by plotting in PC1/PC2 space. The exercise's punchline: complete linkage actually does *worse* than K-means here for $K=2$ on this dataset; single and average linkage don't help. Lesson: hierarchical isn't always best, even when the data look hierarchical.

## How it might appear on the exam

- **Q5-style hand dendrogram (per [[L27-summary]]).** Given a 4×4 dissimilarity matrix, complete linkage. Find smallest entry → fuse → recompute with max → repeat → draw dendrogram with fusion heights labeled → state which observations are in each cluster after a specified cut. **−1 per mistake** on this question type. Show every step.
- **Hand dendrogram with single or average linkage** instead of complete — same recipe, swap max for min or mean.
- **Dendrogram-reading interpretation question.** Given a dendrogram image, "which two observations are most similar?" Look for the lowest fusion. "How many clusters does cutting at height $h$ produce?" Count the number of vertical lines crossing the horizontal cut.
- **Linkage comparison T-F:**
  - "Complete and average linkage tend to yield balanced clusters." → **True**.
  - "Single linkage often produces chained clusters." → **True**.
  - "The horizontal position of leaves on a dendrogram indicates similarity." → **False**.
- **Method comparison:** hierarchical vs K-means — when each is preferred. Hierarchical: don't know $K$, want a tree, suspect nested structure. K-means: known $K$, want a single flat partition, large $n$ where $\binom{n}{2}$ pairwise distances are too expensive.
- **Linkage choice question:** which linkage would you use if you wanted to allow long, snaking clusters? **Single.** Which would you use to discourage them? **Complete or average.**

## Related

- [[k-means-clustering]] — the partitioning counterpart; no nesting, must pick $K$.
- [[distance-metrics]] — Euclidean vs correlation vs others; the metric choice changes the dendrogram entirely.
- [[standardization]] — mandatory preprocessing.
- [[dimensionality-reduction]] — common preprocessing for clustering high-dim data; PCA also used to *visualize* the resulting clusters (Exercise 10.4).
