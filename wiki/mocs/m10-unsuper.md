---
moc: 10-unsuper
title: "M10: Unsupervised Learning"
lectures: [L21, L22]
isl-ch: 12
slides:
  - modules/10Unsuper/10Unsuper.md
  - modules/10Unsuper/unsupervised_learning_presentation_lecture2.md
exercises:
  - exercises/Exercise10/
tags:
  - moc
  - module/10-unsuper
---

# Module 10: Unsupervised Learning

Two lectures (Apr 13, Apr 14) on the "no $Y$" half of the course: PCA as the canonical dimensionality-reduction tool, then K-means and hierarchical clustering as the discrete-summary cousins. Load-bearing for the exam: the PCA explained-variance / score arithmetic (Q3e-style) and the **hierarchical-clustering hand-dendrogram** with the `−1 point per mistake` rubric (Q5-style); k-means hand computations also fair game. Standardization is mandatory for everything in this module.

## Lectures
- [[L21-unsupervised-1]]: wraps module 9 ([[xgboost]], [[partial-dependence-plots]]); opens module 10 with the "unsupervised is dangerous statistics" framing; full PCA from scratch (max-variance optimization, SVD/eigendecomposition, loadings, scree plot, USArrests + eigenfaces); standardization mandate; PCA fails on curved data → motivates clustering
- [[L22-unsupervised-2]]: K-means (algorithm, local-minima rerun fix, $K$ as hyperparameter), hierarchical clustering (agglomerative, dendrogram, complete/single/average linkage), Euclidean-vs-correlation distance via the shopper example; pivots into module 11 NN history preview at the end (out of scope for this MOC)

## Concepts (atoms in this module)
- [[principal-component-analysis]]: rotate to orthogonal directions of decreasing variance; eigenvectors of the (standardized) covariance; loadings interpret PCs in terms of original variables; **PCA is not scale invariant**
- [[explained-variance-and-scree-plot]]: PVE = $\lambda_m / \sum \lambda_k$; chop at the elbow or at a 90/95/99% cumulative threshold; supervised use cases should pick $M$ by CV instead
- [[k-means-clustering]]: partition into $K$ disjoint clusters minimizing within-cluster squared Euclidean distance; iterative centroid → assign loop converges to a *local* minimum, so rerun with `nstart ≥ 20`; $K$ is a hyperparameter you must justify
- [[hierarchical-clustering]]: agglomerative bottom-up merging with complete/single/average linkage; dendrogram cuts give nested partitions; the **−1-point-per-mistake** hand-computation question
- [[distance-metrics]]: Euclidean groups infrequent shoppers, correlation ($1-\rho$) groups people with similar preferences; metric is *the* design choice in clustering
- [[dimensionality-reduction]]: umbrella concept linking PCA / PCR / PLS / LDA-projection / NN-feature-extractor; supervised vs unsupervised is the key axis

## Cross-cutting concepts touched (Specials)
- [[standardization]]: first introduced module 06 for ridge/lasso; this module is where it becomes universally mandatory, PCA, k-means, and hierarchical clustering all break without it ([[L21-unsupervised-1]] / [[L22-unsupervised-2]])
- [[curse-of-dimensionality]]: first introduced module 04; revisits here as the reason Euclidean distance "sucks in high-dimensional spaces", affects clustering and motivates dim-reduction-before-clustering
- [[collinearity]]: first introduced module 03; this module's PCA is the rotation-based fix (orthogonal $Z$'s by construction)

## Exercises
- [[../../exercises/Exercise10]]: full module 10 drill on the NYT stories dataset: 10.1 PCA biplot + PVE / cumulative PVE; 10.2 prove K-means objective is monotone non-increasing using the $\sum_{i,i'} \to \sum_i \|\cdot - \bar x_k\|^2$ identity; 10.3 K-means with `nstart=20`, visualize in PC1/PC2 space, compare to true art/music labels; 10.4 hierarchical with complete + single + average linkage, `cutree` at $K=2$, compare to truth

## Out of scope (this module)
- **Spectral / eigen-decomposition theory of the covariance matrix** - "we don't talk about spectral decomposition" (deferred to Linear Statistical Models) - [[L04-statlearn-3]]; use the eigenvalue = PC variance fact, don't derive it
- **Non-negative matrix factorization (NMF) / "parts" version of eigenfaces** - name-checked only as a contrast to PCA - [[L21-unsupervised-1]]
- **K-means++ initialization, weighted KNN, Ward linkage formula, gap statistic, silhouette** - mentioned as alternatives only - [[L22-unsupervised-2]]
- **Wasserstein, cosine, Manhattan distance internals** - name-checked ("favorite because it sounds cool"); the *idea* "distance choice matters" is in scope, the metric internals are not - [[L22-unsupervised-2]]
- **Soft / mixture-model K-means (EM, mixtures of Gaussians)**: book §12.4.3 mentions; never lectured
- **Divisive (top-down) hierarchical clustering**: book mentions briefly; prof doesn't cover - [[L22-unsupervised-2]]
- **§12.3 Missing values / matrix completion**: book-only, never lectured
- **History questions** (who invented PCA, year of X) - "I'm not going to ask you a history question" - [[L22-unsupervised-2]] / [[L27-summary]]

## ISLP pointer
Chapter 12: Unsupervised Learning. Deep treatment of in-scope concepts (PCA in §12.2, K-means in §12.4.1, hierarchical in §12.4.2, distance choice in §12.4.2 with Figure 12.15) is in `wiki/book/12-unsupervised.md`. Atoms carry section-level `isl-ref:` pointers.
