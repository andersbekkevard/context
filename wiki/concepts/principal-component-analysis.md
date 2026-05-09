---
concept: principal-component-analysis
module: 10-unsuper
lectures: [L08, L14, L15, L21]
isl-ref: 12.2
exercises:
  - Exercise6.7  -  pick how many PCs to use for the Credit dataset, justify
  - Exercise10.1  -  biplot + PVE / cumulative-PVE plots on the NYT stories data
related: [explained-variance-and-scree-plot, dimensionality-reduction, principal-component-regression, standardization, k-means-clustering, hierarchical-clustering, multivariate-normal, collinearity]
tags:
  - concept
  - module/10-unsuper
aliases:
  - PCA
---

# Principal component analysis (PCA)

The prof's first dimensionality-reduction tool, and his first unsupervised method. Frame: **rotate the data into orthogonal directions of decreasing variance**, then keep only the first few. Optimization view = "find the unit-norm direction of maximal variance"; linear-algebra view = eigendecomposition / SVD of the centered covariance matrix. Same algorithm; one is the intuition, the other is the recipe.

## Definition (prof's framing)

> "Find the best way of rotating our shit such that now it has a maximum variance." - [[L21-unsupervised-1]]

The first principal component is the unit-norm linear combination $\phi_1 \in \mathbb{R}^p$ of the (standardized) features whose projection $z_{i1} = \phi_1^\top x_i$ has the largest sample variance. The second PC is the unit-norm direction orthogonal to $\phi_1$ with largest remaining variance. And so on up to $p$.

> "PCA is the thing that I always start with and we often move away from it because it's too simple but everyone uses it. … It doesn't work very well, but you understand it. It's easy. It's fast. It's simple. It's based on a lot of very nice math. It's old. It's not going to do anything too weird." - [[L21-unsupervised-1]]

## Notation & setup

- $X \in \mathbb{R}^{n \times p}$: data matrix, **column-centered (and standardized) before PCA**.
- $\phi_m = (\phi_{1m}, \ldots, \phi_{pm})^\top$: the **loading vector** of PC $m$. Unit norm: $\sum_j \phi_{jm}^2 = 1$.
- $z_{im} = \sum_j \phi_{jm} x_{ij}$: the **score** of observation $i$ on PC $m$. The $z_m \in \mathbb{R}^n$ is a new variable.
- $\lambda_m$: variance of $z_m$ = the $m$-th eigenvalue of the sample covariance $\Sigma = \tfrac{1}{n-1} X^\top X$.
- "Standardize" means subtract column mean *and* divide by column SD, i.e. z-score every variable. See [[standardization]].

## Formula(s) to know cold

The optimization that defines the first PC:

$$
\max_{\phi_1} \;\frac{1}{n}\sum_{i=1}^n \left(\sum_{j=1}^p \phi_{j1} x_{ij}\right)^2
\quad \text{subject to} \quad \sum_{j=1}^p \phi_{j1}^2 = 1.
$$

Equivalently (after centering): the first PC direction is the eigenvector of $\Sigma = \tfrac{1}{n-1} X^\top X$ corresponding to the largest eigenvalue. Subsequent PCs are eigenvectors of decreasing eigenvalues.

The score for observation $i$ on PC $m$:
$$z_{im} = \phi_{1m} x_{i1} + \phi_{2m} x_{i2} + \ldots + \phi_{pm} x_{ip}.$$

This computation is the core of the L27-flagged exam question (see "How it might appear" below): plug an observation into a loading vector to get its score.

## Insights & mental models

**It is a rotation, not a rescaling.** The unit-norm constraint exists *exactly* so the algorithm can't cheat by inflating variance through scaling: "If you didn't subject that constraint, you could just make it … arbitrarily create high numbers, but also because we don't want to rescale the data." - [[L21-unsupervised-1]]

**Decorrelation by construction.** Subsequent PCs are orthogonal $\Rightarrow$ scores are uncorrelated. "You're reducing it into directions that are no longer correlated with each other. You're removing all these annoying correlations in your data." - [[L21-unsupervised-1]] This is the conceptual bridge to [[principal-component-regression|PCR]] and to the [[collinearity]] fix.

**Loadings interpret PCs in terms of original variables.** $\phi_{jm}$ tells you how much variable $j$ participates in PC $m$. Big $|\phi_{jm}|$ → variable $j$ is loud in PC $m$. On `USArrests`: PC1 loads roughly equally on Murder/Assault/Rape (an "overall criminality" axis), PC2 loads heavily on UrbanPop (urbanization). On the NYT stories data (Exercise 10.1): PC1 separates music from art, PC2 separates within-art topics.

**Two equivalent geometries.** (1) Direction of maximal variance. (2) The hyperplane that minimizes squared distance to the points, PCA is the *best* $M$-dimensional linear approximation of the data in squared-error terms (ISL 12.2.2). The prof goes through view (1); view (2) is the book's "another interpretation."

**SVD vs eigendecomposition.** Both work; choice is purely computational. "It's not obvious that it's the same problem … but the eigenvalue is equal to the variance of the principal component." - [[L15-modelsel-4]] / [[L21-unsupervised-1]]. SVD lets you exploit the smaller dimension when $n \ll p$ or vice versa.

**Full ≠ reduced.** Computing all $p$ PCs gives 100% of the variance, you've just rotated coordinates. Dimensionality reduction comes from **truncating** to the first $M < p$, see [[explained-variance-and-scree-plot]].

**Uniqueness up to sign.** "The principal component vector is unique. So there's only going to be one. Of course, the sign flip is boring. It just means which direction it is." - [[L21-unsupervised-1]] If two software packages give opposite-sign loadings, that's not a bug.

**PCA is unsupervised.** It uses only $X$, never $Y$. So when you use it inside [[principal-component-regression|PCR]], the largest-variance directions might be unrelated to the response, this is PCR's signature failure mode. PLS fixes it by using $\text{Cov}(X, Y)$ instead.

**PCA is linear.** Curved data (a circle, an arc) defeats it: "PCA is kind of stuck just making linear stuff. It can't do anything non-linear." - [[L21-unsupervised-1]] That's the bridge to clustering (next lecture).

## Exam signals

> "Find the best way of rotating our shit such that now it has a maximum variance." - [[L21-unsupervised-1]]

> "Standardize before PCA." - [[L21-unsupervised-1]]

> "PCA is not scale invariant." - [[L14-modelsel-3]] (verbatim slide bullet).

> "If you don't standardize, then it'll just be dominated by the strong one." - [[L14-modelsel-3]]

> "This kind of question I would also say is fair game because it tests basic knowledge of PCA that we covered in class, with simple calculations. And again, show your work so that you can get partial credit when the inevitable little mistake happens." - [[L27-summary]] (on a PCA explained-variance / loading / score question)

## Pitfalls

- **Standardize first.** Otherwise the largest-unit variable dominates the first PC purely because its numbers are bigger. The slide deck has a `prcomp(..., scale = FALSE/TRUE)` demo on USArrests showing exactly this. Without scaling, on USArrests the first PC loads almost entirely on `Assault` (variance 6945), meaningless artifact of units. With scaling, PC1 picks up the substantive "overall criminality" axis.
- **PCA is not scale invariant**: unlike OLS, where scaling a column just rescales its $\beta$. Here, scaling changes the answer.
- **Don't over-interpret loadings.** They are correlations in disguise; signs can flip between runs (uniqueness up to sign).
- **PCA only handles linear correlation.** "If your issue is actually not a linear correlation but some sort of complicated thing, then it might not find it." - [[L14-modelsel-3]]
- **Unsupervised**: PCs may not align with what predicts $Y$. For supervised dimension reduction, use [[partial-least-squares|PLS]] or LDA.
- **Computing all $p$ PCs is not dimensionality reduction**: you've just rotated. The reduction comes from chopping at some $M$.

## Scope vs ISLR

- **In scope:** the optimization view (max variance + unit-norm constraint + orthogonality), the eigendecomposition recipe, loadings, scores, the standardization mandate, the USArrests / eigenfaces / ad-spending examples, PCA-as-visualization, PCA inside [[principal-component-regression|PCR]], PCA's failure on curved data.
- **Look up in ISLR:** §12.2, full chapter. §12.2.2 (closest-hyperplane interpretation), §12.2.3 (PVE derivation in the $R^2 = 1 - \text{RSS}/\text{TSS}$ form), §12.2.4 (scaling discussion + uniqueness up to sign + how-many-PCs).
- **Skip in ISLR:**
  - **Spectral / eigen decomposition derivations of covariance** - [[L04-statlearn-3]]: "we don't talk about spectral decomposition" (deferred to Linear Statistical Models). Use the eigenvalue fact, don't derive it.
  - **§12.3 Missing values / matrix completion**: book-only, never lectured. Skip.
  - **Non-negative matrix factorization (NMF) / "parts" version of eigenfaces**: name-checked in [[L21-unsupervised-1]] only. Out per `docs/scope.md`.

## Exercise instances

- **Exercise 6.7**: How many PCs for the Credit dataset? Justify. Pure scree-plot / cumulative-PVE reading. Drills the [[explained-variance-and-scree-plot]] companion atom.
- **Exercise 10.1**: On NYT stories: produce a biplot (interpret), PVE plot, cumulative PVE plot. Realistic "use PCA + read its outputs" workflow. Also illustrates the loading-interpretation drill (PC1 ↔ music vs art).

## How it might appear on the exam

- **Q3e-style calculation (per [[L27-summary]]):** five standardized variables, eigenvalues / loadings / one observation given. Three sub-questions:
  1. Total variance explained by the first $M$ PCs → sum the eigenvalue ratios.
  2. How many PCs needed for 90% / 95% variance → cumulative sum.
  3. Compute the score $z_{im}$ for one observation on one PC → plug into the loading vector. **Show work** for partial credit.
- **Output interpretation:** given a biplot, describe what the data look like in PC1/PC2 space and what the loadings tell you about the original variables.
- **Conceptual MC / T-F:**
  - "PCA is scale invariant." → **False** (this is the canonical trap).
  - "Standardization affects PCA results." → **True**.
  - "The first PC always points in the direction of the response $Y$." → **False** (PCA is unsupervised).
  - "Loadings sum to one." → **False** (their *squares* sum to one; they're a unit vector).
- **Conceptual short-answer:** why standardize before PCA? (units → variance → first PC dominated by largest-unit variable). Why is PCA called "rotation"? (orthonormal change of basis, no rescaling, that's what the unit-norm constraint enforces).
- **Method comparison:** PCR vs lasso vs ridge, which uses PCA, which uses $Y$, which can do exact zeros, etc. (See [[principal-component-regression]].)

## Related

- [[explained-variance-and-scree-plot]]: the "how many PCs to keep" companion: PVE, cumulative PVE, scree.
- [[dimensionality-reduction]]: the umbrella concept; PCA is the canonical instance.
- [[principal-component-regression]]: PCA as the front end of a regression pipeline (module 6 use case).
- [[standardization]]: mandatory preprocessing step. PCA is the textbook reason this matters.
- [[multivariate-normal]]: the theoretical setting where eigenvectors of $\Sigma$ have the cleanest interpretation.
- [[collinearity]]: the problem PCR/PCA fixes by rotating to orthogonal directions.
- [[k-means-clustering]], [[hierarchical-clustering]], the other module-10 unsupervised tools; clustering is the discrete-summary cousin to PCA's continuous low-dim summary.
