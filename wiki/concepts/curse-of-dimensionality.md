---
concept: curse-of-dimensionality
module: 04-classif
lectures: [L07, L08, L15]
isl-ref: 4.5.1
exercises: []
related: [knn-classification, linear-discriminant-analysis, dimensionality-reduction, principal-component-analysis, high-dimensional-regression, naive-bayes]
tags:
  - concept
  - module/04-classif
aliases:
  - high-dimensional geometry
---

# Curse of dimensionality

The headline failure mode of [[knn-classification|KNN]] (and many distance-based methods): in high dimensions, **all pairwise distances become nearly equal**, so "nearest neighbor" stops carrying information. The prof: "the dimensionality is a curse, and in this case, it's a curse for K-nearest neighbours… makes it useless."

## Definition (prof's framing)

> "If there's a number of dimensions and you look at the average distance between points, of course as you add more dimensions that average distance is going to be bigger. But the weird thing is that the variance , like how much those distances vary just by chance , gets smaller and smaller. … So that in this million-dimensional space, the points that are close together have almost the same distance value as points that are far away." - [[L08-classif-2]]

> "Because the nearest neighbors tend to be far away in high dimensions and the method is no longer local. This is referred to as the *curse of dimensionality*." , slide deck

## Notation & setup

- $p$: number of predictors / dimensions.
- $n$: number of observations.
- "Curse" kicks in when $p$ grows faster than $\log n$ (roughly) , there isn't enough data to fill the space.

## What goes wrong (mechanically)

Pick $n$ points uniformly in the unit hypercube $[0, 1]^p$. As $p$ grows:

1. **The volume of any local neighborhood shrinks rapidly.** To enclose a fixed *fraction* of the data, the neighborhood radius must grow toward $1$. So "local" stops meaning local.
2. **The ratio of nearest-to-farthest pairwise distance approaches 1.** The variance of pairwise distances shrinks relative to the mean , every point is roughly equally far from every other. Distance loses discriminating power.
3. **Most of the volume of the hypercube concentrates near the boundary.** The vast majority of points sit close to the "edges" rather than in the interior , KNN's nearest-neighbor estimates extrapolate.

The prof's recommendation: **simulate it yourself** , sample uniform points in $d$ dimensions, plot the distribution of pairwise distances as $d$ grows. The collapse-onto-a-mean is striking.

## Insights & mental models

- **Distance-based methods take the biggest hit.** [[knn-classification]], k-means clustering with Euclidean distance, kernel methods. Anything where "the closest point" is the answer.
- **Parametric methods often survive better.** [[logistic-regression]], [[linear-discriminant-analysis]] , they make stronger assumptions, so they don't have to estimate dense local neighborhoods.
- **LDA actually thrives in high $p$.** "Mapping it down to a dimension specifically to separate out these categories." - [[L09-classif-3]]. $K$ classes → $K - 1$ discriminant scores → fits the curve to the relevant low-d subspace.
- **The fixes are all "use lower dimensions":**
  - **Dimensionality reduction:** [[principal-component-analysis|PCA]], [[principal-component-regression|PCR]], LDA-as-projection.
  - **Regularization:** [[ridge-regression|ridge]], [[lasso]] , shrink to a low-d effective subspace.
  - **Restrictive parametric models:** [[naive-bayes]] (assume diagonal $\Sigma$), simpler families.
  - **Feature selection:** drop irrelevant predictors before running KNN.
- **High-dim multicollinearity is the regression-side analog.** When $p$ is large, predictors are increasingly likely to be near-collinear by chance , XᵀX gets ill-conditioned even before $p$ approaches $n$. See [[collinearity]] / [[high-dimensional-regression]].
- **The fix isn't always more data.** "$n$ much larger than $p$" , but in genomics ($p$ in the tens of thousands) you can't make $n$ keep up. Methods need to assume structure.

## Exam signals

> "The dimensionality is a curse, and in this case, it's a curse for K-nearest neighbours. … makes it useless." - [[L08-classif-2]]

> "Curse of dimensionality. Euclidean distance in high-dimensional spaces becomes large for everyone , this doesn't work too well if we have many, many covariates." - [[L07-classif-1]]

> "The effectiveness of the KNN classifier falls quickly when the dimension of the predictor space is high." , slide deck

The prof returned to it three times across module 4 and module 6 ([[high-dimensional-regression]]). The KNN-killing role is the most flagged framing.

## Pitfalls

- **Treating "high $p$" as a fixed cutoff.** It's relative to $n$. With $n = 10^6$ and $p = 50$, KNN can still work; with $n = 100$ and $p = 50$, it can't.
- **Standardization isn't a fix.** Standardizing puts every dim on the same scale, which is necessary, but doesn't change the underlying geometric concentration.
- **Switching to a different distance (cosine, Manhattan) helps marginally.** Prof: "Cosine distance might help in some cases, something you could play with that might be fun." , but the underlying curse persists; you've just changed the rate.
- **Confusing curse-of-dimensionality with multicollinearity.** Related, but distinct. Multicollinearity is about correlated predictors; CoD is about geometric distance concentration. In high $p$, both happen and reinforce.
- **Believing "all methods break in high $p$."** [[lasso]], LDA, PCA-based methods are designed exactly for this regime.

## Scope vs ISLR

- **In scope:** Definition (geometric concentration of distances), the KNN failure mode, why parametric methods are more robust, what the standard fixes are (dimensionality reduction, regularization).
- **Look up in ISLR:** §4.5.1, p. 161 (KNN-CoD discussion); §6.4 for the full high-dim regression story (modules 6 territory).
- **Skip in ISLR:** Detailed concentration-of-measure proofs (measure-theoretic). Specific bounds on the rate of collapse.

## Exercise instances

None , no exercise problem drills the curse directly. The concept appears as **scaffolding inside [[knn-classification]]** ("why does KNN fail in high $p$?"), as motivation for [[principal-component-regression|PCR]] / [[ridge-regression|ridge]] / [[lasso]] in module 6, and as motivation for [[naive-bayes]] (diagonal $\Sigma$ when $p$ is large).

## How it might appear on the exam

- **Conceptual MCQ:** "Which method is most affected by the curse of dimensionality?" → KNN. (LDA/QDA/logistic less so; PCA/lasso designed for it.)
- **T/F:** "Standardizing predictors removes the curse of dimensionality" → false (helps with comparability, but the geometry persists).
- **Method-comparison:** "For a problem with $n = 100$ and $p = 1000$, which classifier(s) would you avoid?" → KNN; consider LDA, naive Bayes, logistic with regularization.
- **Why naive Bayes wins in high $p$:** parameter count is $O(p)$ not $O(p^2)$, dodges the variance explosion.
- **Why LDA survives:** projects to $K - 1$ dimensions, exploits the low-d structure of the class means.

## Related

- [[knn-classification]]: the headline victim.
- [[linear-discriminant-analysis]]: the headline survivor (dimensionality-reduction method).
- [[dimensionality-reduction]]: the umbrella concept of mapping $p \to M < p$.
- [[principal-component-analysis]]: the standard unsupervised reduction.
- [[principal-component-regression]]: the regression-side application.
- [[high-dimensional-regression]]: the $p > n$ companion in module 6.
- [[naive-bayes]]: large-$p$ rescue via the diagonal-$\Sigma$ assumption.
- [[collinearity]]: the regression-side pathology that high-$p$ amplifies.
