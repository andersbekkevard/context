---
moc: 02-statlearn
title: Statistical Learning
lectures: [L02, L03, L04]
isl-ch: 2
slides:
  - modules/2StatLearn/2StatLearn.1.md
  - modules/2StatLearn/2StatLearn.2.md
exercises:
  - exercises/Exercise2/
  - exercises/compulsory-exercise-1.md
tags:
  - moc
  - module/02-statlearn
---

# Module 02: Statistical Learning

The vocabulary-and-decomposition module: Y = f(X) + ε, supervised vs unsupervised, prediction vs inference, parametric vs nonparametric, the train/test MSE U-shape, and the bias-variance decomposition the prof flags as exam-bait. Three lectures (Jan 12, 13, 19) covering ISL ch. 2, plus the random-vector / covariance / multivariate-normal plumbing that the rest of the course (modules 3 + 4) rides on. Load-bearing for everything downstream: bias-variance recurs in nearly every later module, and the multivariate-normal feeds OLS sampling theory and LDA/QDA.

## Lectures
- [[L02-statlearn-1]]: vocabulary (quantitative/qualitative, supervised/unsupervised, regression/classification, prediction/inference), Breiman's "Two Cultures"; first flag of the bias-variance exam question
- [[L03-statlearn-2]]: Y = f(X) + ε; reducible vs irreducible split; parametric vs nonparametric; polynomial-degree U-shape; full board derivation of the bias-variance decomposition
- [[L04-statlearn-3]]: over-parameterized digression (pseudo-inverse → benign overfitting / double descent); random vectors, covariance and correlation matrices, contrasts, multivariate normal

## Concepts (atoms in this module)
- [[parametric-vs-nonparametric]]: assume a form for f and estimate parameters vs let the data dictate the shape; linear regression vs KNN as the canonical contrast
- [[reducible-vs-irreducible-error]]: E[(Y − Ŷ)²] = (f − f̂)² + Var(ε); the cross term vanishes because E[ε] = 0; irreducible = noise floor
- [[flexibility-overfitting-underfitting]]: training MSE always falls with flexibility, test MSE is U-shaped; KNN-K and polynomial degree are the standard knobs
- [[knn-classification]]: non-parametric majority-vote classifier; small K wiggly islands (overfit), big K over-smoothed; killed by curse of dimensionality
- [[knn-regression]]: average the K nearest training y's; same K-as-flexibility story; the standing CV-target example for picking K
- [[random-vector-and-covariance]]: p-vector with covariance Σ (variances on diagonal, covariances off); E(AXB) = A·E(X)·B, Cov(CX) = CΣCᵀ; correlation = Σ rescaled by sd-diagonal
- [[contrasts]]: linear combination Z = CX; expectations and covariances follow from the random-vector machinery; cork worked example

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: **introduced and derived here** (this module owns its first systematic treatment); revisits in nearly every later module, named on every CE1 problem 1 sub-question
- [[multivariate-normal]]: introduced here as the joint distribution of a random vector; revisited in [[L05-linreg-1]] for the β̂ sampling distribution and in [[L09-classif-3]] as the LDA/QDA class-conditional
- [[double-descent]]: first introduced here in [[L04-statlearn-3]] via the 100,000-degree polynomial pseudo-inverse demo; the prof's hobbyhorse, returns five times across the course

## Exercises
- [[../../exercises/Exercise2]]: full module-2 drill set: prediction vs inference identification (2.1), flexible vs rigid bias-variance reasoning (2.2), correlation from a covariance matrix (2.3g), bivariate-normal simulation under four Σ patterns (2.4), polynomial-regression bias-variance simulation (2.5)
- [[../../exercises/compulsory-exercise-1]]: problem 1 entirely lives in this module: 1a (write expected test MSE), 1b (derive the 3-term bias-variance decomposition), 1c (interpret the three terms), 1d (T/F on the tradeoff), 1e (read a KNN bias-variance plot), 1f (correlation off a 2×2 Σ), 1g (match contour to Σ)

## Out of scope (this module)
- **Spectral / eigen-decomposition derivations of covariance** - "we don't talk about spectral decomposition" - deferred to Linear Statistical Models - [[L04-statlearn-3]]. Eigenvalue-as-PC-variance is captured later in [[principal-component-analysis]] / [[explained-variance-and-scree-plot]]; the full spectral theory is out.
- **Pseudo-inverse / Moore-Penrose mathematics** - used in L04 to demonstrate over-parameterized fits but never formally derived; the *concept* (minimum-norm interpolator) is captured in [[double-descent]], the algebra is out - [[L04-statlearn-3]]; reinforced "explicitly bracketed off" in [[L08-classif-2]]

## ISLR pointer
Chapter 2: Statistical Learning. The deep treatment of in-scope concepts in this module is in `book/02-statlearn.md`; the prof said "it's well written… it's the right source." Specific atoms carry section-level `isl-ref:` pointers (e.g. 2.1.1 reducible/irreducible, 2.1.2 parametric/nonparametric, 2.2.1 flexibility, 2.2.2 bias-variance, 2.2.3 KNN).
