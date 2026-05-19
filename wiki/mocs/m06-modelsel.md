---
moc: 06-modelsel
title: "M06: Model Selection and Regularization"
lectures: [L12, L13, L14, L15]
isl-ch: 6
slides:
  - modules/6ModelSel/selection_regularization_presentation_lecture1.md
  - modules/6ModelSel/selection_regularization_presentation_lecture2.md
exercises:
  - exercises/Exercise6/
  - exercises/compulsory-exercise-1.md
tags:
  - moc
  - module/06-modelsel
---

# Module 06: Model Selection and Regularization

The prof's framing: "regularization is constraint" and the most important variant of model selection in modern ML, trading a little bias for a lot of variance reduction. Four lectures (Feb 23, Feb 24, Mar 2, Mar 3) covering subset selection → shrinkage (ridge / lasso / elastic net) → dimensionality reduction (PCR / PLS) → high-dim motivation. The prof distrusts AIC / BIC / Cp and prefers cross-validation throughout.

## Slides
- <a href="/pdfs/m06-modelsel-slides.pdf" target="_blank" rel="noopener">m06-modelsel-slides.pdf</a>

## Lectures
- [[L12-modelsel-1]]: opens with "regularization as constraint", subset selection (best / forward / backward / hybrid), starts ridge
- [[L13-modelsel-2]]: reframes module as "reducing the variance"; ridge + lasso + elastic net; geometric ellipse-meets-diamond picture
- [[L14-modelsel-3]]: implicit-regularization teaser, Bayesian view of ridge/lasso (excluded), PCA + PCR pipeline
- [[L15-modelsel-4]]: PCR wrap, PLS, and the high-dimensional ($p > n$) motivation for the whole module

## Concepts (atoms in this module)
- [[subset-selection]]: best ($2^p$, infeasible past modest $p$) vs forward / backward / hybrid greedy searches
- [[ridge-regression]]: L2 penalty $\lambda\sum\beta_j^2$; smooth shrinkage that *never* hits zero; closed-form, works when $p > n$
- [[lasso]]: L1 penalty $\lambda\sum|\beta_j|$; corners on the axes drive coefficients to *exactly* zero → variable selection for free
- [[ridge-vs-lasso-geometry]]: RSS ellipses meet the L1 diamond at corners (sparsity) vs the L2 ball at smooth interior points; "capitalist vs socialist"
- [[elastic-net]]: combines L1 + L2; sparsity from lasso plus correlated-variable averaging from ridge
- [[principal-component-regression]]: standardize → PCA → regress $y$ on first $M$ PCs; orthogonal $Z$'s kill multicollinearity; ≈ a discretized ridge
- [[partial-least-squares]]: PCR but maximize $\text{Cov}(Z, Y)$ instead of $\text{Var}(Z)$; supervised dimensionality reduction; "no better than ridge but Swedish"
- [[high-dimensional-regression]]: $p > n$ breaks OLS ($X^\top X$ singular, $R^2 = 1$ always); regularization is *the* reason this module exists

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; this module revisits in [[L13-modelsel-2]] as "we're trading a little bias for a lot of variance reduction" (prof restated as exam-flagged)
- [[regularization]]: this module is its first systematic treatment; prof: "the most important variant of model selection"
- [[cross-validation]]: owned by module 5; the workhorse for picking $\lambda$ / number of PCs throughout this module ([[L12-modelsel-1]], [[L13-modelsel-2]], [[L15-modelsel-4]])
- [[standardization]]: mandatory before ridge, lasso, PCR, PCA; flagged repeatedly across [[L12-modelsel-1]]–[[L15-modelsel-4]] ("PCA is not scale invariant")
- [[double-descent]]: prof returns in [[L13-modelsel-2]] to qualify the bias-variance "tradeoff" framing; minimum-norm solutions among interpolators

## Exercises
- [[../../exercises/Exercise6]]: subset selection, ridge, lasso, PCR, PLS all on the Credit data (problems 6.3–6.9); the canonical drill set for this module
- [[../../exercises/compulsory-exercise-1]]: problem 4a writes 10-fold CV pseudocode (CV mechanic that this module relies on for $\lambda$ selection)

## Out of scope (this module)
- **AIC / BIC / Cp / adjusted-$R^2$ derivations and formulas** - "I really don't think I'm going to ask any questions about this" - [[L12-modelsel-1]] / [[L13-modelsel-2]]. Conceptual "they exist, penalize complexity, prof distrusts them" is in scope via [[aic-bic-conceptual]]; the algebra is not.
- **Bayesian interpretation of ridge / lasso (Gaussian / Laplace priors)** - "I really don't think I'd put this on the test, just because it kind of assumes a lot of knowledge that maybe you don't have" - [[L14-modelsel-3]]
- **L0 norm / "Optimal Brain Damage"** - "we won't go into it because it's not used in practice" - [[L14-modelsel-3]]
- **Detailed PLS history and chemometrics-specific tuning**: PCR is the workhorse; PLS atom captures algorithm + the "Swedish" verdict, no separate history - [[L14-modelsel-3]] / [[L15-modelsel-4]]
- **Elastic Net detailed tuning**: concept named in [[L13-modelsel-2]], no worked example
- **Spectral / eigen-decomposition derivations** - "we don't talk about spectral decomposition" - deferred to Linear Statistical Models - [[L04-statlearn-3]] (eigenvalue = PC variance is in [[explained-variance-and-scree-plot]]; full theory is out)

## ISLP pointer
Chapter 6: Linear Model Selection and Regularization. The deep treatment of in-scope concepts in this module lives in `wiki/book/06-modelsel.md`. Atoms carry section-level `isl-ref:` pointers, e.g. ridge §6.2.1, lasso §6.2.2, PCR §6.3.1, PLS §6.3.2, high-dim §6.4.
