---
concept: partial-least-squares
module: 06-modelsel
lectures: [L14, L15]
isl-ref: 6.3.2
exercises:
  - Exercise6.9  -  apply PLS on Credit, compare to PCR and other module 6 methods
related: [principal-component-regression, principal-component-analysis, ridge-regression, lasso, dimensionality-reduction, standardization, cross-validation]
tags:
  - concept
  - module/06-modelsel
aliases:
  - PLS
  - partial least squares regression
---

# Partial least squares (PLS)

The supervised cousin of [[principal-component-regression|PCR]]: same compress-then-regress pipeline, but the components are chosen to maximize $\text{Cov}(X\phi, Y)$ instead of $\text{Var}(X\phi)$, so $Y$ guides the directions. The prof's editorial verdict, slide-quoted: *"PLS often performs no better than ridge regression or PCR but it's Swedish, so it's like they're meatballs, they're not better but they sound good."* - [[L15-modelsel-4]]. Atom is correspondingly compact: know the algorithm, the contrast with PCR, and the verdict.

## Definition (prof's framing)

> "It's the same idea as the principal component analysis, only now you're finding the principal components not as the directions of maximal variance of $X$, but the maximal covariance of $X$ and $Y$." - [[L15-modelsel-4]]

> "PLS works similar to PCR. … But it uses the response $Y$ in order to identify new features, attempts to find directions that help explain both the response and the predictors.", slide deck (selection_regularization_presentation_lecture2.md)

## Notation & setup

- Same skeleton as PCR: build orthogonal directions $Z_1, \dots, Z_M$ as linear combinations of standardized $X$, then regress $Y$ on the first $M$.
- **Difference from PCR**: each $\phi_{jm}$ is chosen to maximize $\text{Cov}(Z_m, Y)$ (subject to $\sum_j \phi_{jm}^2 = 1$), not $\text{Var}(Z_m)$.
- $M$ chosen by [[cross-validation]].
- **Standardize both $X$ and $Y$** before fitting (ISLP §6.3.2, verbatim: *"We generally standardize the predictors and response before performing PLS."*). This is the asymmetry vs PCR, where only $X$ needs standardizing because $Y$ isn't used to construct components. In PLS, $Y$ enters every loading via $\phi_{j1} \propto \text{Corr}(X_j, Y)$, so $Y$'s scale propagates into the directions; standardizing keeps the covariance objective scale-equivariant in $Y$. `sklearn`'s `PLSRegression` does this by default (`scale=True`).

## The algorithm

To get the first PLS component:

1. **For each $X_j$, regress $Y$ on $X_j$ alone** → simple-regression coefficient $\hat\beta_j$. This is **proportional to** $\text{Corr}(Y, X_j)$.
2. **Set $\phi_{j1} = \hat\beta_j$**. So $Z_1 = \sum_j \phi_{j1} X_j$ weights each $X_j$ by how well it (alone) explains $Y$.

Slide-flagged version:

> "$\phi_{j1}$ is the coefficient from the simple linear regression of $Y$ onto $X_j$. This coefficient is proportional to the correlation between $Y$ and $X_j$. PLS puts highest weight on the variables that are most strongly related to the response.", slide deck

To get $Z_2$ and onward:

> "We regress each variable on $Z_1$ and take the residuals. The residuals are remained info not explained by $Z_1$. We the[n] compute $Z_2$ using this orthogonalized data, similarly to $Z_1$. We can repeat this iteration process $M$ times to get $Z_1, \dots, Z_M$.", slide deck

So the new direction at each step is the maximum-covariance direction in the part of $X$-space not yet explained by previous components. Same Gram-Schmidt-style deflation idea as PCA's "next orthogonal direction with maximum variance," just substituting covariance for variance.

## Insights & mental models

### PLS is supervised; PCR is unsupervised, the only conceptual difference

The whole point. PCA / PCR picks directions of high $X$-variance, ignoring $Y$. PLS picks directions of high $\text{Cov}(X, Y)$, using $Y$ at every step. So PLS partially fixes PCR's biggest weakness, the no-guarantee-the-high-variance-directions-of-X-relate-to-Y problem flagged in [[principal-component-regression]].

> "It can sound like cheating, 'it kind of sounds like you're doing regression twice', but it works." - [[L15-modelsel-4]]

### Verdict: not a clear winner

> "In practice, PLS often performs no better than ridge regression or PCR. Supervised dimension reduction of PLS can reduce bias. It also has the potential to increase variance.", slide deck

> "All of these tend to behave similarly. A lot of things come down to looking like ridge regression, surprisingly. And conveniently, ridge regression behaves smoothly instead of this kind of discrete thing which PCR and PLS both have." - [[L15-modelsel-4]]

The prof's editorialized ranking, verbatim ([[L15-modelsel-4]]):

> "Lasso falls somewhere between ridge and best subset regression and has some nice properties of each. For me, definitely the most interesting things are lasso and also implicit lasso. Many different ways you can get lasso and ridge regression. Those are, I think, the most interesting part of this module. But being exposed to the ideas of partial least squares regression and PCR are good because they're fairly common, certainly PCR is very common."

So: he likes lasso (and ridge); PCR is "fairly common" so worth knowing; PLS is niche-but-good-to-recognize.

### Local color (not testable, but characterizes the prof's framing)

> "Developed in the 70s by Herman Wold (Swedish) for chemometrics, they had lots of variables, needed fast linear methods, computers were terrible, and PCA wasn't doing it for them." - [[L15-modelsel-4]]

> "Locally relevant because… commonly used in this field called chemometrics, and one guy who's fairly prominent in chemometrics and also in the development of these methods is actually Harold Martens, if anyone's heard of him. He was big in this partial least squares stuff. He wrote some of the early papers in the 70s." - [[L14-modelsel-3]]

Per [[scope]] §"Module 06": *"Detailed PLS history and chemometrics-specific tuning"* is out of scope. Local color only.

### PLS, PCR, ridge, converging behaviour

Slide summary:

> "PLS, PCR and ridge regression tend to behave similarly. Ridge regression may be preferred because it shrinks smoothly, rather than in discrete steps."

In summary, all three are non-sparse predictive smoothers. Use [[lasso]] or [[subset-selection]] when you actually need variable selection. Use ridge / PCR / PLS when you just want predictive accuracy and don't care about which raw variables drive it.

## Exam signals

> "PLS often performs no better than ridge regression or PCR but it's Swedish." - [[L15-modelsel-4]] (verbatim slide quote, the prof read this aloud)

> "It's the same idea as the principal component analysis, only now you're finding the principal components not as the directions of maximal variance of $X$, but the maximal covariance of $X$ and $Y$." - [[L15-modelsel-4]]

The PLS-vs-PCR contrast (supervised vs unsupervised) is the most likely exam-handle. The prof's verdict tells you not to over-invest: **know what PLS is and how it differs from PCR; don't memorize tuning details.**

Per [[scope]]: *"Detailed PLS history and chemometrics-specific tuning"*, explicitly out. Concept and one-line distinction from PCR, in.

## Pitfalls

- **Confusing PLS with PCR.** Same compress-then-regress skeleton; the *only* difference is what's being maximized (covariance vs variance) and that PLS uses $Y$.
- **Believing "supervised → strictly better than unsupervised."** PLS can reduce bias *and* increase variance, net effect is often a wash vs ridge / PCR (slide-flagged).
- **Forgetting to standardize.** Same as PCA / PCR.
- **Treating PLS as a variable-selection method.** It isn't, like PCR, all back-transformed $\hat\beta_j$ are typically nonzero.
- **Memorizing the deflation algorithm in detail.** Out of scope; the high-level "regress each $X$ on $Y$ for $\phi_{j1}$, deflate, repeat" is enough.

## Scope vs ISLP

- **In scope:** the PLS pipeline (same as PCR but supervised); the contrast with PCR (uses $Y$ → maximizes $\text{Cov}(X, Y)$ instead of $\text{Var}(X)$); the algorithm at the level "regress each $X_j$ on $Y$ to get $\phi_{j1}$, then deflate"; the prof's verdict ("often no better than ridge or PCR, but supervised").
- **Look up in ISLP:** §6.3.2 (pp. 286–288). Brief, ISLP doesn't dwell on PLS either.
- **Skip in ISLP / out of scope:** detailed deflation algebra; chemometrics tuning; the various PLS variants (PLS-1, PLS-2, kernel PLS); history beyond "Wold, 1970s, chemometrics." Per [[L15-modelsel-4]] and [[scope]].

## Exercise instances

- Exercise6.9, Apply PLS on the Credit dataset and compare with previous methods (OLS / [[subset-selection]] / [[ridge-regression]] / [[lasso]] / [[principal-component-regression|PCR]]).

(Slide labels this "Recommended exercise 12"; it maps to RecEx6 problem 9 in the actual exercise file.)

## How it might appear on the exam

- **PLS vs PCR conceptual question**: "What is the key difference between PLS and PCR?" → PLS is supervised, uses $Y$ to choose components by maximizing $\text{Cov}(X\phi, Y)$, while PCR is unsupervised, uses only $X$ to maximize $\text{Var}(X\phi)$.
- **Multiple choice / true-false**: "PLS uses the response variable $Y$ when constructing components." → True. "PLS guarantees lower test MSE than PCR." → False (slide-flagged: often no better).
- **Method comparison from a results table**: PLS / PCR / ridge / lasso side by side on the same data. Recognize that the first three behave similarly; lasso is the differentiator if sparsity matters.
- **Probably not asked**: detailed PLS algorithm pseudocode; chemometrics applications; history.

## Related

- [[principal-component-regression]]: the unsupervised cousin; PLS is "PCR but uses $Y$."
- [[principal-component-analysis]]: PCA's variance-maximization is what PLS replaces with covariance-maximization.
- [[ridge-regression]]: slide-flagged behaviour: PLS, PCR, ridge "tend to behave similarly," ridge preferred for smoothness.
- [[lasso]]: the variable-selecting alternative; different philosophy entirely.
- [[dimensionality-reduction]]: the umbrella concept.
- [[standardization]]: required preprocessing.
- [[cross-validation]]: for choosing $M$.
