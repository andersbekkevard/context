---
concept: multivariate-normal
modules: [02-statlearn, 03-linreg, 04-classif]
lectures: [L04, L05, L09]
isl-ref: 4.4.2
exercises:
  - Exercise2.4  -  simulate multivariate normal with mvrnorm under different Σ patterns (4 covariance settings)
  - CE1 problem 1g  -  match a contour plot to a given covariance matrix
related: [random-vector-and-covariance, contrasts, linear-regression, sampling-distribution-of-beta, linear-discriminant-analysis, quadratic-discriminant-analysis, naive-bayes, gaussian-error-assumptions, least-squares-and-mle]
tags:
  - concept
  - specials
aliases:
  - MVN
  - multivariate Gaussian
  - joint normal
---

# Multivariate normal distribution

The bridge between modules 2, 3, and 4. Module 2 introduces it as the joint distribution of a [[random-vector-and-covariance|random vector]] $\mathbf{X}$; module 3 uses it for the **sampling distribution of $\hat\beta$** in OLS; module 4 uses it as the **class-conditional density** $f_k(x)$ in [[linear-discriminant-analysis|LDA]] and [[quadratic-discriminant-analysis|QDA]]. Same density, three roles. The exam-bait facts: contour ellipsoids tell you the covariance structure (CE1 1g); marginals are normal; **zero covariance ⇒ independence under joint normality** (only here).

## Definition (prof's framing)

> "Today we're going to talk about random vectors. We're going to talk about the covariance matrix, correlation matrix, what those are, and also the normal distribution, in particular the multivariate case. And this is setting us up to talk about regression." - [[L04-statlearn-3]]

> "Do you guys know the relationship between the normal distribution and regression? … If you minimize the normal distribution, if you assume your data is normally distributed and you have it the mean parameterized by some model, then that's equivalent to linear regression. So this multivariate case is a way of understanding how we do regression in multiple variables." - [[L04-statlearn-3]]

For $\mathbf{x} \in \mathbb{R}^p$ with mean vector $\boldsymbol\mu$ and covariance matrix $\boldsymbol\Sigma$:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\boldsymbol\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2} (\mathbf{x} - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf{x} - \boldsymbol\mu)\right)$$

Reduces to the univariate normal $f(x) = (2\pi\sigma^2)^{-1/2} \exp(-(x-\mu)^2 / (2\sigma^2))$ when $p = 1$, *"you have to go really, really far out, but you can see how it reduces down."* ([[L04-statlearn-3]])

## Returns in other modules

- [[L04-statlearn-3]]: first introduction. Cork-trees-as-running-example. Univariate Gaussian generalized to $p$ dimensions: $x - \mu$ becomes a vector, $\sigma^2$ becomes $\Sigma$, $1/\sigma^2$ becomes $\Sigma^{-1}$ in the exponent, $\sigma$ in the normalizer becomes $|\Sigma|^{1/2}$. Ran out of time mid-contour-matching.
- [[L05-linreg-1]]: finishes the contour-matching exercise from L04. Then makes the **mindset shift from joint $(X, Y)$ distribution to conditional $Y \mid X$**, *"we will look at how things co-vary, but in the sense of how Y varies as a function of X."* This is the bridge to regression. Under Gaussian errors, $\hat\beta \sim N(\beta, \sigma^2 (X^\top X)^{-1})$, a multivariate normal in parameter space.
- [[L09-classif-3]]: used as the **class-conditional density** in LDA and QDA. $f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp(-\tfrac{1}{2}(x - \mu_k)^\top \Sigma^{-1} (x - \mu_k))$ for LDA (shared $\Sigma$); $f_k(x)$ uses $\Sigma_k$ for QDA. The whole module-4 generative-classifier story is multivariate normals. **Naive Bayes** = multivariate normal with **diagonal $\Sigma$** (predictors conditionally independent given class).

## Notation & setup

- $\mathbf{x} = (x_1, \dots, x_p)^\top \in \mathbb{R}^p$: a $p$-dimensional column vector.
- $\boldsymbol\mu = (\mu_1, \dots, \mu_p)^\top \in \mathbb{R}^p$: the mean vector.
- $\boldsymbol\Sigma$: the $p \times p$ [[random-vector-and-covariance|covariance matrix]]: $\Sigma_{ij} = \mathrm{Cov}(X_i, X_j)$, with variances $\sigma^2_j$ on the diagonal and covariances off.
- $|\boldsymbol\Sigma|$: determinant of $\Sigma$. **$|\Sigma| = 0$ is bad** ($\Sigma$ singular), division by zero in the density, "yuck" ([[L04-statlearn-3]]).
- $\boldsymbol\Sigma^{-1}$: the **precision matrix**. Plays the role of $1/\sigma^2$ in the univariate case.

Notation: $\mathbf{X} \sim N_p(\boldsymbol\mu, \boldsymbol\Sigma)$ means $\mathbf{X}$ has the multivariate normal distribution with mean $\boldsymbol\mu$ and covariance $\boldsymbol\Sigma$.

## Formula(s) to know cold

### The density

$$\boxed{\;f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\boldsymbol\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2} (\mathbf{x} - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf{x} - \boldsymbol\mu)\right)\;}$$

Mapping the pieces from the univariate Gaussian:

| Univariate | Multivariate |
|---|---|
| $x - \mu$ | $\mathbf{x} - \boldsymbol\mu$ (vector) |
| $(x-\mu)^2 / \sigma^2$ | $(\mathbf{x}-\boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf{x}-\boldsymbol\mu)$ (Mahalanobis distance) |
| $\sigma$ in normalizer | $|\boldsymbol\Sigma|^{1/2}$ |
| $\sqrt{2\pi}$ | $(2\pi)^{p/2}$ |

### Useful properties (listed by the prof)

From [[L04-statlearn-3]]:

- **Contours are ellipsoids.** Level sets $\{x : (x-\mu)^\top \Sigma^{-1} (x-\mu) = c\}$ are ellipsoids centered at $\boldsymbol\mu$, oriented and stretched by $\boldsymbol\Sigma$.
- **Linear combinations are normal.** $A \mathbf{X} + b \sim N(A\boldsymbol\mu + b, A \boldsymbol\Sigma A^\top)$ for any matrix $A$ and vector $b$. (See [[contrasts]].)
- **Marginals are normal.** Any subset of the components is multivariate normal in its own right.
- **Conditionals are normal.** $\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2$ is multivariate normal, this is what enables linear regression to be exact under joint normality.
- **Zero covariance ⇒ independence, under joint normality only.** This is special to Gaussians; in general, $\mathrm{Cov}(X, Y) = 0$ does not imply $X \perp Y$.

### Use in OLS sampling

Under Gaussian errors $\varepsilon \sim N(0, \sigma^2 I)$:

$$\hat\beta = (X^\top X)^{-1} X^\top \mathbf{y} \;\sim\; N_p(\beta, \;\sigma^2 (X^\top X)^{-1})$$

the sampling distribution of the OLS estimator is a $p$-variate normal. ([[sampling-distribution-of-beta]] for the full derivation; covered in module 3.)

### Use in LDA/QDA discriminants

LDA assumes $\mathbf{X} \mid Y = k \sim N_p(\boldsymbol\mu_k, \boldsymbol\Sigma)$, pooled $\Sigma$, class-specific means. Apply Bayes, take logs, drop $k$-independent terms:

$$\delta_k(\mathbf{x}) = \mathbf{x}^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu_k - \tfrac{1}{2} \boldsymbol\mu_k^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu_k + \log \pi_k$$

**linear in $\mathbf{x}$**, because the $\mathbf{x}^\top \Sigma^{-1} \mathbf{x}$ term is the same for every $k$ and drops out of the $\arg\max$. ([[L09-classif-3]])

QDA assumes class-specific $\boldsymbol\Sigma_k$ → the quadratic term doesn't cancel:

$$\delta_k(\mathbf{x}) = -\tfrac{1}{2} \mathbf{x}^\top \boldsymbol\Sigma_k^{-1} \mathbf{x} + \mathbf{x}^\top \boldsymbol\Sigma_k^{-1} \boldsymbol\mu_k - \tfrac{1}{2} \boldsymbol\mu_k^\top \boldsymbol\Sigma_k^{-1} \boldsymbol\mu_k - \tfrac{1}{2} \log |\boldsymbol\Sigma_k| + \log \pi_k$$

**quadratic in $\mathbf{x}$**, hence "Quadratic Discriminant Analysis." ([[L09-classif-3]], the prof flagged "where does the quadratic come from in QDA?" as a typical exam question.)

## Insights & mental models

### Reading contour plots (CE1 1g, the explicit exam-bait)

Two-dimensional MVN contours are ellipses. Reading them:

| Σ pattern | Contour appearance |
|---|---|
| $\sigma_1^2 = \sigma_2^2$, $\rho = 0$ | **Circle** centered at $(\mu_1, \mu_2)$, equal variances, no correlation |
| $\sigma_1^2 \neq \sigma_2^2$, $\rho = 0$ | **Axis-aligned ellipse** stretched along the larger-variance axis |
| $\rho > 0$ | Ellipse tilted **upward-right** (positive diagonal) |
| $\rho < 0$ | Ellipse tilted **upward-left** (negative diagonal) |
| $\sigma_1^2 \neq \sigma_2^2$, $\rho \neq 0$ | Tilted ellipse with the long axis closer to whichever variance is larger |

[[L05-linreg-1]] worked through these explicitly:
- "Circular ellipse, no diagonal pull → correlation 0, equal variances."
- "Diagonal pull going up → positive correlation."
- "Diagonal pull going down → negative correlation."
- "Stretched only along one axis → unequal variances, no correlation."

### Why MVN is load-bearing for the whole course

> "Today we will discuss Y given X. So not the joint distribution of them, but Y given X. So we're trying to essentially make a model of it… we will look at how things co-vary, but in the sense of how Y varies as a function of X." - [[L05-linreg-1]]

MVN gives a **joint model** of $(X_1, \dots, X_p, Y)$. The conditional $Y \mid X$ extracted from a joint MVN is **linear** in $X$ with **constant variance**, that's exactly the linear regression model. So all the OLS sampling theory (CIs, t-tests, F-tests, prediction intervals) is **exactly correct** when joint normality holds and only **approximately** correct otherwise.

### Why MVN is load-bearing for classification

LDA / QDA / Naive Bayes are all **generative** classifiers: model $P(\mathbf{X} \mid Y = k)$ instead of $P(Y \mid \mathbf{X})$. The choice of $P(\mathbf{X} \mid Y = k)$ as multivariate normal is **the** modeling assumption for module 4's generative half. Three flavors:

- **LDA**: $\mathbf{X} \mid Y = k \sim N(\boldsymbol\mu_k, \boldsymbol\Sigma)$, pooled covariance.
- **QDA**: $\mathbf{X} \mid Y = k \sim N(\boldsymbol\mu_k, \boldsymbol\Sigma_k)$, class-specific covariance.
- **Naive Bayes**: $\mathbf{X} \mid Y = k \sim N(\boldsymbol\mu_k, \mathrm{diag}(\boldsymbol\sigma_k^2))$, predictors conditionally independent given class.

The "where does the quadratic come from" question is direct algebra on the MVN density.

### Mahalanobis distance

The exponent of the MVN density is $-\tfrac{1}{2} D_M^2$, where

$$D_M^2(\mathbf{x}, \boldsymbol\mu) = (\mathbf{x} - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1} (\mathbf{x} - \boldsymbol\mu)$$

is the **Mahalanobis distance** (squared), Euclidean distance after rotating and rescaling by $\Sigma^{-1/2}$. It's the natural distance under joint normality. (Not on the syllabus by name, but the exponent of the density is exactly this.)

### Conditional MVN, why linear regression works

If $(\mathbf{X}, Y)$ is jointly MVN with the appropriate block structure, then

$$Y \mid \mathbf{X} = \mathbf{x} \sim N\!\big(\mu_Y + \boldsymbol\Sigma_{Y\mathbf{X}} \boldsymbol\Sigma_{\mathbf{XX}}^{-1} (\mathbf{x} - \boldsymbol\mu_X),\; \sigma^2_Y - \boldsymbol\Sigma_{Y\mathbf{X}} \boldsymbol\Sigma_{\mathbf{XX}}^{-1} \boldsymbol\Sigma_{\mathbf{X}Y}\big)$$

linear in $\mathbf{x}$ with constant variance. **This is the linear regression model**, derived from joint normality. The prof gestured at this in [[L04-statlearn-3]] without writing it out, *"the multivariate case is a way of understanding how we do regression in multiple variables."*

### Singular Σ

> "If the determinant is zero, then that's typically bad, because you divide by zero and yuck. And I think that's the point they're trying to make." - [[L04-statlearn-3]]

$|\Sigma| = 0$ means the data lies on a lower-dimensional subspace; the density isn't well-defined on the full $p$-dim space. This is the same pathology as **collinearity** in OLS ($X^\top X$ singular), extreme correlation between predictors. Fix: drop a variable, regularize, or use PCA.

## Exam signals

> "Today we will discuss Y given X. So not the joint distribution of them, but Y given X." - [[L05-linreg-1]] (the conditional → regression bridge)

> "If you minimize the normal distribution, if you assume your data is normally distributed and you have it the mean parameterized by some model, then that's equivalent to linear regression. So this multivariate case is a way of understanding how we do regression in multiple variables." - [[L04-statlearn-3]]

> "That's another good exam question, where does the quadratic come from in QDA? Or show that, yeah… it's an interesting point that simply by making the sigma $k$-dependent, we introduced a new term, and that term is quadratic." - [[L09-classif-3]]

CE1 problem 1g (matching contour plot to covariance matrix) is the **direct exam-style application**, the prof has explicitly drilled it into a compulsory exercise.

## Pitfalls

- **Zero covariance ⇒ independence is special to MVN.** In general, $\mathrm{Cov}(X, Y) = 0$ does NOT mean $X \perp Y$, only under joint normality. Don't carry this over to other distributions.
- **Joint normality of marginals ≠ joint normality.** Two normal marginals can have a non-Gaussian joint (counter-example: dependent normals concocted to be marginally Gaussian but jointly not). The MVN assumption is on the joint, not just on each component.
- **$|\Sigma| = 0$ breaks the density**: handle singular cases via dimensionality reduction or regularization.
- **Don't read "the variance is on the diagonal" as a special property**: it's the *definition* of $\Sigma_{ii} = \mathrm{Var}(X_i)$. Off-diagonal entries are covariances; rescaling to $\rho$'s gives the [[random-vector-and-covariance|correlation matrix]].
- **Standardization doesn't make data multivariate normal**: z-scoring sets means to 0 and variances to 1, but doesn't change the joint shape. Skewed data stays skewed after standardization.
- **Spectral / eigen-decomposition of Σ is OUT of scope.** [[L04-statlearn-3]] verbatim: *"we don't talk about spectral decomposition"*, deferred to Linear Statistical Models. You can use the contour-stretching intuition without the full eigen-machinery.
- **Pooled covariance is just a convex combination of class-specific covariances**: $\hat\Sigma_\text{pooled} = \sum_k \frac{n_k - 1}{n - K} \hat S_k$. Don't confuse it with the unconditional sample covariance.
- **The MVN assumption is a *modeling* assumption, not a fact about your data**, flagged for LDA/QDA in [[L09-classif-3]]: *"Maybe it's not a good idea to pretend that the X's are well modeled by a Gaussian, that's a good way to break a model."*
- **Contour shapes are about $\Sigma$, not $\mu$**: the mean vector just shifts the center of the ellipsoid; the shape and orientation are pure $\Sigma$.

## Scope vs ISLR

- **In scope:** the density formula, what each piece means, contour-matching, the role in LDA/QDA discriminants (with the where-does-the-quadratic-come-from derivation), zero-covariance ⇒ independence under normality, the basic properties (linear combos / marginals / conditionals stay normal).
- **Look up in ISLR:** §4.4.2 (LDA for $p > 1$, with the MVN density); §4.4.3 (QDA, where $\Sigma$ becomes $\Sigma_k$); §4.4.4 (Naive Bayes, where $\Sigma_k$ becomes diagonal).
- **Skip in ISLR (book-only, prof excluded):** spectral decomposition / eigenanalysis of $\Sigma$ - [[L04-statlearn-3]]: *"we don't talk about spectral decomposition"*, deferred to Linear Statistical Models. The eigenvalue-as-PC-variance fact comes back in PCA ([[principal-component-analysis]]) but the full spectral theory of $\Sigma$ doesn't.

## Exercise instances

- **Exercise2.4**: simulate from a multivariate normal with `mvrnorm` under four different covariance settings (uncorrelated equal var, uncorrelated unequal var, positive correlation, negative correlation). The hands-on version of the contour-matching exercise.
- **CE1 problem 1g**: match a contour plot to a given covariance matrix. **Single-choice question, explicit exam-style.** $\Sigma = \begin{pmatrix} 1 & 0.2 \\ 0.2 & 4 \end{pmatrix}$ → axis-aligned ellipse with the long axis along $X_2$ (variance 4 vs 1) and slight upward tilt (positive correlation 0.1).

## How it might appear on the exam

- **Contour-matching question (CE1 1g style)**: given a $\Sigma$, pick the right contour plot. Or vice versa. Read the variances off the diagonal (which axis is longer?), read the correlation off the off-diagonal (tilted up or down? how much?).
- **"Where does the quadratic come from in QDA?"**: algebra on the MVN density: when $\Sigma$ is the same for all classes, the $\mathbf{x}^\top \Sigma^{-1} \mathbf{x}$ term cancels in $\arg\max_k$; when $\Sigma_k$ differs by class, the term survives → quadratic in $\mathbf{x}$.
- **Derive the LDA discriminant from scratch**: start from $\arg\max_k \pi_k f_k(\mathbf{x})$, take logs, drop $k$-independent terms, end with $\delta_k(\mathbf{x})$. Same template applies to QDA but you keep the quadratic term.
- **T/F: "Zero covariance implies independence"**: depends on the assumption. Under joint normality, TRUE. In general, FALSE.
- **T/F: "If $X_1$ and $X_2$ are each marginally normal, then $(X_1, X_2)$ is multivariate normal"**: FALSE. Counter-examples exist.
- **"Why is the MVN assumption sometimes a bad idea?"**: flagged in [[L09-classif-3]]: real predictors aren't always Gaussian; LDA/QDA suffer if the MVN class-conditional model is far off.
- **Connection to OLS sampling**: under Gaussian errors, $\hat\beta$ is multivariate normal. From this you derive the sampling distribution (and CIs and t-tests).

## Related

- [[random-vector-and-covariance]]: the build-up: $\Sigma$, $\mathrm{Corr}$, expectation rules. Module 2 prerequisite.
- [[contrasts]]: linear combinations $C\mathbf{X}$; under MVN, $C\mathbf{X}$ is also MVN.
- [[linear-regression]]: the conditional of MVN $(\mathbf{X}, Y)$ gives the linear regression model exactly.
- [[gaussian-error-assumptions]]: the MVN of the error vector $\boldsymbol\varepsilon$; the assumption that makes OLS = MLE.
- [[least-squares-and-mle]]: OLS minimization is MLE under Gaussian errors. Direct consequence of the MVN density's exponent.
- [[sampling-distribution-of-beta]]: under Gaussian errors, $\hat\beta \sim N_p(\beta, \sigma^2 (X^\top X)^{-1})$.
- [[linear-discriminant-analysis]]: uses MVN class-conditionals with pooled $\Sigma$; discriminants linear in $\mathbf{x}$.
- [[quadratic-discriminant-analysis]]: uses MVN class-conditionals with class-specific $\Sigma_k$; discriminants quadratic in $\mathbf{x}$.
- [[naive-bayes]]: uses MVN class-conditionals with **diagonal** $\Sigma_k$; predictors conditionally independent given class.
- [[discriminant-score-and-decision-boundary]]: derived from the MVN-based discriminants.
- [[principal-component-analysis]]: eigenvectors of $\Sigma$ give the directions of maximal variance under joint normality.
