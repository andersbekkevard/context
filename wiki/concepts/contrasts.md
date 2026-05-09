---
concept: contrasts
module: 02-statlearn
lectures: [L04]
isl-ref: null
exercises: []
related: [random-vector-and-covariance, multivariate-normal, linear-regression, categorical-encoding-and-interactions]
tags:
  - concept
  - module/02-statlearn
aliases:
  - linear combinations
  - C X
---

# Contrasts (linear combinations)

A **contrast** is any linear combination of a random vector's components , e.g. N − S, E + W, or (E + W) − (N + S) on the cork-deposit data. Once you write the contrasts as $\mathbf Z = C \mathbf X$ for a constant matrix $C$, their expectations and covariances drop out of the random-vector machinery: $E(\mathbf Z) = C \boldsymbol\mu$ and $\mathrm{Cov}(\mathbf Z) = C \boldsymbol\Sigma C^\top$. The prof flagged the cork worked example as "a good exercise to do in the exercise session."

## Definition (prof's framing)

> "[A contrast is] any linear combination of the variables you find interesting: e.g. N − S, E + W, (E + W) − (N + S). Once you've defined them as new variables, you can take their expectations and covariances using the same machinery." - [[L04-statlearn-3]]

Formally, given a random vector $\mathbf X_{(p \times 1)}$ and a constant matrix $C_{(k \times p)}$, the new random vector
$$\mathbf Z = C \mathbf X = \begin{pmatrix} \sum_{j=1}^p c_{1j} X_j \\ \vdots \\ \sum_{j=1}^p c_{kj} X_j \end{pmatrix}$$
holds the $k$ contrasts.

## Notation & setup

- $\mathbf X_{(p \times 1)}$: original random vector with mean $\boldsymbol\mu$ and covariance $\boldsymbol\Sigma$.
- $C_{(k \times p)}$: contrast matrix. Each row is one linear combination's coefficients.
- $\mathbf Z_{(k \times 1)} = C \mathbf X$: vector of $k$ contrasts.
- "Contrast" in the strict statistical sense means the row-coefficients sum to zero (so the linear combination is invariant under shifts of the mean), but in this course "contrast" is used loosely for *any* linear combination of interest. The cork example "E + W" wouldn't qualify under the strict definition; the prof's usage is the looser one.

## Formula(s) to know cold

The two formulas to know cold here are the same two from [[random-vector-and-covariance]]:
$$\boxed{\;E(\mathbf Z) = E(C \mathbf X) = C \boldsymbol\mu, \qquad \mathrm{Cov}(\mathbf Z) = \mathrm{Cov}(C \mathbf X) = C \boldsymbol\Sigma C^\top\;}$$

If $\mathbf X$ is multivariate normal, then $\mathbf Z = C\mathbf X$ is also multivariate normal , that's one of the four useful properties of the [[multivariate-normal]] from the slides ("linear combinations of components are multivariate normal"). So $\mathbf Z \sim N_k(C\boldsymbol\mu, C \boldsymbol\Sigma C^\top)$.

## Insights & mental models

**Contrasts as feature engineering for the multivariate setting.** You define new variables that capture the comparison you actually care about , e.g. on the cork data, "is the cork deposit denser on the south side than the north?" maps to $Y_1 = X_S - X_N$, and "is the east-west contrast different from the north-south contrast?" maps to $Y_3 = (X_E + X_W) - (X_N + X_S)$. Once you have $C$, the rest is matrix algebra.

**The cork worked example** (`modules/2StatLearn/2StatLearn.2.md` / [[L04-statlearn-3]]):

For three contrasts (N − S, E + W, (E + W) − (N + S)) on $\mathbf X = (X_N, X_E, X_S, X_W)^\top$:
$$C = \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & 1 \\ -1 & 1 & -1 & 1 \end{bmatrix}$$

Then $E(\mathbf Z) = C \boldsymbol\mu$ and $\mathrm{Cov}(\mathbf Z) = C \boldsymbol\Sigma C^\top$, both computed by direct matrix multiplication. In R: `C %*% mu` and `C %*% Sigma %*% t(C)`. The exercise asks you to write down $C$, identify $E(Y_1)$ as $\mu_N - \mu_S$, and find $\mathrm{Cov}(Y_1, Y_3) = $ (row-1 of $C \boldsymbol\Sigma$) dotted with (row-3 of $C$).

**Why contrasts matter beyond M2:**

- **Hypothesis tests in regression** (Q4 of the four "important questions" in M3): testing whether two coefficients are equal, or whether a sum of coefficients differs from a baseline, is a contrast on $\hat\beta$. The covariance machinery $\mathrm{Var}(C\hat\beta) = \sigma^2 C (X^\top X)^{-1} C^\top$ is the foundation for the standard errors of those tests.
- **Categorical predictors with K levels** ([[categorical-encoding-and-interactions]]): the K − 1 dummies define K − 1 contrasts against the reference level. R's `contrasts()` function is named for exactly this. The prof uses "contrast" in this regression sense in M3.
- **PCA** ([[principal-component-analysis]]): each principal component is a contrast , a linear combination of the standardized predictors. The loadings $\phi_{jm}$ are the contrast coefficients, the PC variances are $\mathrm{Cov}(C\mathbf X)$ on the diagonal.
- **LDA / QDA**: the discriminant functions $\delta_k(\mathbf x) = \mathbf x^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu_k - \frac{1}{2} \boldsymbol\mu_k^\top \boldsymbol\Sigma^{-1} \boldsymbol\mu_k + \log \pi_k$ are linear contrasts in $\mathbf x$ , the decision boundary is where two such contrasts are equal.

So while the M2 atom is light, the *machinery* (covariance of a linear transformation) is everywhere downstream.

**Reading the result.** $\mathrm{Cov}(\mathbf Z)$ is a $k \times k$ matrix whose diagonal entries are $\mathrm{Var}(Y_i) = $ (row-$i$ of $C$) $\boldsymbol\Sigma$ (row-$i$ of $C$)$^\top$, and whose off-diagonal entries are the covariances between contrasts. From there you can pull correlations between contrasts the usual way.

## Pitfalls

- **Order matters in $C \boldsymbol\Sigma C^\top$, not $C^\top \boldsymbol\Sigma C$.** Easy transpose mistake.
- **The constant matrix $C$ has to be conformable**: $C$ is $(k \times p)$, $\boldsymbol\Sigma$ is $(p \times p)$, so $C\boldsymbol\Sigma$ is $(k \times p)$ and $C\boldsymbol\Sigma C^\top$ is $(k \times k)$. Check dimensions first.
- **A "contrast" in this course is just any linear combination**: don't get hung up on the stricter definition that requires the coefficients to sum to zero.
- **If $C$ has linearly dependent rows, $\mathrm{Cov}(\mathbf Z)$ will be singular** even if $\boldsymbol\Sigma$ wasn't , the contrasts you defined aren't truly $k$-dimensional.
- **Mean-centering vs not**: since $\mathrm{Cov}$ uses $\mathbf X - \boldsymbol\mu$, the constant intercept term doesn't appear in $\mathrm{Cov}(\mathbf Z)$ , but $E(\mathbf Z) = C\boldsymbol\mu$ does carry the means.

## Scope vs ISLP

- **In scope:** writing down a contrast matrix $C$, computing $E(C\mathbf X)$ and $\mathrm{Cov}(C\mathbf X)$ by hand, the cork example.
- **Look up in ISLP:** ISLP doesn't have a dedicated "contrasts" section in chapter 2 , the closest treatment is the categorical-encoding discussion in §3.3.1 (and the implicit contrast-matrix view of dummy coding). For the matrix-algebra theory, ISLP is light; Johnson & Wichern or any multivariate-stats text covers it formally.
- **Skip in ISLP:** none , this is a matrix-algebra fact, not a textbook topic.

## Exercise instances

No recommended-exercise problem tagged "contrasts" specifically. The cork worked example in the slide deck (`modules/2StatLearn/2StatLearn.2.md`) and the prof's "good exercise to do in the exercise session" remark in [[L04-statlearn-3]] are the de facto exercise instance , write down $C$ for (N − S, E + W, (E + W) − (N + S)), compute $E(\mathbf Y)$ and $\mathrm{Cov}(\mathbf Y)$ analytically and in R.

The downstream applications (regression coefficient testing in M3, PCA loadings in M10, discriminant functions in M4) are exercised heavily in their own atoms.

## How it might appear on the exam

- **Direct hand-calc.** Given $\boldsymbol\Sigma$ and a $C$ (probably 2 × 3 or 2 × 4), compute $\mathrm{Cov}(C\mathbf X) = C \boldsymbol\Sigma C^\top$. Pure plug-and-chug, well-suited to the 2026 short-answer format.
- **Identify a contrast in a regression context.** Given a regression with three group dummies, write the contrast matrix that tests "group 1 vs the average of groups 2 and 3."
- **Combined with multivariate normal.** Show that if $\mathbf X \sim N_p(\boldsymbol\mu, \boldsymbol\Sigma)$, then $C\mathbf X \sim N_k(C\boldsymbol\mu, C\boldsymbol\Sigma C^\top)$. (One-line application of the [[multivariate-normal]] property.)
- **Conceptual: why are PCA loadings called "contrasts"?** Because each PC is a linear combination of the standardized variables, and the loadings give the coefficients of that combination.

## Related

- [[random-vector-and-covariance]]: the parent atom; the formulas $E(C\mathbf X) = C\boldsymbol\mu$ and $\mathrm{Cov}(C\mathbf X) = C\boldsymbol\Sigma C^\top$ live there
- [[multivariate-normal]]: linear combinations of MVN are MVN; that's the route from this atom into LDA/QDA
- [[linear-regression]]: the regression coefficients themselves are a linear-combination story; their distribution follows the same machinery
- [[categorical-encoding-and-interactions]]: dummy-coded categorical predictors define contrasts against a reference level
- [[principal-component-analysis]]: each PC is a contrast in the strict sense (the loadings are the coefficient vector)
