---
concept: sampling-distribution-of-beta
module: 03-linreg
lectures: [L05, L06]
isl-ref: 3.1.2, 3.2.1
exercises:
  - Exercise3.2a - derive distribution of β̂ in matrix form, find Var(β̂ⱼ)
related: [linear-regression, least-squares-and-mle, design-matrix-and-hat-matrix, gaussian-error-assumptions, confidence-and-prediction-intervals, t-test-and-significance, collinearity, multivariate-normal]
tags:
  - concept
  - module/03-linreg
aliases:
  - distribution of beta hat
  - SE of beta
---

# Sampling distribution of β̂

Under the classical Gaussian linear model, $\hat{\boldsymbol\beta}$ is exactly multivariate normal, centered on the true $\boldsymbol\beta$ (unbiased) with covariance $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$. This is the source of all subsequent inference: t-tests, F-tests, CIs, PIs.

## Definition (prof's framing)

> "If you have a billion parameters, what's the uncertainty of them, and they're all working against each other? It becomes very confusing. But in this case, you can do it very well." - [[L05-linreg-1]]

Multiple regression result (proved in Exercise 3.2a, derived in [[L06-linreg-2]]):

$$\hat{\boldsymbol\beta} \sim N_{p+1}\bigl(\boldsymbol\beta,\ \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}\bigr).$$

Simple regression special case: each component is univariate Gaussian, centered on the true value, with variance read off the diagonal of the covariance matrix.

> "That's what we want. If it was biased then we'd be upset because then our model is not going to give us the right shit." - [[L06-linreg-2]]

## Notation & setup

- $\hat{\boldsymbol\beta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$: see [[design-matrix-and-hat-matrix]].
- True $\boldsymbol\beta$ unknown; we estimate $\hat{\boldsymbol\beta}$.
- $\sigma^2$ unknown; estimated by $\hat\sigma^2 = \mathrm{RSS}/(n - p - 1)$.
- $\mathrm{SE}(\hat\beta_j)^2 = \hat\sigma^2 \cdot c_{jj}$, where $c_{jj}$ is the $j$-th diagonal of $(\mathbf{X}^\top\mathbf{X})^{-1}$.

## Formula(s) to know cold

Multiple regression:

$$\boxed{\hat{\boldsymbol\beta} \sim N_{p+1}(\boldsymbol\beta,\ \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1})}$$

Per-coefficient variance:

$$\mathrm{Var}(\hat\beta_j) = \sigma^2 [(\mathbf{X}^\top\mathbf{X})^{-1}]_{jj}.$$

Simple regression closed forms (the only ones easy to write without matrix inversion):

$$\mathrm{SE}(\hat\beta_1)^2 = \frac{\sigma^2}{\sum_i (x_i - \bar x)^2}, \qquad \mathrm{SE}(\hat\beta_0)^2 = \sigma^2 \left[\frac{1}{n} + \frac{\bar x^2}{\sum_i (x_i - \bar x)^2}\right].$$

Residual standard error (estimator of $\sigma$):

$$\hat\sigma = \mathrm{RSE} = \sqrt{\frac{\mathrm{RSS}}{n - p - 1}}.$$

For simple regression, $n - 2$ in the denominator (two df eaten by $\hat\beta_0$ and $\hat\beta_1$). With multiple regression, $n - p - 1$.

## Insights & mental models

### Derivation in three lines

The proof (Exercise 3.2a, sketched in [[L06-linreg-2]]):

Write $\hat{\boldsymbol\beta} = \mathbf{C}\mathbf{y}$ with $\mathbf{C} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. Use $\mathbf{y} \sim N_n(\mathbf{X}\boldsymbol\beta, \sigma^2\mathbf{I})$. Then by the linear-transformation property of the multivariate normal:

- $\mathsf{E}(\hat{\boldsymbol\beta}) = \mathbf{C}\mathbf{X}\boldsymbol\beta = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}\boldsymbol\beta = \boldsymbol\beta$.
- $\mathrm{Cov}(\hat{\boldsymbol\beta}) = \mathbf{C}\sigma^2\mathbf{I}\mathbf{C}^\top = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}$.
- Linear function of multivariate normal → multivariate normal.

Conclusion: $\hat{\boldsymbol\beta} \sim N_{p+1}(\boldsymbol\beta, \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1})$. ∎

### Experiment design from the variance formula

The simple-regression form

$$\mathrm{SE}(\hat\beta_1)^2 = \frac{\sigma^2}{\sum_i (x_i - \bar x)^2}$$

tells you how to design experiments. We can't shrink $\sigma^2$ (it's a property of the noise), but we can:

- **Increase $n$**: more samples.
- **Spread $x$ wider**: sample further apart in $x$.

> "It is kind of weird to think that you can look at these equations and then from that gain an intuition of how you can do your experiment better. But you do." - [[L05-linreg-1]]

### Significance is just sample size

> "If $n$ is infinity… your standard [error] is going to be small as shit, which means it's going to look significant even if it isn't." - [[L05-linreg-1]]

The variance shrinks like $1/n$, so any non-zero effect eventually becomes statistically significant for big enough $n$. See [[t-test-and-significance]].

### Why $\mathbf{X}^\top\mathbf{X}$ matters: collinearity

The variance has the constant-$\sigma^2$ baked in, plus the data-dependent factor $(\mathbf{X}^\top\mathbf{X})^{-1}$. When two predictors are nearly the same, $\mathbf{X}^\top\mathbf{X}$ is near-singular, its inverse blows up, the diagonal entries grow without bound, variances explode. See [[collinearity]].

> "This factor X transpose X comes into play in particular when two variables are basically the same, because then they can trade off each other and then this variance explodes." - [[L06-linreg-2]]

### Estimated SE vs true SE

Strictly, the SE you can compute uses $\hat\sigma$ in place of the unknown $\sigma$. So tests use the **t distribution** (heavy-tailed Gaussian) with $n - p - 1$ df, not the standard normal. For $n \gtrsim 30$ they're indistinguishable.

## Exam signals

> "A lot of the reasons we ask those questions is so we can make tests on them." - [[L06-linreg-2]]

> "Result (proving this is **problem 2 of the recommended exercises**)" - [[L06-linreg-2]]

> "I think this is really why statisticians love these distributions, because you can read out what's going to happen when you look at them." - [[L06-linreg-2]]

## Pitfalls

- **Wrong df.** Simple LR uses $n - 2$; multiple LR uses $n - p - 1$ (where $p$ is the number of slopes, not counting the intercept). Conventions differ in books; say which you're using.
- **Estimated vs known $\sigma$.** With unknown $\sigma$, use t with $n - p - 1$ df, not standard normal. The hat is implicit in how R reports SE.
- **The multivariate covariance has off-diagonal entries.** $\hat\beta_0$ and $\hat\beta_1$ are *not* generally independent in simple LR. Their covariance becomes zero only if $\bar x = 0$.
- **Bias is a function of the model, not the estimator.** $\hat{\boldsymbol\beta}$ is unbiased *for the true $\boldsymbol\beta$ in the assumed model.* If the true model is non-linear, the LS slope is unbiased for the *best linear approximation*, not for the curve.
- **Inflation under collinearity.** A coefficient estimate may be near-zero with a huge SE, looks "insignificant" but the joint test (F) over the correlated set may still be highly significant. See [[t-test-and-significance]] and [[f-test]].

## Scope vs ISLP

- **In scope:** the multivariate normal sampling distribution, derivation of mean and covariance, the simple-regression SE formulas, residual standard error.
- **Look up in ISLP:** §3.1.2 (pp. 63–66, simple LR SE), §3.2.1 (matrix-form result, lighter derivation).
- **Skip in ISLP:** specifics of the t- and F-distributions are referenced but not derived; ISLP is light here. Walpole is the prof's recommended classical reference for the $\chi^2_{n-p-1}$ distribution of $\hat\sigma^2$.

## Exercise instances

- Exercise3.2a: full derivation: show $\hat{\boldsymbol\beta}$ has the stated distribution; what assumptions are needed; what does this imply for $\hat\beta_j$; how to compute $\mathrm{Var}(\hat\beta_j)$.

## How it might appear on the exam

- **Write the distribution of $\hat{\boldsymbol\beta}$** (and the assumptions under which it holds): could be a true/false or short-derivation question.
- **Derive the per-coefficient variance** in simple LR. Standard "show your work" question; hand-derive $\mathrm{SE}(\hat\beta_1)^2 = \sigma^2/\sum(x_i - \bar x)^2$.
- **What happens to the SE when...?** Add more data (down by $\sqrt n$); spread $x$ wider (down); two predictors become highly correlated (up, collinearity).
- **Read SE from regression output.** 2025 Q6a-style: given the table, what is the estimate, what is the 95% CI? Use $\hat\beta_j \pm t_{0.975, n-p-1} \cdot \mathrm{SE}(\hat\beta_j)$.

## Related

- [[linear-regression]]: the underlying model
- [[least-squares-and-mle]]: derivation of $\hat{\boldsymbol\beta}$
- [[design-matrix-and-hat-matrix]]: source of $(\mathbf{X}^\top\mathbf{X})^{-1}$
- [[gaussian-error-assumptions]]: what gives us the multivariate-normal result
- [[confidence-and-prediction-intervals]]: built directly from this distribution
- [[t-test-and-significance]]: the standardized coefficient test
- [[collinearity]]: what blows up the variance
- [[multivariate-normal]]: the cross-cutting prerequisite distribution
