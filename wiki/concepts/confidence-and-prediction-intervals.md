---
concept: confidence-and-prediction-intervals
module: 03-linreg
lectures: [L05, L06]
isl-ref: 3.2.2
exercises:
  - Exercise3.2b  -  simulate to demonstrate the frequentist CI interpretation
  - Exercise3.2c  -  simulate the PI at a fixed x₀
  - Exercise3.2d  -  CI for x₀ᵀβ vs PI for Y at x₀
related: [linear-regression, sampling-distribution-of-beta, t-test-and-significance, gaussian-error-assumptions]
tags:
  - concept
  - module/03-linreg
aliases:
  - CI
  - PI
  - prediction interval
  - confidence interval
---

# Confidence and prediction intervals

Two intervals around a regression prediction. **CI = where the true *mean response* lies**; **PI = where a *future observation* lands**. PI is always wider because it adds the irreducible noise $\sigma^2$. Both narrowest where data is densest, fanning out at the extremes.

## Definition (prof's framing)

For a fixed test point $\mathbf{x}_0$:

- **Confidence interval** = uncertainty in $\hat y_0 = \mathbf{x}_0^\top \hat{\boldsymbol\beta}$ as an estimator of the *expected response* $\mathbf{x}_0^\top \boldsymbol\beta$.
- **Prediction interval** = uncertainty in $\hat y_0$ as a *prediction of a future observation* $y_{\text{new}} = \mathbf{x}_0^\top \boldsymbol\beta + \varepsilon_{\text{new}}$ at $\mathbf{x}_0$, including the irreducible noise.

> "Plotting the confidence and prediction intervals around all predicted values $\hat Y_0$ one obtains the **confidence range** or **confidence band** for the expected values of $Y$. … The prediction range is much broader than the confidence range." , module 3 slides

CI for an individual coefficient $\beta_j$ ([[L05-linreg-1]]): $\hat\beta_j \pm t_{1-\alpha/2,\,n-p-1} \cdot \mathrm{SE}(\hat\beta_j)$. For 95% with reasonable $n$: $t \approx 1.96 \approx 2$.

## Notation & setup

- $\mathbf{x}_0$ = test point (fixed covariates).
- $\hat y_0 = \mathbf{x}_0^\top \hat{\boldsymbol\beta}$ = point prediction.
- Use the t-distribution with $n - p - 1$ df (since $\sigma$ is estimated). For large $n$, t ≈ N.
- $1 - \alpha$ = nominal confidence level (typically 0.95).

## Formula(s) to know cold

Confidence interval for a single coefficient:

$$\hat\beta_j \pm t_{1-\alpha/2,\,n-p-1} \cdot \mathrm{SE}(\hat\beta_j).$$

Confidence interval for the mean response at $\mathbf{x}_0$:

$$\mathbf{x}_0^\top \hat{\boldsymbol\beta} \pm t_{1-\alpha/2,\,n-p-1} \cdot \hat\sigma \sqrt{\mathbf{x}_0^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}_0}.$$

Prediction interval for a future observation at $\mathbf{x}_0$:

$$\mathbf{x}_0^\top \hat{\boldsymbol\beta} \pm t_{1-\alpha/2,\,n-p-1} \cdot \hat\sigma \sqrt{1 + \mathbf{x}_0^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}_0}.$$

The PI carries an extra **+1** under the square root , that's the irreducible $\sigma^2$ contribution. It's why PI > CI **always**.

## Insights & mental models

### Two sources of uncertainty in PI, one in CI

CI accounts for: uncertainty in $\hat{\boldsymbol\beta}$ (only).

PI accounts for: uncertainty in $\hat{\boldsymbol\beta}$ + irreducible noise $\varepsilon_0$.

> "To answer this question [PI], we have to sum uncertainty over two components: (1) the uncertainty in the predicted value $\hat y_0$ (due to uncertainty in $\hat{\boldsymbol\beta}$); (2) the irreducible error $\varepsilon_0 \sim N(0, \sigma^2)$." , module 3 slides

### The frequentist CI interpretation

> "There is a 95% probability that the interval [from the random procedure] will contain the *true* value of $\beta_j$." , module 3 slides

Crucially: the *interval* is random, the parameter is fixed. Repeat the experiment many times → ~95% of the constructed intervals cover the true $\beta_j$. CE1 problem 2g (true/false on p-values) hammers the related "$1 - p$ is the probability $H_0$ is true" trap; CIs have the same misinterpretation risk.

### Interval shape

Both CI and PI are narrowest near the centroid of the data and fan out at the extremes , the $\mathbf{x}_0^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_0$ term grows with distance from the mean. Visually: the CI band hugs the line; the PI band is a wide envelope.

### Why CI for x₀ᵀβ ≠ PI for Y at x₀

The CI is for the *mean* , the expected response. The PI is for an *individual* observation , a single random draw from $N(\mathbf{x}_0^\top\boldsymbol\beta, \sigma^2)$. Even with infinite data ($\hat{\boldsymbol\beta} \to \boldsymbol\beta$), the CI shrinks to a point but the PI stays wide because $\sigma^2 > 0$.

## Exam signals

> "We will discuss confidence and prediction ranges in the (more general) multiple linear regression setup." , module 3 slides

> "**Confidence intervals (CIs) are a much more informative way to report results than $p$-values!**" , module 3 slides

(Both intervals are derived in **problem 2 of the recommended exercises**.) , [[L06-linreg-2]]

## Pitfalls

- **CI vs PI confusion.** If asked "what's the uncertainty around an individual prediction at $x_0 = 50$?" → PI. If asked "what's the uncertainty in the *average* response at $x_0 = 50$?" → CI. Mixing them up is the canonical exam slip.
- **CI for $\beta_j$ vs CI for $\mathbf{x}_0^\top\boldsymbol\beta$ vs PI for $y$ at $\mathbf{x}_0$.** Three distinct objects, three different formulas; don't conflate. Exercise 3.2d makes you walk through all three.
- **Misinterpreting "95% probability."** It's *the procedure's coverage rate*, not "this specific interval has 95% probability of containing $\beta$." Once you compute the interval, $\beta$ is either in it or not.
- **PI fails if assumptions fail.** Both rely on Gaussian errors; PI especially relies on the residual variance estimate being valid. Heteroscedasticity → PI is wrong.
- **Always wider for PI.** A common slip: PI ⊃ CI strictly. If you draw a band that has CI > PI, you've swapped them.

## Scope vs ISLP

- **In scope:** difference between CI and PI; their derivation in matrix form; the t-distribution-based formulas; the band shape; the frequentist interpretation.
- **Look up in ISLP:** §3.2.2 (pp. 81–82, *Predictions*) , concise treatment with the +1 in the PI; figure 3.6 shows the band shape.
- **Skip in ISLP:** Bayesian credible intervals , out of scope. Bonferroni / multiple-testing corrections to the CI , never covered.

## Exercise instances

- Exercise3.2b: simulate $Y = 1 + 3X + \varepsilon$ for $\sim 1000$ datasets; check empirically that the 95% CI covers $\beta_0$ and $\beta_1$ ~95% of the time
- Exercise3.2c: same simulation philosophy, but for the PI at a fixed $x_0 = 0.4$
- Exercise3.2d: construct CI for $\mathbf{x}_0^\top\boldsymbol\beta$; explain the connection between CI for $\beta_j$, CI for $\mathbf{x}_0^\top\boldsymbol\beta$, and PI for $Y$ at $\mathbf{x}_0$

## How it might appear on the exam

- **Distinguish CI and PI.** Definition or T/F question on which is wider, what each represents, which contains $\sigma^2$.
- **Read intervals from a band plot.** Given a regression with the usual two-band plot, identify which is CI which is PI; predict at a new $x$ value and quote the appropriate interval.
- **Frequentist interpretation T/F.** "If we computed 100 95% CIs from 100 random samples, ~95 would cover the true $\beta$." Correct interpretation.
- **CI from regression output.** Given $\hat\beta_j$ and $\mathrm{SE}(\hat\beta_j)$ from a table, compute the 95% CI as $\hat\beta_j \pm 2 \cdot \mathrm{SE}$ (1.96 ≈ 2 trick).
- **Derive PI from CI by adding $\sigma^2$.** Conceptual question: how does the formula change?

## Related

- [[linear-regression]]: the underlying model
- [[sampling-distribution-of-beta]]: both intervals derive from this
- [[t-test-and-significance]]: same machinery, different question
- [[gaussian-error-assumptions]]: both intervals require these
