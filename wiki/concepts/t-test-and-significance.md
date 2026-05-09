---
concept: t-test-and-significance
module: 03-linreg
lectures: [L05, L06]
isl-ref: 3.1.2, 3.2.2
exercises:
  - CE1 problem 2g — true/false on what a p-value does and does not mean
related: [linear-regression, sampling-distribution-of-beta, confidence-and-prediction-intervals, f-test, gaussian-error-assumptions]
tags:
  - concept
  - module/03-linreg
aliases:
  - t-test
  - p-value
  - statistical significance
  - practical significance
---

# t-test, p-values, statistical vs practical significance

The per-coefficient inference machine — and the prof's recurring sermon: **"significance is just sample size."** Big $n$ makes everything look significant; the slope size is what tells you whether it actually matters. "Ideally you want both."

## Definition (prof's framing)

Test the null that a single regression coefficient is zero.

$$H_0: \beta_j = 0 \quad \text{vs.} \quad H_1: \beta_j \neq 0.$$

Test statistic:

$$t = \frac{\hat\beta_j - 0}{\mathrm{SE}(\hat\beta_j)} \sim t_{n - p - 1} \text{ under } H_0.$$

Two-sided p-value: $P(|T| \ge |t_{\text{obs}}|)$ under $t_{n-p-1}$.

> "We assume the most boring thing and then we try to reject it being boring — you know, just like we do with people in jail. We assume they're innocent, which is boring, and then we try to argue that they're guilty, which is more interesting because everyone's innocent." — [[L05-linreg-1]]

## Notation & setup

- Df = $n - p - 1$ (multiple regression) or $n - 2$ (simple).
- $t \approx 1.96$ for the 5% two-sided cutoff once df is large enough; $\approx 2$ as a working value.
- The p-value, formally: probability of observing a test statistic at least as extreme as $t_{\text{obs}}$, **assuming $H_0$ is true.**
- Default null is $\beta_j = 0$; you can test against another value $\beta_j = c$ with $t = (\hat\beta_j - c)/\mathrm{SE}(\hat\beta_j)$.

## Formula(s) to know cold

$$\boxed{t_j = \frac{\hat\beta_j}{\mathrm{SE}(\hat\beta_j)}, \qquad \mathrm{df} = n - p - 1.}$$

Under multiple regression: $\mathrm{SE}(\hat\beta_j) = \hat\sigma \sqrt{c_{jj}}$ where $c_{jj}$ is the $j$-th diagonal of $(\mathbf{X}^\top\mathbf{X})^{-1}$, and $\hat\sigma = \sqrt{\mathrm{RSS}/(n - p - 1)}$.

Equivalence with the F-statistic at $p=1$: $F_{1, n-p-1} = t_{n-p-1}^2$. So a t-test on a single coefficient ↔ an F-test on the singleton.

## Insights & mental models

### "Significance is just sample size"

The most-quoted line of L05:

> "If $n$ is infinity… your standard [error] is going to be small as shit, which means it's going to look significant even if it isn't." — [[L05-linreg-1]]

Variance of $\hat\beta_j$ shrinks like $1/n$ → SE shrinks like $1/\sqrt n$ → t grows like $\sqrt n$. Any non-zero effect, no matter how small, becomes statistically significant for large enough $n$. "Significance is a notion of how many samples you have."

### Statistical vs practical significance

The distinction the prof keeps coming back to:

> "You could have a statistically very reliable, very confident, very, very confident that there is a relationship there, and it might not matter at all. The trend is like basically zero." — [[L05-linreg-1]]

- **Statistical significance**: is the effect real (small p, narrow CI)?
- **Practical significance**: is the slope big enough to matter?

The first is about reliability; the second is about effect size. You want both.

### Discipline differences

- **Engineering**: nobody cares about p-values. Effect size is everything. *"If you need statistics to show a relationship is meaningful, you don't study it, you just ignore it."* — [[L05-linreg-1]]
- **Biology / softer sciences**: effects are squishy and small; you have to lean on statistics. So you care about p-values, and the danger is conflating significance with meaning.

### Why not just look at individual p-values?

Slide flag, then prof gloss:

> "The variables can actually be correlated, and then none of them actually look significant, but overall the test is very significant." — [[L06-linreg-2]]

Sequence: F-test first ([[f-test]]) for "is anything good at all"; then drill into individual t-tests. Skipping Q1 leads to wrong inferences when predictors are correlated.

### What the p-value is *not*

A canonical exam trap (CE1 problem 2g) — all of the following are FALSE:

- "$1 - p$ is the probability that $H_0$ is true."
- "If $p > 0.05$, then $H_1$ is not true."
- "$p$ tells you the probability that the results happened by random chance."

The correct definition: "$p$ is the probability to observe a data summary under $H_0$ that is at least as extreme as the one observed."

## Exam signals

> "Statistical vs practical significance" — slide section header that the prof returned to multiple times in [[L05-linreg-1]] and [[L06-linreg-2]].

> "If $n$ is infinity… your standard [error] is going to be small as shit, which means it's going to look significant even if it isn't." — [[L05-linreg-1]]

> "Only checking individual p-values is dangerous." — slide flag in [[L06-linreg-2]]

> "I might say in the multivariate linear regression case, how would I test if at least one of the predictors is useful in predicting the response, or I might ask: why would I want to know this, what's the point?" — [[L06-linreg-2]]

> "What are the odds, the log odds? How do you compute them? What do they mean?" — [[L27-summary]] (analogous "compute and interpret" pattern from logistic regression — same per-coefficient inference muscle).

## Pitfalls

- **p-value misinterpretation.** All four CE1 2g traps. The p-value is conditional on $H_0$, not a probability of $H_0$.
- **Confusing rejection with truth.** Failing to reject $H_0$ ≠ "the null is true." It's "we don't have enough evidence to reject."
- **Multiple testing.** Looking at $p$ p-values and picking the smallest one inflates the family-wise false-positive rate. Module 12 doesn't cover formal corrections (Bonferroni / FDR are out of scope), but the *concept* of "more tests, more false positives" is fair.
- **Engineering vs biology trap.** A statistically significant slope of 0.01 may be practically meaningless; a non-significant slope of 5 may matter (but is unreliable). Always quote effect size alongside the p.
- **Ignoring correlation among predictors.** Individual t-tests can both be insignificant while the overall F is highly significant — see [[f-test]] and [[collinearity]].
- **Df off by one.** Simple LR: $n - 2$. Multiple LR: $n - p - 1$ (with $p$ slopes excluding intercept). Ignoring the $-1$ is benign for moderate $n$ but wrong on principle.

## Scope vs ISLR

- **In scope:** the t-statistic formula, what the p-value means and doesn't mean, statistical vs practical significance, why you sequence F-test before t-tests, the danger of relying on p-values alone.
- **Look up in ISLR:** §3.1.2 (pp. 67–68) for the simple-LR t-test; §3.2.2 (pp. 75–77, *Is There a Relationship?*) for the multiple-LR setup including the F-statistic.
- **Skip in ISLR:** formal multiple-testing correction theory (ISLR ch. 13) — entirely out of scope. Permutation / bootstrap-based p-values for regression — not covered until module 5 (and only conceptually).

## Exercise instances

- CE1 problem 2g — true/false on four canonical p-value misinterpretations. The correct definition: "$p$ is the probability to observe a data summary under $H_0$ that is at least as extreme as the one observed."

(Other module 3 exercises implicitly use t-tests when interpreting `lm()` output — Exercise 3.1c, 3.1d — but the per-coefficient t-test isn't the standalone learning target there.)

## How it might appear on the exam

- **CE1-style true/false on what a p-value means.** Almost certainly an exam question variant. Verbatim correct definition: "probability of observing a data summary under $H_0$ at least as extreme as the one observed."
- **Read significance from output.** Given a regression table with `Estimate, Std. Error, t value, Pr(>|t|)`, identify significant predictors at $\alpha = 0.05$. Quote both the p-value and the effect size.
- **Significance vs effect size T/F.** "A coefficient with $p < 0.001$ is practically meaningful" — not necessarily; could be a tiny effect with huge $n$.
- **"How would you test whether $\beta_j = 0$?"** State the t-statistic, the df, the t-distribution under $H_0$. Don't have to compute the p-value.
- **2025 Q7 / odds-ratio analog.** Logistic regression but same machinery — identify the per-coefficient z-test, interpret. The question style transfers.

## Related

- [[linear-regression]] — the model whose coefficients are tested
- [[sampling-distribution-of-beta]] — derivation of the t-statistic
- [[confidence-and-prediction-intervals]] — same machinery, alternative report
- [[f-test]] — joint test (Q1) before drilling into individual t's
- [[gaussian-error-assumptions]] — what the t-test relies on
