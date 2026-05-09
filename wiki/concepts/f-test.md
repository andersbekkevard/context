---
concept: f-test
module: 03-linreg
lectures: [L06]
isl-ref: 3.2.2
exercises: []
related: [linear-regression, t-test-and-significance, sampling-distribution-of-beta, collinearity, categorical-encoding-and-interactions]
tags:
  - concept
  - module/03-linreg
aliases:
  - F-statistic
  - ANOVA
---

# F-test for regression

Q1 of the four "important questions" — **is *at least one* of the predictors useful?** Generalizes the t-test to multiple coefficients; reduces to the t-test when $p = 1$ ($F_{1, n-p-1} = t^2_{n-p-1}$). The prof was emphatic: **"won't ask you to compute it"** — only why you'd use it and what null it tests.

## Definition (prof's framing)

Test the joint null that all slopes are zero:

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0 \quad \text{vs.} \quad H_1: \text{at least one } \beta_j \neq 0.$$

> "It's like we're assuming innocence so that we can prove guilty… We don't prove anything in statistics… at least I've never proved anything with data." — [[L06-linreg-2]]

Test statistic:

$$F = \frac{(\mathrm{TSS} - \mathrm{RSS})/p}{\mathrm{RSS}/(n - p - 1)} \sim F_{p,\, n-p-1} \text{ under } H_0.$$

Subset variant — test that a *group* of $q$ coefficients are simultaneously zero (e.g. all dummies of a $K$-level factor):

$$F = \frac{(\mathrm{RSS}_0 - \mathrm{RSS})/q}{\mathrm{RSS}/(n - p - 1)} \sim F_{q,\, n-p-1}.$$

## Notation & setup

- Large model: $p+1$ regression parameters, RSS.
- Small model: $p+1-q$ parameters (drop $q$ predictors), $\mathrm{RSS}_0$.
- Numerator df = $p$ (all slopes) or $q$ (subset variant).
- Denominator df = $n - p - 1$.
- Two equivalent computations: from $R^2$ ($F = R^2/(1-R^2) \cdot (n-p-1)/p$), or from the RSS difference.

## Insights & mental models

### Reduces to t when $p = 1$

> "The F-statistic generalizes [the t-test] and degrades back to the t-statistic when p = 1." — [[L06-linreg-2]]

Specifically, $F_{1, n-p-1} = t^2_{n-p-1}$. So F isn't *new* machinery — it's the multivariate generalization. Both come out of the same Gaussian-error sampling distribution.

### Why ask Q1 first

Slide flag from [[L06-linreg-2]]: **"only checking individual p-values is dangerous."** Why?

> "The variables can actually be correlated, and then none of them actually look significant, but overall the test is very significant." — [[L06-linreg-2]]

So sequence: F-test ("is anything good at all?") → drill into individual t's. Skipping to t-tests can hide a collinearity-masked signal. This is the case-in-point connection between [[collinearity]] and [[t-test-and-significance]].

### The subset / partial F

Useful for testing whether a *block* of predictors matters — most importantly, all $K{-}1$ dummies for a categorical factor. The R idiom is `anova(small, large)` — comparing nested models. The denominator uses the *larger* model's RSS / df.

### Where it doesn't generalize

The prof's bigger reservation:

> "I'm more interested in things that can generalize to other distributions. This thing's rather specific to the case of linear regression." — [[L06-linreg-2]]

Outside Gaussian linear regression (logistic, GLM, GAM), the F machinery doesn't transfer cleanly. Use likelihood-ratio tests / deviance comparisons instead — taught later.

### "It matters for ANOVA"

The prof noted F is the workhorse of **AN**alysis **O**f **VA**riance, heavily used in design of experiments / engineering. That's a different course; the machinery is the same idea (block-of-coefficients null hypothesis).

## Exam signals

> [!important] F-test scope flag — verbatim
> "I'm going to say right now I probably won't ask any questions about an F-test, which is what I'm going to show you in a minute. I think it's too boring for this class, to be honest… I don't think I'm going to say anything about an F-test. I'm more interested in things that can generalize to other distributions. This thing's rather specific to the case of linear regression." — [[L06-linreg-2]]

> "If I was going to ask you something about this, it'd be more about the null hypothesis. And then you could say, 'I would use an F-test.' But I'm not going to make you compute it because I honestly don't care." — [[L06-linreg-2]]

> "I might say in the multivariate linear regression case, how would I test if at least one of the predictors is useful in predicting the response, or I might ask: why would I want to know this, what's the point?" — [[L06-linreg-2]]

> 2025 Q6a reformulation: *"Is there evidence that variable Z is relevant?" → he'd say what test would you use (F-test for ANOVA), without making you compute it numerically.* — [[L27-summary]]

## Pitfalls

- **Computing the statistic.** *Don't bother memorizing the formula* — prof said he won't ask. Memorize the null hypothesis and the qualitative reasoning instead.
- **Confusing the partial F with the global F.** The global F tests *all* slopes = 0. The partial F tests a *subset* (e.g., all dummies of a factor). Same machinery, different RSS comparison.
- **Skipping F.** Only checking individual t's misses correlated-predictor cases — see [[collinearity]].
- **Generalizing to GLMs.** Doesn't transfer; use likelihood-ratio tests in module 4 onward.
- **F-statistic ≈ 1 under H₀.** A common conceptual checkpoint: if the model explains nothing, numerator and denominator are both estimating the noise, so $F \approx 1$. Big F → real signal.

## Scope vs ISLR

- **In scope:** the null hypothesis of the global F-test, the conceptual reason to use it (avoid the t-test trap under collinearity / many predictors), the partial F for testing categorical predictors, the equivalence with t when $p = 1$.
- **Look up in ISLR:** §3.2.2 (pp. 75–77, *Is There a Relationship?*) — concise treatment with the formula and the partial F via `anova()`.
- **Skip in ISLR (book-only / prof excluded):** F-test mechanics, p-value computation from the F-distribution, ANOVA tables in detail. Walpole's "good for classic statistics" reference is what the prof points at if you want the full classical treatment.

## Exercise instances

None directly tagged. The F-test appears implicitly in Exercise 3.1d (testing the 3-level factor `origin` with `anova()`), but the *learning target* there is categorical encoding, not the F-test itself.

## How it might appear on the exam

- **State the null hypothesis.** Most likely: "What test would you use to check whether *any* predictor is useful?" Answer: F-test on $H_0: \beta_1 = \cdots = \beta_p = 0$.
- **Why not just use t-tests?** Because under collinearity, individual t's can both be insignificant while the joint F is highly significant. State the trap.
- **Test a categorical predictor.** "How would you test whether a 3-level factor matters?" → F-test (or `anova()`) on the joint null that all $K{-}1$ dummies are zero.
- **F = t² when p = 1.** Conceptual: the F-test reduces to the t-test for a single coefficient.
- **Don't compute.** Prof has explicitly said he won't make you compute the F-statistic. If asked anyway, write the formula and stop.

## Related

- [[linear-regression]] — the parent model
- [[t-test-and-significance]] — same machinery for single coefficients; F=t² when p=1
- [[sampling-distribution-of-beta]] — derivation foundation
- [[collinearity]] — *why* you ask Q1 before drilling into t's
- [[categorical-encoding-and-interactions]] — partial F is how you test factor predictors
