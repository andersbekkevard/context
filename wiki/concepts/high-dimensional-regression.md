---
concept: high-dimensional-regression
module: 06-modelsel
lectures: [L15]
isl-ref: 6.4
exercises: []
related: [ridge-regression, lasso, elastic-net, principal-component-regression, subset-selection, regularization, cross-validation, collinearity, curse-of-dimensionality, double-descent, aic-bic-conceptual]
tags:
  - concept
  - module/06-modelsel
aliases:
  - high-dim-regression
  - p-greater-than-n
  - p > n regression
---

# High-dimensional regression ($p > n$)

The closing motivation for *why* module 6 exists: when you have more predictors than observations, vanilla OLS is useless. $X^\top X$ is singular, training $R^2 = 1$ regardless of whether anything actually predicts anything, $\hat\sigma^2$ is unreliable, [[aic-bic-conceptual|Cp/AIC/BIC]] all break, and [[collinearity]] becomes pathological. The whole module's machinery — [[subset-selection|forward stepwise]] (capped at $n-1$ submodels), [[ridge-regression|ridge]], [[lasso]], [[elastic-net]], [[principal-component-regression|PCR]], [[partial-least-squares|PLS]] — exists to make the problem well-posed in this regime. The prof's framing: *"It's very important. And now it's also very common."* — [[L15-modelsel-4]].

## Definition (prof's framing)

> "High dimension problems: $n < p$. More common nowadays." — slide deck (selection_regularization_presentation_lecture2.md)

> "Most of statistics has really been built on the notion that you have more data points than your parameters. So a lot of what happens when the situation is reversed, it's just not well understood. And so it's very important. And now it's also very common." — [[L15-modelsel-4]]

The prof's own example (neuroscience): used to record 8–15 neurons simultaneously; now records 6–7 thousand, plus eye cameras, whisker angles, head tilt, body posture, behavioral decomposition. Both $X$ and $Y$ exploded.

## Notation & setup

- $n$ = number of observations.
- $p$ = number of predictors.
- "**High-dimensional**" = $p \gtrsim n$ (the problematic regime is $p \geq n$, with $p \gg n$ the most extreme).
- The whole module's tools assume you can either constrain the model (regularization, dimension reduction) or use a procedure that works at $p > n$ (forward stepwise capped at $n-1$ submodels).

## What goes wrong with vanilla OLS in high dimensions

### 1. Training $R^2 \to 1$ regardless of truth

> "If you have a lot of variables, then you're going to perfectly fit the data, even though the relationship might be completely meaningless." — [[L15-modelsel-4]]

> "Standard linear regression cannot be applied. Perfect fit to the data, regardless of relationship." — slide deck

Once $p \geq n$, the design matrix $X$ has at least as many columns as rows, and you can find a perfect-fit linear combination of *any* random predictors — including pure noise. Training error → 0, test error → blown up. (Same phenomenon as the day-1 $p$-vs-test-MSE figure, replayed.)

### 2. $X^\top X$ is singular → no unique OLS solution

There are infinitely many $\hat\beta$'s that achieve zero training error. Without a constraint (regularization, minimum-norm via pseudo-inverse, dimension reduction), there's no principled way to pick one.

(This connects to [[double-descent]]: the minimum-norm interpolator is what the pseudo-inverse picks, and that's *implicitly* doing ridge regression — see [[L04-statlearn-3]].)

### 3. Cp / AIC / BIC don't save you

> "They're problematic, partially because it's hard to estimate the $\sigma^2$. I would say they're more problematic because their assumptions are always wrong, and they're typically always wrong. So I wouldn't recommend using those." — [[L15-modelsel-4]]

> "Unfortunately, the $C_p$, AIC, and BIC approaches are problematic (hard to estimate $\sigma^2$)." — slide deck

The penalty criteria require an estimate of the noise variance $\sigma^2$, which itself requires a fitted model with reliable residuals. In high dimensions you can't get one. → Use [[cross-validation]] instead.

### 4. Multicollinearity becomes pathological

> "In the high-dimensional setting, the multicollinearity problem is extreme. We can never know exactly which variables, if any, truly are predictive of the outcome. We can never identify the best coefficients for use in the regression. At most, we can hope to assign large regression coefficients to variables that are correlated with the variables that truly are predictive of the outcome. We will find one of possibly many suitable predictive models." — slide deck (the prof read this aloud, [[L15-modelsel-4]])

Slide-flagged definition the prof endorsed:

> "Multicollinearity: any variable in the model can be written as a linear combination of all of the other variables in the model." — slide deck

In ordinary regression, [[collinearity]] makes individual coefficients unstable. In high-dim, *every* variable is essentially a linear combination of the others — coefficient identification is hopeless.

### 5. Adding noise features always (eventually) hurts

> "Adding more features can help, smaller can help if you have the right ones, but adding noise features that are not associated with the response increases test error. Noise features [exacerbate] the risk of overfitting because you start fitting to the noise instead of the variables that matter." — [[L15-modelsel-4]]

> "Noise features exacerbating the risk of overfitting. Previous example shows that regularizations does not eliminate the problem." — slide deck

Slide-flagged simulated example: $n = 100$, three values of $p$, of which $20$ predictors are truly associated with the response. *"When $p = 2000$ the lasso performed poorly regardless of the amount of regularization."*

> "Regularization does not entirely eliminate the problem. It certainly helps. And these things are actually ongoing and improving over the years." — [[L15-modelsel-4]]

"Noise" can mean either actual measurement noise *or* real-but-irrelevant features. Including details about Japan's weather in a model of what's happening in this classroom is "not noise in the statistical sense, just irrelevant — the effect is the same."

### 6. Sometimes you don't even have enough data for a test set

> "Worse — sometimes you don't even have enough data to construct a test set." — [[L15-modelsel-4]]

In which case you fall back to CV / bootstrap to estimate generalization, and even those degrade.

## Insights & mental models

### What rescues the problem

The whole rest of module 6 is the answer:

| Tool | What it does in high dim |
|---|---|
| [[subset-selection|Forward stepwise]] (capped at $n-1$) | Builds models incrementally; never needs to fit the full $p$-predictor model. Slide-flagged: *"Forward stepwise selection can be applied even in the high-dimensional setting where $n < p$, by limiting the algorithm to submodels $\mathcal{M}_0, \dots, \mathcal{M}_{n−1}$."* |
| [[subset-selection|Backward stepwise]] | **Doesn't work** — needs to fit the full model first ([[L12-modelsel-1]]). |
| [[ridge-regression|Ridge]] | Adds $\lambda I$ to $X^\top X$ → unique solution even when $p > n$. *"If you have too many parameters, you can add a regularizer to keep the model finding a unique solution."* — [[L13-modelsel-2]]. |
| [[lasso|Lasso]] | Bounded active set ($\le n$ nonzero coefficients); explicit sparsity. |
| [[elastic-net|Elastic net]] | Same as lasso plus correlated-variable averaging. |
| [[principal-component-regression|PCR]] / [[partial-least-squares|PLS]] | Compress to $M < p$ orthogonal components first, then OLS. |

### Exploratory vs confirmatory science (prof's editorial aside)

> "Right now I'm saying you're trying to explore for relationships in your data. That's not often what we want to do scientifically. Often we want to test relationships. So this would be more in the exploratory phase of trying to understand what relationships are there. … Here, we're not testing things, right? We're just seeing what relationships are the strongest. So it's still called statistics, but we're not doing hypothesis testing. We're not doing good science in a way. We're just kind of fishing around. It would be like a first step — you fish, then maybe you have another set of data where you would test something." — [[L15-modelsel-4]]

High-dim model selection is "fishing around." The proper scientific use is: fish on dataset 1, formulate a hypothesis, test it on dataset 2.

### Curse of dimensionality (prof's add-on, not on slides)

> "One of the key problems with high-dimensional stuff is that distances between points become less variable. So when you have a lot of dimensions, you have this curse of dimensionality, so then it's harder to distinguish between things." — [[L15-modelsel-4]]

Same KNN-killing phenomenon from earlier modules. In high dim, every pair of points is roughly equidistant. See [[curse-of-dimensionality]].

### Connection to double descent

In the over-parameterized regime ($p \gg n$), test error after the interpolation point can come *back down* — see [[double-descent]] and [[L04-statlearn-3]]. The prof flagged this as "implicit ridge" — the minimum-norm solution among infinite interpolators acts as a regularizer.

## Exam signals

> "Standard linear regression cannot be applied. Perfect fit to the data, regardless of relationship." — slide deck

> "In the high-dimensional setting, the multicollinearity problem is extreme. We can never know exactly which variables, if any, truly are predictive of the outcome." — slide deck (prof read aloud, [[L15-modelsel-4]])

> "Adding noise features that are not associated with the response increases test error." — paraphrase of slide and [[L15-modelsel-4]]

> "The Cp, AIC, and BIC approaches are problematic (hard to estimate $\sigma^2$)." — slide deck

The 2024 exam fill-in-the-blank had: *"Ridge regression is possible even if $p > n$ [correct]; Lasso requires that $p < n$ [wrong]"* — the high-dim distinction (which methods survive $p > n$) is exam-tested directly.

## Pitfalls

- **Trying OLS at $p > n$.** $X^\top X$ singular. Use ridge or lasso or PCR.
- **Trusting Cp / AIC / BIC at $p > n$.** $\hat\sigma^2$ unreliable. Use CV.
- **Believing regularization eliminates the noise-features problem.** It doesn't — it just shifts the breakdown threshold further right. Slide-flagged: lasso fails at $p = 2000$ regardless of regularization.
- **Believing "perfect training fit means good model."** In high dim, you can perfectly fit *any* random labels with random features — see Exercise5.3 ([[nested-cv-and-cv-pitfalls|wrong-way CV]]).
- **Confusing "$p > n$" with "high signal-to-noise."** They're orthogonal. You can have $p > n$ with strong signal (then regularization works) or with mostly noise (regularization helps but can't save you).
- **Reading "this regression has high $R^2$" in high dim as evidence of model quality.** $R^2 = 1$ is automatic when $p \geq n$. Always evaluate on held-out / CV error.
- **Identifying "the best $k$ features" from a high-dim regularized fit as confidently predictive.** Per the prof's quote — at most you've identified features *correlated with* the truly predictive ones. Many "suitable predictive models" exist.

## Scope vs ISLR

- **In scope:** the high-dim setting and why it exists; the four failure modes (perfect fit / singular $X^\top X$ / unreliable penalty criteria / pathological multicollinearity); the noise-features-always-hurt result; the toolset that does work ([[subset-selection|forward stepwise]] capped, [[ridge-regression|ridge]], [[lasso]], [[elastic-net]], [[principal-component-regression|PCR]], [[partial-least-squares|PLS]]); the exploratory-vs-confirmatory framing; the connection to [[curse-of-dimensionality]] and [[double-descent]].
- **Look up in ISLR:** §6.4 (pp. 290–294) — High-Dimensional Data, What Goes Wrong, Regression in High Dimensions, Interpreting Results in High Dimensions. Short and focused.
- **Skip in ISLR:** none — §6.4 is short and the prof covered most of it. The detailed derivation of why $\hat\sigma^2$ breaks (which lives in linear-models theory) is not needed; the conceptual claim is.

## Exercise instances

None directly. The high-dim discussion in [[L15-modelsel-4]] is closing motivation, not exercise material. Tangentially related: Exercise5.3 (the wrong-way CV simulation with $p = 5000$ random predictors and random labels) demonstrates the high-dim failure mode in CV terms.

## How it might appear on the exam

- **True / false**: "When $p > n$, OLS produces a unique $\hat\beta$." → False (multiple solutions; $X^\top X$ singular). "Ridge regression can be fit when $p > n$." → True. "Lasso requires $p < n$." → False (this exact MC trap was on the 2024 exam).
- **True / false**: "When $p > n$, training $R^2 = 1$ is evidence of a good model." → False (any random labels can be perfectly fit).
- **Method comparison**: "Why might [[principal-component-regression|PCR]] outperform OLS when $p > n$?" → because OLS doesn't have a unique solution; PCR squashes $X$ to $M < n$ orthogonal components first, restoring well-posedness.
- **Conceptual short answer**: "Why are Cp/AIC/BIC problematic in high dimensions?" → they require a reliable $\hat\sigma^2$, which requires a fitted full model — not available when $X^\top X$ is singular. Use CV instead.
- **Output interpretation**: given a noise-features simulation plot (test MSE vs $p$ with $20$ true features and varying noise features), explain why the curve climbs as more noise is added — even with regularization.
- **Conceptual essay**: "What is multicollinearity in high dimensions, and what can we hope for in terms of identifying predictive variables?" → answer paraphrasing the slide / prof's quote: every variable is approximately a linear combination of others; we can never identify the truly predictive variables, only ones correlated with them; we'll find one of possibly many suitable predictive models.

## Related

- [[ridge-regression]] — the canonical high-dim regression tool.
- [[lasso]] — the high-dim variable-selection tool (with caveats: active set bounded by $n$).
- [[elastic-net]] — practical default for high-dim with correlated predictors.
- [[principal-component-regression]] / [[partial-least-squares]] — dimension-reduction approaches.
- [[subset-selection]] — only forward stepwise survives high-dim; backward fails.
- [[regularization]] — the cross-cutting Special; this concept is the "*why* regularization is critical" section.
- [[cross-validation]] — only honest model-selection criterion when penalty criteria break.
- [[collinearity]] — what makes high-dim pathological; in low-dim it's annoying, in high-dim it's fundamental.
- [[curse-of-dimensionality]] — the distance-blurring phenomenon that also degrades KNN, etc.
- [[double-descent]] — the modern view: past the interpolation point, test error can come down again.
- [[aic-bic-conceptual]] — the penalty criteria that break here.
