---
concept: r-squared
module: 03-linreg
lectures: [L05, L06]
isl-ref: 3.1.3, 3.2.2
exercises: []
related: [linear-regression, sampling-distribution-of-beta, t-test-and-significance, cross-validation, bias-variance-tradeoff]
tags:
  - concept
  - module/03-linreg
aliases:
  - coefficient of determination
  - adjusted R squared
  - explained variance
---

# R² and adjusted R²

The classical "fraction of variance explained" measure. **Prof distrusts it**, never decreases when you add a predictor (so it always favors the bigger model on training data) and "what is good is very different depending on the context." Adjusted $R^2$ patches the parameter-count temptation; the prof would still prefer test-set error.

## Definition (prof's framing)

$$R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} = \frac{\mathrm{TSS} - \mathrm{RSS}}{\mathrm{TSS}}$$

where $\mathrm{TSS} = \sum_i (y_i - \bar y)^2$ (total variability in $Y$) and $\mathrm{RSS} = \sum_i (y_i - \hat y_i)^2$ (leftover after the fit).

> "How much does your shit vary in general versus how much can you actually explain of that." - [[L05-linreg-1]]

Prof's framing: $R^2$ is the *first and crudest* of many model-accuracy measures. "I would always use the test error", foreshadowing modules 5–6.

## Notation & setup

- $\mathrm{TSS}$ = total sum of squares.
- $\mathrm{RSS}$ = residual sum of squares.
- $R^2 \in [0, 1]$ (in OLS with intercept). Higher = more variance explained on training data.
- In **simple linear regression**, $R^2 = \mathrm{Cor}(\mathbf{x}, \mathbf{y})^2$, squared sample correlation.

## Formula(s) to know cold

$$\boxed{R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}}$$

Adjusted $R^2$:

$$\boxed{R^2_{\mathrm{adj}} = 1 - \frac{\mathrm{RSS}/(n - p - 1)}{\mathrm{TSS}/(n - 1)} = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}}$$

Note: adjusted $R^2$ *can decrease* when you add a useless variable; plain $R^2$ cannot.

## Insights & mental models

### Why $R^2$ alone fails

Adding a predictor on training data **cannot make the fit worse**. The new model contains the old as a special case (set the new $\beta$ to zero); the optimizer can always do at least as well. So $R^2$ rises monotonically with $p$:

> "It seems easy, you're just like, 'just tell me the error.' Yes, but… should you look at the data that you fit the model on, should you look at held-out data, should you penalize it in some way?" - [[L06-linreg-2]]

Worked example from the slides: BMI alone gives $R^2 \approx 0.50$; add age → 0.58; add age + neck + hip + abdomen → 0.72. Is the bigger model "better"? Without held-out evaluation, you can't tell.

### Why the prof distrusts $R^2$

> "What is good is very different depending on the context of the data. … In my field, that's like unheard of. That's fine." - [[L06-linreg-2]]

Plus:

> "I have reported this [adjusted $R^2$] in articles, but I would never really, if it was up to me, we wouldn't include it. Especially in the analysis I do, typically there's so much stuff that we've left out that this number doesn't mean anything. But it's still interesting." - [[L06-linreg-2]]

His preferred metric: **test-set error**, module 5's [[cross-validation]] is what he reaches for instead.

### Adjusted $R^2$, the patch

The $(n-1)/(n-p-1)$ factor penalizes parameter count. With this:

> "Adding more parameters can actually give you a smaller $R^2$ if they don't do anything." - [[L06-linreg-2]]

So adjusted $R^2$ behaves more like a model-comparison metric, but still uses training data, still distrusted.

### $R^2$ in simple LR

Equals the squared sample correlation:

$$R^2 = \mathrm{Cor}(\mathbf{x}, \mathbf{y})^2.$$

Verifiable in R: `summary(lm(...))$r.squared` should equal `cor(x, y)^2`.

## Exam signals

> "I would always use the test error." - [[L05-linreg-1]]

> "Kind of just the first one that they came up with. A lot of people don't like it. There's an adjusted version. There's other versions." - [[L05-linreg-1]]

> "The bigger point: any model-comparison metric must account for the parameter-count temptation.", paraphrase of [[L06-linreg-2]] adjusted $R^2$ slide.

The prof in [[L27-summary]] explicitly links the "test vs train" pattern to $R^2$ analogues for multiple model types. The keyword "training" vs "testing" in an exam question changes the answer.

## Pitfalls

- **$R^2$ as model quality.** A high $R^2$ doesn't mean the model is correct, useful, or generalizes. It just means it fits the training data.
- **$R^2$ comparisons across different sample sizes.** TSS scales with $n$, so direct comparisons across datasets of different size are meaningless.
- **$R^2$ on a model without intercept.** The decomposition $\mathrm{TSS} = \mathrm{RSS} + \mathrm{ESS}$ only holds when the model has an intercept. Without one, $R^2$ can be negative or > 1.
- **$R^2$ outside OLS.** For ridge / lasso / GAM / GLM, you can compute analogous quantities, but they don't carry the same meaning. Use the test-set error instead.
- **Adjusted $R^2$ as a panacea.** Better than $R^2$ but still a training metric and still distrusted by the prof. CV is the principled answer.
- **Interpretation slip.** "$R^2 = 0.7$" means 70% of the *variance* in $Y$ is explained, *not* "the model is 70% accurate."

## Scope vs ISLP

- **In scope:** the formula, what it means, why it monotonically increases with $p$, the adjusted form and its penalty, the prof's distrust, that test error is preferred.
- **Look up in ISLP:** §3.1.3 (pp. 70–71, $R^2$ in simple regression); §3.2.2 (pp. 79–81, adjusted $R^2$ and the four important questions).
- **Skip in ISLP:** the derivation of adjusted $R^2$ from Mallow's $C_p$, module 6 covers this conceptually only; full algebra is out of scope per [[L12-modelsel-1]] / [[L13-modelsel-2]].

## Exercise instances

None directly tagged in the manifest, $R^2$ shows up as a side-output in essentially every regression-fitting exercise (e.g. Exercise3.1c interprets `summary(lm)`, where $R^2$ is one of the lines).

## How it might appear on the exam

- **Compute $R^2$ from a small table.** Given $\mathrm{RSS}$ and $\mathrm{TSS}$ (or $\bar y$ and the fitted values), compute $1 - \mathrm{RSS}/\mathrm{TSS}$.
- **True/false on monotonicity.** "Adding a predictor can never decrease $R^2$ on training data", TRUE for plain $R^2$, FALSE for adjusted.
- **Train vs test trap.** "If model B has more predictors than model A, then $R^2_B > R^2_A$", true on training, not necessarily on test (could overfit). The 2025 Q4 polynomial trap is the template, see [[L27-summary]].
- **Compare models by $R^2$ vs adjusted $R^2$.** Identify when they disagree; explain why.
- **What's good $R^2$?** Open-ended, "it depends on the field." Engineering wants 0.95+; biology might be thrilled with 0.30.

## Related

- [[linear-regression]]: the model whose fit is being assessed
- [[sampling-distribution-of-beta]]: adjusted $R^2$ uses the same df accounting
- [[t-test-and-significance]]: alternative per-coefficient measure
- [[cross-validation]]: the prof's preferred replacement for $R^2$
- [[bias-variance-tradeoff]]: why training $R^2$ is misleading on flexible models
