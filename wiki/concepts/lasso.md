---
concept: lasso
module: 06-modelsel
lectures: [L13, L14, L15, L24, L26]
isl-ref: 6.2.2
exercises:
  - Exercise6.6 - apply lasso to Credit, compare with ridge and OLS
related: [ridge-regression, ridge-vs-lasso-geometry, elastic-net, subset-selection, regularization, cross-validation, standardization, high-dimensional-regression, collinearity, double-descent]
tags:
  - concept
  - module/06-modelsel
aliases:
  - L1 regularization
  - L1 penalty
---

# Lasso

The L1 cousin of [[ridge-regression]]. Same penalized-RSS objective but with $|\beta_j|$ instead of $\beta_j^2$, and that single change makes the constraint region a *diamond* whose corners sit on the axes, so the solution typically lands on a corner with some $\beta_j = 0$ exactly. **Variable selection for free**, without the $2^p$ combinatorial cost of [[subset-selection|best-subset]]. The prof's editorial verdict: *"Tibshirani, Hastie, most of their research, if you look at what they've done, is all about lasso. They love this thing. It's like their favourite topic."* - [[L13-modelsel-2]].

## Definition (prof's framing)

> "The Lasso regression coefs $\beta^L$ are the ones that minimize $\text{RSS} + \lambda \sum_{j=1}^p |\beta_j|$, with $\lambda > 0$ being a tuning parameter.", slide deck (selection_regularization_presentation_lecture1.md)

> "Just replace this squared with an absolute value." - [[L13-modelsel-2]]

> "The $\ell_1$ penalty has the effect of forcing some of the coefficients to be exactly zero when $\lambda$ is large enough.", slide deck

## Notation & setup

- $\beta_0$ = intercept (**not penalized**); $\beta_1, \dots, \beta_p$ = penalized slope coefficients.
- $\lambda \geq 0$ = tuning parameter, chosen by [[cross-validation]].
- $X$ = $n \times p$ design matrix; **must be standardized** before fitting (same as ridge).
- Constraint form: minimize RSS subject to $\sum_j |\beta_j| \le s$.

## Formula(s) to know cold

**Penalized objective:**
$$\hat\beta^L_\lambda = \arg\min_\beta \left\{ \sum_{i=1}^n \left(y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij}\right)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

**Constraint form:**
$$\min_\beta \text{RSS} \quad \text{subject to} \quad \sum_{j=1}^p |\beta_j| \le s$$

Two extremes:
- $\lambda = 0$ → recover OLS.
- $\lambda$ large enough → all $\hat\beta_j = 0$ exactly. Crucially, **all-zero happens at finite $\lambda$**, not at infinity (unlike ridge, where you'd need $\lambda = \infty$).

No closed form (the absolute value is non-differentiable at zero). Fit via coordinate descent or LARS, the prof noted *"there are tricks; most just end up kind of rounding this out, but the upshot is L1 is still cheap to fit."* - [[L13-modelsel-2]]. Mechanics out of scope.

## Insights & mental models

### Why the L1 corners produce sparsity

The prof sketched $|\beta|$ vs $\beta^2$ on the board. Two key contrasts vs ridge:

1. **Linear growth far from zero** → less aggressive on big coefficients you actually need.
2. **Constant $\pm 1$ gradient right up to zero** → constantly pushes coefficients toward zero. (Compare ridge: $\frac{d}{d\beta}\beta^2 = 2\beta$, which **vanishes** at zero.)
3. **Non-differentiable at $\beta = 0$** (gradient jumps from $-1$ to $+1$) → the optimizer can sit *exactly* at zero stably.

> "The constant gradient actually drags coefficients all the way down." - [[L13-modelsel-2]]

This is the same story the [[ridge-vs-lasso-geometry|geometric picture]] tells: the diamond's corners on the axes are where the RSS ellipses tend to first touch the constraint region.

### "Capitalist" personality

> "Lasso is like very capitalist. Just shoot everyone, let all the poor people die, let the one rich guy win. And L2 is more socialist… probably both are too bad. Which is why we have something called the elastic net." - [[L13-modelsel-2]]

> "[Lasso] encourages winners and losers." - [[L14-modelsel-3]]

When two predictors are correlated, lasso picks one and zeros the other. The choice is data-dependent, small noise can flip which one wins. *"Imagine $\beta_1$ and $\beta_2$ are two things that are relatively correlated… we might end up where actually $\beta_1$ is just sent straight to 0 and it's all driven by $\beta_2$."* - [[L13-modelsel-2]].

### Lasso = subset selection without combinatorial cost

The selling point that the prof emphasized hardest:

> "You could throw away all the ones that are zero and rerun the model with just the two parameters. You wouldn't need any regularization, because it would be well-behaved. And you'd get exactly the same solution as if you'd done forward / backward subset selection, but instead of trying many, many, many models, you just run lasso once. Boom, same place." - [[L13-modelsel-2]]

This is the practical pitch: you get the variable-selection effect of [[subset-selection]] from a *single* convex optimization, not $2^p$ of them. ISLR §6.2.2 echoes: lasso "performs variable selection."

### Simulated example: $p = 45$, $n = 50$, only 2 truly nonzero (slide-flagged)

The deck's headline lasso illustration:
- $n \approx p$, so OLS is on the edge of singular.
- Only 2 of 45 predictors actually relate to $y$.
- **OLS**: one true predictor's $\hat\beta$ is fine; the other true predictor's $\hat\beta$ is *smaller* than several noise predictors' $\hat\beta$'s. OLS does a "shit job" picking the right ones.
- **CV-tuned lasso**: lands at a $\lambda$ where the two true coefficients are clearly nonzero and **all 43 others are exactly zero**. Identifies the right model.

> "If you can do better than [predicting all-zero], your model really sucks, because… plain OLS [on this example] sits *above* that line, strictly worse than predicting nothing." - [[L13-modelsel-2]]

The right-hand panel showed CV error along $\lambda$, with a horizontal reference line for "predict zero everywhere except intercept", the OLS end of the curve was *above* it. Pathological-but-instructive.

### What lasso doesn't do well

When predictors are correlated and you'd benefit from averaging over them, lasso arbitrarily drops one. Ridge averages. [[elastic-net]] combines both, *"this is probably the one that people use the most."* - [[L13-modelsel-2]].

### Coefficient trace plot

Coefficients vs $\|\hat\beta_\lambda\|_1 / \|\hat\beta_\text{OLS}\|_1$ on the x-axis (in the deck), at ratio = 0, $\lambda$ is so big everything is zero; at ratio = 1, $\lambda$ is so small we recover OLS. Same information as $\lambda$ axis, rescaled. **Distinguishing feature of a lasso trace vs a ridge trace**: lasso paths drop to zero and stay there; ridge paths approach zero asymptotically but never touch.

### Refit-on-active-set step

> "Optionally, especially with lasso, note which coefficients went exactly to zero, drop those variables, then refit without the penalty on the surviving ones. Then you've essentially done model selection." - [[L14-modelsel-3]]

Two-step workflow: lasso for variable selection → unpenalized OLS on the survivors. Removes the lasso bias on the retained coefficients.

## Exam signals

> "Tibshirani, Hastie, most of their research, if you look at what they've done, is all about lasso. They love this thing." - [[L13-modelsel-2]]

> "Worth learning how to interpret this geometric interpretation of ridge and lasso." - [[L14-modelsel-3]]

> "For me, definitely the most interesting things are lasso and also implicit lasso. Many different ways you can get lasso and ridge regression. Those are, I think, the most interesting part of this module." - [[L15-modelsel-4]]

> "If model interpretability is desirable, lasso is preferred.", slide deck

The 2023 / 2024 exams asked direct lasso questions: fit it, choose $\lambda$, compare coefficients with OLS, explain why some are zero. 2026 will be the open-book interpretation analogue.

## Pitfalls

- **Forgetting to standardize.** Same as ridge, lasso is not scale-invariant. *"Yeah, also that way the betas all have the same amplitude or it would have a similar amplitude. So it is typically standardized."* - [[L14-modelsel-3]].
- **Reading the active set as "the truly important variables."** Highly sample-dependent under [[collinearity]]. The 2 winners in this dataset may not be the 2 winners in the next.
- **Believing lasso is uniformly better than ridge.** It isn't, lasso wins when the true model is sparse; ridge wins when many predictors contribute moderately and roughly equally.
- **"Lasso requires $p < n$"**: false (this was the wrong-answer trap in the 2024 exam MC). Lasso works at $p > n$; the active set is bounded by $\min(n, p)$.
- **Penalizing the intercept.** Don't.
- **Direction of $\lambda$ confusion.** Same as ridge: $\lambda \uparrow$ → more shrinkage and more zeros; $\lambda \downarrow 0$ → recover OLS.

## Scope vs ISLR

- **In scope:** the L1 objective and constraint forms; the all-zero-at-finite-$\lambda$ behaviour; the geometric picture (corners → sparsity, see [[ridge-vs-lasso-geometry]]); standardization requirement; the $p=45, n=50$ simulated example; "capitalist" personality and when to prefer lasso over ridge; refit-on-active-set workflow; the conceptual claim that lasso = subset selection without combinatorial cost.
- **Look up in ISLR:** §6.2.2 (pp. 256–264) for the orthonormal-design closed-form (lasso's per-coefficient soft-thresholding $\text{sign}(\hat\beta_j)(|\hat\beta_j| - \lambda/2)_+$ vs ridge's $\hat\beta_j/(1+\lambda)$, §6.2.2 "Simple Special Case" pp. 269–270, the cleanest formal contrast). §6.2.3 (pp. 274–279) for CV tuning workflow.
- **Skip in ISLR (prof-excluded):**
  - **Bayesian (Laplace prior) interpretation of lasso**: §6.2.2 pp. 271–273. *"I really don't think I'd put this on the test, just because it kind of assumes a lot of knowledge that maybe you don't have."* - [[L14-modelsel-3]].
  - **L0 norm / "Optimal Brain Damage."** - [[L14-modelsel-3]]: *"It's actually related to the model selection stuff, but again, we won't go into it because it's not used in practice."*
  - Full coordinate-descent / LARS algorithm details.

## Exercise instances

- Exercise6.6, apply lasso to the Credit dataset, compare with OLS *and* ridge. Standard `cv.glmnet(alpha=1)` workflow: standardize, CV-select $\lambda$, refit, compare coefficient table; observe which variables get zeroed.

(Note: CE1 doesn't use lasso. Slide labels this "Recommended exercise 6.")

## How it might appear on the exam

- **Coefficient-table comparison** (canonical, recurs in past exams). Given OLS / ridge / lasso outputs side by side: identify the lasso column (some coefficients exactly zero); explain *why* (L1's corners on axes); explain *why ridge wouldn't do the same* (L2 ball is smooth, see [[ridge-vs-lasso-geometry]]). The 2023 exam answer key called for *"the coefficients are shrunken (0.5P) and some are zero (0.5P). For ridge we would also have expected to see shrinkage (0.5P), but none of the coefficients had gone to zero (0.5P)."*
- **MC / true-false**: "Lasso performs variable selection." → True. "Lasso requires $p < n$." → False. "Ridge sets some coefficients exactly to zero." → False (ridge never does, only lasso does).
- **Trace plot reading**: identify lasso traces (drop to zero, stay) vs ridge traces (asymptote to zero, never touch).
- **Choose-your-method**: given a problem description ("we want interpretability, only ~5 of 100 features matter") → lasso. ("All features are weakly correlated with $y$, no sparsity expected") → ridge. ("Both correlated features and sparsity") → [[elastic-net]].
- **CV plot reading**: same as ridge. `lambda.min` vs `lambda.1se` per the [[one-standard-error-rule]].
- **Pseudocode**: "How would you fit lasso and pick $\lambda$?", same workflow as ridge: standardize → CV grid → pick min or 1-SE → refit.

## Related

- [[ridge-regression]]: the L2 cousin; smooth shrinkage, never zero, "socialist."
- [[ridge-vs-lasso-geometry]]: the canonical ellipse-meets-diamond figure; *"worth learning how to interpret"* per the prof.
- [[elastic-net]]: L1 + L2 together; sparsity from lasso plus correlated-variable averaging from ridge.
- [[subset-selection]]: the combinatorial cousin; lasso achieves the same end (variable selection) without the $2^p$ cost.
- [[regularization]]: the cross-cutting Special; lasso is one of two flagship instances.
- [[cross-validation]]: for choosing $\lambda$.
- [[standardization]]: required preprocessing.
- [[high-dimensional-regression]]: lasso is the go-to method when $p > n$ and you want interpretability.
- [[collinearity]]: explains why lasso's choice between correlated predictors is unstable.
- [[double-descent]]: implicit lasso shows up in modern over-parameterized models too ([[L26-nnet-3]]).
