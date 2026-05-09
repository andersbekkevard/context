---
concept: subset-selection
module: 06-modelsel
lectures: [L12, L13]
isl-ref: 6.1
exercises:
  - Exercise6.3a — best-subset on Credit via regsubsets, pick by Cp/BIC/adjR²
  - Exercise6.3b — best-subset selection scored by 10-fold CV
  - Exercise6.3c — compare CV vs penalty-criterion choices on Credit
  - Exercise6.4a — forward, backward, hybrid stepwise on Credit (regsubsets)
  - Exercise6.4b — compare stepwise vs best-subset
related: [ridge-regression, lasso, cross-validation, aic-bic-conceptual, high-dimensional-regression, collinearity]
tags:
  - concept
  - module/06-modelsel
aliases:
  - best-subset
  - stepwise-selection
  - forward-stepwise
  - backward-stepwise
  - hybrid-stepwise
---

# Subset selection

The discrete branch of model selection: pick a subset of the $p$ predictors and throw the rest away. Four flavours — best-subset (try them all), forward stepwise (greedy add), backward stepwise (greedy remove), hybrid (forward + backward interleaved). The prof teaches all four together as one slide deck and one atom; he treats them as motivation for *why* shrinkage exists, since best-subset is "slow as shit for $p$ big" — [[L12-modelsel-1]].

## Definition (prof's framing)

> "Identifying a subset of the $p$ predictors that we believe to be related to the response." — slide deck (selection_regularization_presentation_lecture1.md)

Then within each subset size $k$ pick the model with smallest RSS / largest $R^2$, then choose across $k$ via [[cross-validation]] (preferred) or a penalty criterion ([[aic-bic-conceptual|Cp/AIC/BIC/adjR²]]).

> "We're trying to find the best of all of these possible models." — [[L12-modelsel-1]]

## Notation & setup

- $p$ = total predictors available; $k$ = number of predictors in candidate model; $\mathcal{M}_k$ = best $k$-predictor model.
- $\mathcal{M}_0$ = null model (intercept only).
- Choice across $k$ requires a complexity-penalty criterion or held-out / CV estimate, since RSS strictly decreases with $k$.

## Formula(s) to know cold

**Best-subset model count**:
$$\binom{p}{1} + \binom{p}{2} + \dots + \binom{p}{p} = 2^p$$

**Stepwise (forward, backward, hybrid) model count**:
$$1 + \sum_{k=0}^{p-1}(p-k) = 1 + \frac{p(p+1)}{2}$$

For $p=20$: best-subset fits $2^{20} \approx 10^6$ models, stepwise fits $211$.

## The four algorithms

### 1. Best subset selection

1. $\mathcal{M}_0$ = intercept-only.
2. For $k = 1, \dots, p$: fit all $\binom{p}{k}$ models with $k$ predictors. Pick the one with smallest RSS / largest $R^2$ → call it $\mathcal{M}_k$.
3. Choose among $\mathcal{M}_0, \dots, \mathcal{M}_p$ using **CV** (preferred) or a penalty criterion (Cp, BIC, adjR²).

> [!important] Verbatim — best-subset's death sentence
> "Slow as shit for $p$ big — or even like impossibly slow. Like just never going to happen." — [[L12-modelsel-1]]

Why RSS / $R^2$ are okay for step 2 but not step 3: at fixed $k$, all candidates have the same parameter count → fair comparison on training data. Across different $k$, more parameters always lower training RSS → need a complexity-aware criterion.

The prof's nuance:

> "If they change in dimensionality or they have very different statistics, then use cross-validation for that too." — [[L12-modelsel-1]]

(His example: a continuous "pupil dilation" vs a binary "rearing" event — both "one variable" but with wildly different statistics, so RSS comparisons mislead.)

### 2. Forward stepwise selection

1. $\mathcal{M}_0$ = intercept-only.
2. For $k = 0, 1, \dots, p-1$: starting from $\mathcal{M}_k$, fit all $p-k$ models that add one more predictor. Pick the one improving RSS / $R^2$ most → $\mathcal{M}_{k+1}$.
3. Choose among $\mathcal{M}_0, \dots, \mathcal{M}_p$ as in best-subset.

**Guided search — not guaranteed to find the best subset.** Once a predictor enters, it cannot leave.

> "It's a bit annoying that you might not get the best best model, but it's probably pretty good." — [[L12-modelsel-1]]

The Credit-data example illustrates: best-subset and forward stepwise agree at $k=1, 2, 3$. At $k=4$, best-subset drops `rating` and adds `cards`; forward stepwise can't backtrack — keeps `rating`, adds `limit`. Different 4-variable answers.

> "Forward stepwise selection can be applied even in the high-dimensional setting where $n < p$ — by limiting the algorithm to submodels $\mathcal{M}_0, \dots, \mathcal{M}_{n-1}$ only." — slide deck

### 3. Backward stepwise selection

Mirror image: start with the full model $\mathcal{M}_p$, drop the predictor whose removal hurts the fit the least, repeat.

> [!important] Hard requirement
> "Backwards selection requires that the number of samples $n$ is larger than the number of parameters." — [[L12-modelsel-1]]

Because step 1 fits the full OLS model, which needs $n > p$ for $X^\top X$ to be invertible. Forward stepwise has no such requirement.

> "Need more number of data points than predictors for least squares without tricks. By 'tricks' I mean other stuff we're going to talk about later." — [[L12-modelsel-1]]

(Foreshadowing [[ridge-regression|ridge]] / [[lasso]].)

### 4. Hybrid (sequential replacement) stepwise

Forward + backward interleaved. After adding a predictor, check whether any previously-added predictor can now be dropped.

> "You go up and then you remove one because maybe it's no longer [helpful]. In that case here, probably what would happen — if you went forward to four parameters and then went backwards, it would kill off the rating. And then if you went forward again, you would get cards." — [[L12-modelsel-1]]

In principle hybrid can recover a best-subset answer that pure forward misses. Same $\sim p(p+1)/2$ cost as forward / backward.

## Insights & mental models

- **Subset selection is the discrete sibling of [[regularization|shrinkage]].** Both reduce effective parameter count. Best-subset does it combinatorially; ridge / lasso do it continuously through a penalty.
- **Penalty criteria vs CV — prof's preference**: he distrusts Cp/AIC/BIC because they assume a $\sigma^2$ that's hard to estimate when $p$ is large. Always uses CV. *"I almost always in almost everything I do, I use cross-validation of some sort… because assumptions are always wrong, right?"* — [[L12-modelsel-1]]
- **The Credit dataset story** (best-subset vs forward stepwise diverge at $k=4$): correlated predictors flip which subset is "best." Once you understand this you understand why subset selection is fragile and why [[lasso]]'s automatic selection is appealing.
- **Why subset selection breaks at modern scale**: *"What if you're in the modern machine learning framework — almost a trillion parameters? $2^{\text{trillion}}$. It is too much."* — [[L13-modelsel-2]]. Regularization is the only viable path for large $p$.

## Exam signals

> "Slow as shit for $p$ big — or even like impossibly slow." — [[L12-modelsel-1]]

> "Backwards selection requires that the number of samples $n$ is larger than the number of parameters." — [[L12-modelsel-1]]

> "I really don't think I'm going to ask any questions about [Cp/AIC/BIC]. I'm not going to ask you to derive them." — [[L12-modelsel-1]]

> "It's a bit annoying that you might not get the best best model, but it's probably pretty good." — [[L12-modelsel-1]]

The prof drilled the **counts** ($2^p$ vs $1 + p(p+1)/2$) and the **constraint** ($n > p$ for backward). Both are clean MC / fill-in candidates.

## Pitfalls

- **Comparing across $k$ on RSS / $R^2$ alone.** Always loses to the largest model. Use CV or a complexity-penalty criterion.
- **Backward stepwise when $n \leq p$.** Cannot fit the starting full model. Use forward (or shrinkage).
- **Trusting a penalty criterion in high dimension.** $\hat\sigma^2$ becomes unreliable when $p \approx n$ — Cp/AIC/BIC degrade. Use CV (per [[L15-modelsel-4]]).
- **Forgetting that forward stepwise is greedy.** Can produce a different 4-variable model than best-subset on the *same* data — see Credit-data example.
- **Reading "stepwise picked these 5 variables" as a confidence claim about which 5 matter.** Highly sample-dependent under [[collinearity]].

## Scope vs ISLR

- **In scope:** all four algorithms (best, forward, backward, hybrid); the model counts; the $n > p$ requirement for backward; CV vs penalty-criterion choice; the Credit-data example (different $k=4$ winners); the $n < p$ option for forward stepwise (slide-flagged).
- **Look up in ISLR:** §6.1 (pp. ~244–251), Algorithms 6.1–6.3 in the textbook box format.
- **Skip in ISLR (book-only / prof-excluded):** the algebra of $C_p$, AIC, BIC, adjusted $R^2$ — *"I'm not going to ask you to derive them"* — [[L12-modelsel-1]]. The conceptual claim "they penalize complexity" stays in scope (see [[aic-bic-conceptual]]); the formulas don't.

## Exercise instances

- Exercise6.3a — best-subset on Credit via `regsubsets`, pick model by $C_p$, BIC, adjusted $R^2$.
- Exercise6.3b — same Credit data, score subsets via 10-fold CV instead.
- Exercise6.3c — compare CV vs penalty-criterion picks; do they agree?
- Exercise6.4a — forward, backward, hybrid (sequential replacement) stepwise on Credit.
- Exercise6.4b — compare stepwise picks against best-subset; observe where they diverge.

## How it might appear on the exam

- **Multiple choice / fill-in**: "How many models does best-subset fit for $p$ predictors?" → $2^p$. "How many for forward stepwise?" → $1 + p(p+1)/2$.
- **True / False**: "Backward stepwise can be applied when $p > n$." → False (forward can; backward cannot).
- **True / False**: "Best-subset is guaranteed to find the lowest-RSS model of each size $k$, but stepwise is not." → True for forward / backward; the Credit-data divergence at $k=4$ is the canonical example.
- **Method comparison**: given two procedures' chosen subsets on the same data, explain why they differ (correlated predictors → forward stepwise gets locked in, can't backtrack).
- **Conceptual**: "Why prefer CV over Cp/AIC/BIC for choosing across $k$?" → CV makes no $\sigma^2$ or distributional assumptions; penalty criteria assume a clean $\hat\sigma^2$ that's hard to get when $p \approx n$.
- The 2024 / 2025 exams asked direct ridge vs lasso comparisons, not subset selection. But subset selection is on slides + exercises so any of the above patterns is fair game.

## Related

- [[ridge-regression]] — the continuous shrinkage cousin; same goal (reduce variance) without combinatorial search.
- [[lasso]] — does subset selection "for free" via the L1 penalty's corners.
- [[cross-validation]] — the preferred criterion for choosing across $k$.
- [[aic-bic-conceptual]] — the penalty criteria mentioned only conceptually; not on the exam.
- [[high-dimensional-regression]] — backward stepwise breaks here; only forward (or regularized) works.
- [[collinearity]] — explains why best-subset and forward stepwise can disagree on the Credit data.
