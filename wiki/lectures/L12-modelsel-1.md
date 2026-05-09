---
lecture: 12
date: 2026-02-23
module: 06-modelsel
title: Model Selection and Regularization 1
slides: modules/6ModelSel/selection_regularization_presentation_lecture1.md
topics:
  - regularization
  - bias-variance-tradeoff
  - subset-selection
  - cross-validation
  - aic-bic-conceptual
  - r-squared
  - ridge-regression
  - shrinkage
  - standardization
  - hyperparameter
tags:
  - lecture
  - module/06-modelsel
aliases:
  - Lecture 12
---

# L12: Model Selection and Regularization 1

The prof opens module 6 by framing **regularization as constraint**, a way to make over-parameterized models behave by trading a little bias for a lot of variance reduction. He covers the three subset-selection algorithms (best, forward, backward, plus hybrid), is explicit that the training-error penalties (Cp, AIC, BIC, adjusted $R^2$) are not exam material and that he doesn't trust them, then introduces [[ridge-regression]] (= L2) as the first shrinkage method. Stops mid-ridge; continues tomorrow.

## Key takeaways

- **Regularization is constraint.** "We're going to look at the regularizations or constraints on these models that they do the thing that we talk about." Often the constraint you write down (push betas to zero) is not directly what you want — but it produces the model behavior you want (low variance, generalizes).
- **Bias-variance recap, applied to predictor count.** Standard linear: $n \gg p$ → low variance; $n \approx p$ → high variance; $n < p$ → variance explodes, model unusable without extra constraints. "Often we can substantially reduce the variance at the cost of a negligible increase in bias", the prof flags he is *always* surprised how cheap the bias cost is.
- **Best subset selection** fits all $2^p$ models, picks best of each order $k$ by RSS / $R^2$, then chooses across orders by **cross-validation** (or the penalty criteria, which the prof distrusts). Verbatim: **"slow as shit for $p$ big."**
- **Forward / backward / hybrid stepwise** are guided searches: $1 + p(p+1)/2$ fits, much cheaper, **not guaranteed** to find the best subset. Demonstrated on Credit data where adding `limit` flips the best 4-variable set from `{rating, ...}` to one with `cards` instead — predictors are correlated in annoying ways.
- **Penalty criteria (Cp, AIC, BIC, adjusted $R^2$): explicitly not exam material.** "I really don't think I'm going to ask any questions about this." All you need to know: they exist as ways to penalize training RSS by model complexity. Prof doesn't trust them; recommends cross-validation instead.
- **Ridge regression = L2.** Augment the RSS objective with $\lambda \sum_j \beta_j^2$. Tug-of-war: RSS pulls betas away from zero, the penalty pulls them back. $\lambda = 0$ → standard OLS; $\lambda = \infty$ → all betas → 0. **Standardize the X's first** — ridge is not scale-invariant.
- **The general lens introduced today**: many of the "tricks" in this module are forms of variance reduction by adding constraints to the optimization. Subset selection does it discretely; shrinkage does it continuously.

## Framing: regularization as constraint

The prof opens with a long analogy spree (kids, American suburbs, parking-lot zoning) all pointing at one idea: **the constraint you put on a system isn't always the thing you literally want — it's a lever that produces the system you want as a side effect.**

> "Regularization is really a way of constraining the model. And constraining it in such a way that what comes out is good."

Why care: many modern models — neural nets being the obvious case — have far more parameters than data points. Without constraints they cannot be fit at all. Regularization is "one of the main tricks that is used."

The transferable framing for this lecture:

> "The constraint that we're putting on this system is not 'build parking lots,' but 'push your parameters towards zero.' And the result is you have a model that will generalize better on data that it hasn't seen."

So the explicit constraint (sparsity, shrinkage) accomplishes the implicit goal (low variance, better out-of-sample prediction).

## Recap: linear setting and the n vs. p story

Setting is the standard linear model $Y = X\beta + \varepsilon$. Two derivations of $\hat\beta$ recapped: closed-form OLS via $(X^\top X)^{-1} X^\top Y$, and **maximum likelihood under Gaussian errors**, which gives the same estimator. (Recommended exercise — the prof says he hasn't shown the derivation himself; do it.)

The two-second sketch he wrote on the board: the Gaussian likelihood is $\prod_i \exp(\dots(y_i - \beta x_i)^2 \dots)$, so taking $\log$ converts the product into a sum of squared residuals — same minimizer as OLS.

The variance-vs-$p$ regime that motivates the rest of the module:

- $n \gg p$: low variance.
- $n \approx p$: high variance.
- $n < p$: **infinite variance**, model cannot be used (without extra constraints).

> "We have ways of constraining the model so this isn't really a problem [in the $n<p$ case] — but it's still worth thinking about."

### Bias-variance, again

Same decomposition introduced day 1 ([[bias-variance-tradeoff]]): error = irreducible + bias + variance. Adding parameters trades bias for variance along the classic U-shape.

> "Often we can substantially reduce the variance at the cost of a negligible increase in bias. I think that's been always surprising to me — just how little bias you need to get a lot of reduction in variance."

The reason variance matters is generalization: high variance means the fitted model swings around with each new training set, so predictions on held-out data are unreliable. The trivial low-variance model (predict zero always) makes the point that low variance alone isn't enough — you also need low bias.

## Goal: interpretable, simple models

The module's goal: improve prediction accuracy and/or interpretability **by replacing OLS with something else**. Interpretable here means few parameters and simple form.

> "Some of them might be irrelevant. And even if they're not entirely irrelevant, maybe they're not reliable enough… we want to reduce the model."

The general term is **model selection**; the specific instance is feature/variable selection. The three families he lists from the slides:

1. **Subset selection**: pick a subset of $X$'s, throw the rest away.
2. **Shrinkage**: keep all $X$'s, regularize the coefficients toward zero.
3. **Dimensionality reduction**: squish $p$ predictors into fewer composite ones (next lecture).

Today: subset selection in full, shrinkage started (ridge only).

## Subset selection

### Best subset selection

Fit **every possible model** of every order. For $p$ predictors:

- Order 1: $\binom{p}{1}$ models.
- Order 2: $\binom{p}{2}$ models.
- … through $\binom{p}{p}$.
- Total: $\sum_k \binom{p}{k} = 2^p$.

The prof did the binomial recap on the board (and wasn't fully sure of the factorial form — flagged as "easy to Google").

> "$2^{20}$ is over a million. So you have this can be a big number… It just gets big very quickly. So the point is, we're trying to find the best of all of these possible models."

> [!important] Verbatim signal
> "Slow as shit for $p$ big — or even like impossibly slow. Like just never going to happen."

The algorithm (slide-deck steps):

1. $M_0$ = null model (intercept only).
2. For each $k = 1, \dots, p$: fit all $\binom{p}{k}$ models with $k$ predictors. Pick the one with **smallest RSS / largest $R^2$** — call it $M_k$.
3. Choose among $M_0, M_1, \dots, M_p$ using **cross-validation** (preferred) or one of the penalty criteria.

Why RSS / $R^2$ are okay for step 2 but not step 3: within a fixed $k$, all candidate models have the same number of parameters, so their potential to overfit is the same — the comparison is fair on training data. Across different $k$, more parameters always help training fit, so you need either CV or a complexity penalty.

The prof adds a caveat to the "RSS is fine within a $k$" claim:

> "If they change in dimensionality or they have very different statistics, then use cross-validation for that too."

His concrete example: pupil dilation fluctuates continuously; "rearing" is a binary event happening for ~10s in a 10-minute recording. Both are "one variable" but with wildly different statistics — RSS comparisons can be misleading.

> "I almost always in almost everything I do, I use cross-validation of some sort… because assumptions are always wrong, right? They're just so wrong."

### The training-error penalty criteria: explicitly not exam material

The slide deck lists $C_p$, $AIC$, $BIC$, adjusted $R^2$. All have the shape: training RSS plus a term that grows with the number of parameters (and often involves $\hat\sigma^2$, an unbiased noise estimate).

> [!important] Verbatim - exam scope on penalties
> "I really don't think I'm going to ask any questions about this… I'm not going to ask you to use these. I'm not going to ask you to derive them. The derivations are actually, I would argue, the most interesting part. But that's not part of the course."

What the prof wants you to take away:

> "If I was going to ask you anything, I would just want you to know that there exist penalties on the training error that you can use that attempt to account for the increased complexity of models with more parameters."

He doesn't trust them — they make assumptions about $\sigma^2$ and the form of the penalty that don't hold in his fields (aerospace, neuroscience). Use cross-validation instead. The names and forms only matter to the extent that you recognize them.

The plot of RSS vs. $p$ from the slides illustrates the saturation point: error stops dropping past some $k$ (in the example, $k=3$). "You want to find the smallest model that explains the data the best."

### Forward stepwise selection

Best subset is infeasible past modest $p$. **Forward stepwise** is a guided alternative.

Algorithm:

1. $M_0$ = null model.
2. For $k = 0, 1, \dots, p-1$: starting from $M_k$, fit all $p - k$ models that add **one** predictor. Pick the one improving fit most (RSS / $R^2$, or cross-validated).
3. Choose among $M_0, \dots, M_p$ as before (CV or penalty).

The prof's worked example with predictor letters:

- Start with $\beta_0$ alone.
- Try adding A, B, C, D, E one at a time. Suppose C wins → $M_1 = \{\beta_0, C\}$.
- Try adding each of {A, B, D, E} to $\{\beta_0, C\}$. Suppose E wins → $M_2 = \{\beta_0, C, E\}$.
- Continue until adding doesn't help (or the slide algorithm: until all in).

Cost: $1 + p(p+1)/2$ fits. For $p = 20$, that's **211 fits** instead of the million of best subset.

> "211 is a lot less than a million. And so you save a lot of time. And in this case, you know, with 20, you probably just made it feasible to actually do it."

But it's a **guided search**; you can miss the true best subset:

> "We building it up - so we might actually have the wrong combination."

Demonstrated on Credit data (slide): the best 1-, 2-, and 3-variable subsets agree between best-subset and forward stepwise. At $k = 4$, the best-subset answer drops `rating` and adds `cards`, but forward stepwise can't backtrack — it carries `rating` along because `limit` came in at step 4.

> "It's a bit annoying that you might not get the best best model, but it's probably pretty good."

### Backward stepwise selection

Mirror image. Start with the full model, drop one predictor at a time — at each step, drop the predictor whose removal hurts the fit the least.

Same $1 + p(p+1)/2$ cost. Same "guided search, no guarantee" caveat.

> [!important] Hard requirement
> "Backwards selection requires that the number of samples $n$ is larger than the number of parameters."

Because you have to fit the full model first. Which means in the **standard OLS setting with no other tricks, $n > p$ is required.**

> "Need more number of data points than predictors for least squares without tricks. By 'tricks' I mean other stuff we're going to talk about later."

This is the same $n < p$ failure mode as the variance recap above — without regularization, OLS literally cannot be fit. Forward stepwise doesn't have this problem because it starts small.

### Hybrid stepwise

Forward + backward interleaved. After adding a variable, check whether any previously-added variable can now be dropped. Repeat.

> "You go up and then you remove one because maybe it's no longer [helpful]. In that case here, probably what would happen — if you went forward to four parameters and then went backwards, it would kill off the rating. And then if you went forward again, you would get cards."

So in principle, hybrid could recover the best-subset 4-variable answer that pure forward missed on the Credit data. The prof said he hadn't actually run it, but speculated.

### Subset selection: wrap

End of subset selection. Three approaches (best, forward/backward, hybrid). The discrete combinatorial problem of "which $X$'s belong" is what we just solved. Now we shift framing.

## Shrinkage methods: introduction

The third approach to model reduction. Don't pick a subset of $X$'s — keep them all, but shrink the coefficients.

> "We're going to fit a model containing all the parameters, so we're going to use all of them, we're not going to select explicitly which ones. And then we're going to add a constraint to the model. So we're not going to optimize, we're not going to minimize the least squares alone. We're going to minimize that and something else."

Why this counts as variance reduction: if a $\beta_j$ is shrunk to zero, that predictor effectively drops out — same end as subset selection. If shrunk partway, the effective model complexity drops without going all the way to discrete selection.

> "It tries to reduce the number of parameters effectively — effective parameters — because if the beta is zero, then essentially that parameter is not in there."

The prof's optimization framing (a recurring lens this module):

> "Think about a lot of statistics problems from the lens of optimization to be very helpful. If you don't know optimization stuff, I'd recommend taking a course on it someday — which is very useful."

Two classical shrinkage methods: **ridge** (today) and **lasso** (next time).

## Ridge regression (L2)

> "This is also known as L2. It's entirely possible that I will just start calling it L2 one day. So it would be nice if you can remember that L2 and ridge mean exactly the same thing."

The "L2" name comes from the 2-norm — the penalty squares the betas. (Forgot why "ridge" is called ridge, said he'd try to draw it later.)

The objective:

$$
\min_\beta \;\; \underbrace{\sum_{i=1}^n \left(y_i - \beta_0 - \beta_1 x_{i1} - \dots - \beta_p x_{ip}\right)^2}_{\text{RSS}} \;+\; \lambda \underbrace{\sum_{j=1}^p \beta_j^2}_{\text{L2 penalty}}
$$

(Penalty is on $\beta_1, \dots, \beta_p$ only — the intercept $\beta_0$ is **not** penalized.)

### Tug of war

The prof's central intuition for ridge:

> "RSS… it wants to make those parameters beta whatever it can to fit the data. So it's pulling them away from zero, either to big numbers, positive or negative — whatever. It's pulling them away from zero. And then this thing [the penalty] pulls them back to zero. So one is pushing away, one is pulling back. So you have this tug of war on the betas."

Consequence: the optimizer prefers solutions where **a few betas are large** (the most useful ones) and **the rest are small**. Because all betas are summed in the same penalty, letting one large beta in costs the same as several small ones — but if the data really needs that one beta, the RSS gain outweighs the penalty.

> "There's only so many ways… there's going to be a unique way where these things trade off each other such that this is happy and this is happy."

### The hyperparameter $\lambda$

> "The lambda is a tuning parameter or a hyperparameter… it's a bit annoying because you either have to pick it or there's ways of choosing what value should be."

Two extremes:

- **$\lambda = 0$**: penalty does nothing, you recover plain OLS.
- **$\lambda = \infty$**: penalty dominates, all $\beta_j \to 0$.

The interesting answer is somewhere in between. How to pick $\lambda$ — covered next time.

### Standardize the X's

> [!important] Pre-step that matters
> "Importantly, ridge regression is **not scale-invariant**, meaning that it matters what the amplitude of the beta is."

If one predictor varies on scale $10^{-4}$ and another on scale $10^3$, their corresponding $\beta_j$'s will be on wildly different scales — the small-scale predictor needs a huge $\beta$, the large-scale one needs a tiny $\beta$. Ridge sums squared betas indiscriminately, so the small-scale predictor's beta gets penalized far more for no good reason.

**Fix**: standardize each $X_j$ (e.g., divide by its standard deviation) before fitting. Then all betas are on a comparable scale and the penalty is fair.

> "So the betas will also be comparable, be at similar scales."

### Reading the trace plot

Slide showed standardized coefficients (the betas) plotted against $\lambda$:

- **Far left** ($\lambda \approx 0$): coefficients sit at their OLS values — large in magnitude, mixed signs.
- **Far right** ($\lambda$ very large): all coefficients squashed near zero.
- **In between**: smooth shrinkage paths. Coefficients don't all shrink at the same rate.

> "Somewhere in here we would want to pick a value where we have a nice model."

There's also an alternative axis where the standardized $\beta$'s are plotted against $\|\hat\beta(\lambda)\|/\|\hat\beta(0)\|$ (running 0 → 1 from full shrinkage to OLS values).

### How fitting actually proceeds

To fit ridge: take the total loss (RSS + penalty), differentiate w.r.t. each $\beta_j$, solve. Each $\beta_j$ has a contribution from both terms — they're optimized simultaneously.

(The prof gestured at the math, didn't write the closed form — that's coming tomorrow.)

## Closing — what's next

> "We'll talk more about this tomorrow. We'll finish up this part of the module six. And then we will also talk about the next part, which has to do with using some dimensionality reduction. So you actually take your $p$ predictors, you squish them into a smaller set of predictors and then use those. And there's different ways of squishing them. That's the gist of that one."

Tomorrow ([[L13-modelsel-2]]): finish ridge (closed form, choosing $\lambda$, shrinkage geometry), then [[lasso]] / L1, then start dimensionality reduction (PCR / PLS).

## Recommended exercises flagged

- The two prior-module exercises he believes have already been done: derive $\hat\beta = (X^\top X)^{-1} X^\top Y$, and show MLE = OLS under Gaussian errors. *"If not, please do it."*
- **Run subset selection on the Credit data set** (best-subset and stepwise, both forward and backward), and compare the chosen models using both a penalty criterion (BIC is the most-trusted by tradition — *"the one that people believe the most is Bayesian information criteria"*) **and** cross-validation. Form your own opinion about which to trust.
- For the hybrid approach: run it on Credit and see whether it recovers the best-subset 4-variable answer that pure forward missed.
