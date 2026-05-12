---
lecture: 13
date: 2026-02-24
module: 06-modelsel
title: "L13: Model Selection and Regularization 2 (Ridge)"
slides: modules/6ModelSel/selection_regularization_presentation_lecture1.md
topics:
  - bias-variance-tradeoff
  - subset-selection
  - aic-bic-conceptual
  - ridge-regression
  - lasso
  - elastic-net
  - l1-l2-regularization
  - shrinkage
  - cross-validation
tags:
  - lecture
  - module/06-modelsel
aliases:
  - Lecture 13
---

# L13: Model Selection and Regularization 2 (Ridge)

Session 2 of module 6. The prof reframes the whole module as **"reducing the variance,"** revisits the [[bias-variance-tradeoff]] from week 1 to motivate why penalising parameters helps, briefly notes [[aic-bic-conceptual|Mallow's Cp]] (penalty $\propto$ expected variance increase per added parameter, not examinable), then walks through [[ridge-regression]] (L2) and [[lasso]] (L1) end-to-end: formulation, the lambda sweep, the figures, what each shape of penalty does to the geometry of the solution, and finishes with [[elastic-net]] as the practical compromise. The geometric "ellipse-meets-diamond" picture is the climax of the lecture.

## Key takeaways

- **The unifying frame**: model selection and regularization are both "reduce the variance by simplifying the model." Removing or shrinking parameters costs you a tiny bit of bias and buys you a lot of variance reduction. The bias-variance decomposition is the lens, and *will be on the exam* (the prof restated this).
- **Ridge / L2**: $\text{RSS} + \lambda \sum \beta_j^2$. Closed-form, easy to fit, shrinks coefficients smoothly toward zero **but never exactly to zero**. Works even when $p > n$ (least squares blows up; ridge stays unique).
- **Lasso / L1**: $\text{RSS} + \lambda \sum |\beta_j|$. Same idea but the absolute value's non-flat behaviour at zero forces some coefficients **exactly to zero** → does model selection for free.
- **Geometric picture**: ridge constraint region is a circle/ellipsoid (smooth) → intersection with the RSS ellipses tends *not* to land on an axis. Lasso constraint region is a diamond (sharp corners on the axes) → intersection *tends* to land on a corner → sparsity. The pointy bits matter.
- **Lasso = capitalist (one rich guy wins, the rest die). Ridge = socialist (averages over correlated parameters, no one dominates).** Elastic net = the centrist compromise.
- **Cross-validation picks lambda.** You never see the bias and variance directly; if you did, you wouldn't be doing statistics. The CV curve has a sweet spot between the high-variance/low-bias extreme ($\lambda = 0$) and the high-bias/low-variance extreme ($\lambda \to \infty$).

## Reframing the module: "reducing the variance"

Opens by re-naming the module. Model selection and regularization are both ways of **reducing variance**:

> "We could also try to call this reducing the variance. And I kind of like that idea because it really gets at at least one way of looking at what these regularization terms are doing as well as the model selection - which you could argue is a kind of regularization, a kind of constraint on the model."

Scrolls back to the [[bias-variance-tradeoff]] derivation from the first/second day of class. Reproduces the decomposition: expected squared error of $\hat{f}$ vs the truth splits into a reducible part and an irreducible part. The reducible part further splits:

$$\text{Reducible} = \underbrace{(\text{truth} - \mathbb{E}[\hat{f}])^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\big[(\hat{f} - \mathbb{E}[\hat{f}])^2\big]}_{\text{Variance}}$$

He flags this as exam-critical:

> [!important] Exam flag (restated)
> "I mentioned that this is definitely going to be on the exam. I mean, just this concept. And I don't want to say how the question will be, but this is an important concept to really wrap your head around because it's deceptively confusing. It seems really simple. It's not. It's like an onion with layers."

### The trade-off intuition

Bias = (truth $-$ expected prediction). Variance = how much the fitted prediction varies across realizations of the training sample at a fixed $X_0$. The variance is taken over realizations of the data: different samples → different fits → how much they vary. The argument: if you can pay a *little* bias and the bias is squared, you can buy a *lot* of variance reduction.

> "If you reduce the bias a little bit - sorry, if you increase the bias a little bit - you can reduce the variance a lot. Because you have the squared term there."

He explicitly walks the thought experiment: pretend the *reducible* error is fixed (it isn't really; that's the whole point of "reducible"). Multiply the bias by some small factor; squared, it grows slowly. If the sum has to stay constant, the variance must drop by a *lot*. Stylised, but it's the right mental picture for why slightly-wrong-but-stable beats unbiased-but-jittery.

This is the lever both subset selection and regularization pull on. Removing parameters that don't matter barely moves bias but reduces variance a lot, and "what's not obvious is that it actually reduces the variance by a lot, while the bias just doesn't get worse."

> "In the other book, written by the same authors, they go through a more formal decomposition of this to show that this variance term depends heavily on the number of parameters. We're not going to go through that because it's left out of this course, but we kind of do show it."

So formally deriving "variance scales with $p$" is **out of scope**; the conceptual claim ("more parameters → more variance") is in scope.

## Recap of yesterday's selection methods

Notation: $p$ is the total number of regressors you *could* include; $d \leq p$ is what survives selection.

Three families for picking $d$ out of $p$ parameters, all judged by [[cross-validation]]:

- [[subset-selection|best subset selection]]: try every combination of $d$ predictors, pick the best. Combinatorial, $2^p$ models.
- [[subset-selection|forward stepwise]]: start with one parameter; greedy add the best next predictor; **always keep prior decisions**. Not guaranteed to find the global optimum but "often does pretty well."
- [[subset-selection|backward stepwise]]: start with all $p$; greedy remove. Same caveat.
- **Hybrid (forward-then-backward)**: can reverse decisions; go forward, then backward, then forward again until satisfied. "Often works well. I don't know why you don't see it that often in published articles. At least intuitively it works well."

None of the greedy methods are guaranteed to find the right model.

### Mallow's $C_p$ aside (not examinable)

Mentioned just to share an intuition. The penalty term in [[aic-bic-conceptual|Mallow's Cp]] of the form $2 d$ (where $d$ is the number of parameters in the candidate model and $p$ is the total available) is, derivation aside, *literally* the expected increase in variance from adding one parameter. So the penalty is making the variance term explicit in the objective.

> "Their derivations are beyond the scope of the course and they all kind of suck in different ways. People use them, but whatever. I don't want to go through that whole story again."

The prof acknowledges the pitch didn't land cleanly: "Maybe that wasn't so helpful in understanding. I thought it was interesting." Takeaway: he's *not* testing the formulas for $C_p$, AIC, BIC, adjusted $R^2$; only the conceptual claim that an information criterion penalises model complexity. **They don't show up on the exam.**

## Shrinkage / regularization: the second big idea

Calls it shrinkage, sparsity priors, regularization, interchangeably. Notes the field has rediscovered these "in every field - engineers figured it out, physicists figured it out, statisticians did. Hard to know who to credit." Also calls L2 by "Tikhonov" (he can't pronounce it) and notes the "ridge" name is from a 1970s statistics paper.

The unifying form:

$$\underbrace{\sum (y_i - x_i^\top \beta)^2}_{\text{loss / log-likelihood}} + \underbrace{\lambda \cdot \text{penalty}(\beta)}_{\text{regularization}}$$

> "If you think of it in terms of [GLMs], the first term could be the log likelihood of anything - Poisson log-likelihood, whatever distribution your data is. So the first term is very flexible, and you can stick this regularization term on any of those log-likelihoods."

Engineering framing: this is just adding a constraint to an over-flexible optimization (e.g. fit a rectangle subject to fixed area). Bayesian framing teased, promised for later.

## Ridge regression (L2)

$$\hat{\beta}_\text{ridge} = \arg\min_\beta \Big\{ \sum_i (y_i - x_i^\top\beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \Big\}$$

Reminds the class: **not scale-invariant** (must standardize predictors first; already covered yesterday).

### Fitting and the lambda sweep

The math is "just as easy as before": take derivatives, get a closed form. The hard part is choosing $\lambda$.

For each $\lambda$, refit. Two extremes:

- $\lambda \approx 0$: recovers OLS. **High variance, low bias.**
- $\lambda \to \infty$: all $\hat\beta_j \to 0$. **Lowest variance (perfect repeatability!), worst bias.**

> "At [the all-zeros solution], your variance over data sets is going to be great. They're always giving you zeros. Perfect repeatability. But your bias sucks because the model is very far away from [the truth]."

Somewhere between is the sweet spot; find it via [[cross-validation]]. The CV error curve dips in the middle because it implicitly captures the bias-variance balance.

> "We don't typically know what the bias and the variance are. These are just things we think about. If we knew what the bias was, we wouldn't be fitting a model - we would just use the true model and we wouldn't do any statistics, because that would be stupid."

The MSE-vs-$\lambda$ figure (in the deck) shows three curves: bias, variance, and test MSE / CV error. Bias starts low and rises with $\lambda$; variance starts high and drops sharply; test MSE = bias$^2$ + variance + noise dips through a minimum where the two curves cross-trade. The prof openly hedges on which colour-coded curve in the deck is which ("this guy did other stuff... I didn't make this figure"), but the point stands: variance can shrink to **about half** (or less) of the OLS variance for the cost of a small bias bump.

### Ridge's headline benefit: works when $p \geq n$

Standard linear regression blows up when $p \geq n$ (design matrix singular, no unique solution). Ridge does **not** blow up.

> "If you have too many parameters, you can add a regularizer to keep the model finding a unique solution. It maintains the model as convex, meaning it will still have a unique solution for a given value of $\lambda$."

This is the same trick that makes giant modern models trainable. He calls back to a comment from yesterday ("sometimes you think you're doing one thing but really you're doing another"):

> "You think you're penalizing just the number of parameters or shrinking them down to smaller numbers, but really you're allowing the model to generalize better to data you haven't seen. How weird is that?"

There's a deeper connection too: ridge in the over-parameterised regime lets you **average over multiple equally-good solutions**, directly related to [[boosting]] (fit the model 10 times, take the ensemble). With heavy regularization on a too-flexible model, "you can do that even with least squares, by simply having way too many parameters and adding a regularizer." Promised to revisit when the boosting module arrives.

### Computational angle vs subset selection

Subset selection requires fitting $2^p$ models. Hybrid forward/backward fits many too.

> "What if you're in the modern machine learning framework - almost a trillion parameters? $2^{\text{trillion}}$. It is too much. There's no way you can try every possible model that has a trillion parameters."

Regularization just augments one optimization with one extra term. Peanuts compared to combinatorial subset search.

### Ridge's drawbacks

- **Never sends coefficients exactly to zero.** Tends toward zero, but doesn't get there. So you can't use ridge to *select* a subset; just to shrink everyone. "Often it's zero enough that you can just throw them away if you want, but it won't guarantee it." → if your goal is interpretability ("which 3 of these 6 predictors actually matter?"), ridge is a worse choice than subset selection or lasso. The retained-but-shrunken coefficient still sits in the model.
- **Squaring penalises big-but-needed coefficients hard.** If a $\beta$ should genuinely be 10, then $\beta^2 = 100$ is a huge contribution to the total objective; ridge will pull it down, introducing bias precisely when one variable should *dominate* the model. The very feature you most want to keep is the one ridge fights hardest.

Recommended exercise: apply ridge to the credit dataset; compare with OLS and with subset selection.

## From ridge to lasso: the shape of the penalty

Sketches $\beta^2$ vs $\beta$ on the board (symmetrically; that symmetry is *also* why squaring works: positive and negative $\beta$ get pulled to zero equally). Two regions of the squared penalty are bad:

1. **Far from zero**: penalty grows fast, over-aggressive on big coefficients.
2. **Near zero**: the function is *flat* there ($\frac{d}{d\beta} \beta^2 = 0$ at $\beta = 0$). No gradient pushing things *to* zero; the penalty barely cares whether $\beta$ is $0.01$ or exactly $0$.

> "This is exactly why this ridge regression term doesn't ever get you to zero - because the squared function just doesn't [have a kink] there."

What we want: a penalty that's gentler at the extremes and *more* aggressive near zero. The natural candidate is the absolute value $|\beta|$:

- Linear growth → less aggressive on large $\beta$.
- Constant gradient ($\pm 1$) right up to zero → constantly pushes coefficients toward zero.
- Caveat: derivative is *not* continuous at $\beta = 0$ (jumps from $-1$ to $+1$). Makes fitting trickier ("you have to do the right thing at zero"); there are tricks (most "just end up kind of rounding this out"), but the upshot is L1 is still cheap to fit and you keep all the speed advantages over subset selection.

## Lasso (L1)

$$\hat{\beta}_\text{lasso} = \arg\min_\beta \Big\{ \sum_i (y_i - x_i^\top\beta)^2 + \lambda \sum_{j=1}^p |\beta_j| \Big\}$$

> "Just replace this squared with an absolute value."

Same lambda sweep. Same extremes (small $\lambda$ → OLS; big $\lambda$ → all zero). But here, **all-zero happens at finite $\lambda$**, not at infinity, because the constant gradient actually drags coefficients all the way down.

Notes the L0 "norm" (count nonzero coefficients) exists as the most aggressive sparsity penalty; it's basically subset selection. **Out of scope, not on the test.**

> "Tibshirani, Hastie - most of their research, if you look at what they've done, is all about lasso. They love this thing. It's like their favourite topic."

### The $\|\beta\|_1 / \|\hat\beta_\text{OLS}\|_1$ figure

The figure in the deck plots coefficients against $\|\beta_\lambda\|_1 / \|\beta_\text{OLS}\|_1$ on the x-axis (instead of $\lambda$ directly). At ratio = 0, $\lambda$ is so big that everything is exactly zero. At ratio = 1, $\lambda$ is so small that we recover OLS. Same information as a $\lambda$ axis, just rescaled.

### Simulated example: $p = 45$, $n = 50$, only 2 truly nonzero

The deck's simulated example: 45 predictors, 50 samples, **only 2 are actually related to $y$** (the other 43 are pure noise). $n \approx p$ so OLS is on the edge of singular (design matrix near-square, "almost the same number of equations and unknowns"). Without regularization, the fit is iffy. With it, things get a lot better.

- **OLS ($\lambda = 0$)**: one true coefficient (purple in the deck) is reasonably large, good. The other true coefficient (green) has a magnitude *smaller than several of the noise coefficients*. OLS does a "shit job" of identifying the right parameters. (Caveat: comparing magnitudes of nonzero coefficients is "not always the best idea"; zero vs nonzero is meaningful.)
- **Cross-validated lasso**: minimises CV error, lands at a $\lambda$ where the two true coefficients are clearly nonzero and **all 43 others are exactly zero**. Identifies the right model.

The left panel of the figure shows CV error along the $\lambda$ sweep. A horizontal reference line marks "predict all-zero except the intercept" (i.e., the CV error you'd get if every $\beta_j = 0$). Plain OLS (right end of the figure) sits *above* that line, strictly worse than predicting nothing.

> "If you can do better than that, your model really sucks - because that all-zero. And so this is lambda big. Here, lambda is essentially zero. And you can see that standard linear regression has much bigger error than it does if it just assumes zero. I'd like to think this is such a fictitious situation that it would never occur, but I'm sure we could find many publications that made this mistake."

The CV-minimum point lies somewhere in the middle and beats both extremes substantially.

> "You could throw away all the ones that are zero and rerun the model with just the two parameters. You wouldn't need any regularization, because it would be well-behaved. And you'd get exactly the same solution as if you'd done forward / backward subset selection - but instead of trying many, many, many models, you just run lasso once. Boom, same place."

This is the practical pitch: lasso = "subset selection without the combinatorial cost." Bonus aside on the CV mechanics in this regime: with $n=50, p=45$, leave-one-out (or any CV scheme) actually fits the model on **even fewer** samples than parameters; yet ridge/lasso don't blow up because the regularizer keeps the optimization well-posed.

## The geometric interpretation (ridge vs lasso)

The classic side-by-side picture: two coefficients $\beta_1, \beta_2$ in the plane. Two competing objectives:

- **RSS** is constant on **ellipses** centred at $\hat{\beta}_\text{OLS}$.
- **Lasso penalty** $|\beta_1| + |\beta_2|$ is constant on **diamonds** (rotated squares) centred at the origin.
- **Ridge penalty** $\beta_1^2 + \beta_2^2$ is constant on **circles** centred at the origin.

The solution is where the RSS ellipse first *touches* the constraint region (intersection point).

### Why lasso produces sparsity

The diamond has **sharp corners on the axes**. Ellipses tend to first hit the constraint region at one of those corners, and a corner means one $\beta = 0$. Higher-dimensional analogue: the L1 ball has sharp edges/faces along axes, and ellipsoids intersect them on those low-dimensional faces.

> "It's the same idea that both this [ellipse] and this [diamond] have constant value of the objective - and where they intersect is one of these corners. It doesn't have to be, but they tend to."

### Why ridge does not

The circle is *smooth*. The intersection lands somewhere generic; neither $\beta_1$ nor $\beta_2$ tends to be zero.

### Ridge encourages *averaging* (the deeper distinction)

Beyond "ridge doesn't go to zero": ridge **encourages ties / combinations**. If $\beta_1$ and $\beta_2$ are correlated (solve roughly the same problem), lasso picks one and zeros the other. Ridge averages over them, keeping both at moderate values.

> "Imagine $\beta_1$ and $\beta_2$ are two things that are relatively correlated. They solve the same problem. Given that your data happens to have whatever properties, $\beta_1$ is going to be a little bit worse than $\beta_2$. Then if we use [lasso], we might end up where actually $\beta_1$ is just sent straight to 0 and it's all driven by $\beta_2$. ... If we're in this ridge case, when we use ridge regression, we're actually averaging over the two. So we're saying: don't let any one of these dominate."

> "Lasso is like very capitalist. Just shoot everyone, let all the poor people die, let the one rich guy win. And L2 is more socialist. Honestly, if you go to the extremes of either one - like where the US is tending - and if you went to the extreme of socialism - probably both are too bad. Which is why we have something called the elastic net. And it's also why we try not to have extreme governments."

The averaging is robust: in the *next* dataset, maybe $\beta_2$ is the noisier one and $\beta_1$ helps; ridge has hedged.

## Comparison summary

Neither is universally better:

- **Lasso** wins when you genuinely have **few important parameters** (sparse truth).
- **Ridge** wins when **all parameters are roughly equally important** (when you want to average over multiple plausible solutions, no single one dominates).

You usually don't know a priori. CV often (not always) picks the better one for your data.

## Elastic net

$$\hat{\beta}_\text{enet} = \arg\min_\beta \Big\{ \text{RSS} + \lambda \sum \beta_j^2 + \gamma \sum |\beta_j| \Big\}$$

Both penalties together; tune both via CV. If CV says "all L1," $\lambda \to 0$ and $\gamma > 0$. If CV says "all L2," vice versa. "Often parameterized slightly differently [in libraries], but the idea is the same - you can have both." Inherits sparsity from L1 *and* the correlated-variable averaging of L2; solves the failure mode where lasso arbitrarily picks one of two correlated features.

> "This is probably the one that people use the most."

## Closing notes

Recommends repeating the credit-dataset exercise with lasso and comparing against ridge / OLS / subset selection. Two slides remain in the deck (the Bayesian interpretation of ridge/lasso as Gaussian/Laplace priors on $\beta$, and a final geometric / "credit dataset" example), deferred to next session.

> "We'll talk about it next time. No reason to keep you. So have a nice week. I'll see you guys on Monday."

No reading assigned beyond what's in the deck.

### Pointers for the exam

- The bias–variance decomposition itself: derivable, conceptually understood, **on the exam**.
- The conceptual claim that more parameters → more variance: in scope. The formal proof: out of scope.
- $C_p$, AIC, BIC, adjusted $R^2$ formulas: out of scope.
- Ridge vs lasso: formulation, geometric picture, when each wins, what each does to coefficients; all in scope.
- Elastic net: know the form and its motivation as a hybrid; details of practical tuning probably not.
- Bayesian / prior interpretation of ridge & lasso: covered next session, watch for scope flags there.
