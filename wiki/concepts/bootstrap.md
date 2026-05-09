---
concept: bootstrap
module: 05-resample
lectures: [L11]
isl-ref: 5.2
exercises:
  - Exercise5.4  -  compute probability that observation i is in a bootstrap sample, derive 1 − 1/e ≈ 0.632
  - Exercise5.5  -  describe bootstrap algorithm for SE / CI of a regression coefficient
  - Exercise5.6  -  implement bootstrap for SE of an OLS coefficient (for-loop and boot package), compare to (XᵀX)⁻¹σ̂²
  - CE1 problem 4d  -  B=1000 bootstrap to estimate SE and 95% CI of a logistic-regression-derived probability
related: [bagging, out-of-bag-error, sampling-distribution-of-beta, confidence-and-prediction-intervals, cross-validation]
tags:
  - concept
  - module/05-resample
aliases:
  - bootstrap method
  - Efron bootstrap
  - resampling with replacement
---

# Bootstrap

The prof's "feels like magic" tool. *"Your best model for the real world is the data itself."* Resample with replacement, compute your statistic on each resample, histogram the resulting distribution. Sidesteps closed-form sampling-distribution theory entirely. Born in 1979 (Efron); the prof: *"I'm pretty sure Fisher, if he had a better computer, would have done bootstrap instead of all the distribution stuff."*

## Definition (prof's framing)

> "Use the data itself to get more information about a statistic." , slide deck, [[L11-resample-2]]

The **empirical distribution** $\hat f$ puts mass $1/n$ on each observed point. **Sampling from $\hat f$ with replacement is the bootstrap.** We can't sample more from the unknown true $f$, but we can sample as much as we want from $\hat f$ , which is our best estimate of $f$.

> "Just keep it as it is, repick points, take the same number of points, and now you have a new data set. It's different from the original one, and it's as close to what you can expect as possible because you're just resampling your best estimate of the true process." - [[L11-resample-2]]

## Notation & setup

- Original sample $X_1, \dots, X_n$ (could be vectors / regression rows / pairs $(x_i, y_i)$).
- A **bootstrap sample** $X^*_1, \dots, X^*_n$ , drawn with replacement from $\{X_1, \dots, X_n\}$, **same size $n$**. Some originals appear multiple times; some (~1/3) don't appear at all.
- $B$ = number of bootstrap samples. *"You typically want to use a big number like a thousand or ten thousand."* - [[L11-resample-2]]
- For each bootstrap sample, compute the statistic of interest $\hat\theta^*_b$.
- The collection $\{\hat\theta^*_1, \dots, \hat\theta^*_B\}$ is your estimate of the sampling distribution of $\hat\theta$.

## The algorithm (slide deck form)

**For estimating SE / sampling distribution of a statistic $\hat\theta$:**

1. For $b = 1, \dots, B$:
   a. Draw $X^*_1, \dots, X^*_n$ from $\{X_1, \dots, X_n\}$ with replacement.
   b. Compute $\hat\theta^*_b = \theta(X^*_1, \dots, X^*_n)$.
2. Estimate the standard error:

   $$\widehat{\text{SE}}_{\text{boot}}(\hat\theta) = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \big(\hat\theta^*_b - \bar\theta^*\big)^2}, \quad \bar\theta^* = \frac{1}{B} \sum_b \hat\theta^*_b$$

3. The histogram of $\{\hat\theta^*_b\}$ is your estimate of the sampling distribution.

4. Confidence intervals: percentile method (take 2.5% and 97.5% quantiles of the bootstrap distribution for a 95% CI), or normal approximation $\bar\theta^* \pm 1.96 \cdot \widehat{\text{SE}}_{\text{boot}}$.

## Formula(s) to know cold

**Probability obs $i$ is in a bootstrap sample** (Exercise 5.4 , the canonical hand-calculation):

- $P(\text{obs } i \text{ drawn on a single draw}) = 1/n$.
- $P(\text{obs } i \text{ NOT drawn on a single draw}) = 1 - 1/n$.
- $P(\text{obs } i \text{ NOT drawn in any of } n \text{ draws}) = (1 - 1/n)^n$.
- $P(\text{obs } i \text{ IS in the bootstrap sample}) = 1 - (1 - 1/n)^n$.

For large $n$:

$$\boxed{(1 - 1/n)^n \to 1/e \quad \Rightarrow \quad P(\text{obs in bootstrap}) \to 1 - 1/e \approx 0.632}$$

So roughly **63.2% of original observations are in any given bootstrap sample**, and **36.8%** are out (the "out-of-bag" fraction , see [[out-of-bag-error]]).

## Insights & mental models

- **The central insight (verbatim):** *"Your best model for the real world , like for the real data, not just the sample that you have but the bigger sample everywhere , your best model for that is the data itself. And so if you want to look at different realizations of the data, you resample from that same data with replacement, because it's always going to be the best model for the world."* - [[L11-resample-2]]
- **Why "bootstrap":** the loop on the back of a boot. Pulling yourself up by it is impossible , Baron von Munchausen famously did it. Efron picked the name because *"it feels like magic. It feels like you shouldn't be able to do it."* - [[L11-resample-2]]
- **Why with replacement:** without replacement at full $n$, you just permute the data → useless. With replacement, you genuinely draw different samples from the empirical distribution. *"The reason you do replacement is that you want to have the same length of data every single time."* - [[L11-resample-2]]
- **The "picture of the picture" analogy:** *"It's like taking a picture and then taking a picture of the picture 100 times and then averaging them, and somehow it's better than the original picture. It doesn't sound like it makes sense, but it does."* - [[L11-resample-2]]
- **One bootstrap sample tells you nothing.** A single resample gives you a different statistic from the original; that alone is meaningless. The whole point is the **distribution** across $B$ resamples.
- **Use cases:** SE, CI, hypothesis test, ask if a new value plausibly came from the distribution. The histogram-of-the-statistic is your sampling distribution; do whatever you'd normally do with one.

## The sample-median worked example (canonical slide flow)

Setup: we have $X_1, \dots, X_n \sim N(0, 1)$ (truth known for demonstration). Want to estimate $\text{SD}(\tilde X)$ where $\tilde X$ is the sample median. There's no clean closed form for the median's SD (unlike the mean's $\sigma/\sqrt n$).

1. **The "we know the truth" version**: repeatedly sample fresh data of size $n = 101$ from $N(0, 1)$, compute the median each time, take the SD across $B$ medians. Result: ~0.125. *"This isn't actually bootstrap , they're showing what the true sampling distribution looks like."* - [[L11-resample-2]]
2. **The bootstrap version**: take *one* sample of $n = 101$, then resample with replacement from it $B = 1000$ times, compute the median each time, take the SD. Result: ~0.1365.

> "Surprisingly close to the truth, especially considering it seems like magic." - [[L11-resample-2]]

## Bootstrap for regression (Exercise 5.5–5.6)

For multiple linear regression we already know $\text{Cov}(\hat\beta) = \sigma^2 (\mathbf X^\top \mathbf X)^{-1}$ , but only **under** the standard distributional assumptions on the residuals. If those assumptions fail, the closed-form SE is wrong. The bootstrap gives an alternative that makes fewer assumptions.

> "All these distributional assumptions are assumptions, and for those to be right your data has to be of a certain type, and maybe they're not. So often using a bootstrap will be better , will be making fewer assumptions and can give you a different result." - [[L11-resample-2]]

**Algorithm (Exercise 5.5 / 5.6):**

```
For b = 1, ..., B:
    Draw n rows from (X, y) with replacement → bootstrap data (X*, y*)
    Fit OLS: β̂*_b = ((X*)ᵀX*)⁻¹(X*)ᵀy*
    Store β̂*_b
SE_boot(β̂_j) = sample SD of {β̂*_{1,j}, ..., β̂*_{B,j}}
95% CI: quantile method or normal approximation
```

Compare to the theoretical $\sigma^2(X^\top X)^{-1}$ , they should agree when assumptions hold (Exercise 5.6).

> "It's just really very easy to make yourself." - [[L11-resample-2]]

A `for`-loop over `sample(x, size=n, replace=TRUE)` and a few lines around it is enough. R's `boot::boot()` automates it.

## Bootstrap for a derived quantity (CE1 problem 4d)

Estimate uncertainty of a *predicted probability* (not just a coefficient):

```
Fit logistic regression on original data → predicted prob p̂(x₀) for x₀ = (sbp=140, sex=male)
For b = 1, ..., 1000:
    Draw n rows of (X, y) with replacement
    Refit logistic regression
    Compute p̂*_b(x₀)
SE_boot(p̂(x₀)) = sample SD of {p̂*_b(x₀)}
95% CI: 2.5% and 97.5% quantiles of {p̂*_b(x₀)}
```

This is one of the prof-flagged exam-relevant patterns: **bootstrap a derived quantity** where no closed-form SE exists.

## Connection from intro stats

The prof's aside: when teaching the two-sample $t$-test, some courses also show a **bootstrap version** , resample the labels, recompute the difference of means many times, see how often it exceeds the observed difference. Same logic, no distributional assumption.

> "In my mind this is much easier to understand than the notion that a given estimator, like a mean or a median, has a distribution." - [[L11-resample-2]]

## Exam signals

> "Your best model for the world is the data itself." - [[L11-resample-2]]

> "Could we do it without replacement? I mean, I guess you could, but it's not really the same model… If you do it without replacement, then either you're always getting the same data every single time, just reordered , so that's useless." - [[L11-resample-2]]

> "Often using a bootstrap will be better , will be making fewer assumptions and can give you a different result." - [[L11-resample-2]] (regression context)

The prof spent ~half of L11 on bootstrap. **Definitely on the exam in some form** , likely either Exercise 5.4-style hand calculation, or a CE1 4d-style "estimate SE/CI of [some quantity]" pseudocode.

## Pitfalls

- **Without replacement** at full size = permutation = useless. Common confusion.
- **Treating one bootstrap sample as the answer.** It's the distribution across $B$ samples that matters.
- **$B$ too small.** A few hundred is fine for SE; CIs need 1,000+ for the percentile method to be stable.
- **Bootstrap doesn't fix bias of the estimator**: only quantifies variability. If $\hat\theta$ is biased, the bootstrap distribution is centered around $\hat\theta$, not the true $\theta$.
- **Independence trap (carries over from CV).** Bootstrap assumes the original data is iid from $f$. For time series / spatial data, naïve bootstrap breaks the dependency structure and gives wrong SEs. Block bootstrap is the fix (not in scope here).

## Scope vs ISLP

- **In scope:** the central idea ("data is its own model"), the algorithm, the with-replacement requirement, the $1 - 1/e \approx 0.632$ probability, regression / derived-quantity SE estimation, CI via percentile or normal approximation, bootstrap as alternative to closed-form SEs.
- **Look up in ISLP:** §5.2 (pp. 209–212) for the conceptual treatment; §5.3.4 (pp. 224–227) for the lab on bootstrap (R syntax , ignore per prof's exam policy).
- **Skip in ISLP (book-only, prof excluded):** detailed CI methods (BCa, studentized bootstrap), bootstrap hypothesis testing in depth, parametric vs nonparametric bootstrap distinction.

## Exercise instances

- **Exercise5.4**: derive $P(\text{obs } i \in \text{bootstrap sample}) = 1 - (1 - 1/n)^n$, take the limit to $1 - 1/e \approx 0.632$. Hand calculation. (Result reused in module 8 for [[out-of-bag-error]].)
- **Exercise5.5**: describe the bootstrap algorithm for SE / 95% CI of a regression coefficient. State the regression assumptions you make.
- **Exercise5.6**: implement the bootstrap from 5.5 in R (for-loop and `boot::boot()`); compare bootstrap SE to the theoretical $\sigma^2(X^\top X)^{-1}$ value from `summary(lm())$coeff`. Use SLID dataset, predict `wages ~ .`, focus on `age`.
- **CE1 problem 4d**: $B = 1000$ bootstrap to estimate SE and 95% CI for a logistic-regression-derived probability $\hat P(\text{chd} = 1 \mid \text{sbp} = 140, \text{sex} = \text{male})$. Interpret. The prof-flagged "bootstrap a derived quantity" pattern.

## How it might appear on the exam

- **Hand calculation:** derive $1 - (1 - 1/n)^n \to 1 - 1/e \approx 0.632$ for $n$ large. Direct port of Exercise 5.4. Could be asked numerically (compute for $n = 5$, then take the limit).
- **Pseudocode / equation-writing:** "Describe the bootstrap algorithm for estimating the standard error of $\hat\beta_j$ in multiple linear regression." Direct port of Exercise 5.5.
- **Conceptual / true-false:**
  - "Bootstrap samples are drawn with replacement from the original data" → true.
  - "The bootstrap can correct for bias in the original estimator" → false (it quantifies variability, not bias).
  - "A single bootstrap sample is informative about the estimator's distribution" → false (need $B$ samples).
- **Output interpretation:** given a histogram of $B$ bootstrap statistics, read off SE, percentile CI, and the original estimate's location in the distribution.
- **Use-case justification:** "Why use the bootstrap for the SE of $\hat\beta$ when the closed-form $\sigma^2(X^\top X)^{-1}$ exists?" → because the closed form depends on Gaussian-error / IID assumptions; the bootstrap requires neither.
- **The "with vs without replacement" question**: almost certain to appear as T/F somewhere.

## Related

- [[bagging]]: bootstrap aggregation; uses the same resampling but for prediction, not uncertainty
- [[out-of-bag-error]]: uses the $1 - 1/e$ result; the leftover ~37% becomes a free test set
- [[sampling-distribution-of-beta]]: the closed-form alternative when assumptions hold
- [[confidence-and-prediction-intervals]]: bootstrap is one route to a CI
- [[cross-validation]]: the other big resampling family in module 5
