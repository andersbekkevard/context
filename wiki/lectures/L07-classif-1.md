---
lecture: 7
date: 2026-01-27
module: 04-classif
title: Classification 1
slides: modules/4Classif/4Classif.md
topics:
  - classification-setup
  - logistic-regression
  - logit-link
  - bernoulli-distribution
  - least-squares-and-mle
  - newtons-method
  - odds-and-log-odds
  - discriminant-score-and-decision-boundary
  - misclassification-rate
  - knn-classification
  - curse-of-dimensionality
  - diagnostic-vs-sampling-paradigm
tags:
  - lecture
  - module/04-classif
aliases:
  - Lecture 7
---

# L07 — Classification 1

Substitute lecture: a PhD student from Ben's group covers the first half of module 4 (Ben is in Oslo). He motivates classification via the `Default` credit-card dataset, shows why naive linear regression on a 0/1 response is unsatisfying, then walks through [[logistic-regression]] end-to-end (model, link function, MLE, odds-ratio interpretation, prediction). Closes with the [[classification-setup|Bayes classifier]], the test/train error setup, and [[knn-classification]]. Stops short of LDA/QDA, which Ben will pick up next time. The substitute speaks fast and "tells less stories" than Ben — denser but lower on personality signals.

## Key takeaways

- Classification = response variable is **categorical (discrete)**, not continuous. Binary case is the easy one; multi-class needs dummy coding or methods like LDA / KNN.
- Naive linear regression on a 0/1 response *can* work but isn't satisfying — the fit isn't bounded to [0, 1] and "if we have many, many more cases" of one class the line may never cross the 0.5 threshold.
- Logistic regression assumes Yᵢ ~ Bernoulli(pᵢ); the **logistic link** log(p / (1 − p)) = β₀ + β₁x₁ + … keeps p in (0, 1) and gives the S-curve.
- It's still called a "linear model" because the linear predictor η is linear in the βs even though p(η) is not.
- MLE has **no closed form**; you solve the score equations numerically with **Newton's method** (Newton–Raphson / Fisher scoring).
- Coefficients interpret as **log odds-ratios**: increasing xⱼ by one unit multiplies the odds by exp(βⱼ). This is *the* story for reading logistic-regression output.
- The [[classification-setup|Bayes classifier]] (assign to argmax pₖ(x)) has the smallest test error rate — the **base error rate** is the classification analogue of irreducible error.
- KNN: non-parametric, hyperparameter K, majority vote among the K nearest neighbors (Euclidean by default). Suffers from **curse of dimensionality** and is hard to interpret per-covariate.
- Two paradigms for estimating Pr(Y = k | X): **diagnostic** (logistic, KNN — model the posterior directly) vs **sampling / generative** (LDA, QDA — model class-conditional densities + prior, flip via Bayes).

## Setup: what changes with classification

The substitute opens by noting today is "very similar to the linear regression stuff you talked about yesterday, other than… the main difference now being that the response variable can be categorical, meaning the Y is not continuous." Material is mostly **chapter 4** of ISL, with a small piece in **chapter 2** on KNN.

The response Y is now discrete — binary or multi-class (3, 4, however many). We have training data with both covariates and ground-truth labels, and we want to build a classifier for new samples. The standard approach: **build probabilities of a new sample belonging to each of the different classes**, then decide.

We assess performance by how many test samples we get right — the **misclassification rate** (a proportion). In medical contexts you might also balance false-positive vs false-negative explicitly ("if we really, really don't want to misdiagnose someone… we can sort of tune those thresholds"), but the book doesn't dwell on that.

Three methods for the module:

- **Logistic regression** — binary case.
- **K-nearest neighbors** — any number of classes, non-parametric, harder to interpret.
- **Linear and quadratic discriminant analysis** — next time, with Ben.

The motivating dataset is `Default` (credit-card defaults) from ISLR. Two predictors: `income` and `balance`. A scatter plot already tells you "balance seems to be very important because you can sort of draw a line here and that would discriminate the two classes pretty well, but income seems less important. So we should be able to see that in the numbers later when we try to fit something."

## Why naive linear regression on a 0/1 response is unsatisfying

Dummy-encode Y as 0/1 (e.g., 1 = paid, 0 = defaulted), fit OLS, threshold at 0.5. You get a number for each new sample. **In principle it can work**, but two problems:

- Nothing in linear regression enforces 0 ≤ ŷ ≤ 1.
- Class imbalance breaks it. "If we have many, many more cases [of one class], then the linear regression line… will try to fit those samples and it might not even ever pass the 0.5 threshold, meaning that regardless of whatever new data we get in here, will predict no, will predict zero." Only by extrapolating way past the data range (e.g., balance ≈ 4000) does the threshold ever get crossed.

> "It's not super nice that it's not bounded between zero and one, as probability should be."

So we want a curve that's **bounded in [0, 1]**, with parameters controlling **how sharply it switches** and **where it shifts**. The S-curve.

For multi-class, linear regression is even worse: assigning numbers 1, 2, 3 to {stroke, drug overdose, epileptic seizure} **imposes an artificial ordering**. "It's not like if you have two times the amount of stroke you suddenly get a drug overdose." Stick to dummy coding, which adds one dummy per extra class, and remove one to avoid degeneracy. Multi-class logistic regression exists but isn't covered — LDA and KNN handle it.

## Logistic regression

### Model

Assume Yᵢ ~ **Bernoulli(pᵢ)** with P(Yᵢ = 1) = pᵢ. Reference class is Y = 1 (could be Y = 0 — "just remember that later when we get the coefficients out").

We need to **link** the covariates to pᵢ. Form the linear predictor η = β₀ + β₁x₁ + … + βₚxₚ as in linear regression, then push it through a **link function**. For [[logistic-regression]]:

$$\log\!\left(\frac{p_i}{1-p_i}\right) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}.$$

Solve for p:

$$p_i = \frac{e^{\beta_0 + \beta_1 x_{i1} + \cdots}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \cdots}}.$$

This is bounded in (0, 1), gives the S-shape, and "should be able to solve these problems." For a single covariate you get the standard logistic curve.

> "We still call it a linear model in the generalized linear models, because this term here that actually includes all the parameters we fit, that is still linear, even though [the link] is not, which is why I still call it a linear model."

A student asks **why this link function specifically** — is it just convenient, or is it forced? Answer: in general there are many link functions, but for a Bernoulli response "I believe that you can prove that this link function is optimal for the Bernoulli distribution… every time we use the logistic regression that the log of [odds] — yeah, exactly… related to GLMs." The deeper canonical-link story sits in the GLM course; flagged as **outside scope** here.

### Maximum likelihood estimation

Estimate the βs by [[least-squares-and-mle|maximum likelihood]]. Likelihood is a product of Bernoulli pmfs:

$$L(\boldsymbol\beta) = \prod_i p_i^{y_i}(1 - p_i)^{1-y_i},$$

substitute pᵢ in terms of η, take logs (turns products into sums — "a lot nicer to deal with"), differentiate w.r.t. each βⱼ, set to zero. You get **p + 1 nonlinear equations** with no closed form.

Solved numerically — typically **Newton's method** (Newton–Raphson). The substitute also notes: "as you'll probably get to at the end of this course different neural networks" use related iterative schemes.

> "Does that make sense? Good, good, good."

### Interpreting the coefficients: odds and odds ratios

In linear regression, βⱼ tells you how much ŷ changes if xⱼ goes up by 1. Here that breaks because of the link function.

Define the **odds**:

$$\text{odds} = \frac{p}{1-p} = \frac{P(Y = 1 \mid X)}{P(Y = 0 \mid X)}.$$

> "Odds is used in betting or horse races… 5-to-1 chance that something happens. Then there's a 5-over-6 chance that this thing happens because it happens in 5 out of 6 cases."

The model rewrites multiplicatively:

$$\frac{p_i}{1-p_i} = e^{\beta_0} \cdot e^{\beta_1 x_{i1}} \cdots e^{\beta_p x_{ip}}.$$

Now compare odds at xⱼ + 1 vs xⱼ — almost everything cancels:

$$\frac{\text{odds}(X_j = x_{ij} + 1)}{\text{odds}(X_j = x_{ij})} = e^{\beta_j}.$$

> "By increasing the covariate by one unit, we change the odds for y to be in class 1 by a factor of the exponent of beta."

Equivalently, **βⱼ is a log odds-ratio**.

Worked on the `Default` data with `balance + income + student`:

- `income` not significant; multiplying its β by 10,000 changes odds barely at all.
- `balance`: multiplying its β by 100 changes odds **a lot more**.

So the coefficient size, the unit of the covariate, and the significance all matter when you read the output.

A reminder on the **reference class trap**: if 1 = default and we increase balance, odds-of-default go up. Flip the encoding (0 = default as reference) and the sign flips. Always remember which class you encoded as 1.

### Prediction

Given fitted β̂ and a new x₀, plug into the inverse-logit:

$$\hat p(x_0) = \frac{e^{\hat\eta_0}}{1 + e^{\hat\eta_0}}.$$

R: `predict(glm_heart, newdata=..., type="response")`. Or write the formula by hand. With cutoff 0.5, classify to class 1 if p̂ > 0.5.

Worked example: a student with balance 2000 and income 40 000 → plug in → above 0.5 → predict default.

### Worked example: South African heart disease

The `SAheart` dataset: 462 males (160 cases of CHD, 302 controls). Covariates include `sbp`, `tobacco`, `ldl`, `famhist` (binary), `obesity`, `alcohol`, `age`. Response `chd`. "Lots of very heartwarming data sets there. They're all so positive and nice."

`pairs()` plot doesn't reveal a single covariate that splits the classes cleanly — "we probably need some combination of these variables." `glm(chd ~ ., family="binomial")` fits in one line. From the summary: `age`, `famhist`, `tobacco`, `ldl` all look important; intercept too. "Interestingly, all of these seems to increase the probability of heart disease, which is fun. That's maybe why they recorded those variables."

Caveat on reading "important":

- Look at **both** p-values and **effect size** (β). Tiny β is uninteresting even if significant.
- **Units matter**: age in minutes vs years vs decades changes the size of β by orders of magnitude — comparing raw βs across covariates with different units is meaningless.
- "If we get enough samples we can always get it to be significant. Almost always." (Same point the prof hammered in L05 / L06 about big-n significance inflation.)

Interactions work exactly as in linear regression — include x₁, x₂, and x₁·x₂ as predictors. "All of the same fitting tricks… can be used here." Block-wise fits, multicollinearity diagnostics, etc.

## The Bayes classifier

We've estimated p(x) = Pr(Y | X) with logistic regression, but haven't said precisely how to *use* this for classification beyond "0.5 is probably a decent cutoff."

The [[classification-setup|Bayes classifier]] makes that precise: **assign each observation to the most likely class given its predicted values**. In binary, that's the 0.5 cutoff. For K classes, it's argmaxₖ pₖ(x).

### Bayes error rate (the irreducible error of classification)

The boundary at p = 0.5 (or argmax) is the **Bayes decision boundary**.

The **Bayes error rate** is what you'd get if you knew the *true* distribution Pr(Y | X) and classified optimally — but still had irreducible noise. "Best case performance… directly analogous to irreducible error" from the regression bias-variance story.

> "We might still get some error just because of noise in the data or maybe because the variables recorded are irrelevant. So this is the best case or the lowest amount of error we can possibly get with this data set."

In practice we don't have the true Pr(Y | X) — we estimate it via the Bernoulli + logistic model.

## Training / test error and loss

Standard split: 70/30 or 80/20 train/test. (He stumbles on the percentages — "80, 30, no, 80, 70 percent and make that test training data, sorry.") You evaluate on data the model wasn't trained on so you don't overfit — same [[bias-variance-tradeoff]] story as in regression.

Optionally a **validation set** for hyperparameter tuning — irrelevant for plain logistic regression, but if you have a fancier model with knobs (neural net layers, learning rate), use a third split.

Loss: **0/1 loss** via an indicator function. Training error rate:

$$\frac{1}{n}\sum_{i=1}^n \mathbf{1}(y_i \neq \hat y_i),$$

with the indicator returning 1 on **mismatch** ("for some reason"). Test error rate is the same formula on test samples. A good classifier has **low test error** — that's the criterion that respects the bias-variance trade-off.

## K-nearest neighbors

> "Classification is now the y, the response variable… so it's still supervised. We do know how many classes we have. So it's not k-means. K-means is clustering where we don't have a y."

Important to disambiguate: the K in [[knn-classification]] is the **number of neighbors**, not the number of classes.

### The algorithm

KNN is **non-parametric** — no distribution is fit. You set a hyperparameter K. Given a new observation x₀:

1. Find the K closest points in the training set (Euclidean distance by default — "but there's nothing stopping you from using any other notion of distance if you feel like it").
2. Look at their labels. **Majority vote**.
3. Assign x₀ to the most common class in its neighborhood.

That's it.

### Synthetic example: two Gaussian clouds

Two 100-point Gaussian clouds in 2D, μ_A = (1, 1), μ_B = (3, 3), Σ = 2·I. Eyeball: the separating line should be diagonal, around x₁ + x₂ ≈ 4. Make a grid of test points and classify each one.

- **K = 1**: just take the nearest point's class. Decision regions are ragged, with isolated islands of red inside blue territory and vice versa, "because of how noisy these two clouds are." High variance.
- **Increasing K**: islands disappear, the boundary smooths.
- **K = 150** (out of 200 total): boundary is nearly straight. "We include almost the entire dataset… every point looks at almost the entire dataset to decide where it belongs, which is maybe too much." High bias — in the limit, KNN just returns the majority class everywhere.

So the same [[bias-variance-tradeoff]] applies, parameterized by K.

### Choosing K

Pick K via cross-validation (proper treatment in **module 5**). Plot test error vs K on a fixed training set; pick K where the curve plateaus. In the synthetic example, error saturates around K = 15–20.

> "If k includes the entire dataset, then the classifier is kind of useless. Then it just picks the most likely, the class that has the most samples in the training data."

Implicit assumption: classes should have roughly similar sample counts for K to behave well. **Class imbalance** is a weakness.

### Limitations

Two main ones:

- **Curse of dimensionality.** Euclidean distance in high-dimensional spaces becomes large for everyone — "this doesn't work too well if we have many, many covariates." Cosine distance might help in some cases, "something you could play with that might be fun."
- **Hard to interpret.** No coefficients to assign to covariates. "Then you need to do more work… some extra steps on top of this classification."

Also: **scale matters**. Mixing units (minutes vs years) makes Euclidean distances "really funny." Standardize.

> "It's a very different way of classifying from the like super statistical logistic regression way of doing it."

## Two paradigms: diagnostic vs sampling

After the break, the substitute closes with the conceptual frame for what comes next.

There are two ways to estimate Pr(Y = k | X = x):

### Diagnostic paradigm

Estimate Pr(Y = k | X = x) **directly**. Logistic regression and KNN do this — they target the posterior right away. This is "what we've talked about so far."

### Sampling paradigm (generative)

Estimate it indirectly. Model:

- Class-conditional densities **fₖ(x) = Pr(X = x | Y = k)** — the distribution of the covariates *within* each class.
- Class **priors** πₖ = Pr(Y = k) — usually estimated as nₖ / n.

Then flip via Bayes' theorem:

$$\Pr(Y = k \mid X = x) = \frac{f_k(x)\,\pi_k}{\sum_l f_l(x)\,\pi_l}.$$

The substitute calls the denominator "the partition function if you're from physics."

This is the route LDA and QDA take. "But yeah, both linear and quadratic discriminant analysis is doing that, which we'll talk about with Ben next time."

> "This is sometimes convenient if we want to make our statistical assumptions on the X's instead of the Y's, or at least to a larger extent on the X's."

## Closing

He ends early. "So unless you have any super important questions or want to go through logistic regression more carefully, then I think those were the slides. No reactions. Okay. Well, let's stop early then."

Picks up next time with Ben on LDA / QDA (L08 — Classification 2).
