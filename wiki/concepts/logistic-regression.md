---
concept: logistic-regression
module: 04-classif
lectures: [L07, L08, L09, L17, L27]
isl-ref: 4.3
exercises:
  - Exercise4.4a - given β₀=−6, β₁=0.05, β₂=1, compute P(A) for a student with 40 study-hours and GPA 3.5
  - Exercise4.4b - find required study-hours so P(A) = 0.5
  - Exercise4.6b - fit glm(Direction~., binomial) on Weekly, interpret summary
  - Exercise4.6c - confusion matrix for full-data logistic regression
  - Exercise4.6d - train/test split logistic on Lag2, confusion matrix on held-out
  - Exercise4.6i - compare logistic with LDA/QDA/KNN on Weekly
  - Exercise4.6j - ROC and AUC for the four classifiers
  - CE1 problem 3a - show logit(p) is linear in covariates
  - CE1 problem 3b - interpret β₁ for an additional ace
  - CE1 problem 3c - fit logistic with ACEdiff/UFEdiff, derive class boundary, plot, sensitivity/specificity
related: [odds-and-log-odds, classification-setup, linear-discriminant-analysis, confusion-matrix, sensitivity-specificity, roc-auc, diagnostic-vs-sampling-paradigm, collinearity, gaussian-error-assumptions, generalized-additive-models]
tags:
  - concept
  - module/04-classif
aliases:
  - binary logistic regression
  - Bernoulli GLM
---

# Logistic regression

The prof's go-to binary classifier, a Bernoulli GLM with the logistic link, fit by maximum likelihood. He returned to it in five lectures and walked through it again on the 2025 exam (Q7), making it the highest-load concept in module 4. Coefficient = **log odds-ratio** is the interpretive payload; the model's assumptions parallel OLS, including the same collinearity failure mode.

## Definition (prof's framing)

> "We may assume that $Y_i$ follows a Bernoulli distribution with probability of success $p_i$.", slide deck

> "We still call it a linear model in the generalized linear models, because this term here that actually includes all the parameters we fit, that is still linear, even though [the link] is not." - [[L07-classif-1]]

The model has three pieces:

1. **Random component:** $Y_i \sim \text{Bernoulli}(p_i)$.
2. **Linear predictor:** $\eta_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$.
3. **Logistic (logit) link:** $\log\!\left(\dfrac{p_i}{1 - p_i}\right) = \eta_i$.

Inverted, this gives the **sigmoid** for $p_i$, bounded, smooth, fittable.

## Notation & setup

- $Y_i \in \{0, 1\}$, "success" = $Y_i = 1$ (state your encoding).
- $p_i = \Pr(Y_i = 1 \mid X = x_i)$.
- $\eta_i$, linear predictor; the prof writes it as $\eta$ on the board (occasionally $H$): "I typically use the letter H because I can remember what that thing's called." - [[L08-classif-2]].
- $\boldsymbol{\beta} = (\beta_0, \ldots, \beta_p)^\top$, intercept $\beta_0$.

A student asked **why the logit specifically?** Prof's answer: "I believe that you can prove that this link function is optimal for the Bernoulli distribution… every time we use the logistic regression that the log of [odds], yeah, exactly… related to GLMs." - [[L07-classif-1]]. The full canonical-link story sits in the GLM course; **outside scope** here.

## Formula(s) to know cold

**Sigmoid (inverse logit):**

$$p_i = \frac{e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}}} = \frac{1}{1 + e^{-\eta_i}}$$

**Logit (log-odds):**

$$\log\!\left(\frac{p_i}{1 - p_i}\right) = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$$

**Likelihood (factorizes by independence):**

$$L(\boldsymbol\beta) = \prod_{i=1}^n p_i^{y_i}(1 - p_i)^{1 - y_i}$$

**Log-likelihood (the form the prof sets to zero by partial differentiation):**

$$\ell(\boldsymbol\beta) = \sum_i \left[y_i \eta_i - \log(1 + e^{\eta_i})\right]$$

No closed form for $\hat{\boldsymbol\beta}$, solved numerically by Newton-Raphson / Fisher scoring.

**Prediction at a new $x_0$:**

$$\hat p(x_0) = \frac{e^{\hat\eta_0}}{1 + e^{\hat\eta_0}}, \quad \hat\eta_0 = \hat\beta_0 + \hat\beta_1 x_{01} + \cdots + \hat\beta_p x_{0p}$$

Classify to class 1 if $\hat p(x_0) > 0.5$ (default cutoff; tunable, see [[sensitivity-specificity]]).

## Insights & mental models

- **Coefficient = log odds-ratio.** A one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$. See [[odds-and-log-odds]] for the interpretation engine.
- **Same assumptions as OLS on the linear predictor.** Independence of observations; same units of $x$ matter the same way. The prof flags **collinearity** as the same failure mode: the maximum becomes non-unique. - [[L08-classif-2]]
- **Linearity in $\boldsymbol\beta$, nonlinearity in $p$.** Still a "linear model" because $\eta$ is linear in the betas; the sigmoid is just the link.
- **Per-class probabilities are useful but not always trustworthy.** "If you're a doctor, don't really trust this." - [[L08-classif-2]]
- **Direct (diagnostic) approach.** Models $\Pr(Y\mid X)$ directly, contrast with [[linear-discriminant-analysis]] which goes via Bayes' rule. See [[diagnostic-vs-sampling-paradigm]].

## Inference: the GLM table

Same R-style summary as OLS, estimate, SE, $z$-value, $p$-value (the SE comes from the Fisher information; $z$ is approximately $N(0,1)$ under standard regularity). The prof warns about absurdly small reported p-values:

> "These numbers are of course ridiculously small. Never write that in an article, people will laugh at you, because a probability of negative 200 is, you know, more likely we don't exist. … Some assumption is wrong regardless of what it is." - [[L08-classif-2]]

CIs and hypothesis tests on individual $\beta_j$'s work the same way as in OLS, all "large-$n$ approximate" because of the asymptotic normality of MLE.

## Worked example: South African heart disease

`SAheart`, 462 males, $Y$ = `chd`, predictors `sbp`, `tobacco`, `ldl`, `famhist`, `obesity`, `alcohol`, `age`. `glm(chd ~ ., family="binomial")` in one line. Output flags `age`, `famhist`, `tobacco`, `ldl` as important.

> "Interestingly, all of these seems to increase the probability of heart disease, which is fun. That's maybe why they recorded those variables." - [[L07-classif-1]]

Caveats on reading "important":
- Look at **both** $p$-value **and** effect size.
- **Units matter:** `age` in minutes vs years vs decades changes the coefficient by orders of magnitude.
- "If we get enough samples we can always get it to be significant. Almost always." - [[L07-classif-1]]

Interactions, factor variables, and prediction work exactly as in OLS, `glm()` mirrors `lm()`.

## Same failure modes as OLS

> "The collinearity problem we talked about a minute ago, that can happen here, and then this thing's no longer having a single maximum and it gets weird." - [[L08-classif-2]]

[[collinearity|Collinearity]] kills the unique-MLE property. Independence violations break inference. Standardization isn't *required* (the model is shift- and scale-equivariant in $\boldsymbol\beta$) but helps numerical stability and interpretability.

## Multi-class logistic regression, out of scope

> "We're not going to talk about it… mostly because LDA and KNN can deal with this case." - [[L07-classif-1]]

The slide deck shows the multinomial form for completeness, but the prof skipped it. **Don't atomize.** For multi-class, route to [[linear-discriminant-analysis]] / [[knn-classification]].

## Exam signals

> "What are the odds, the log odds? How do you compute them? What do they mean?" - [[L27-summary]]

> "This is the kind of question I would ask. It's simple. You calculate it. It's why you need a calculator." - [[L27-summary]] (re: odds ↔ probability)

> "How does the feature pay-zero influence the odds to default? … We need to be able to do it for the men and the women." - [[L27-summary]] (interaction trap)

> "Logistic regression would be easier to explain than LDA, especially to doctors." - [[L09-classif-3]]

The 2025 exam's biggest problem (Q7) was a multi-part logistic-regression walkthrough: fit, interpret coefficients (with interaction), define and compute sensitivity/specificity, interpret ROC, compare with KNN and tree methods. The prof said this is **the** classification flagship.

## Pitfalls

- **Reporting "$\beta_j = 0.05$ means probability goes up by 0.05", wrong.** It's the change in log-odds; probability change is non-constant (depends where on the sigmoid).
- **Reference-class flip flips signs.** State your encoding ("default = 1, non-default = 0") in your answer.
- **Interaction terms invalidate the simple "$e^{\beta_j}$ per unit" reading**: see [[odds-and-log-odds]] interaction trap.
- **Collinearity** in the predictors breaks the unique MLE the same way it breaks OLS, fix by dropping a variable, regularizing, or PCR.
- **Don't use OLS on a 0/1 response with class imbalance.** Predictions stick at 0; never crosses 0.5. (Slides hammer this.)
- **Trusting tiny $p$-values literally.** $p = 10^{-200}$ is "more likely we don't exist", some assumption is wrong.

## Scope vs ISLP

- **In scope:** Bernoulli GLM, sigmoid/logit link, MLE objective + Newton-Raphson (high-level), coefficient = log odds-ratio, GLM output table, prediction formula, comparison with LDA/QDA/KNN, ROC/AUC for model comparison.
- **Look up in ISLP:** §4.3, pp. 130–143. The MLE derivation (4.5) and the Default-data tables (4.1, 4.2, 4.3) are the canonical reference.
- **Skip in ISLP (book-only or excluded):**
  - **Multinomial logistic regression** (§4.3.5) - [[L07-classif-1]]: "we're not going to talk about it."
  - **Probit / complementary log-log link functions** - [[L07-classif-1]]: "outside the scope of this course."
  - **Poisson regression / GLMs in greater generality** (§4.6), never covered.
  - **Detailed Newton-Raphson algebra / IRLS derivation**: say "fit by MLE via Newton, no closed form" and stop.

## Exercise instances

- **Exercise4.4a**: given $\beta_0 = -6$, $\beta_1 = 0.05$ (study hours), $\beta_2 = 1$ (GPA), compute $P(A)$ for a student with 40 hours and GPA 3.5. Pure plug-into-sigmoid.
- **Exercise4.4b**: invert the sigmoid: find required study hours for $P(A) = 0.5$ at GPA 3.5. Solve $\eta = 0$.
- **Exercise4.6b**: fit `glm(Direction ~ ., binomial)` on the `Weekly` data; interpret the summary table. Which lag is significant?
- **Exercise4.6c**: full-data confusion matrix, overall correct fraction.
- **Exercise4.6d**: train/test split (1990–2008 vs 2009–2010) on `Lag2` only; held-out confusion matrix.
- **Exercise4.6i**: compare logistic with LDA/QDA/KNN on the Weekly data.
- **Exercise4.6j**: ROC curves and AUC for all four classifiers.
- **CE1 problem 3a**: derive that $\text{logit}(p_i)$ is linear in the covariates from the sigmoid form.
- **CE1 problem 3b**: interpret $\beta_1$: "one extra ace for player 1 multiplies the odds of winning by $e^{\beta_1}$."
- **CE1 problem 3c**: fit logistic on `ACEdiff` + `UFEdiff`; derive the $\hat p = 0.5$ class boundary as $x_2 = bx_1 + a$; plot; compute sensitivity/specificity on the test set.

## How it might appear on the exam

- **Calculator MCQ:** Given $\beta_0, \beta_1$ and $x_0$, compute $\hat p$ and predicted class.
- **Output interpretation:** Given a `glm()` summary table, identify significant predictors, sign of effect, and interpret coefficients on the odds scale (with proper unit-aware multiplications).
- **Class boundary derivation:** From a fitted logistic with two predictors, set $\hat p = 0.5$, solve for the line $x_2 = bx_1 + a$. Plot. (CE1.3c-style.)
- **Sensitivity/specificity from confusion matrix.** Standard companion to a logistic-regression output.
- **Method comparison:** "Why might you choose logistic regression over LDA?" → no Gaussian assumption on $X$, more interpretable, two classes.
- **Interaction trap:** Multi-step problem with `sex × x_j`; compute the per-group odds ratio. **Show your work for partial credit** even if the calculator slips.
- **MLE vs LS true/false**: the L27 mathy template applies here: "MLE under Gaussian noise = LS" generalizes to "logistic MLE has no closed form, fit numerically."

## Related

- [[odds-and-log-odds]]: the interpretive engine: $\beta_j = \log\text{(odds ratio)}$.
- [[classification-setup]]: Bernoulli responses, 0/1 loss, threshold-based assignment.
- [[diagnostic-vs-sampling-paradigm]]: logistic models $\Pr(Y\mid X)$ directly (diagnostic side).
- [[linear-discriminant-analysis]]: same linear log-odds form, different parameter-estimation route (Gaussian + Bayes vs MLE).
- [[confusion-matrix]], [[sensitivity-specificity]], [[roc-auc]]: companion performance metrics.
- [[collinearity]]: same failure mode as OLS, breaks the unique MLE.
- [[gaussian-error-assumptions]]: independence is the killer assumption here as in OLS.
- [[generalized-additive-models]]: logistic GAM extends this with $f_j(x_j)$ basis pieces on the linear predictor.
- [[bias-variance-tradeoff]]: choosing the cutoff trades sensitivity for specificity; choosing model complexity trades bias for variance.
