---
concept: diagnostic-vs-sampling-paradigm
module: 04-classif
lectures: [L07, L09]
isl-ref: 4.4
exercises: []
related: [logistic-regression, knn-classification, linear-discriminant-analysis, quadratic-discriminant-analysis, naive-bayes, classification-setup, multivariate-normal]
tags:
  - concept
  - module/04-classif
aliases:
  - generative vs discriminative
  - diagnostic paradigm
  - sampling paradigm
---

# Diagnostic vs sampling (generative) paradigm

The conceptual divider in module 4. Two routes to estimating $\Pr(Y = k \mid X = x)$:

- **Diagnostic**: model the posterior **directly**. ([[logistic-regression]], [[knn-classification]].)
- **Sampling / generative**: model the **class-conditional density** $f_k(x) = \Pr(X \mid Y = k)$ and the **prior** $\pi_k = \Pr(Y = k)$, then flip via Bayes. ([[linear-discriminant-analysis|LDA]], [[quadratic-discriminant-analysis|QDA]], [[naive-bayes]].)

The prof's slide deck calls these the "two approaches to estimate $\Pr(Y \mid X)$." The choice maps to **what assumptions you're willing to make on $X$** , none (logistic, KNN) vs Gaussian (LDA, QDA) vs Gaussian-with-independence (naive Bayes).

## Definition (prof's framing)

> "There are two ways to estimate $\Pr(Y = k \mid X = x)$." - [[L07-classif-1]]

### Diagnostic paradigm

> "Estimate $\Pr(Y = k \mid X = x)$ directly. Logistic regression and KNN do this , they target the posterior right away." - [[L07-classif-1]]

The fitted model **is** an estimate of the posterior. Logistic regression: parametric (sigmoid of linear $\eta$). KNN: non-parametric (fraction of $K$ neighbors in each class).

### Sampling paradigm (generative)

> "Estimate it indirectly. Model the class-conditional densities $f_k(x) = \Pr(X = x \mid Y = k)$ , the distribution of the covariates *within* each class , and class priors $\pi_k = \Pr(Y = k)$ , usually estimated as $n_k / n$. Then flip via Bayes' theorem." - [[L07-classif-1]]

> "We're flipping it around. So now, instead of modeling this probability of Y given X directly, we want to make a model of what is the prior distribution of Y, like how likely is this class A or class B, and the probability of X given Y." - [[L09-classif-3]]

## The Bayes-flip formula

Both paradigms ultimately want $\Pr(Y \mid X)$. The diagnostic models it; the sampling derives it via:

$$\Pr(Y = k \mid X = x) = \frac{f_k(x)\,\pi_k}{\sum_\ell f_\ell(x)\,\pi_\ell}$$

The denominator is the "partition function if you're from physics" (substitute lecturer's framing , [[L07-classif-1]]).

## The roster , which method goes where

| Method | Paradigm | What it models |
|--------|----------|----------------|
| Logistic regression | Diagnostic | $\Pr(Y \mid X)$ via sigmoid of linear $\eta$ |
| KNN | Diagnostic | $\Pr(Y \mid X)$ as fraction of $K$ nearest neighbors |
| LDA | Sampling | $f_k(x) \sim \mathcal{N}(\mu_k, \Sigma)$ + $\pi_k$, then Bayes |
| QDA | Sampling | $f_k(x) \sim \mathcal{N}(\mu_k, \Sigma_k)$ + $\pi_k$, then Bayes |
| Naive Bayes | Sampling | $f_k(x) = \prod_j f_{kj}(x_j)$ + $\pi_k$, then Bayes |

## Insights & mental models

- **The trade-off is in the assumptions on $X$.** Diagnostic methods assume nothing about $\Pr(X)$ , they only model $\Pr(Y \mid X)$. Generative methods commit to a parametric form for the class-conditional density, which is more efficient when correct, riskier when wrong.
- **When the assumption holds, generative wins.** With Gaussian class-conditionals and shared $\Sigma$, LDA is the maximum-likelihood Bayes-optimal classifier. Logistic with the same data approaches the same boundary but with more variance (no Gaussian assumption to lean on).
- **When the assumption fails, diagnostic wins.** Wonky non-Gaussian features, mixed-type predictors, multi-modal class densities , logistic regression is more robust because it doesn't commit.
- **The boundary form is shared** between LDA and logistic regression , both are linear in the log-odds. The slide deck makes this explicit:

> "For a two-class problem, one can show that for LDA $\log(p_1(x)/(1 - p_1(x))) = c_0 + c_1 x_1$, thus the same linear form. The difference is in how the parameters are estimated." , slide deck

So LDA ↔ logistic ≈ "same model, different fitting." LDA leans on Gaussian-MLE; logistic leans on direct-MLE with no $X$ distribution.

- **Multi-class is easier for the generative side.** LDA / QDA / naive Bayes naturally handle $K > 2$. Logistic needs the multinomial extension (which the prof skipped , [[logistic-regression]]).
- **Computing class probabilities is more direct for the generative side**: once you have $f_k$ and $\pi_k$, you have everything by Bayes. Logistic gives you $\Pr(Y \mid X)$ directly anyway, so this matters mostly for naive Bayes (mixed-type features).

## Why this matters for the exam

The prof's L09 closing comparison maps **method choice** to **which assumptions are met**:

> "LDA: small $n$, classes well separated, Gaussian assumption holds, lots of classes. QDA: Gaussian holds but $\Sigma_k$ unequal. Need enough data. Naive Bayes: large $p$. Skip the cross-covariance. Logistic: two classes, no distributional commitment on $X$, want interpretability. KNN: wonky non-parametric boundary. Bad for large $p$." - [[L09-classif-3]]

Knowing which paradigm a method belongs to is the first step of justifying the choice.

## Exam signals

> "This is sometimes convenient if we want to make our statistical assumptions on the X's instead of the Y's, or at least to a larger extent on the X's." - [[L07-classif-1]]

> "The annoying thing about statistics is you don't really know when to use what." - [[L09-classif-3]]

> "Logistic regression and KNN _directly estimate_ $\Pr(Y = k \mid X = x)$ (diagnostic paradigm). LDA, QDA and naive Bayes _indirectly estimate_ $\Pr(Y = k \mid X = x) \propto f_k(x) \cdot \pi_k$ (sampling paradigm)." , slide deck

A standard MCQ pattern: **classify a method into the right paradigm** , fair-game conceptual question.

## Pitfalls

- **Calling KNN "Bayesian" or "generative."** It's neither. KNN is non-parametric *diagnostic* , it estimates the posterior via local frequencies, not via a generative model of $X$.
- **Calling logistic regression "generative."** It's the canonical diagnostic / discriminative method.
- **Confusing the **Bayes classifier** (the optimal abstract decision rule) with **naive Bayes** (one specific generative model).** The Bayes classifier uses the *true* $\Pr(Y \mid X)$; naive Bayes is a specific generative classifier with the conditional-independence assumption. The "Bayes" in both refers to Bayes' theorem flipping conditioning.
- **Treating the paradigm choice as a choice between paradigms.** It's really a choice between **which assumptions you want to make on the $X$ distribution.** None → diagnostic. Gaussian → LDA. Class-specific Gaussian → QDA. Diagonal Gaussian → naive Bayes.

## Scope vs ISLR

- **In scope:** Definition of both paradigms, the Bayes-flip formula, which methods belong to which side, when to prefer which (assumption-driven argument).
- **Look up in ISLR:** §4.4 introduction (pp. 144–145) for the Bayes-theorem framing; §4.5.1 for the formal LDA-vs-logistic-vs-naive-Bayes analytical comparison.
- **Skip in ISLR:** Discriminative-vs-generative literature outside ISLR's framing , never covered.

## Exercise instances

None , no exercise drills the paradigm distinction in isolation. The concept is the **scaffolding** that lets Anders make sense of why module 4 covers five methods rather than one. Used as conceptual framing for method-choice discussions in [[linear-discriminant-analysis]] / [[logistic-regression]] / [[quadratic-discriminant-analysis]] / [[naive-bayes]] / [[knn-classification]].

## How it might appear on the exam

- **MCQ:** "Which of the following classifiers is *generative*?" → LDA, QDA, naive Bayes (any of them). "Which is *discriminative*?" → logistic, KNN.
- **T/F:** "Naive Bayes models $\Pr(Y \mid X)$ directly" → false (it models $\Pr(X \mid Y)$ and uses Bayes' theorem). "Logistic regression makes assumptions about the distribution of $X$" → false (only about the linear predictor).
- **Method-choice justification:** "Why might LDA be preferred to logistic regression here?" → "the Gaussian assumption seems plausible and the classes are well-separated, so the parametric efficiency of LDA pays off."
- **The Bayes-flip:** "Given $f_k(x)$ and $\pi_k$, write $\Pr(Y = k \mid X = x)$." → just the canonical formula.

## Related

- [[logistic-regression]]: diagnostic, two-class, parametric.
- [[knn-classification]]: diagnostic, non-parametric.
- [[linear-discriminant-analysis]]: sampling, Gaussian + pooled $\Sigma$.
- [[quadratic-discriminant-analysis]]: sampling, Gaussian + class-specific $\Sigma_k$.
- [[naive-bayes]]: sampling, conditional independence.
- [[classification-setup]]: both paradigms aim to approximate the Bayes classifier.
- [[multivariate-normal]]: the class-conditional density assumption for LDA/QDA.
