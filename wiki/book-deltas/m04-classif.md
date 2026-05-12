---
title: "M04: Classification — Book delta"
module: 04-classif
isl-ch: 4
lectures: [L07, L08, L09]
---

# Module 04: Classification — Book delta

ISLP §4.3–§4.5 is a remarkably complete treatment of [[logistic-regression|logistic regression]], [[linear-discriminant-analysis|LDA]], [[quadratic-discriminant-analysis|QDA]], [[naive-bayes|naive Bayes]], and [[roc-auc|ROC/AUC]]: the model assumptions, the [[discriminant-score-and-decision-boundary|discriminant-score]] formulas (eqs. 4.18, 4.24, 4.28), and the [[bias-variance-tradeoff|bias-variance]] argument for choosing among them are all in the book. The deltas in this module are not large doctrinal pieces — they are concrete formulas and worked artifacts that the prof (or the slide deck he taught from) wrote down explicitly, and that ISLP either omits, sketches but never closes, or hides inside a verbal aside.

The biggest single delta is the **MLE machinery for logistic regression**: ISLP gives the product-form likelihood (eq. 4.5) and then says "the mathematical details of maximum likelihood are beyond the scope of this book," whereas Benjamin's slide deck writes the log-likelihood, takes derivatives, and names Newton–Raphson. Second-biggest: the **explicit multivariate pooled covariance estimator** (the 1D version is in ISLP eq. 4.20 but the $p>1$ formula is only described verbally). Third: the **prof's two flagged decision-boundary derivation patterns** (the 1D-LDA boundary in the form he asked for in R, and the 2D worked example $\mu_A=(1,1)$, $\mu_B=(3,3)$, $\Sigma=2I$ → $x_2 = 4 - x_1$, which he flagged as *the* exam template for module 4).

Everything below is in scope per `docs/scope.md`. Out-of-scope items (multinomial logistic regression beyond the brief mention, probit/cloglog link functions, Fisher's eigenvalue derivation of LDA, asymmetric ROC analysis for imbalanced classes) get no atom here.

---

## 1. Logistic regression: the MLE machinery

### 1.1 The three equivalent forms of the log-likelihood

[[[L07-classif-1|L07]], [[L08-classif-2|L08]], slide deck §"Estimating the regression coefficients with ML"; concept: [[logistic-regression]]]

ISLP eq. (4.5) gives only the product-form likelihood:

$$
L(\boldsymbol{\beta}) \;=\; \prod_{i: y_i = 1} p(x_i) \prod_{i': y_{i'}=0} \bigl(1 - p(x_{i'})\bigr)
\;=\; \prod_{i=1}^n p_i^{y_i}(1 - p_i)^{1 - y_i}.
$$

Then ISLP says "the mathematical details of maximum likelihood are beyond the scope of this book." Benjamin's slide deck explicitly derives the **log-likelihood** in three equivalent forms — this is the object you actually differentiate, and the form he taught from:

$$
\begin{aligned}
\ell(\boldsymbol{\beta})
&= \sum_{i=1}^n \Bigl[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \,\Bigr] \\
&= \sum_{i=1}^n \Bigl[\, y_i \log\!\bigl(\tfrac{p_i}{1 - p_i}\bigr) + \log(1 - p_i) \,\Bigr] \\
&= \sum_{i=1}^n \Bigl[\, y_i\,\eta_i \;-\; \log\!\bigl(1 + e^{\eta_i}\bigr) \,\Bigr],
\end{aligned}
$$

where $\eta_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$ is the linear predictor and $p_i = e^{\eta_i}/(1 + e^{\eta_i})$.

The third form is the clean one: $y_i \eta_i$ is linear in $\boldsymbol\beta$, and $\log(1 + e^{\eta_i})$ is the **log-partition** that makes $\ell$ concave in $\boldsymbol\beta$ — that concavity is why the MLE is unique (when it exists) and why the iterative solver converges.

### 1.2 Score equations and Newton–Raphson

[[[L07-classif-1|L07]], slide deck §"Estimating the regression coefficients with ML"; concept: [[logistic-regression]]]

ISLP does **not** name Newton–Raphson; the slide deck does. The prof's framing (L07):

> "MLE has no closed form; you solve the score equations numerically with Newton's method (Newton–Raphson / Fisher scoring)."

The mechanics, reproduced from the slide deck and L07:

1. Differentiate $\ell$ in (1.1) with respect to each $\beta_j$:

   $$
   \frac{\partial \ell}{\partial \beta_j}
   \;=\; \sum_{i=1}^n \bigl(y_i - p_i\bigr)\,x_{ij},
   \qquad j = 0, 1, \ldots, p.
   $$

   (Take $x_{i0} \equiv 1$ for the intercept.) Stack into a vector:

   $$
   \boxed{\; \nabla\ell(\boldsymbol{\beta}) \;=\; X^\top (y - p), \;}
   $$

   where $X$ is the $n \times (p+1)$ design matrix, $y$ the $n$-vector of responses, $p$ the $n$-vector of fitted probabilities. **Setting this to zero gives $p + 1$ nonlinear equations in $\boldsymbol\beta$ — no closed form.**

2. Solve numerically by **Newton–Raphson**: at iterate $\boldsymbol\beta^{(t)}$,

   $$
   \boldsymbol\beta^{(t+1)} \;=\; \boldsymbol\beta^{(t)} \;-\; \bigl[H(\boldsymbol\beta^{(t)})\bigr]^{-1}\,\nabla\ell(\boldsymbol\beta^{(t)}),
   $$

   where the Hessian is

   $$
   H(\boldsymbol\beta) \;=\; \frac{\partial^2 \ell}{\partial\boldsymbol\beta\,\partial\boldsymbol\beta^\top} \;=\; -X^\top W X,
   \qquad W = \operatorname{diag}\bigl(p_i(1 - p_i)\bigr).
   $$

   **Fisher scoring** replaces $H$ with its expectation $\mathbb{E}[-H] = X^\top W X$; for canonical-link GLMs (the logit is canonical for Bernoulli) the two are identical, so the distinction is verbal here.

3. Convergence is fast (quadratic near the optimum) because $\ell$ is concave. The standard error of $\hat{\boldsymbol\beta}$ comes from the inverse Fisher information at convergence: $\widehat{\operatorname{Var}}(\hat{\boldsymbol\beta}) = (X^\top W X)^{-1}$ evaluated at $\hat p_i$. This is the engine behind the R-style GLM `summary` table (estimate / SE / $z$).

> [!note] Exam-relevant takeaway
> You will not be asked to run a Newton step by hand. You may be asked to write — in math, English, or pseudocode — *how* logistic-regression coefficients are estimated. The answer is: write the log-likelihood, differentiate, set $\nabla\ell = 0$, observe no closed form, iterate by Newton–Raphson. ISLP gives you the first half; this section gives you the second.

### 1.3 Closed-form inversion: predicted-probability ↔ required covariate value

[[[L07-classif-1|L07]]; Exercise 4.4b; concept: [[logistic-regression]]]

Not in ISLP as a named result, but worth pinning down for hand calculation. Given fitted $\hat\beta_0, \hat\beta_1, \ldots, \hat\beta_p$, the predicted probability at $x_0$ is the sigmoid

$$
\hat p(x_0) \;=\; \frac{e^{\hat\eta_0}}{1 + e^{\hat\eta_0}}, \qquad \hat\eta_0 = \hat\beta_0 + \sum_j \hat\beta_j x_{0j}.
$$

Inverting for "what $x_{0j}$ gives $\hat p(x_0) = p^\star$?" with all other covariates held fixed: take logit of both sides,

$$
\log\!\frac{p^\star}{1 - p^\star} \;=\; \hat\beta_0 + \hat\beta_j x_{0j} + \sum_{k \neq j} \hat\beta_k x_{0k},
$$

solve linearly:

$$
\boxed{\; x_{0j} \;=\; \frac{1}{\hat\beta_j}\!\left[\, \log\!\tfrac{p^\star}{1 - p^\star} \;-\; \hat\beta_0 \;-\; \sum_{k \neq j} \hat\beta_k x_{0k} \,\right]. \;}
$$

Special case $p^\star = 0.5$: $\log\frac{0.5}{0.5} = 0$, so the threshold-crossing covariate value is $x_{0j} = -(\hat\beta_0 + \sum_{k\neq j}\hat\beta_k x_{0k})/\hat\beta_j$. Calculator-friendly, the kind of problem Exercise 4.4b drills.

---

## 2. LDA: the full pooled-covariance toolkit

### 2.1 Multivariate pooled covariance estimator (the $p > 1$ formula)

[[[L09-classif-3|L09]], slide deck §"Estimators for p>1"; concept: [[linear-discriminant-analysis]]]

ISLP gives the 1D pooled-variance estimator explicitly in eq. (4.20):

$$
\hat\sigma^2 \;=\; \frac{1}{n - K} \sum_{k=1}^K \sum_{i: y_i = k}(x_i - \hat\mu_k)^2.
$$

ISLP then says, of the $p > 1$ case, only that "the formulas are similar to those used in the one-dimensional case, given in (4.20)." It does not write them out. The slide deck does, and Anders will want this at the exam table:

**Per-class sample covariance:**

$$
\hat{\boldsymbol{\Sigma}}_k \;=\; \frac{1}{n_k - 1} \sum_{i: y_i = k} (x_i - \hat{\boldsymbol\mu}_k)(x_i - \hat{\boldsymbol\mu}_k)^\top \;\;\in\;\; \mathbb{R}^{p \times p}.
$$

**Pooled covariance (the LDA $\hat\Sigma$):**

$$
\boxed{\; \hat{\boldsymbol\Sigma} \;=\; \sum_{k=1}^K \frac{n_k - 1}{n - K}\,\hat{\boldsymbol\Sigma}_k \;=\; \frac{1}{n - K}\sum_{k=1}^K \sum_{i: y_i = k}(x_i - \hat{\boldsymbol\mu}_k)(x_i - \hat{\boldsymbol\mu}_k)^\top. \;}
$$

Equivalently: a weighted average of per-class sample covariances, weighted by $(n_k - 1)$ degrees of freedom. The denominator is $n - K$ (degrees of freedom of the pooled estimator), not $n$. **In 1D the formula collapses to ISLP eq. (4.20).** Exercise 4.2a drills exactly this for a 2-class bank-note example.

### 2.2 1D-LDA decision boundary, two classes, unequal priors

[[[L09-classif-3|L09]], slide deck §"Parameter estimators", R code chunk; concept: [[discriminant-score-and-decision-boundary]]]

ISLP gives the **equal-priors** 1D boundary (eq. 4.19):

$$
x \;=\; \frac{\mu_1 + \mu_2}{2}.
$$

The **unequal-priors** version is in the slide deck's R example but never written algebraically. The prof's R rule line:

```
rule = 0.5*(mean(train1) + mean(train2))
     + var.pool*(log(n2train/n) - log(n1train/n)) / (mean(train1) - mean(train2))
```

is exactly the closed-form solution of $\delta_1(x) = \delta_2(x)$ for 1D LDA. Reproducing the derivation:

Set $\delta_1(x) = \delta_2(x)$ with $\delta_k(x) = x \mu_k/\sigma^2 - \mu_k^2/(2\sigma^2) + \log\pi_k$ (the 1D LDA discriminant of ISLP eq. 4.18):

$$
x\,\frac{\mu_1}{\sigma^2} - \frac{\mu_1^2}{2\sigma^2} + \log\pi_1 \;=\; x\,\frac{\mu_2}{\sigma^2} - \frac{\mu_2^2}{2\sigma^2} + \log\pi_2.
$$

Collect:

$$
x\,\frac{\mu_1 - \mu_2}{\sigma^2} \;=\; \frac{\mu_1^2 - \mu_2^2}{2\sigma^2} \;+\; \log\pi_2 - \log\pi_1
\;=\; \frac{(\mu_1 - \mu_2)(\mu_1 + \mu_2)}{2\sigma^2} \;+\; \log\!\frac{\pi_2}{\pi_1}.
$$

Solve for $x$ (dividing both sides by $(\mu_1 - \mu_2)/\sigma^2$, valid whenever the means differ):

$$
\boxed{\; x \;=\; \frac{\mu_1 + \mu_2}{2} \;+\; \frac{\sigma^2}{\mu_1 - \mu_2}\,\log\!\frac{\pi_2}{\pi_1}. \;}
$$

Equal priors $\Rightarrow$ second term vanishes $\Rightarrow$ ISLP's midpoint rule. **Direction of effect:** increasing $\pi_1$ makes $\log(\pi_2/\pi_1)$ more negative, which (with $\mu_1 > \mu_2$, so the first factor is positive) **decreases $x$** — the boundary moves *away from* class 1, i.e. more $x$-values get classified as class 1. The prof flagged the sign direction as a common trap.

### 2.3 The multivariate worked example, $\mu_A = (1,1)$, $\mu_B = (3,3)$, $\Sigma = 2I$ — full algebra

[[[L09-classif-3|L09]], slide deck §"Back to our synthetic example"; concept: [[discriminant-score-and-decision-boundary]]]

This is **the** prof's flagged exam template ("And that's often an exam question … you have the pies, you have the mu's, and you'd have a value for the standard deviation, and then you would solve for where is the decision point"). The slide deck stops at the answer $x_2 = 4 - x_1$; reproducing every step.

Setup: $\mu_A = (1,1)^\top$, $\mu_B = (3,3)^\top$, $\Sigma = 2I$ so $\Sigma^{-1} = \tfrac{1}{2}I$, equal priors $\pi_A = \pi_B = 0.5$.

The multivariate LDA discriminant (ISLP eq. 4.24):

$$
\delta_k(x) \;=\; x^\top \Sigma^{-1} \mu_k \;-\; \tfrac{1}{2}\mu_k^\top \Sigma^{-1} \mu_k \;+\; \log\pi_k.
$$

Equating $\delta_A(x) = \delta_B(x)$:

$$
x^\top \Sigma^{-1}(\mu_A - \mu_B) \;-\; \tfrac{1}{2}\mu_A^\top \Sigma^{-1}\mu_A \;+\; \tfrac{1}{2}\mu_B^\top\Sigma^{-1}\mu_B \;+\; \log\pi_A - \log\pi_B \;=\; 0.
$$

Compute each piece.

**Cross-term coefficient $\Sigma^{-1}(\mu_A - \mu_B)$:**

$$
\Sigma^{-1}(\mu_A - \mu_B) \;=\; \tfrac{1}{2}I\,\begin{pmatrix} -2 \\ -2 \end{pmatrix} \;=\; \begin{pmatrix} -1 \\ -1 \end{pmatrix}.
$$

So $x^\top \Sigma^{-1}(\mu_A - \mu_B) = -x_1 - x_2$.

**Intercept piece $-\tfrac{1}{2}\mu_A^\top \Sigma^{-1}\mu_A + \tfrac{1}{2}\mu_B^\top \Sigma^{-1}\mu_B$:**

$$
\mu_A^\top \Sigma^{-1} \mu_A \;=\; (1,1)\,\tfrac{1}{2}I\,(1,1)^\top \;=\; \tfrac{1}{2}(1 + 1) \;=\; 1.
$$

$$
\mu_B^\top \Sigma^{-1} \mu_B \;=\; (3,3)\,\tfrac{1}{2}I\,(3,3)^\top \;=\; \tfrac{1}{2}(9 + 9) \;=\; 9.
$$

So the intercept piece is $-\tfrac{1}{2}(1) + \tfrac{1}{2}(9) = -\tfrac{1}{2} + \tfrac{9}{2} = 4$.

**Prior piece:** equal priors $\Rightarrow \log\pi_A - \log\pi_B = 0$.

**Assemble:**

$$
-x_1 - x_2 + 4 \;=\; 0 \quad\Longleftrightarrow\quad \boxed{\; x_2 \;=\; 4 - x_1. \;}
$$

A line, falling straight out of the LDA assumption — the prof's emphasized observation: *"I didn't ask for a line. I just equated the two things and then I solved for them and then it became a line."*

> [!note] Procedural template for exam day
> Whenever a 2-class LDA problem in 2D drops with given $\mu_A, \mu_B, \Sigma, \pi_A, \pi_B$:
> 1. Compute $\Sigma^{-1}$ (small dimension, hand-invert).
> 2. Compute the cross-term row vector $v = \Sigma^{-1}(\mu_A - \mu_B)$. The boundary's $x$-coefficient row is $v$.
> 3. Compute the constant $c = -\tfrac{1}{2}\mu_A^\top \Sigma^{-1}\mu_A + \tfrac{1}{2}\mu_B^\top\Sigma^{-1}\mu_B + \log(\pi_A/\pi_B)$.
> 4. Boundary equation: $v^\top x + c = 0$. In 2D, solve for $x_2$ as a linear function of $x_1$.
> 5. **Normal direction is $\Sigma^{-1}(\mu_A - \mu_B)$, not $\mu_A - \mu_B$.** This is the subtlety ISLP glosses: $\Sigma^{-1}$ rotates and rescales the obvious connecting vector. Only when $\Sigma \propto I$ do the two coincide.

### 2.4 Posterior-probability recovery: the softmax of discriminant scores

[[[L09-classif-3|L09]], slide deck §"Posterior probabilities"; concept: [[linear-discriminant-analysis]]]

ISLP gives the Bayes-rule expression $p_k(x) = \pi_k f_k(x)/\sum_\ell \pi_\ell f_\ell(x)$ (eq. 4.15) and notes you compute it explicitly. The slide deck adds a much cleaner recovery formula directly from the discriminant scores, which is what you'd actually use once you have the $\delta_k$'s:

$$
\boxed{\; \hat P(Y = k \mid X = x) \;=\; \frac{e^{\hat\delta_k(x)}}{\sum_{\ell=1}^K e^{\hat\delta_\ell(x)}}. \;}
$$

**Why this works.** $\delta_k(x) = \log[\pi_k f_k(x)] + (\text{terms not depending on }k) + \text{const}$. Exponentiating and normalizing kills the additive class-independent constant in the numerator and denominator simultaneously, so the softmax of $\delta$'s returns the correct posterior. **It works identically for QDA** (just use the QDA $\delta_k$) and for naive Bayes.

Practical consequence: once you've computed $\delta_k(x)$ for all $k$, you have everything — the classification (largest $\delta_k$) **and** the probability estimates — without ever explicitly evaluating the Gaussian density.

### 2.5 LDA as dimensionality reduction: $K$ classes → $K-1$ scores

[[[L08-classif-2|L08]], [[L09-classif-3|L09]]; concept: [[linear-discriminant-analysis]]]

Not in ISLP §4.4 (it lives behind the door of §4.4 "Fisher's discriminant" treatment, which is the optional section the prof skipped). The prof's framing:

> "$K$ classes → $K-1$ discriminant scores … you're basically taking a big matrix and transforming it into a new matrix, now of discriminant scores, that if you have fewer categories than you did originally, then actually you have a new transformation of the matrix specifically designed such that these things are best separated with a line."

Concretely: the $K$ discriminant functions $\delta_1, \ldots, \delta_K$ sum to a constant up to a single $k$-free term (because the $K$ posteriors $\hat P(Y = k \mid X = x)$ sum to 1), so only $K - 1$ of them are linearly independent. The effective representation of $x$ for classification purposes is the $(K-1)$-vector of *contrasts* $(\delta_k - \delta_K)_{k=1}^{K-1}$, which lives in $\mathbb{R}^{K-1}$, independent of $p$.

This is why **LDA is robust in high $p$ where KNN dies of the curse**: LDA implicitly projects $p$-dimensional $x$ down to a $(K-1)$-dimensional discriminant-score representation where classes are linearly separable by construction. For $K = 2$, you get a single discriminant score and the entire problem reduces to thresholding it.

---

## 3. QDA: the compact form and the parameter count

### 3.1 QDA discriminant in compact (Mahalanobis) form

[[[L09-classif-3|L09]], slide deck §"Quadratic Discriminant Analysis"; concept: [[quadratic-discriminant-analysis]]]

ISLP gives the expanded form (eq. 4.28) — both equivalent forms are shown in the slide deck, and the compact one is much easier to keep in working memory:

$$
\boxed{\; \delta_k(x) \;=\; -\tfrac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) \;-\; \tfrac{1}{2}\log|\Sigma_k| \;+\; \log\pi_k. \;}
$$

Reading: **(negative half-squared Mahalanobis distance to class $k$ mean) + (volume penalty for class $k$) + (log-prior of class $k$)**.

- The Mahalanobis term measures how far $x$ is from $\mu_k$ accounting for the *shape* of $\Sigma_k$.
- The $-\tfrac{1}{2}\log|\Sigma_k|$ is a "volume penalty" — wider class distributions (large $|\Sigma_k|$) lose, because a wide-density class explains a fixed $x$ less well per unit volume than a tight-density class. This term is what survives the QDA derivation that didn't survive LDA's (in LDA, $|\Sigma|$ is the same across $k$ and cancels).
- The log-prior boosts more-populous classes.

This three-piece reading is the prof's go-to mental model and the cleanest setup for an exam-day derivation of "where does the quadratic come from."

### 3.2 Parameter count: LDA vs QDA vs naive Bayes (the bias-variance ledger)

[[[L09-classif-3|L09]], slide deck §"LDA vs QDA"; concepts: [[linear-discriminant-analysis]], [[quadratic-discriminant-analysis]], [[naive-bayes]]]

ISLP states (§4.4.3, p. 152) "QDA estimates a separate covariance matrix for each class, for a total of $Kp(p+1)/2$ parameters" and contrasts with "LDA … there are $Kp$ linear coefficients to estimate" — the latter is loose (it's counting the discriminant-function coefficients, not the underlying covariance parameters). Benjamin's slide deck gives the clean apples-to-apples count of **covariance parameters only**, which is the headline trade-off:

| Method        | Covariance parameters                    | Notes                                                                 |
| ------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| **LDA**       | $\dfrac{p(p+1)}{2}$                      | One shared $\Sigma$; symmetric matrix has $p(p+1)/2$ free entries     |
| **QDA**       | $K \cdot \dfrac{p(p+1)}{2}$              | $K$ class-specific $\Sigma_k$'s, each symmetric                       |
| **Naive Bayes (Gaussian, class-specific $\sigma_{kj}^2$)** | $K \cdot p$ (variances) | Diagonal $\Sigma_k$'s; no off-diagonal parameters                     |
| **Naive Bayes (Gaussian, pooled $\sigma_j^2$)** | $p$                            | Single diagonal $\Sigma$; equivalent to LDA-with-diagonal-$\Sigma$    |

Add $K \cdot p$ mean parameters and $K - 1$ priors to each row to get the full count, but the **covariance row is the one the bias-variance argument lives on**.

Slide-deck numerical example: $p = 100$, $K = 5$:
- LDA covariance parameters: $100 \cdot 101 / 2 = 5{,}050$.
- QDA covariance parameters: $5 \cdot 5{,}050 = 25{,}250$.
- Gaussian naive Bayes (class-specific): $5 \cdot 100 = 500$.

For total parameter count, naive Bayes scales as $\mathcal{O}(pK)$, LDA as $\mathcal{O}(p^2)$, QDA as $\mathcal{O}(Kp^2)$. **That's the bias-variance ledger.**

**Symmetry constraint trap:** a $p \times p$ covariance has $p(p+1)/2$ free parameters, **not $p^2$**. The matrix is symmetric, so only the upper triangle (including the diagonal) is free: $p$ diagonal entries + $p(p-1)/2$ off-diagonal = $p(p+1)/2$.

---

## 4. ROC and AUC: the formal pieces ISLP leaves verbal

### 4.1 Threshold-indexed TPR and FPR formulas

[[[L09-classif-3|L09]], [[L10-resample-1|L10]], slide deck §"ROC curves and AUC"; concept: [[roc-auc]]]

ISLP §4.4.2 describes the ROC curve verbally and shows Figure 4.8, but never writes the threshold-indexed empirical formulas. For pseudocode questions ("write how you'd compute X"), the prof wants these:

For a probabilistic classifier producing scores $\hat p(x_i) \in [0, 1]$ on a test set $\{(x_i, y_i)\}_{i=1}^n$ with binary $y_i \in \{0, 1\}$, at threshold $t \in [0, 1]$:

$$
\operatorname{TPR}(t) \;=\; \operatorname{sensitivity}(t) \;=\; \frac{\#\{i : \hat p(x_i) > t \text{ and } y_i = 1\}}{\#\{i : y_i = 1\}},
$$

$$
\operatorname{FPR}(t) \;=\; 1 - \operatorname{specificity}(t) \;=\; \frac{\#\{i : \hat p(x_i) > t \text{ and } y_i = 0\}}{\#\{i : y_i = 0\}}.
$$

**Constructing the ROC curve.** Sweep $t$ over the distinct values of $\hat p(x_i)$ (in practice, also include $t = 0$ and $t = 1$ for the endpoints), compute $(\operatorname{FPR}(t), \operatorname{TPR}(t))$, plot. By construction:

- $t = 1$ → never predict positive → both rates $= 0$ → origin.
- $t = 0$ → always predict positive → both rates $= 1$ → corner $(1, 1)$.
- Curve is monotone non-decreasing in both axes as $t \downarrow 0$.

### 4.2 Probabilistic interpretation of AUC

[[[L09-classif-3|L09]], [[L10-resample-1|L10]]; concept: [[roc-auc]]]

ISLP says only that AUC is the area under the ROC curve and that an ideal classifier has AUC $= 1$, chance gives AUC $= 0.5$. The **probabilistic interpretation** is the load-bearing one — it's how the prof characterizes "what AUC = 0.7 actually means":

$$
\boxed{\; \operatorname{AUC} \;=\; \Pr\!\bigl[\, \hat p(X^+) \,>\, \hat p(X^-) \,\bigr], \;}
$$

where $X^+$ is a randomly drawn positive ($Y = 1$) example, $X^-$ a randomly drawn negative ($Y = 0$), independently. **AUC is the probability that a random positive scores higher than a random negative.**

Consequences (the prof's exam-style direction-of-effect facts):
- AUC $= 0.5$: chance; the score has no separating power.
- AUC $< 0.5$: classifier is genuinely informative but ordered backwards. **Invert predictions** to get $1 - \operatorname{AUC} > 0.5$.
- AUC $= 1$: perfect ordering (every positive scores higher than every negative).
- AUC is **invariant under any monotone transformation of $\hat p$**. So you can compare classifiers across different score scales; only the ranking matters.
- AUC is **independent of class prevalence**, unlike accuracy. That's why it's the stable metric in medicine.

### 4.3 Qualitative AUC scale (prof-specific)

[[[L09-classif-3|L09]], slide deck; concept: [[roc-auc]]]

ISLP doesn't quantify "what's a good AUC?" The slide deck and L09 give a working scale:

| AUC range | Reading                                                 |
|-----------|---------------------------------------------------------|
| $\approx 0.5$ | Useless (chance).                                       |
| $\approx 0.7$ | "OK" — informative but not great.                       |
| $\approx 0.8$ | "Good."                                                 |
| $\geq 0.9$    | "Very good."                                            |
| $1.0$         | Perfect (suspicious if seen in a real-world dataset).   |

Slide-deck reference points: LDA on the `Default` data gets AUC = 0.95 ("close to the maximum of 1.0, so would be considered very good"); logistic on the SAheart data gets AUC = 0.78.

### 4.4 LDA and logistic produce nearly identical ROC curves

[[[L09-classif-3|L09]], slide deck §"Linearity"; concepts: [[roc-auc]], [[linear-discriminant-analysis]], [[logistic-regression]]]

ISLP mentions in passing (§4.4.2 caption of Fig. 4.8) "the ROC curve for the logistic regression model … is virtually indistinguishable from this one for the LDA model." The slide deck makes the **reason** explicit and reproduces here as a formal observation:

For a two-class problem, **both LDA and logistic regression produce a posterior with the same functional form**:

$$
\log\!\frac{p_1(x)}{1 - p_1(x)} \;=\; c_0 + c_1 x_1 + \cdots + c_p x_p.
$$

The two methods estimate the coefficients $(c_0, \ldots, c_p)$ differently — LDA plugs in Gaussian MLEs and Bayes' rule; logistic regression maximizes the Bernoulli likelihood directly — but produce the **same family of decision boundaries** (linear in $x$) and **monotonically related score functions**. Since AUC depends only on the ranking, **the two AUCs are forced to be close** when the parameter estimates are close.

When do they actually differ? When the Gaussian assumption is badly off (logistic wins, because it doesn't lean on the assumption) or when the classes are very well separated (LDA's MLE is more stable; logistic's MLE can be unstable or fail to exist due to perfect separation).

---

## 5. Notation and naming differences

The prof and the slide deck deviate from ISLP notation in a few small ways. These are pure relabelings, not separate concepts.

| Quantity                       | ISLP notation                                    | Prof / slide deck notation                   |
| ------------------------------ | ------------------------------------------------ | -------------------------------------------- |
| Linear predictor               | "$\beta_0 + \beta_1 X_1 + \cdots$" written out   | $\eta_i$ (occasionally $H$)                  |
| Logistic link                  | "logistic function" (no symbol)                  | Logit link: $\eta = \log(p/(1-p))$           |
| Loss function (classification) | Implicit, called "error rate"                    | "0/1 loss" $\mathbf{1}(y \neq \hat y)$       |
| Estimator paradigms            | Not named; described in §4.4 prose              | **Diagnostic** (logistic, KNN) vs **sampling/generative** (LDA, QDA, naive Bayes) |
| ROC curve $x$-axis             | "false positive rate" (Fig. 4.8 label)           | "1 − specificity" (used interchangeably)     |
| Justice-system analogy for sens/spec trade-off | Not present                          | Lecture-only verbal device                   |
| Bayes-flip denominator         | Just "sum over classes" prose                    | "Partition function if you're from physics" (substitute lecturer, L07) |
| Naive Bayes alternate name     | "naive Bayes"                                    | "Idiot's Bayes" (slide deck variant)         |

ISLP also reserves "Bayes classifier" for the abstraction (the optimal decision rule under the true posterior) and uses "naive Bayes" for the specific generative classifier — the prof keeps this distinction strictly and warned about confusing the two; both terms mean what ISLP means.
