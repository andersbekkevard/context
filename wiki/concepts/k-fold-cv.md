---
concept: k-fold-cv
module: 05-resample
lectures: [L10, L11, L18, L19]
isl-ref: 5.1.3
exercises:
  - Exercise5.1 - describe the k-fold CV algorithm with figure, aggregation, polynomial / KNN examples
  - Exercise5.2 - compare k-fold to validation-set and LOOCV (bias / variance / compute)
  - Exercise6.3b - wrap CV around best-subset selection on Credit
  - Exercise8.2c - cv.tree() to find optimal pruning size for regression tree
  - Exercise8.3e - cv.tree() with prune.misclass for classification spam tree
  - CE1 problem 4a - write 10-fold CV pseudocode for KNN regression with MSE
  - CE1 problem 4b - true/false on k-fold vs LOOCV
related: [leave-one-out-cv, validation-set-approach, one-standard-error-rule, nested-cv-and-cv-pitfalls, cross-validation, bias-variance-tradeoff]
tags:
  - concept
  - module/05-resample
aliases:
  - k-fold cross-validation
  - 5-fold CV
  - 10-fold CV
---

# k-fold cross-validation

The prof's preferred everything-tuner. Split the data into $k$ blocks, hold each out once, average the $k$ MSE estimates. $k = 5$ or $10$ is the standard bias-variance compromise, *"more accurate estimates of the test error rate than does LOOCV"* (slide deck) because the folds are less correlated, while still using $\sim 80$–$90\%$ of the data per fit.

## Definition (prof's framing)

Partition the data into $k$ disjoint folds $C_1, \dots, C_k$ of (roughly) equal size $n_j = n/k$. For each $j = 1, \dots, k$:

1. Train the model on the $k - 1$ other folds.
2. Predict on fold $C_j$, compute fold MSE $\text{MSE}_j$ (or misclassification rate for classification).

Average:

$$\text{CV}_{(k)} = \frac{1}{n} \sum_{j=1}^{k} n_j \, \text{MSE}_j$$

(Equivalently $\frac{1}{k} \sum_j \text{MSE}_j$ when folds are equal-sized.)

For classification, replace MSE with the misclassification indicator: $\text{Err}_j = \frac{1}{n_j} \sum_{i \in C_j} \mathbb{1}(y_i \neq \hat y_i)$.

> "Setting $k = n$ gives LOOCV.", slide deck

## Notation & setup

- $k$ = number of folds; standard choices: $k = 5$ or $k = 10$.
- $C_j$ = indices of observations in fold $j$, $|C_j| = n_j$.
- $\hat y_i$ for $i \in C_j$ = prediction from the model trained without fold $j$.
- Folds are usually random; with structured data (spatial / temporal / clustered) you should construct folds that respect the dependency.

## Formula(s) to know cold

**Regression:**

$$\boxed{\text{CV}_{(k)} = \frac{1}{n} \sum_{j=1}^{k} n_j \, \text{MSE}_j \qquad \text{MSE}_j = \frac{1}{n_j} \sum_{i \in C_j} (y_i - \hat y_i)^2}$$

**Classification:**

$$\text{CV}_{(k)} = \frac{1}{k} \sum_{j=1}^{k} \text{Err}_j \qquad \text{Err}_j = \frac{1}{n_j} \sum_{i \in C_j} \mathbb{1}(y_i \neq \hat y_i)$$

**Standard error of the CV estimate** (used by [[one-standard-error-rule]]):

$$\widehat{\text{SE}}\big(\text{CV}_{(k)}(\theta)\big) = \sqrt{\frac{1}{k - 1} \sum_{j=1}^{k} \big(\text{MSE}_j(\theta) - \overline{\text{MSE}}(\theta)\big)^2}$$

(per the slide deck, sample SD of the per-fold MSEs).

## Insights & mental models

- **Why $k = 5$ or $10$, the bias-variance compromise** (slide enumeration):
  1. Result depends on how folds are made, but **less variation** than the validation set approach.
  2. Computationally cheap, $5$ or $10$ fits, not $n$.
  3. Training set is $(k-1)/k$ of the data → biased *upward* (overestimates true test error).
  4. Bias smallest at $k = n$ (LOOCV), but LOOCV has the high-variance problem.
  5. By [[bias-variance-tradeoff]], $k = 5$ or $10$ is the standard compromise.

- **The "less correlated folds" argument**: the heart of why k-fold beats LOOCV in variance: with 5-fold, two training sets share only 60% of the data, not 99.9%. So per-fold estimates are less correlated → averaging them yields lower variance. The prof's clean line: *"You're winning by having less variance, because you want to know, you want to pick a model that will do well if you have another data set, and it's that variance across data sets that you really want to reduce."* - [[L11-resample-2]]

- **The "less variability across reruns" demo** (Auto data slide): 10 reruns of 10-fold CV give curves that are very tight to each other, where 10 reruns of the validation set give curves that disagree wildly. Visual proof that k-fold stabilizes the estimate.

- **The classification curve** (logistic regression + polynomial degrees 1–10 on the ISLR 2-D classification data, slide Fig. 5.7): 10-fold CV (black) closely tracks the true test error (orange); training error (blue) trends upward weirdly because *"the error rate that they use on the y is not actually computed the same way as the log likelihood"*: logistic regression maximizes likelihood, not misclassification rate. So *"more flexibility" no longer guarantees "lower reported error"*, see [[L11-resample-2]] for the detailed explanation.

- **Pseudocode** (CE1 problem 4a expected form, KNN regression flavor):
  ```
  Inputs: data (X, y) of size n, candidate K-values K_set, number of folds k
  Randomly assign each observation to one of k folds
  For each K in K_set:
      For j = 1, ..., k:
          Train: KNN with parameter K on all folds except j
          Predict: on fold j → get ŷ_i for i in C_j
          MSE_j = (1/n_j) Σ_{i in C_j} (y_i - ŷ_i)^2
      CV(K) = (1/k) Σ_j MSE_j
  Choose K* = argmin_K CV(K)
  Refit KNN with K = K* on the full training data; report final MSE on held-out test set
  ```

- **Independence trap** still applies (slide deck + L10). Random folds break under spatial/temporal correlation. Mitigation in k-fold is *easier* than in LOOCV, *"you can deliberately construct folds that respect the dependency structure (e.g., put a whole spatial block in one fold)."* - [[L10-resample-1]]

## Exam signals

> "Typically in this setting, you're winning by having less variance, because you want to know, you want to pick a model that will do well if you have another data set." - [[L11-resample-2]]

> "By doing the repeated folds, we get a much lower variability across reruns. So there is variability, but much less than the other approach, which is nice." - [[L10-resample-1]]

> "Be careful with independence, independence of points… If you just randomly sort them into the test and fit and validation without any concern of these things, it's going to suck." - [[L10-resample-1]]

CE1 problem 4b explicitly compares k-fold and LOOCV directions:

- *"5-fold CV will generally lead to an estimate of the prediction error with less bias, but more variance, than LOOCV"* → **false** (directions reversed; LOOCV has lower bias, higher variance).
- *"10-fold CV is computationally cheaper than LOOCV"* → **true** (10 fits vs $n$).

## Pitfalls

- **Reversing the bias/variance direction.** Easy to flip on T/F.
- **Random folds for time-series / spatial data**: the canonical "ruin everything" mistake.
- **Reusing the CV-estimated error as a test-set estimate after also picking the model on it**: that's the [[nested-cv-and-cv-pitfalls|nested CV story]]. CV-for-selection ≠ CV-for-assessment.
- **Stratification not enforced** for classification with rare classes, folds can end up with no positives. Standard fix: stratified k-fold (assign observations to folds within each class).

## Scope vs ISLR

- **In scope:** the algorithm (regression and classification), the bias/variance compromise, the standard $k = 5$ or $10$ recommendation, the independence trap, the SE formula (used by 1-SE rule), the comparison with validation set and LOOCV.
- **Look up in ISLR:** §5.1.3 (pp. 203–205) for the algorithm and Figures 5.5/5.6; §5.1.4 (pp. 205–206) for the bias-variance trade-off discussion. Lab in §5.3.2 demonstrates `cv.glm()` (R); ignore the package syntax per the prof's exam-policy.

## Exercise instances

- **Exercise5.1**: describe the algorithm with a figure; show how the per-fold MSE/Err are aggregated; relate to polynomial regression and KNN classification examples.
- **Exercise5.2**: explicit comparison with validation set and LOOCV; bias / variance / computational complexity. Recommended values for $k$ (5 or 10) and why.
- **Exercise6.3b**: wrap k-fold CV around best-subset selection on the Credit data (cross-check against AIC/BIC/Cp/adj-$R^2$ choices).
- **Exercise8.2c**: `cv.tree()` to find optimal pruning size for a regression tree on Carseats; compare pruned vs unpruned MSE.
- **Exercise8.3e**: `cv.tree()` with `prune.misclass` for the classification tree on spam.
- **CE1 problem 4a**: write 10-fold CV pseudocode for KNN regression with MSE as the validation error.
- **CE1 problem 4b**: true/false on k-fold vs LOOCV (bias, variance, compute, "validation set = 2-fold CV", "LOOCV = bootstrap").

## How it might appear on the exam

- **Pseudocode / equation-writing** ("write the 10-fold CV procedure for choosing $K$ in KNN regression"), direct CE1 4a port. Math, English, or pseudocode all OK per the prof.
- **True/false on bias/variance/compute properties**: CE1 4b style. Always show reasoning even if T/F is the only required answer.
- **Compare-and-contrast** with validation-set and LOOCV, the standard 3-row table.
- **Output interpretation:** given a CV-vs-hyperparameter plot (CV error on $y$, $K$ or polynomial degree on $x$), pick the optimal hyperparameter; possibly with the [[one-standard-error-rule]].
- **Independence-trap conceptual:** given temporally correlated data, why is random k-fold a problem and how would you fix it?
- **Method-comparison:** why is k-fold typically preferred over LOOCV in practice, and over the validation set approach?

## Related

- [[leave-one-out-cv]]: the $k = n$ extreme; lower bias, higher variance
- [[validation-set-approach]]: the simplest, most variable cousin
- [[one-standard-error-rule]]: the simplest-within-1-SE selection criterion built on k-fold CV
- [[nested-cv-and-cv-pitfalls]]: when you need both selection AND assessment, and the wrong-way trap
- [[cross-validation]]: global cross-cutting picture
- [[bias-variance-tradeoff]]: the lens for choosing $k$
