---
concept: standardization
modules: [06-modelsel, 08-trees, 10-unsuper, 11-nnet]
lectures: [L12, L13, L14, L15, L21, L22, L24]
isl-ref: 6.2.1
exercises:
  - Exercise11.3 - preprocess Boston housing with mean/sd before NN
related: [ridge-regression, lasso, principal-component-analysis, k-means-clustering, hierarchical-clustering, knn-classification, knn-regression, feedforward-network, regularization, distance-metrics]
tags:
  - concept
  - specials
aliases:
  - z-score
  - z-scoring
  - scaling
  - normalization
---

# Standardization (z-score)

The one-line preprocessing rule the prof restates in every module that uses it: **subtract the column mean, divide by the column standard deviation, before fitting**. Required by every method in the course that is **not scale-invariant**, ridge, lasso, elastic net, PCA/PCR/PLS, k-means, hierarchical clustering, KNN, neural networks. The diagnostic when results look weird: did you standardize? If no, your largest-unit predictor is dominating the geometry for no good reason.

## Definition (prof's framing)

> "Importantly, ridge regression is **not scale-invariant**, meaning that it matters what the amplitude of the beta is." - [[L12-modelsel-1]]

> "PCA is not scale invariant. So if you don't standardize them so that their mean is zero and their variance is one, then if one had a standard deviation of like a million, then that will be your strongest variable." - [[L14-modelsel-3]]

> "This is true of so many algorithms that it's almost like a guarantee that you're going to have to do this. So both PCA and clustering, they're sensitive to the metric you're using." - [[L22-unsupervised-2]]

For each predictor $X_j$:

$$\boxed{\;z_{ij} = \frac{x_{ij} - \bar x_j}{s_j}, \qquad \bar x_j = \frac{1}{n}\sum_i x_{ij}, \qquad s_j = \sqrt{\frac{1}{n-1}\sum_i (x_{ij} - \bar x_j)^2}\;}$$

After standardization, every column has mean 0 and (sample) variance 1. Distances, penalties, and inner products treat all coordinates on equal footing, which is what every method below implicitly assumes.

## Returns in other modules

- [[L12-modelsel-1]]: first explicit appearance, in the context of [[ridge-regression]]. Standardize before adding $\lambda \sum \beta_j^2$, otherwise small-scale predictors get crushed.
- [[L13-modelsel-2]]: re-emphasized for ridge, then carried into [[lasso]]. Same reason: penalty treats all $\beta_j$ symmetrically, so all $X_j$ must be on a common scale.
- [[L14-modelsel-3]]: *"Standardize $X$ first so the betas are on comparable scales (and the penalty hits them comparably)."* Then re-stated for **PCA** at the start of the dimensionality-reduction segment: PCA chases total variance, so a $X$ in km vs cm changes the answer entirely.
- [[L15-modelsel-4]]: PCR pipeline (*"Standardize → PCA → regress on first M PCs"*). Emphasized again as a slide bullet: *"PCA is not scale invariant, standardize first."*
- [[L21-unsupervised-1]]: full PCA treatment, with a slide-deck demo of `prcomp(..., scale=TRUE, center=TRUE)` vs without. *"It really depends [on] the scaling of the variables, right? Because it's trying to capture the total variance. … then the variable would appear to have a higher variance along that dimension just because that has bigger numbers."*
- [[L22-unsupervised-2]]: extended to clustering: **k-means** and **hierarchical clustering** both rely on Euclidean distance, so the same scale issue. *"If you have three measurements, one in centimeters, one in nanograms, the nanogram column is numerically huge for any real-world object, so it dominates pairwise Euclidean distance."* Z-score is the standard fix; alternatives (divide by max, divide by IQR) exist for specific cases.
- [[L24-nnet-2]]: for [[feedforward-network|NN]] inputs. *"You don't want one variable to basically suck up all the variance, just like in the PCA."* Boston-housing demo normalizes before training. Without standardization, gradients are imbalanced across input dimensions and training is much harder.

## Notation & setup

- $X$ = $n \times p$ data matrix; columns are predictors.
- $\bar x_j$ = column-$j$ mean; $s_j$ = column-$j$ sample SD.
- After standardization, the column means are 0 and column variances are 1. (Some packages use population SD with denominator $n$; functionally equivalent for the purposes of every method below.)

## Formula(s) to know cold

$$z_{ij} = \frac{x_{ij} - \bar x_j}{s_j}$$

That's the whole formula. The thing to remember is **which methods need it**.

### Methods that require standardization (course coverage)

| Method | Why standardize | Lecture flag |
|---|---|---|
| [[ridge-regression]] | Penalty $\sum \beta_j^2$ treats $\beta$'s symmetrically; needs $X_j$ on a common scale | [[L12-modelsel-1]] |
| [[lasso]] | Same, penalty $\sum \|\beta_j\|$ is also scale-asymmetric | [[L13-modelsel-2]] |
| [[elastic-net]] | Both penalties scale-asymmetric | [[L13-modelsel-2]] |
| [[principal-component-analysis|PCA]] / [[principal-component-regression|PCR]] / [[partial-least-squares|PLS]] | Maximizes variance, biggest-unit variable wins by default | [[L14-modelsel-3]], [[L15-modelsel-4]], [[L21-unsupervised-1]] |
| [[k-means-clustering]] | Euclidean distance dominates by largest-scale dimension | [[L22-unsupervised-2]] |
| [[hierarchical-clustering]] | Same, Euclidean distance issue | [[L22-unsupervised-2]] |
| [[knn-classification]] / [[knn-regression]] | Same | exam_analysis §4b |
| [[feedforward-network|NN inputs]] | Gradient magnitudes imbalanced across inputs of wildly different scales | [[L24-nnet-2]] |

### Methods that do NOT require standardization

- **Plain OLS**: scale-invariant in the fit (the $\beta$'s rescale, but predictions don't change). Standardization is a *cosmetic* choice for OLS, useful for interpretation when comparing $\beta$ magnitudes, not required for the math.
- **Trees, random forests, boosting**: splits compare values within one variable at a time; no cross-variable distance computation. Insensitive to scale. (Why XGBoost / LightGBM / random forest don't need it.)
- **Logistic regression**: same as OLS, scale-invariant in fit (coefficients rescale to compensate). Standardization for interpretability only.
- **Naive Bayes**: uses density estimates per class, not pairwise distances.
- **GAMs**: each $f_j$ is fit separately; rescaling $X_j$ just rescales $f_j$.

## Insights & mental models

### The "diagnostic when results look weird"

When ridge / lasso / PCA / k-means gives a bizarre answer, one variable dominates everything, or one PC explains 99% of variance, or one cluster contains everyone, the **first thing to check is whether you standardized**.

> "If one had a standard deviation of like a million, then that will be your strongest variable. That will be the thing that gives you the highest variance, which is annoying because you don't want it to just be that the scale of the variable is bigger." - [[L14-modelsel-3]]

### The "comparable betas" framing

Standardize so that the $\beta$'s have **comparable magnitudes**, which is what makes the penalty fair. After standardization, $|\hat\beta_j|$ measures the predictor's contribution per *standard-deviation* change, not per *unit* change. So you can also use this to rank predictors after fitting.

> "So the betas will also be comparable, be at similar scales." - [[L12-modelsel-1]]

### Standardizing the response (not usually)

You typically **don't** standardize $Y$, the response stays on its original scale so predictions are interpretable. (Exception: NN regression sometimes scales $Y$ for stable training; reverse the scaling for predictions.)

### What about the intercept

For ridge / lasso / elastic net: the **intercept $\beta_0$ is NOT penalized** ([[L12-modelsel-1]]). After standardizing $X$ to mean 0 and centering $Y$, the intercept is just $\bar y$, it absorbs the shift. The penalty applies only to $\beta_1, \dots, \beta_p$.

### Train/test convention (subtle but important)

Compute $\bar x_j$ and $s_j$ on the **training set only**, then apply the *same* mean and SD to the test set. Don't recompute on the test set, that would leak test information. Same logic for k-fold CV: compute statistics inside each training fold.

### Alternatives the prof mentions

> "Other normalizations (e.g. divide by max) are possible, pick what fits the data." - [[L22-unsupervised-2]]

For bounded data (e.g. images on $[0, 255]$), dividing by max gives $[0, 1]$ ranges. For percentages, no scaling needed. The general principle: get all variables onto a common scale that makes the algorithm's geometric assumptions reasonable.

## Exam signals

> "Importantly, ridge regression is **not scale-invariant**." - [[L12-modelsel-1]]

> "PCA is not scale invariant." - [[L14-modelsel-3]] (verbatim slide bullet)

> "This is true of so many algorithms that it's almost like a guarantee that you're going to have to do this." - [[L22-unsupervised-2]]

> "You don't want one variable to basically suck up all the variance, just like in the PCA." - [[L24-nnet-2]]

The exam_analysis "direction-of-effect" cheat sheet (§4b) has *"PCA without standardization → result dominated by largest-scale variable"* and *"KNN distance with mixed units → Euclidean becomes meaningless without scaling."* Both are canonical T/F flips.

## Pitfalls

- **Forgetting to standardize before ridge/lasso**: penalty disproportionately punishes small-scale predictors. Most common ML beginner mistake.
- **Forgetting to standardize before PCA**: first PC ends up being whichever variable has the biggest unit. Re-run after standardizing and you get a totally different answer.
- **Forgetting to standardize before k-means / hierarchical clustering**: distance dominated by big-scale variable; clusters reflect units, not structure.
- **Computing mean/SD on the full dataset (including test)**: minor data leakage. Compute on training, apply to test.
- **Standardizing trees / RF / GBM**: unnecessary; doesn't hurt, but cosmetic. The prof never bothers.
- **Standardizing the response $Y$ in regression**: usually wrong; predictions become uninterpretable. Only standardize $Y$ for NN training stability and reverse the transform for output.
- **"Standardize" vs "normalize"**: the terms are inconsistent across packages. The prof uses **standardize** for z-score; "normalize" sometimes means divide-by-max (to $[0,1]$) and sometimes means z-score. Read the docs.
- **OLS doesn't need it**: but if you standardize and read off $\hat\beta$'s, those are *standardized* coefficients (per-SD effect). Don't compare to unstandardized $\hat\beta$'s from the same dataset.
- **Categorical dummy encodings**: don't z-score binary/dummy variables, leave them as 0/1. Z-scoring would muddle their interpretation.

## Scope vs ISLR

- **In scope:** the z-score formula, **which methods require it** (ridge / lasso / PCA / k-means / hierarchical / KNN / NNs) and **which don't** (OLS / trees / GAMs), the diagnostic role ("did you standardize?"), the train-only computation rule.
- **Look up in ISLR:** §6.2.1 (ridge-regression standardization, equation 6.6 specifically); §10.2 (PCA standardization); §12.4.1 (k-means / scaling).
- **Skip in ISLR (book-only, prof excluded):** elaborate scaling schemes (robust scaling via IQR/MAD, quantile normalization, etc.), name-checked at most. Z-score is the in-scope answer to every "should I scale this?" question.

## Exercise instances

- **Exercise11.3**: preprocess Boston housing with mean/sd before fitting an NN. The standard worked example.

(All other exercises that use ridge/lasso/PCA/k-means/hierarchical implicitly require standardization, but the recommended-exercise sheets often hand you pre-standardized data or call `scale=TRUE` automatically inside the relevant function.)

## How it might appear on the exam

- **T/F: "PCA is scale-invariant"**: FALSE. Direction: PCA chases total variance, so largest-unit variable dominates if you don't standardize.
- **T/F: "Ridge regression is scale-invariant"**: FALSE.
- **T/F: "Random forests need standardized predictors"**: FALSE. (Trees split per-variable; no distance computation.)
- **T/F: "KNN works equally well on raw vs standardized data"**: FALSE.
- **One-line interpretation prompts**: "Why do we standardize before ridge?" → "Because the penalty $\sum \beta_j^2$ treats all $\beta_j$ symmetrically, so the $X_j$'s must be on a common scale or small-scale predictors get over-penalized."
- **Diagnostic question**: "Why does my k-means clustering put everyone in one giant cluster?" → "You probably didn't standardize and one variable's scale is dominating the Euclidean distance." Or: "Why does my first PC explain 99% of the variance?" → "Same answer."
- **Pseudocode** for a pipeline that includes standardization (e.g. ridge with CV): "compute $\bar x_j, s_j$ on the training fold; apply $(x - \bar x_j)/s_j$ to all data; fit on training, evaluate on validation; repeat per fold."
- **Method-comparison question**: "Why does the prof prefer XGBoost over a NN on small tabular data?", partial answer: NN needs standardization + careful regularization; trees don't.

## Related

- [[ridge-regression]]: first canonical example; standardization is a pre-step
- [[lasso]]: same
- [[elastic-net]]: same
- [[principal-component-analysis]]: PCA without standardization is dominated by largest-unit variable
- [[principal-component-regression]]: PCR pipeline starts with standardize-then-PCA
- [[partial-least-squares]]: same
- [[k-means-clustering]]: Euclidean distance needs comparable scales
- [[hierarchical-clustering]]: same
- [[knn-classification]] / [[knn-regression]]: Euclidean nearest neighbors
- [[feedforward-network]]: input standardization for stable gradients
- [[regularization]]: every regularizer in the course assumes you've standardized first
- [[distance-metrics]]: clustering / KNN choice of distance is the corollary; standardization handles the units, distance choice handles the geometry
- [[curse-of-dimensionality]]: in high dim, distances become uniform; standardization doesn't fix this but unstandardized data makes it worse
