---
title: "Module 5: Solutions to Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "February 12, 2025"
---

## Problem 1

a)  See Figure 1

![ ](cv.jpeg)


b) In run $1$, fold $1$ is kept aside and folds $2$ to $k$ are used to train the method. The error is calculated on fold $1$ ($\text{MSE}_1$). We do this for the remaining $k - 1$ runs. The cross-validation error is given by

$$
CV_k = \frac{1}{n} \sum_{j=1}^k n_j\cdot \text{MSE}_j \ ,
$$
where $n_j$ is the size of the $j^\text{th}$ fold.

For a regression problem $\text{MSE}_j = \frac{1}{n_j}\sum_{i \in V_j} (y_i - \hat y_i)^2$ for all $i$ in the validation set $V_j$.
For a binary classification problem the error might be the average of correctly classified observations.


c) Finding the optimal number of neighbors $K$ in KNN or the optimal polynomial degree in polynomial regression.

d) For a classification problem we can use CV to choose between QDA or LDA.


## Problem 2

- a) Advantages and disadvantages of $k$-fold Cross-Validation
  - i) The validation set approach:
    - D: $k$-fold cross validation is computationally more expensive.
    - A: The advantage is that $k$-fold cross validation has less variance and less bias (the validation set approach unifies the worst of both worlds, so to say).
  - ii) In LOOCV there is no randomness in the splits.
    - A: $k$-fold-CV is computationally less expensive, as we run the model $k$ times where $k \ll n$
    - A: $k$-fold-CV gives less variance in the estimates of the test errors, because in LOOCV we are averaging from $n$ fitted models that are trained on nearly the same data, therefore we have positively correlated estimates, which increases variance.
    - D: $k$-fold-CV has more bias, as in LOOCV we use a larger data set to fit the model, which gives a less biased version of the test error.
- b) We know that if $k=n=$ LOOCV the estimator of test error will have small bias but high variance and it is computationally expensive. If $k$ is too small (for example 2), the estimator will have larger bias but lower variance. Experimental research (simulations) has found that $k = 5$ or $k=10$ to often be good choices.

## Problem 3

No solution is provided on top of the guidelines in the exercise sheet.

## Problem 4


a) $P(\text{draw } X_i) = \frac{1}{n}$ and $P(\text{not draw } X_i) =1-P(\text{draw } X_i) = 1 - \frac{1}{n}$


b) $P(\text{not draw any } X_i) = (1 - \frac{1}{n})^n$ and $P(\text{draw at least one } X_i) =1 - \left(1 - \frac{1}{n}\right)^n$


c) $P(X_i \text{ in bootstrap sample}) = 1 - \left(1 - \frac{1}{n}\right)^n \approx 1-\exp(-1) \approx 0.632$

d)

```python
import numpy as np

rng = np.random.default_rng(0)
n = 100
B = 10000
j = 0
res = np.empty(B)
for b in range(B):
    res[b] = np.sum(rng.choice(n, size=n, replace=True) == j) > 0
res.mean()
```

The approximation becomes quickly very good as $n$ increases, as the following graph shows:

```python
import numpy as np
import matplotlib.pyplot as plt

ns = np.arange(1, 101)
probs = 1 - (1 - 1 / ns) ** ns

plt.plot(ns, probs)
plt.axhline(1 - 1 / np.e, linestyle="--", color="grey")
plt.ylim(0.6, 1)
plt.xlabel("n")
plt.ylabel(r"$1-(1-1/n)^n$")
plt.show()
```


## Problem 5

We repeat the following for $b = 1, \dots, B$:

* Draw with replacement a bootstrap sample.
* Fit the model.
* Store $\hat{\beta_b}$.

Calculate $\hat{SD}(\hat{\beta}) =\sqrt{\frac{1}{B-1} \sum_{b=1}^B (\hat{\beta_b} - \frac{1}{B}\sum_{b=1}^B \hat{\beta_b})^2}$.

For the $95\%$ CI, we can calculate the $0.025$ and $0.975$ quantiles of the sample $\hat{\beta_b}$, $b=1,\dots,B$.


## Problem 6

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS, summarize
from sklearn.base import clone

SLID = pd.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/carData/SLID.csv",
    index_col=0,
).dropna()

design = MS(["education", "age", "sex", "language"])

def boot_OLS(model_matrix, response, D, idx):
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params

rng = np.random.default_rng(0)
B = 1000
beta_hat = np.empty(B)
for i in range(B):
    idx = rng.choice(SLID.index, size=SLID.shape[0], replace=True)
    params = boot_OLS(design, "wages", SLID, idx)
    beta_hat[i] = params["age"]
```

We can for example look at the histogram of the samples of $\hat\beta$ to get an idea of the distribution:

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

xs = np.linspace(beta_hat.min(), beta_hat.max(), 200)
norm_den = norm.pdf(xs, loc=beta_hat.mean(), scale=beta_hat.std())

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(beta_hat, bins=30, density=True,
        color="lightgrey", edgecolor="black")
ax.plot(xs, norm_den, color="red")
ax.set_xlabel(r"$\hat{\beta}_{\mathrm{age}}$")
plt.show()
```

The $95\%$ CI for $\hat{\beta}_{\text{age}}$ can now be derived by either using the $2.5\%$ and $97.5\%$ quantiles of the samples, or by using the $\hat\beta \pm 1.96\cdot \text{SD}(\hat\beta)$ idea:

```python
print(beta_hat.std())
print(np.quantile(beta_hat, [0.025, 0.975]))
print((beta_hat.mean() - 1.96 * beta_hat.std(),
       beta_hat.mean() + 1.96 * beta_hat.std()))
```

We can compare these results to what we would obtain directly from fitting the model and calculating the confidence interval

```python
X = design.fit_transform(SLID)
y = SLID["wages"]
SLID_lm = sm.OLS(y, X).fit()
SLID_lm.conf_int()
```

The 95% CI for `age` is essentially the same as the one we obtained from bootstrapping.

However, it can happen that the CI would differ between the bootstrap and the linear regression output, for example when some of the modeling assumptions are violated. See again section 5.3.4 in the course book for a similar example where the standard errors and CIs differ.



If you prefer to use the `boot_SE()` helper from the ISLP lab (Section 5.3.3):

```python
from functools import partial

def boot_SE(func, D, n=None, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B):
        idx = rng.choice(D.index, n, replace=True)
        value = func(D, idx)
        first_ += value
        second_ += value ** 2
    return np.sqrt(second_ / B - (first_ / B) ** 2)

age_func = partial(boot_OLS, design, "wages")
boot_SE(age_func, SLID, B=1000, seed=0)
```

---

# Summing up

## Take home messages

* Use $k=5$ or $10$ fold cross-validation for model selection or assessment.
* Use bootstrapping to estimate the standard deviation of an estimator, and understand how it is performed before module 8 on trees.


# Further reading

* [Videos on YouTube by the authors of ISL, Chapter 5](https://www.youtube.com/playlist?list=PL5-da3qGB5IA6E6ZNXu7dp89_uv8yocmf), and corresponding [slides](https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/cv_boot.pdf)
* [Solutions to exercises in the book, chapter 5](https://rstudio-pubs-static.s3.amazonaws.com/65561_43c0eaaa8565414eae333b47038f716c.html)
