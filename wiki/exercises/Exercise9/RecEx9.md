---
title: "Module 9: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author:
  - Kenneth Aase, Sara Martino, Stefanie Muff
  - Department of Mathematical Sciences, NTNU
date: "March 19, 2025"
---


---

**We recommend you to work through Section 8.3.4 in the course book (Lab: Boosting)**

---


## Problem 1

Both **bagging** and **boosting** are *ensemble* methods.

* What is an ensemble method?
* What are the main conceptual differences between bagging and boosting?


## Problem 2

Consider the Gradient Tree Boosting Algorithm (given on slide 34 in this week's lectures).

Explain the meaning behind each of the following symbols. If relevant, also explain what the impact of changing them will be, and explain how one might find or choose them.


* $L(.)$
* $f_m(.)$
* $M$
* $J_m$


## Problem 3 - Learning rate

The algorithm mentioned in Problem 2 does not include the so-called *learning rate* parameter $\nu$. In the context of boosting methods, what is the interpretation behind this parameter, and how would one choose it? If you were to include the parameter in the Gradient Tree Boosting Algorithm, which equation(s) in the algorithm would you change and how?

## Problem 4 - Boosting methods

In the code chunk below we simulate data meant to mimic a genomic data set for a set of markers (locations in the human genome) that underlie a complex trait (such as height, or whether you have cancer or not). You don't need to understand the details of the simulation, but the main takeaways are

* $y$ is a continuous response, representing some biological trait (for example height) depending on both genetic (the predictors) and environmental (here included as random noise) factors.
* We have many more predictors (genetic markers) $M$ than observations (individuals) $N$.
* The predictors have a complicated structure: most of them have a very small effect on $y$, but a few of them have a larger effect.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
N = 1000   # number of individuals
M = 10000  # number of genotyped markers

# Allele frequencies
p = rng.beta(2.3, 7.4, size=M)
p = np.where(p < 0.5, p, 1 - p)

# Genotype matrix: each column j is Binomial(2, p[j]) sampled N times
X = rng.binomial(n=2, p=p, size=(N, M))

# Dirichlet weights for the four effect-size components
effect_weights = rng.dirichlet(alpha=[0.9, 0.08, 0.015, 0.005], size=M)

# Effect sizes: first component is 0, the others are N(0, sd=sqrt(...))
sds = np.sqrt([1e-4, 1e-3, 1e-2])
effects = np.column_stack([
    np.zeros(M),
    rng.normal(0, sds[0], size=M),
    rng.normal(0, sds[1], size=M),
    rng.normal(0, sds[2], size=M),
])
marker_effects = (effect_weights * effects).sum(axis=1)

genetics = X @ marker_effects
environment = rng.normal(0, np.sqrt(np.var(genetics) / 2), size=N)
y = genetics + environment

dat = pd.DataFrame(X, columns=[f"V{j+1}" for j in range(M)])
dat.insert(0, "y", y)

# Looking at a few entries in the data:
dat.iloc[:6, :20]
```

We will now try to predict $y$ using various tree boosting methods, adapted from code provided in <https://bradleyboehmke.github.io/HOML/gbm.html>.

### a) A basic GBM

We first implement a standard gradient boosting tree using `GradientBoostingRegressor` from `sklearn.ensemble`. To make sure everything works, we run a very small ($M=3$) model below.

```python
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.model_selection import cross_val_score

X_dat = dat.drop(columns="y").values
y_dat = dat["y"].values

gbm1 = GBR(
    loss="squared_error",   # SSE loss function
    n_estimators=3,
    learning_rate=0.1,
    max_depth=7,
    min_samples_leaf=10,
    random_state=0,
)

# 10-fold CV (analogous to cv.folds = 10 in gbm)
cv_scores = cross_val_score(
    gbm1, X_dat, y_dat, cv=10, scoring="neg_mean_squared_error"
)
gbm1.fit(X_dat, y_dat)
```

We note that running the above code block is very slow - even for a *very* low number of trees $M=3$ ($M$ is typically in the thousands). How could we modify the algorithm to be less computationally intensive?

### b) Stochastic GBMs

We now move on to stochastic gradient boosting tree models for the same data set. Stochastic GBM subsamples both rows (observations) and columns (predictors). In `sklearn`, `GradientBoostingRegressor` supports row subsampling via `subsample` and column subsampling per tree via `max_features`. Per-split column subsampling (an `h2o.gbm`-style option) is not exposed by `GradientBoostingRegressor`, so we approximate it via `max_features` only.

Explain what is done in the below code (the [HOML chapter on GBMs](https://bradleyboehmke.github.io/HOML/gbm.html) and the `sklearn` docs may be helpful).


```python
from sklearn.model_selection import RandomizedSearchCV, KFold

# Hyperparameter grid
param_dist = {
    "subsample": [0.2, 0.5],          # row subsampling
    "max_features": [0.1],            # col subsampling per tree (fraction)
    "min_samples_leaf": [10],
    "learning_rate": [0.05],
    "max_depth": [3, 5, 7],
}

# Base estimator: large n_estimators + early stopping via validation_fraction/n_iter_no_change
base_gbr = GBR(
    n_estimators=10000,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-3,
    random_state=123,
)

search = RandomizedSearchCV(
    estimator=base_gbr,
    param_distributions=param_dist,
    n_iter=6,
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=123),
    random_state=123,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_dat, y_dat)

# Sort results by mean test MSE (low to high)
results = pd.DataFrame(search.cv_results_)
results["mean_test_mse"] = -results["mean_test_score"]
results.sort_values("mean_test_mse")[
    ["params", "mean_test_mse", "std_test_score"]
]
```

### c) XGBoost

Below we also provide code for applying cross-validated XGBoost to the same data set. Expand this code to perform a search of the hyperparameter space, similar to b).

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_dat, label=y_dat)

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.2,
    "colsample_bytree": 0.1,
}

cv_res = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=6000,
    nfold=5,
    early_stopping_rounds=50,
    seed=123,
    verbose_eval=True,
)

# minimum test CV RMSE
cv_res["test-rmse-mean"].min()
```

### d) Model evaluation

Finally, experimenting with the code from a), b) and c), what is the best model you are able to find? Fit that model using all available training data (no cross-validation).
