---
title: "Module 9: Recommended Exercises (Solution)"
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

*Solution*:

* What is an ensemble method?

A statistical learning model that combines the output from many statistical learning models, to perform better than any of the models individually.

* What are the main conceptual differences between bagging and boosting?

Bagging: Bootstrap aggregation. Based on building many models *independently of each other*, and combining the results.
Boosting: Based on building many models *sequentially*, where we let each model improve the fit for the parts of the problem that the model in the previous iteration struggled with.

## Problem 2

Consider the Gradient Tree Boosting Algorithm (given on slide 34 in this week's lectures).

Explain the meaning behind each of the following symbols. If relevant, also explain what the impact of changing them will be, and explain how one might find or choose them.

*Solution*:

* $L(.)$

This is the loss function we are trying to minimize when fitting the base learners in each iteration. Our choice of loss function will of course depend on the type of data we are modelling (i.e., regression vs. classification), but the choice can also be highly impactful on the robustness of our method - for example, the quadratic loss function might place too much weight on extreme observations.

* $f_m(.)$

This is our gradient boosted tree at the $m^{\text{th}}$ iteration of the algorithm. It is found by fitting a tree to the pseudo-residuals $r_{im}$, and within the terminal regions of that tree, adding a "correction" ($\gamma_{jm}$ for terminal region $j$) to the current $f_{m-1}$ in a way that minimizes the loss function $L$. For details on how to fit the individual trees, see the previous module.

* $M$

This is the number of trees (base learners) that we fit. This is a hyperparameter that should be tuned with e.g. cross-validation, or chosen adaptively with some stopping criterion. Too low $M$ leads to underfitting (we don't have enough base learners to capture the structure in the data), while too high $M$ leads to overfitting.

* $J_m$

This is the number of terminal regions of the tree (base learner) we fit in iteration $m$. $J_m$ will be determined by the procedure we use to fit each base learner, but we commonly constrain what the maximum value of $J_m$ can be. This constraint (the maximum value of $J_m$) is a hyperparameter that should be tuned with e.g. cross-validation. The lower we set the maximum allowed $J_m$ to be, the less complex each tree will be - i.e. the simpler the base learners in the ensemble will be. And vice versa. More complicated base learners will also mean a higher computational cost per iteration.

## Problem 3 - Learning rate

The algorithm mentioned in Problem 2 does not include the so-called *learning rate* parameter $\nu$. In the context of boosting methods, what is the interpretation behind this parameter, and how would one choose it? If you were to include the parameter in the Gradient Tree Boosting Algorithm, which equation(s) in the algorithm would you change and how?

*Solution*:

The learning rate $0<\nu<1$ is a factor that we use to scale the base learner at each iteration of the boosting algorithm. Thus, the lower its value, the less impact each base learner will have on the ensemble. Empirically, this parameter is often found to be crucial in determining the accuracy of the model. The smaller the value of this parameter, the higher we need $M$ to be, since we slow down the learning. $\nu$ can be chosen by cross-validation, or set to a fixed (small) value if $M$ is chosen adaptively (see above).

The learning rate would be added to step 2d) on lecture slide 35 (Algorithm 10.3 in "The Elements of Statistical Learning"), so that it would read

$$
\text{Update} \quad f_m(x)=f_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm}I(x\in R_{jm}).
$$

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

*Solution*:

We could for example:

* Reduce the allowed complexity of each base learner (reduce `max_depth`).
* Subsample observations (like in bagging) and/or predictors (like in random forest). In other words, go for a stochastic gradient boosting approach (`subsample` < 1 and/or `max_features` < 1 in `GradientBoostingRegressor`).
* Change our validation approach. The above code does a 10-fold cross validation for each $M$ up to $3$. To save on computations, we could go to 5-fold CV, or even a training/test set approach.

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

*Solution*:

We run a stochastic gradient boosting tree with subsampling of both rows (observations) and columns (predictors). For each tree we subsample $10$% of the predictors (`max_features=0.1`), so $10\,000 \cdot 0.1 = 1000$ predictors are considered per tree. (The original `h2o` code further subsampled $10$% of these *per split*, giving $100$ predictors per split; `GradientBoostingRegressor` does not expose per-split column subsampling, so we omit that level.) The fraction of observations used to fit each tree ($20$% or $50$%) is chosen by 5-fold cross-validation, as is the allowed complexity of each tree (`max_depth`).

We fix the learning rate to a low value of $\nu = 0.05$, and choose $M$ adaptively: we either stop training if the validation loss has not improved by $10^{-3}$ over the last $10$ iterations (via `n_iter_no_change=10` and `tol=1e-3`), or we reach the cap of $10\,000$ iterations. (`sklearn` does not have a built-in wall-clock time stopping rule equivalent to `max_runtime_secs`; if needed, this can be enforced externally.)


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

*Solution*:

Below we do a hyperparameter tuning, again adapted from <https://bradleyboehmke.github.io/HOML/gbm.html>. Note that XGBoost has *many* parameters that can be tuned on a much finer grid than what is done here.

```python
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb

dtrain = xgb.DMatrix(X_dat, label=y_dat)

# hyperparameter grid
hyper_grid = pd.DataFrame(
    list(itertools.product(
        [0.05],          # eta
        [3, 5, 7],       # max_depth
        [3],             # min_child_weight
        [0.2, 0.5],      # subsample
        [0.1],           # colsample_bytree
    )),
    columns=["eta", "max_depth", "min_child_weight",
             "subsample", "colsample_bytree"],
)
hyper_grid["rmse"] = 0.0   # a place to dump RMSE results
hyper_grid["trees"] = 0    # a place to dump required number of trees

# grid search
for i in range(len(hyper_grid)):
    params = {
        "objective": "reg:squarederror",
        "eta": hyper_grid.loc[i, "eta"],
        "max_depth": int(hyper_grid.loc[i, "max_depth"]),
        "min_child_weight": hyper_grid.loc[i, "min_child_weight"],
        "subsample": hyper_grid.loc[i, "subsample"],
        "colsample_bytree": hyper_grid.loc[i, "colsample_bytree"],
    }
    m = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=6000,
        nfold=5,
        early_stopping_rounds=50,
        seed=123,
        verbose_eval=False,
    )
    hyper_grid.loc[i, "rmse"] = m["test-rmse-mean"].min()
    hyper_grid.loc[i, "trees"] = int(m["test-rmse-mean"].idxmin() + 1)

(hyper_grid
 .query("rmse > 0")
 .sort_values("rmse"))

# minimum test CV RMSE across the grid
hyper_grid["rmse"].min()
```


### d) Model evaluation

Finally, experimenting with the code from a), b) and c), what is the best model you are able to find? Fit that model using all available training data (no cross-validation).

*Solution*:

The answer will depend on what models you decided to investigate, and how you decided to tune the hyperparameters. The final fit should involve a call to `GradientBoostingRegressor.fit()` or `xgb.train()` (using the chosen hyperparameters and a suitable number of trees, e.g. `best_iteration` from the cross-validated XGBoost run).

Notice that here we have skipped a model assessment of the final model, as we used all the available data to train the final model. If we were in a data-rich situation we might have held out an independent test set until now to check the accuracy of the final model. Or, if we had a lot of computing power, we might have done a nested cross-validation to perform the model assessment.
