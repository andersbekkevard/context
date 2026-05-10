---
title: "Module 7: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "February 28, 2025"
---

The original version of this exercise sheet was developed in 2019 by Andreas Strand and colleagues. Thanks for the permission to build on it.

## Problem 1

Let us take a look at the `Auto` data set. We want to model miles per gallon `mpg` by engine horsepower `horsepower`. Separate the observations into training and test. A training set is plotted below.

Perform polynomial regression of degree $1$, $2$, $3$ and $4$. Add the fitted curves to the plot below.

Also plot the test error depending on polynomial degree.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, poly

Auto = load_data('Auto')
# extract only the two variables from Auto
ds = Auto[['horsepower', 'mpg']]
n = ds.shape[0]
# which degrees we will look at
deg = [1, 2, 3, 4]

rng = np.random.default_rng(1)
# training ids for training set
tr = rng.choice(n, size=round(n / 2), replace=False)
# plot of training data
fig, ax = subplots()
ax.scatter(ds.iloc[tr]['horsepower'], ds.iloc[tr]['mpg'],
           color='darkgrey')
ax.set_title('Polynomial regression')
ax.set_xlabel('horsepower')
ax.set_ylabel('mpg');
```

## Problem 2

We will continue working with the `Auto` data set. The variable `origin` is $1$, $2$ or $3$, corresponding to American, European or Japanese origin, respectively. Convert `origin` to a categorical variable (e.g. `Auto['origin'].astype('category')`). Predict `mpg` by origin with a linear model. Plot the fitted values and approximative $95\%$ confidence intervals. The `get_prediction()` method on a fitted `statsmodels` result, together with `conf_int(alpha=0.05)`, gives confidence bands for the predictions.

  * Hint: build a small `pandas.DataFrame` of the three origins (as a categorical) and pass it to `MS(...).transform()` to obtain the prediction design matrix.

  * Hint: to plot the confidence intervals, use `ax.vlines(origin, lwr, upr)` (or equivalently draw vertical segments from `lwr` to `upr` at each `origin`), where `origin`, `lwr` and `upr` come from a dataframe with `lwr` as the lower bound and `upr` as the upper bound.


## Problem 3

Now, let us look at the `Wage` data set. The section on Additive Models [(in this week's slides)](https://github.com/stefaniemuff/statlearning2/blob/master/7BeyondLinear/7BeyondLinear_2024.pdf) explains how we can create an additive model by adding components together. One type of component we saw is natural cubic splines. Derive the design matrix $\mathbf{X}$ for a natural cubic spline with one internal knot at `year` $= 2006$, from the natural cubic spline basis:

$$
b_1(x_i) = x_i, \quad b_{k+2}(x_i) = d_k(x_i)-d_K(x_i),\; k = 0, \ldots, K - 1,\\
$$

$$
d_k(x_i) = \frac{(x_i-c_k)^3_+-(x_i-c_{K+1})^3_+}{c_{K+1}-c_k}.
$$

## Problem 4

We will continue working with the same additive model as in problem 3, but we now also include a cubic spline for the covariate `age`, and a linear term for `education`. The `ISLP` model spec `MS([bs('age', internal_knots=[40, 60]), ns('year', internal_knots=[2006]), 'education'])` gives a design matrix for the model. This matrix is what `pygam`/`sm.OLS` would use under the hood. However, it does not equal our design matrix for the additive model $\mathbf X = (\mathbf{1}, \mathbf{X}_1, \mathbf{X}_2, \mathbf{X}_3)$. Despite this, the predicted responses will still be the same.

Write code that produces $\mathbf X$. The code below may be useful.

```python
import numpy as np
import pandas as pd

# X_1
def mybs(x, knots):
    x = np.asarray(x)
    cols = [x, x**2, x**3]
    cols += [np.maximum(0, x - k)**3 for k in knots]
    return np.column_stack(cols)

def d(c, cK, x):
    return (np.maximum(0, x - c)**3 - np.maximum(0, x - cK)**3) / (cK - c)

# X_2
def myns(x, knots):
    x = np.asarray(x)
    kn = [x.min()] + list(knots) + [x.max()]
    K = len(kn)
    sub = d(kn[K - 2], kn[K - 1], x)
    cols = [x] + [d(kn[i], kn[K - 1], x) - sub for i in range(K - 2)]
    return np.column_stack(cols)

# X_3
def myfactor(x):
    return pd.get_dummies(x, drop_first=True).to_numpy(dtype=float)
```

If the code is valid, the predicted response $\hat{\mathbf y} = \mathbf X(\mathbf X^\mathsf{T} \mathbf X)^{-1} \mathbf X^\mathsf{T} \mathbf y$ should be the same as when fitting the same model with `sm.OLS` on the `MS([bs(...), ns(...), 'education'])` design matrix.

**Hints:**

```python
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, bs, ns

Wage = load_data('Wage')
# Figure out how to define your X-matrix (this is perhaps a bit tricky!)
X = ...
# fitted model with our X
myhat = sm.OLS(Wage['wage'], X).fit().fittedvalues
# fitted model with the ISLP design matrix
spec = MS([bs('age', internal_knots=[40, 60]),
           ns('year', internal_knots=[2006]),
           'education'])
Xspec = spec.fit_transform(Wage)
yhat = sm.OLS(Wage['wage'], Xspec).fit().fittedvalues
# are they equal?
np.allclose(myhat, yhat)
```

How can `myhat` equal `yhat` when the design matrices differ?

## Problem 5

In this exercise we take a quick look at different non-linear regression methods. We continue using the `Auto` dataset from above, but with more variables.

Fit an additive model using `LinearGAM` from `pygam`. Call the result `gam_full`.

  * `mpg` is the response,
  * `displacement` is a cubic spline (hint: `bs`) with one knot at 290,
  * `horsepower` is a polynomial of degree 2 (hint: `poly`),
  * `weight` is a linear function,
  * `acceleration` is a smoothing spline with `df = 3` (hint: `s_gam` from `pygam`),
  * `origin` is a categorical variable (which can be interpreted as a step function, which can be seen from the dummy variable coding).

Plot the resulting partial-dependence curves. Comment on what you see.

**Hints:** because `LinearGAM` only handles smoothing-spline (`s`), linear (`l`) and factor (`f`) terms, mix the two approaches:

  * For `displacement` (cubic spline at knot 290), `horsepower` (degree-2 polynomial), `weight` (linear) and `origin` (factor): build the relevant columns "by hand" using `MS([bs('displacement', internal_knots=[290]), poly('horsepower', degree=2), 'weight', 'origin'])` and fit with `sm.OLS`, OR
  * Fit the full model with `LinearGAM` using `s_gam` for `acceleration` and `displacement`, `l_gam` for `weight`, and `f_gam` for `origin`, with `poly`-style polynomial entered as additional pre-built columns.

Use `from ISLP.pygam import plot as plot_gam` to draw the partial-dependence plots, with subplots arranged via `fig, axes = subplots(2, 3, figsize=(...))`.


### Acknowledgements

This document was originally adapted from the R-based recommended exercises by Sara Martino, Stefanie Muff and Kenneth Aase (Department of Mathematical Sciences, NTNU).
