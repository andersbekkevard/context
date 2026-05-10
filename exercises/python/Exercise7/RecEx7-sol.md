---
title: "Module 7: Solutions to Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "February 28, 2025"
---

## Problem 1

The code below performs polynomial regression of degree $1$, $2$, $3$ and $4$. We loop over the degrees, fit each model on the training set, draw the fitted curve and accumulate the test MSE. Finally we plot the test error against the polynomial degree.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from matplotlib import cm
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
tr = rng.choice(n, size=n // 2, replace=False)
te = np.setdiff1d(np.arange(n), tr)

train = ds.iloc[tr]
test = ds.iloc[te]

# colors for the four lines
colors = cm.rainbow(np.linspace(0, 1, len(deg)))

fig, ax = subplots()
ax.scatter(train['horsepower'], train['mpg'], color='darkgrey')
ax.set_title('Polynomial regression')
ax.set_xlabel('horsepower')
ax.set_ylabel('mpg')

# iterate over all degrees (1:4)
MSE = np.zeros(len(deg))
order = np.argsort(train['horsepower'].values)

for i, d in enumerate(deg):
    # fit model with this degree
    poly_d = MS([poly('horsepower', degree=d)]).fit(train)
    Xtr = poly_d.transform(train)
    mod = sm.OLS(train['mpg'], Xtr).fit()
    # add lines to the plot using fitted values from the training set
    ax.plot(train['horsepower'].values[order],
            mod.fittedvalues.values[order],
            color=colors[i],
            label=f'd = {d}')
    # test MSE
    Xte = poly_d.transform(test)
    pred = mod.predict(Xte)
    MSE[i] = np.mean((pred - test['mpg'].values)**2)

ax.legend(loc='upper right')

# plot MSE
fig2, ax2 = subplots()
ax2.plot(deg, MSE, marker='o')
ax2.set_xlabel('degree')
ax2.set_title('Test error')
ax2.set_ylabel('MSE');
```

The same solution using a per-degree dataframe and a single line plot is shown below.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, poly

Auto = load_data('Auto')
ds = Auto[['horsepower', 'mpg']]
n = ds.shape[0]
deg = [1, 2, 3, 4]

rng = np.random.default_rng(1)
tr = rng.choice(n, size=n // 2, replace=False)
te = np.setdiff1d(np.arange(n), tr)
train = ds.iloc[tr]
test = ds.iloc[te]

# scatter of training data
fig, ax = subplots()
ax.scatter(train['horsepower'], train['mpg'], color='darkgrey')
ax.set_title('Polynomial regression')
ax.set_xlabel('horsepower')
ax.set_ylabel('mpg')

# iterate over all degrees and collect predictions
records = []
MSE = np.zeros(len(deg))
order = np.argsort(train['horsepower'].values)

for i, d in enumerate(deg):
    poly_d = MS([poly('horsepower', degree=d)]).fit(train)
    Xtr = poly_d.transform(train)
    mod = sm.OLS(train['mpg'], Xtr).fit()
    fit_df = pd.DataFrame({
        'horsepower': train['horsepower'].values,
        'mpg': mod.fittedvalues.values,
        'degree': d
    })
    records.append(fit_df)
    Xte = poly_d.transform(test)
    MSE[i] = np.mean((mod.predict(Xte) - test['mpg'].values)**2)

dat = pd.concat(records, ignore_index=True)

# overlay fitted curves coloured by degree
for d, sub in dat.groupby('degree'):
    sub_sorted = sub.sort_values('horsepower')
    ax.plot(sub_sorted['horsepower'], sub_sorted['mpg'],
            label=f'degree = {d}')
ax.legend()

# plot MSE
fig2, ax2 = subplots()
ax2.plot(deg, MSE, marker='o')
ax2.set_xlabel('degree')
ax2.set_ylabel('MSE')
ax2.set_title('Test error');
```

## Problem 2

We convert `origin` to a categorical variable. The `get_prediction()` method on a fitted `statsmodels` result yields fitted values and (via `conf_int`) standard-error-based confidence bands.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS

Auto = load_data('Auto')
Auto['origin'] = Auto['origin'].astype('category')

# fit model
spec = MS(['origin']).fit(Auto)
X = spec.transform(Auto)
fit = sm.OLS(Auto['mpg'], X).fit()

# new dataset of the three origins
new = pd.DataFrame({'origin': pd.Categorical([1, 2, 3],
                                             categories=Auto['origin'].cat.categories)})
Xnew = spec.transform(new)

# predicted values and confidence bands
preds = fit.get_prediction(Xnew)
bands = preds.conf_int(alpha=0.05)

dat = pd.DataFrame({
    'origin': [1, 2, 3],
    'mpg': preds.predicted_mean,
    'lwr': bands[:, 0],
    'upr': bands[:, 1]
})

# plot the fitted/predicted values and CI
fig, ax = subplots()
ax.scatter(dat['origin'], dat['mpg'])
ax.vlines(dat['origin'], dat['lwr'], dat['upr'])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['1.American', '2.European', '3.Japanese'])
ax.set_xlabel('origin')
ax.set_ylabel('mpg');
```

## Problem 3

The request is a design matrix $X$ for a natural cubic spline with $X =$ `year` and one internal knot at $2006$.
The definition of the basis for a natural cubic spline is
$$
b_1(x_i) = x_i, \quad b_{k+2}(x_i) = d_k(x_i)-d_K(x_i),\; k = 0, \ldots, K - 1,\\
$$

$$
d_k(x_i) = \frac{(x_i-c_k)^3_+-(x_i-c_{K+1})^3_+}{c_{K+1}-c_k},
$$

where $K$ is the number of internal knots, $c_k$ is the location of the $k^\text{th}$ knot, and $(.)_+$ denotes $\max (., 0)$.
In our case we have one internal knot at $c_1 = 2006$, thus $K = 1$.
Since this is a natural spline, the boundary knots should be the extreme values of `year`, that is $c_0 = 2003$ and $c_2 = 2009$.
Since $K=1$, the index $k$ takes only the value $0$, so we end up with only having to find $b_1(x_i)$ and $b_2(x_i)$.

By inserting into the above expressions, we find that the two basis functions are

\begin{align*}
b_1(x_i) &= x_i,\\
b_2(x_i) &= d_0(x_i)-d_1(x_i)\\
&= \frac{(x_i-c_0)^3_+-(x_i-c_2)^3_+}{c_2-c_0} - \frac{(x_i-c_1)^3_+-(x_i-c_2)^3_+}{c_2-c_1}\\
&= \frac{1}{c_2-c_0}(x_i-c_0)^3_+ - \frac{1}{c_2-c_1}(x_i-c_1)^3_+ + \left(\frac{1}{c_2-c_1}-\frac{1}{c_2-c_0}\right)(x_i-c_{2})^3_+\\
&= \frac{1}{6}(x_i-2003)^3_+ - \frac{1}{3}(x_i-2006)^3_+ + \frac{1}{6}(x_i-2009)^3_+.
\end{align*}

We can simplify the final term in the second basis function by using the fact that the boundary knots are the extreme values of $x_i$, that is $2003 \leq x_i \leq 2009$, and thus $\frac{1}{6}(x_i-2009)^3_+=0$. Thus,

$$
b_2(x_i) = \frac{1}{6}(x_i-2003)^3 - \frac{1}{3}(x_i-2006)^3_+.
$$

The design matrix (here also including an intercept term) is given by

$$
\mathbf X = \begin{pmatrix}
1 & b_{1}(x_1) & b_{2}(x_1) \\
1 & b_{1}(x_2) & b_{2}(x_2) \\
\vdots & \vdots & \vdots \\
1 & b_{1}(x_n) & b_{2}(x_n) \\
\end{pmatrix},
$$

where $b_1(x_i)$ and $b_2(x_i)$ are as found above.

## Problem 4

The matrix $\mathbf X$ is obtained by horizontally stacking an intercept column, a cubic spline, a natural cubic spline and a factor expansion.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, bs, ns

Wage = load_data('Wage')

# Write a couple of helper functions first, which will be used to produce the
# components of the design matrix.
# We write separate functions to generate X_1, X_2 and X_3 (the three components
# of the model).

# X_1: mybs() generates basis functions for the cubic spline
def mybs(x, knots):
    x = np.asarray(x, dtype=float)
    cols = [x, x**2, x**3]
    cols += [np.maximum(0, x - k)**3 for k in knots]
    return np.column_stack(cols)

# X_2: myns() generates basis functions for the natural cubic spline;
# d() is a helper function
def d(c, cK, x):
    return (np.maximum(0, x - c)**3 - np.maximum(0, x - cK)**3) / (cK - c)

def myns(x, knots):
    x = np.asarray(x, dtype=float)
    kn = [x.min()] + list(knots) + [x.max()]
    K = len(kn)
    sub = d(kn[K - 2], kn[K - 1], x)
    cols = [x] + [d(kn[i], kn[K - 1], x) - sub for i in range(K - 2)]
    return np.column_stack(cols)

# X_3: myfactor() generates the dummy-basis functions for a factor covariate
def myfactor(x):
    return pd.get_dummies(x, drop_first=True).to_numpy(dtype=float)

# Once these helpers are prepared, we can generate the model matrix
# X = (1, X_1, X_2, X_3) as a one-liner
n = Wage.shape[0]
X = np.column_stack([
    np.ones(n),
    mybs(Wage['age'].values, [40, 60]),
    myns(Wage['year'].values, [2006]),
    myfactor(Wage['education'])
])
```

We have now created a model matrix $\mathbf X$ "by hand". The thing we wanted to illustrate with this exercise is that this hand-made matrix does not correspond to the internal representation of the model matrix that `MS([bs(...), ns(...), 'education'])` produces:

```python
spec = MS([bs('age', internal_knots=[40, 60]),
           ns('year', internal_knots=[2006]),
           'education'])
X_spec = spec.fit_transform(Wage)
```

```python
# If we now use our model matrix to fit a linear regression model
# (we already include a column of ones in X, so no extra intercept is needed)
myhat = sm.OLS(Wage['wage'], X).fit().fittedvalues

# Comparing to the fitted values with the ISLP-style spec shows that they agree
yhat = sm.OLS(Wage['wage'], X_spec).fit().fittedvalues
np.allclose(myhat, yhat)
```

The fitted values `myhat` and `yhat` are equal. Both the design matrices $\mathbf X$ and the coefficients $\hat{\beta}$ differ, but $\mathbf X \hat{\beta}$ is the same. How can this be? Well, just as there were several ways to represent polynomials, there are also many equivalent ways to represent splines or factor variables using different choices of basis functions.

## Problem 5

Fit additive model and commenting:

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import ModelSpec as MS, bs, poly
from pygam import LinearGAM, s as s_gam, l as l_gam, f as f_gam
from ISLP.pygam import approx_lam, plot as plot_gam

Auto = load_data('Auto')
# first set origin as a categorical variable
Auto['origin'] = Auto['origin'].astype('category')

# Strategy: pygam's LinearGAM only supports s/l/f terms directly, so we build a
# combined design with smoothing splines via pygam for `acceleration` and a
# pre-built basis for the remaining terms (cubic spline of displacement with knot
# at 290, degree-2 polynomial of horsepower, linear weight, and origin as factor).

# Pre-built (parametric) part: bs(displacement) + poly(horsepower, 2) + weight + origin
spec = MS([bs('displacement', internal_knots=[290]),
           poly('horsepower', degree=2),
           'weight',
           'origin']).fit(Auto)
X_param = np.asarray(spec.transform(Auto), dtype=float)

# Smoothing-spline part: acceleration alone, df = 3
X_acc = Auto['acceleration'].values.reshape(-1, 1)

# Combined feature matrix: parametric columns first, then acceleration as the
# last column, which we will smooth with s_gam.
Xgam = np.column_stack([X_param, X_acc])
acc_idx = Xgam.shape[1] - 1

# Build a GAM where every parametric column is treated as a linear term and the
# last column is a smoothing spline; lam=0 disables shrinkage on the parametric
# block.
terms = sum((l_gam(j, lam=0) for j in range(acc_idx)), s_gam(acc_idx))
gam_full = LinearGAM(terms)
gam_full.fit(Xgam, Auto['mpg'])

# Set df = 3 for the smoothing spline on `acceleration`
acc_term = gam_full.terms[acc_idx]
acc_term.lam = approx_lam(Xgam, acc_term, df=3 + 1)
gam_full = gam_full.fit(Xgam, Auto['mpg'])

# Plot covariates: arrange in a 2x3 grid
fig, axes = subplots(2, 3, figsize=(12, 7))
plot_gam(gam_full, acc_idx, ax=axes[0, 0])
axes[0, 0].set_title('acceleration (smoothing spline, df=3)')

# For partial dependence on the parametric covariates (displacement, horsepower,
# weight, origin) we build them by hand from sm.OLS on the same design matrix.
mod = sm.OLS(Auto['mpg'], sm.add_constant(X_param)).fit()
print(mod.summary())
```

Equivalently, the simpler approach is to fit the parametric part with `sm.OLS` on the ISLP design matrix and the smoothing-spline part with `LinearGAM` on `acceleration` separately, then read off partial-dependence plots.

```python
# Simpler: fit the whole model as an OLS using the ISLP design matrix, replacing
# the smoothing spline with a natural spline of df=3 on acceleration. The fits
# are similar in spirit to gam() in R.
from ISLP.models import ns

spec_full = MS([bs('displacement', internal_knots=[290]),
                poly('horsepower', degree=2),
                'weight',
                ns('acceleration', df=3),
                'origin']).fit(Auto)
Xfull = spec_full.transform(Auto)
fitgam = sm.OLS(Auto['mpg'], Xfull).fit()
print(fitgam.summary())
```

Partial-dependence plots for each term can then be generated by holding the other columns at their column means and varying one covariate at a time over a grid (the same recipe as in §7.8.3 of the ISLP lab).

We see `displacement` has two peaks, `horsepower` has the smallest CI for low values, the linear function in `weight` is very variable for small and high values, `acceleration` looks rather like a cubic function, and there is a clear effect of `origin`.


### Acknowledgements

This document was originally adapted from the R-based recommended exercises by Sara Martino, Stefanie Muff and Kenneth Aase (Department of Mathematical Sciences, NTNU).
