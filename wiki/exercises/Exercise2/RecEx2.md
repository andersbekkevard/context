---
title: "Module 2: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author:
  - Sara Martino, Stefanie Muff, Kenneth Aase
  - Department of Mathematical Sciences, NTNU
date: "January 22, 2025"
---


## Problem 1

a) Describe a real-life application in which *classification* might be useful. Identify the response and the predictors. Is the goal inference or prediction?
b) Describe a real-life application in which *regression* might be useful. Identify the response and the predictors. Is the goal inference or prediction?

## Problem 2

a) Take a look at Figure 2.9 in the course book (p. 31). Do the flexible or rigid methods have the highest test error? Is this always the case?
b) Does a very small variance imply that the data has been under- or overfit?
c) Relate the problem of over- and underfitting to the bias-variance trade-off.

## Problem 3 -- Exercise 2.4.9 from ISL textbook (modified)

This exercise involves the `Auto` dataset from the `ISLP` package. Load the data into your Python session by running the following commands:

```python
from ISLP import load_data
Auto = load_data('Auto')
```

PS: if the `ISLP` package is not installed you can install it by running `pip install ISLP` before you load the package the first time.

a) View the data. What are the dimensions of the data? Which predictors are quantitative and which are qualitative?

b) What is the range (min, max) of each quantitative predictor? Hint: use the `min()` and `max()` methods on a pandas DataFrame, or `.agg(['min', 'max'])`.

c) What is the sample mean and sample standard deviation for each quantitative predictor?

d) Now, make a new dataset called `ReducedAuto` where you remove the 10th through 85th observations. What is the range, mean and standard deviation of the quantitative predictors in this reduced set?

e) Using the full dataset, investigate the quantitative predictors graphically using a scatter plot. Do you see any strong relationships between the predictors? Hint: try out the `pd.plotting.scatter_matrix()` function.

f) Suppose we wish to predict gas mileage (`mpg`) on the basis of the other variables (both quantitative and qualitative). Make some plots showing the relationships between `mpg` and the qualitative predictors (hint: `boxplot`). Which predictors would you consider helpful when predicting `mpg`?

g) The correlation of two variables $X$ and $Y$ are defined as
$$
\text{cor}(X,Y) = \frac{\text{cov}(X,Y)}{\sigma_X\sigma_Y}.
$$
The correlation matrix and covariance matrix can be easily found in Python with the pandas `.corr()` and `.cov()` methods, respectively. Use only the covariance matrix (as shown below) to find the correlation between `mpg` and `displacement`, `mpg` and `horsepower`, and `mpg` and `weight`. Do your results coincide with the correlation matrix you find using `Auto[quant].corr()`?

```python
quant = ['mpg', 'cylinders', 'displacement', 'horsepower',
         'weight', 'acceleration', 'year']
covMat = Auto[quant].cov()
```

## Problem 4 -- Multivariate normal distribution

The pdf of a multivariate normal distribution is on the form
$$ f(\boldsymbol{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma|}} \exp\{-\frac{1}{2}(\boldsymbol{x-\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{x-\mu)}\},$$ where $\bf{x}$ is a random vector of size $p\times 1$, $\boldsymbol{\mu}$ is the mean vector of size $p\times 1$ and $\boldsymbol{\Sigma}$ is the covariance matrix of size $p\times p$.

- **a)** Use `np.random.default_rng().multivariate_normal()` to simulate 1000 values from multivariate normal distributions with
  - **i)**
  $$
  \boldsymbol{\mu} = \begin{pmatrix}
  2 \\
  3
  \end{pmatrix} \quad \text{and} \quad \boldsymbol{\Sigma} = \begin{pmatrix}
  1 & 0\\
  0 & 1
  \end{pmatrix},
  $$
  - **ii)**
  $$
  \boldsymbol{\mu} = \begin{pmatrix}
  2 \\
  3
  \end{pmatrix} \quad \text{and} \quad \boldsymbol{\Sigma} = \begin{pmatrix}
  1 & 0\\
  0 & 5
  \end{pmatrix},
  $$
  - **iii)**
  $$
  \boldsymbol{\mu} = \begin{pmatrix}
  2 \\
  3
  \end{pmatrix} \quad \text{and} \quad \boldsymbol{\Sigma} = \begin{pmatrix}
  1 & 2\\
  2 & 5
  \end{pmatrix},
  $$
  - **iv)**
  $$
  \boldsymbol{\mu} = \begin{pmatrix}
  2 \\
  3
  \end{pmatrix} \quad \text{and} \quad \boldsymbol{\Sigma} = \begin{pmatrix}
  1 & -2\\
  -2 & 5
  \end{pmatrix}.
  $$
- **b)** Make a scatter plot of the four sets of simulated data sets. Can you see which plot belongs to which distribution?

## Problem 5 -- Theory and practice: training and test MSE; bias-variance

We will now look closely into the simulations and calculations performed for the training error (`trainMSE`), test error (`testMSE`), and the bias-variance trade-off in lecture 1 of module 2.

Below, the code to run the simulation is included. The data is simulated according to the following specifications:

* True function $f(x)=x^2$ with normal noise $\varepsilon \sim N(0,2^2)$.
* $x= -2.0, -1.9, ... ,4.0$ (grid with 61 values).
* Parametric models are fitted (polynomials of degree 1 to degree 20).
* M=100 simulations.

### a) Problem set-up

Look at the code below, copy it and run it yourself. Explain roughly what is done (you do not need not understand the code in detail), for example by commenting the code after copying it into your report.

We will learn more about fitting linear models in Module 3 - now just think of the polynomial fit as fitting a polynomial regression and then `predict` gives the fitted curve in our grid points. `predictions_list` is just a way to save $M$ simulations of 61 grid-points in $x$ and 20 polynomial models.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

rng = np.random.default_rng(2)  # to reproduce

M = 100   # repeated samplings, x fixed
nord = 20  # order of polynomials

# ------

x = np.arange(-2, 4 + 0.1, 0.1)


def truefunc(x):
    return x ** 2


true_y = truefunc(x)
error = rng.normal(loc=0, scale=2, size=(M, len(x)))
ymat = np.tile(true_y, (M, 1)) + error  # Each row is a simulation

# ------

# predictions_list[i] is a M x len(x) matrix holding the fitted values for
# polynomial degree i+1 across the M simulations.
predictions_list = [np.full((M, len(x)), np.nan) for _ in range(nord)]
for i in range(nord):
    deg = i + 1
    for j in range(M):
        coeffs = np.polyfit(x, ymat[j, :], deg)
        predictions_list[i][j, :] = np.polyval(coeffs, x)

# Plotting -----

# We just want to plot for degrees 1, 2, 10 and 20.
degrees_to_plot = [1, 2, 10, 20]
fig, axes = subplots(nrows=2, ncols=2, figsize=(10, 8), sharex=True, sharey=True)
for ax, deg in zip(axes.flat, degrees_to_plot):
    for j in range(M):
        ax.plot(x, predictions_list[deg - 1][j, :], linewidth=0.5, alpha=0.6)
    ax.plot(x, x ** 2, color='black', linewidth=2)
    ax.set_title(f'poly_degree: {deg}')
fig.tight_layout()
```

What do you observe in the produced plot? Which polynomial fits the best to the true curve?

---

### b) Train and test MSE

First we produce predictions at each grid point based on our training data (`x` and `ymat`). Then we draw new observations to calculate test MSE, see `testymat`.

Observe how `trainMSE` and `testMSE` are calculated, and then run the code.

```python
rng = np.random.default_rng(2)  # to reproduce
M = 100  # repeated samplings, x fixed but new errors
nord = 20

x = np.arange(-2, 4 + 0.1, 0.1)


def truefunc(x):
    return x ** 2


true_y = truefunc(x)
error = rng.normal(loc=0, scale=2, size=(M, len(x)))
testerror = rng.normal(loc=0, scale=2, size=(M, len(x)))
ymat = np.tile(true_y, (M, 1)) + error
testymat = np.tile(true_y, (M, 1)) + testerror

predictions_list = [np.full((M, len(x)), np.nan) for _ in range(nord)]
for i in range(nord):
    deg = i + 1
    for j in range(M):
        coeffs = np.polyfit(x, ymat[j, :], deg)
        predictions_list[i][j, :] = np.polyval(coeffs, x)

trainMSE = [np.mean((predictions_list[i] - ymat) ** 2, axis=1)
            for i in range(nord)]
testMSE = [np.mean((predictions_list[i] - testymat) ** 2, axis=1)
           for i in range(nord)]
```

Next, we plot the training and test error for each of the 100 data sets we simulated, as well as two different plots that show the means across the simulations.

```python
# Build a long-format DataFrame of (poly_degree, simulation_num, error_type, error)
records = []
for i in range(nord):
    deg = i + 1
    for j in range(M):
        records.append({'poly_degree': deg, 'simulation_num': j + 1,
                        'error_type': 'train', 'error': trainMSE[i][j]})
        records.append({'poly_degree': deg, 'simulation_num': j + 1,
                        'error_type': 'test', 'error': testMSE[i][j]})
stacked_errors_df = pd.DataFrame.from_records(records)

# All lines: one line per simulation, faceted by error type
fig, axes = subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
for ax, etype in zip(axes, ['train', 'test']):
    sub = stacked_errors_df[stacked_errors_df['error_type'] == etype]
    for sim in range(1, M + 1):
        s = sub[sub['simulation_num'] == sim]
        ax.plot(s['poly_degree'], s['error'], linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('MSE')
    ax.set_title(etype)
fig.tight_layout()

# Boxplots and means side by side
fig, axes = subplots(nrows=1, ncols=2, figsize=(14, 5))

# Boxplot per polynomial degree, grouped by error type
positions = np.arange(1, nord + 1)
box_train = [stacked_errors_df.query("error_type=='train' and poly_degree==@d")['error'].values
             for d in positions]
box_test = [stacked_errors_df.query("error_type=='test' and poly_degree==@d")['error'].values
            for d in positions]
width = 0.35
axes[0].boxplot(box_train, positions=positions - width / 2, widths=width,
                patch_artist=True, boxprops=dict(facecolor='C0'))
axes[0].boxplot(box_test, positions=positions + width / 2, widths=width,
                patch_artist=True, boxprops=dict(facecolor='C1'))
axes[0].set_xticks(positions)
axes[0].set_xlabel('Polynomial degree')
axes[0].set_ylabel('MSE')
axes[0].legend(['train', 'test'])

# Mean curves
means = (stacked_errors_df
         .groupby(['error_type', 'poly_degree'])['error']
         .mean()
         .reset_index())
for etype in ['train', 'test']:
    sub = means[means['error_type'] == etype]
    axes[1].plot(sub['poly_degree'], sub['error'], label=etype)
axes[1].set_xlabel('Polynomial degree')
axes[1].set_ylabel('MSE')
axes[1].legend(title='Error type')
fig.tight_layout()
```

- Which polynomial degree gives the smallest mean testMSE?
- Which polynomial degree gives the smallest mean trainMSE?
- Which should you use to predict a new value of $y$?

### c) Bias and variance - we use the truth!

Finally, we want to see how the expected quadratic loss can be decomposed into

* irreducible error: $\text{Var}(\varepsilon)=4$
* squared bias: difference between mean of estimated parametric model chosen and the true underlying curve (`truefunc`)
* variance: variance of the estimated parametric model

Notice that the test data is not used -- only predicted values in each x grid point.

Study and run the code. Explain the plots produced.

```python
meanmat = np.zeros((nord, len(x)))
varmat = np.zeros((nord, len(x)))
for j in range(nord):
    # We now take the mean over the M simulations -
    # to mimic E and Var at each x value and each poly model
    meanmat[j, :] = predictions_list[j].mean(axis=0)
    varmat[j, :] = predictions_list[j].var(axis=0, ddof=1)

# Here the truth is finally used!
bias2mat = (meanmat - np.tile(true_y, (nord, 1))) ** 2
```

Plotting the polys as a function of x:

```python
# Long-format DataFrame: one row per (x, poly_degree, type)
n_x = len(x)
df = pd.DataFrame({
    'x': np.tile(x, nord),
    'poly_degree': np.repeat(np.arange(1, nord + 1), n_x),
    'bias2': bias2mat.flatten(),
    'variance': varmat.flatten(),
    'irreducible_error': np.full(nord * n_x, 4.0),  # irr is just sigma^2
})
df['total'] = df['bias2'] + df['variance'] + df['irreducible_error']

df_long = df.melt(id_vars=['x', 'poly_degree'], var_name='type', value_name='value')

df_select_poly = df_long[df_long['poly_degree'].isin([1, 2, 10, 20])]

fig, axes = subplots(nrows=2, ncols=2, figsize=(12, 8))
for ax, deg in zip(axes.flat, [1, 2, 10, 20]):
    sub = df_select_poly[df_select_poly['poly_degree'] == deg]
    for t in ['bias2', 'variance', 'irreducible_error', 'total']:
        s = sub[sub['type'] == t]
        ax.plot(s['x'], s['value'], label=t)
    ax.set_title(f'poly_degree: {deg}')
    ax.legend()
fig.tight_layout()
```

Now plotting effect of more complex model at 4 chosen values of x, compare to Figures in 2.12 on page 36 in ISL (our textbook).

```python
df_select_x = df_long[df_long['x'].isin([-1, 0.5, 2, 3.5])]

fig, axes = subplots(nrows=2, ncols=2, figsize=(12, 8))
for ax, x_val in zip(axes.flat, [-1, 0.5, 2, 3.5]):
    sub = df_select_x[np.isclose(df_select_x['x'], x_val)]
    for t in ['bias2', 'variance', 'irreducible_error', 'total']:
        s = sub[sub['type'] == t]
        ax.plot(s['poly_degree'], s['value'], label=t)
    ax.set_title(f'x: {x_val}')
    ax.legend()
fig.tight_layout()
```

Study the final plot you produced: when the flexibility increases (poly increase), what happens with
i) the squared bias,
ii) the variance,
iii) the irreducible error?

---

### d) Repeat a-c
Try to change the true function `truefunc` to something else - maybe order $3$? What does this do the the plots produced? Maybe you then also want to plot poly3?

Also try to change the standard deviation of the noise added to the curve (now it is sd=2). What happens if you change this to `sd=1` or `sd=3`?

Or, change to the true function so that is not a polynomial?

---

# Acknowledgements

We thank Mette Langaas and her PhD students (in particular Julia Debik) from 2018 and 2019 for building up the original version of this exercise sheet.
