---
title: "Module 2: Recommended Exercises - Solution"
subtitle: "TMA4268 Statistical Learning V2025"
author:
  - Sara Martino, Stefanie Muff, Kenneth Aase
  - Department of Mathematical Sciences, NTNU
date: "January 22, 2025"
---


# Problem 1
a) Classification

Example 1: Diagnosing cancer. Response: cancer (yes/no). Predictors: smoking status, age, family history, gene expression etc. Goal: prediction.

Example 2: Understanding stock market price direction. Response: up/down. Predictors: yesterday's price movement change, two previous days price movement etc. Goal: inference.

Example 3: Flower species. Response: species. Predictors: color, height, leaves, etc. Goal: prediction.

b) Regression

Example 1: Life expectancy. Response: age of death. Predictors: current age, gender, resting heart rate, resting breath rate etc. Goal: prediction.

Example 2: Price of a house. Response: Price. Predictors: age of house, average price in neighborhood, crime rate, distance to city center, distance to school, etc. Goal: prediction.

Example 3: What affects O2-uptake. Response: O2-uptake. Predictors: gender, age, amount of weekly exercise, type of exercise, smoking, heart disease, etc. Goal: inference.

# Problem 2

a) Based on the three different models compared here, the rigid method has the highest test MSE. However, this does not always have to be the case. For instance, in figure 2.10, we see that the most flexible model has the highest test MSE. See also figure 2.12 that summarizes the last three figures and shows some of the variation. So in general we cannot say anything certain about how the test MSE will vary based on model flexibility, this will depend on how the training data compares to the test data, and other model characteristics. However (though the question does not ask this), the *training* MSE will always decrease as the flexibility increases.
b) A small (test) variance implies an underfit to the data.
c) See figure 2.12. Underfit - low variance - high bias. Overfit - high variance - low bias. We wish to find a model that lies somewhere in-between, with low variance *and* low bias.

# Problem 3

```python
from ISLP import load_data
Auto = load_data('Auto')
```

a)

```python
Auto.info()
Auto.describe()
```

The dimensions are $392$ observations (rows) of $9$ variables (columns).
Looking at `Auto.info()` and `Auto.describe()`, `origin` (encoded by values $1,2,3$) and `name` (name of the car) are qualitative predictors. The rest of the predictors are quantitative.

b)

To see the range of the quantitative predictors, either apply `min()` and `max()` to each column with a quantitative predictor separately

```python
print(Auto['mpg'].min(), Auto['mpg'].max())
print(Auto['displacement'].min(), Auto['displacement'].max())
print(Auto['horsepower'].min(), Auto['horsepower'].max())
print(Auto['weight'].min(), Auto['weight'].max())
print(Auto['acceleration'].min(), Auto['acceleration'].max())
print(Auto['year'].min(), Auto['year'].max())
```

or use `agg` to compute the range across the specified columns with a single line of code:

```python
quant = ['mpg', 'cylinders', 'displacement', 'horsepower',
         'weight', 'acceleration', 'year']
Auto[quant].agg(['min', 'max'])
```

c)

To get the sample mean and sample standard deviation of the quantitative predictors, we can again use `agg`, or apply `mean()` and `std()` column-wise.

```python
# mean
Auto[quant].mean()
# or
Auto['mpg'].mean()
Auto['cylinders'].mean()
Auto['displacement'].mean()
Auto['horsepower'].mean()
Auto['weight'].mean()
Auto['acceleration'].mean()
Auto['year'].mean()

# sd
Auto[quant].std()
# or
Auto['mpg'].std()
Auto['cylinders'].std()
Auto['displacement'].std()
Auto['horsepower'].std()
Auto['weight'].std()
Auto['acceleration'].std()
Auto['year'].std()
```

d)

Remove 10th to 85th observations and look at the range, mean and standard deviation of the reduced set. We now only show the solutions using `agg` to save space.

```python
# remove observations (rows 10 through 85, inclusive of 10, exclusive of 86)
ReducedAuto = Auto.drop(Auto.index[9:85]).reset_index(drop=True)

# range, mean and sd
ReducedAuto[quant].agg(['min', 'max'])
ReducedAuto[quant].mean()
ReducedAuto[quant].std()
```

e)

We make a scatter plot of the full dataset using `pd.plotting.scatter_matrix()`.

```python
import pandas as pd
pd.plotting.scatter_matrix(Auto[quant], figsize=(10, 10));
```

We see that there seems to be strong relationships (based on curve trends and correlation) between the pairs: `mpg` and `displacement`,  `mpg` and `horsepower`, `mpg` and `weight`,  `displacement` and `horsepower`, `displacement` and `weight`, `horsepower` and `weight`, and `horsepower` and `acceleration`.

f) Wish to predict gas mileage based on the other variables. From the scatter plot we see that `displacement`, `horsepower` and `weight` could be good choices for prediction of `mpg`. Check if the qualitative predictors could also be good choices by plotting them against `mpg`.

```python
from matplotlib.pyplot import subplots

fig, ax = subplots(figsize=(8, 6))
Auto.boxplot('mpg', by='cylinders', ax=ax)
ax.set_title('mpg vs cylinders')

fig, ax = subplots(figsize=(8, 6))
Auto.boxplot('mpg', by='origin', ax=ax)
ax.set_title('mpg vs origin')
```

From these plots we see that both `cylinders` and `origin` could be good choices for prediction of `mpg`, because the miles per gallon (mpg) seems to depend on these two variables.

g) To find the correlation of the given variables, we need the covariance of these variables as well as the standard deviations, which are both available in the covariance matrix. Remember that the variance of each variable is given in the diagonal of the covariance matrix.

```python
import numpy as np

covMat = Auto[quant].cov()
# mpg vs displacement
covMat.loc['mpg', 'displacement'] / (np.sqrt(covMat.loc['mpg', 'mpg'])
                                      * np.sqrt(covMat.loc['displacement', 'displacement']))
# mpg vs horsepower
covMat.loc['mpg', 'horsepower'] / (np.sqrt(covMat.loc['mpg', 'mpg'])
                                    * np.sqrt(covMat.loc['horsepower', 'horsepower']))
# mpg vs weight
covMat.loc['mpg', 'weight'] / (np.sqrt(covMat.loc['mpg', 'mpg'])
                                * np.sqrt(covMat.loc['weight', 'weight']))
Auto[quant].corr()
```

We see that the obtained correlations coincide with the given elements in the correlation matrix.

# Problem 4

a) Simulate values from the four multivariate distributions using `multivariate_normal()`.

```python
import numpy as np
import pandas as pd

# Simulate 1000 values from the multivariate normal distribution 4 times, with
# 4 different covariance matrices
rng = np.random.default_rng(0)
n = 1000
mu = np.array([2, 3])

set1 = pd.DataFrame(
    rng.multivariate_normal(mu, np.array([[1, 0], [0, 1]]), size=n),
    columns=['x1', 'x2'])

set2 = pd.DataFrame(
    rng.multivariate_normal(mu, np.array([[1, 0], [0, 5]]), size=n),
    columns=['x1', 'x2'])

set3 = pd.DataFrame(
    rng.multivariate_normal(mu, np.array([[1, 2], [2, 5]]), size=n),
    columns=['x1', 'x2'])

set4 = pd.DataFrame(
    rng.multivariate_normal(mu, np.array([[1, -2], [-2, 5]]), size=n),
    columns=['x1', 'x2'])
```

b) Plot the simulated distributions

```python
from matplotlib.pyplot import subplots

fig, axes = subplots(nrows=2, ncols=2, figsize=(10, 10))
for ax, (data, title) in zip(axes.flat,
                              [(set1, 'set1'), (set2, 'set2'),
                               (set3, 'set3'), (set4, 'set4')]):
    ax.scatter(data['x1'], data['x2'], s=10)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
fig.tight_layout()
```


# Problem 5

a) We sample from the model $y=x^2+\epsilon$ where $\epsilon \sim \mathcal{N}(0,2^2)$ and $x\in \{-2,-1.9,-1.8,...,3.8,3.9,4\}$. This means that $y \sim \mathcal{N}(x^2,2^2)$. A total of $100$ samples from this model are generated for each of the $61$ $x$'s.
See comments in code for further explanations.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

rng = np.random.default_rng(2)  # to reproduce

M = 100   # repeated samplings, x fixed
nord = 20  # order of polynomials

# ------
# We make a sequence of 61 points, x.
# These are the points for which we evaluate the function f(x).
x = np.arange(-2, 4 + 0.1, 0.1)


def truefunc(x):
    return x ** 2  # The true f(x) = x^2.


# We find f(x) for each element in vector x.
true_y = truefunc(x)

# Noise (epsilon) is sampled from a normal distribution and stored in this matrix.
# Each column corresponds to one value of x.
error = rng.normal(loc=0, scale=2, size=(M, len(x)))

# The 100 samples or the observations are stored in this matrix. Each row is a simulation.
ymat = np.tile(true_y, (M, 1)) + error

# ------
# Based on the response y_i and the x_i's, we fit a polynomial model of degree
# 1, ..., 20.
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

The upper left plot shows 100 predictions when we assume that $y$ is a linear function of $x$, the upper right plot shows 100 predictions when we assume that $y$ is function of polynomials up to $x^2$, the lower left plot shows 100 predictions when we assume $y$ is a function of polynomials up to $x^{10}$ and the lower right plot shows 100 predictions when assuming $y$ is a function of polynomials up to $x^{20}$.

b) Run the code attached and consider the following plots:

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

# Plotting -----

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

The plots show that the test MSE in general is larger than the train MSE. This is makes sense, since the model is fitted based on the training set. Thus, the error will be smaller for the train data than for the test data. Furthermore, the plots show that the difference between the MSE for the test set and the training set increases when the degree of the polynomial increases. When the degree of the polynomial increases, we get a more flexible model. The fitted curve will try to pass through the training data if possible, which typically leads to an overfitted model that performs bad for test data.

* We observe that poly 2 gives the smallest mean testMSE, while poly 20 gives the smallest trainMSE. Based on these plots, we would choose poly 2 for prediction of a new value of $y$, as the testMSE tells us how the model performs on new data, which was not used to train the model.

c) Run the code and consider the following plots:

```python
meanmat = np.zeros((nord, len(x)))
varmat = np.zeros((nord, len(x)))
for j in range(nord):
    # We now take the mean over the M simulations
    # - to mimic E and Var at each x value and each poly model
    meanmat[j, :] = predictions_list[j].mean(axis=0)
    varmat[j, :] = predictions_list[j].var(axis=0, ddof=1)

# Here the truth is finally used!
bias2mat = (meanmat - np.tile(true_y, (nord, 1))) ** 2

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

We see that the irreducible error remains constant with the complexity (flexibility) of the model and the variance increases with the complexity. A linear model gives variance close to zero, while a polynomial of degree 20 gives variance close to 1 (larger at the borders). A more complex model is more flexible as it can turn up and down and change direction fast. This leads to larger variance. (Look at the plot in 2a, there is a larger variety of curves you can make when the degree is 20 compared to if the degree is 1.)

Furthermore, we see that the bias is large for poly1, the linear model. The linear model is quite rigid, so if the true underlying model is non-linear, we typically get large deviations between the fitted line and the training data. If we study the first plot, it seems like the fitted line goes through the training data in $x=-1$ and $x=3$ as the bias is close to zero here (this is confirmed by looking at the upper left plot in 2a).

The polynomial models with degree larger than one lead to lower bias. Recall that this is the training bias: The polynomial models will try to pass through the training points if possible, and when the degree of the polynomial is large they are able to do so because they have large flexibility compared to a linear model.


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

Compare to Figures in 2.12 on page 36 in ISL (our textbook).


d) To change $f(x)$, replace

```python
def truefunc(x):
    return x ** 2
```

by for example

```python
def truefunc(x):
    return x ** 4
```

or

```python
def truefunc(x):
    return np.exp(2 * x)
```

and rerun the code. Study the results.

If you want to set the variance to 1 for example, set `scale=1` in these parts of the code in 2a and 2b:

```python
rng.normal(loc=0, scale=1, size=(M, len(x)))
```

Also change the following part in 2c:

```python
df = pd.DataFrame({
    'x': np.tile(x, nord),
    'poly_degree': np.repeat(np.arange(1, nord + 1), n_x),
    'bias2': bias2mat.flatten(),
    'variance': varmat.flatten(),
    'irreducible_error': np.full(nord * n_x, 4.0),  # irr is just sigma^2
})
```

to get correct plots of the irreducible error. Here, `np.full(nord * n_x, 4.0)` is replaced by `np.full(nord * n_x, 1.0)`.


# Python packages

If you want to run this notebook, you need to first install the following packages (only once).

```python
# pip install ISLP
# pip install numpy
# pip install pandas
# pip install matplotlib
```
