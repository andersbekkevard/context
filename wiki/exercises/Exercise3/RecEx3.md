---
title: "Module 3: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "January 29, 2025"
---

---

**We strongly recommend you to work through the Section 3.6 in the course book (Lab on linear regression)**

---

# Problem 1 (Extension from Book Ex. 9)

This question involves the use of multiple linear regression on the `Auto` data set from the `ISLP` package (you may use `Auto?` to see a description of the data). First we exclude from our analysis the variable `name` and look at the data summary and structure of the dataset.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

Auto = load_data('Auto')
Auto = Auto.drop(columns=['name'])
# Auto['origin'] = Auto['origin'].astype('category')
Auto.describe()
Auto.info()
```

We obtain a summary and see that all variables are numerical (continuous). However, when we check the description of the data (again with `Auto?`) we immediately see that `origin` is actually encoding for either American (`origin = 1`), European (`origin = 2`) or Japanese (`origin = 3`) origin of the car, thus the values $1$, $2$ and $3$ do not have any actual numerical meaning. We therefore need to first change the data type of that variable to let `Python` know that we are dealing with a qualitative (categorical) variable, instead of a continuous one (otherwise we will obtain wrong model fits). In `pandas` such variables are called _categorical_ (a synonymous for "qualitative predictor"), and before we continue to do any analyses we first need to convert `origin` into a categorical variable:

```python
Auto['origin'] = Auto['origin'].astype('category')
```


## a)
Use `seaborn`'s `pairplot()` (or `pandas.plotting.scatter_matrix`) to produce a scatterplot matrix which includes all of the variables in the data set.

## b)
Compute the correlation matrix between the variables. You will need to remove the categorical covariate `origin`, because this is no longer a continuous variable.

## c)
Use `MS()` together with `sm.OLS()` to perform a multiple linear regression with `mpg` (miles per gallon, a measure for fuel consumption) as the response and all other variables (except `name`) as the predictors. Use `summarize()` (or `results.summary()`) to print the results. Comment on the output. In particular:

i. Is there a relationship between the predictors and the response?

ii. Is there evidence that the weight of a car influences `mpg`? Interpret the regression coefficient $\beta_{\text{weight}}$ (what happens if a car weights 1000kg more, for example?).

iii. What does the estimated coefficient for the `year` variable suggest?

## d)
Look again at the regression output from question **c)**. Now we want to test whether the `origin` variable is important. How does this work for a categorical variable with more than only two levels?


## e)
Produce diagnostic plots of the linear regression fit (residuals vs. fitted, QQ plot of standardized residuals, scale-location, residuals vs. leverage). Comment on any problems you see with the fit. Do the residual plots suggest any unusually large outliers? Does the leverage plot identify any observations with unusually high leverage?

## f)
For beginners, it can be difficult to decide whether a certain QQ plot looks "good" or "bad", because we only look at it and do not test anything. A way to get a feeling for how "bad" a QQ plot may look, even when the normality assumption is perfectly OK, we can use simulations: We can simply draw from the normal distribution and plot the QQ plot. Use the following code to repeat this six times:

```python
from scipy import stats

rng = np.random.default_rng(2332)
n = 100

fig, axes = subplots(2, 3, figsize=(9, 6))
for ax in axes.ravel():
    sim = rng.standard_normal(n)
    stats.probplot(sim, dist='norm', plot=ax)
    ax.set_title('')
fig.tight_layout()
```

## g)
Let us look at interactions. These can be included in the model matrix by passing tuples to `MS()` (see Section 3.6.4 in the course book).

Fit another model for `mpg`, including only `displacement`, `weight`, `year` and `origin` as predictors, plus an interaction between `year` and `origin`. The interaction `year*origin` (in the R formula sense) corresponds in `MS()` to including the main effects together with the interaction tuple, i.e. `MS(['displacement', 'weight', 'year', 'origin', ('year', 'origin')])`. Is there evidence that the interactions term is relevant? Give an interpretation of the result.



## h)
Try a few different transformations of the variables, such as $\log(X),$ $\sqrt{X},$ $X^2$. See Section 3.6.5 in the course book for how to do this. Perhaps you manage to improve the residual plots that you got in e)? Comment on your findings.


# Problem 2

## a)
A core finding for the least-squares estimator $\hat{\boldsymbol\beta}$ of linear regression models is
$$ \hat{\boldsymbol\beta} = ({\bf X}^T{\bf X})^{-1} {\bf X}^T {\bf Y} \ , $$
with $\hat{\boldsymbol\beta}\sim N_{p}(\boldsymbol\beta,\sigma^2({\bf X}^T{\bf X})^{-1})$.

* Show that $\hat{\boldsymbol\beta}$ has this distribution with the given mean and covariance matrix.
* What do you need to assume to get to this result?
* What does this imply for the distribution of the $j$th element of $\hat{\boldsymbol\beta}$?
* In particular, how can we calculate the variance of $\hat{\beta}_j$?

## b)
What is the interpretation of a 95% confidence interval? Hint: repeat experiment (on $Y$), on average how many CIs cover the true $\beta_j$? The following code shows an interpretation of a $95\%$ confidence interval. Study and fill in the code where is needed

* Model: $Y = 1 + 3X + \varepsilon$, with $\varepsilon \sim \mathsf{N}(0,1)$.

```python
beta0 = ...
beta1 = ...
true_beta = np.array([beta0, beta1])  # vector of model coefficients
true_sd = 1  # choosing true sd
nobs = 100

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=nobs)  # simulate the predictor variable X
df = pd.DataFrame({'x': X})
Xmat = MS(['x']).fit_transform(df)  # design matrix

# Count how many times the true value is within the confidence interval
ci_int = np.zeros(1000)
ci_x = np.zeros(1000)
nsim = 1000
for i in range(nsim):
    y = rng.normal(loc=Xmat.values @ true_beta, scale=true_sd, size=nobs)
    mod = sm.OLS(y, Xmat).fit()
    ci = mod.conf_int()  # array shape (2, 2): rows = params, cols = (low, high)

    # if true value of beta0 is within the CI then 1 else 0
    ci_int[i] = 1 if ... else 0

    # if true value of beta_1 is within the CI then 1 else 0
    ci_x[i] = 1 if ... else 0

np.array([ci_int.mean(), ci_x.mean()])
```


## c)
What is the interpretation of a 95% prediction interval? Hint: repeat experiment (on $Y$) for a given ${\boldsymbol x}_0$. Write `Python` code that shows the interpretation of a 95% PI. Hint: In order to produce the PIs use the data point $x_0 = 0.4.$ Furthermore, you may use a similar code structure as in b).

## d)
Construct a 95% CI for ${\boldsymbol x}_0^T \beta$. Explain the connections between a CI for $\beta_j$, a CI for ${\boldsymbol x}_0^T \beta$ and a PI for $Y$ at ${\boldsymbol x}_0$.

## e)
Explain the difference between _error_ and _residual_.  What are the properties of the raw residuals? Why don't we want to use the raw residuals for model check? What is our solution to this?
