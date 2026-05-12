---
title: "Module 3: Recommended Exercises - Solution"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "January 29, 2025"
---

---


# Problem 1


## a)

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)

Auto = load_data('Auto')
Auto = Auto.drop(columns=['name'])
# Auto['origin'] = Auto['origin'].astype('category')
Auto.describe()
Auto.info()

Auto['origin'] = Auto['origin'].astype('category')
```

```python
import seaborn as sns
sns.pairplot(Auto, plot_kws={'s': 4}, height=1.2)
```

## b)

Correlation matrix for the `Auto` data set where we omit the column `origin`:

```python
Auto.drop(columns=['origin']).corr()
```

## c)

```python
terms = Auto.columns.drop('mpg')
X = MS(list(terms)).fit_transform(Auto)
y = Auto['mpg']
results = sm.OLS(y, X).fit()
summarize(results)
results.summary()
```

i. The F-statistic in the very last line of the summary output shows that it is extremely unlikely that we would see such a result if none of the variables had any explanatory power -- the $p$-value is $<2.2e-16$! So we can be sure that the set of predictors is explaining the response to a quite large degree. This is also confirmed by both a quite high $R^2$ and $R^2_{\text adjust}$ (both around 0.82).
ii. Yes, there is very strong evidence for a relationship between the weight of a car and the response. The $p$-value is extremely small. The interpretation is that, if a car weights 1000 kg more, we have a decrease of $1000 \cdot (-6.710\cdot 10^{-3}) = -6.71$ miles per gallon (mpg), that is, the car can drive 6.71 miles less far per gallon of fuel.
iii. The estimated coefficient $\hat{\beta}_\text{year}= 0.77$ suggests that a newer car model has higher `mpg` compared to an older one. A one year newer model should thus, in average, be able to drive $0.77$ miles longer per gallon fuel compared to the older car.


## d)

Remember that if we want to test whether a categorical variable with more than two levels should be removed from the model, we actually have to test whether all slope coefficients that are associated with the respective binary dummy variables are zero simultaneously. Look again at equation (3.30) in the course book (p.85).

Consequently, we have to test here whether $\beta_\text{origin2} = \beta_\text{origin3} = 0$ at the same time, and for this we need the $F$-test. We can fit the reduced model (without `origin`) and compare to the full model with `anova_lm()`:

```python
terms_no_origin = [t for t in terms if t != 'origin']
X_red = MS(terms_no_origin).fit_transform(Auto)
results_red = sm.OLS(y, X_red).fit()
anova_lm(results_red, results)
```

The $p$-value associated with adding `origin` to the model is very small, thus the origin of the car has an influence on the response `mpg`.

## e)

```python
fig, axes = subplots(2, 2, figsize=(9, 7))

# Residuals vs Fitted
axes[0, 0].scatter(results.fittedvalues, results.resid, s=8)
axes[0, 0].axhline(0, c='k', ls='--')
axes[0, 0].set_xlabel('Fitted value')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].set_title('Residuals vs Fitted')

# Normal QQ
from scipy import stats
infl = results.get_influence()
std_resid = infl.resid_studentized_internal
stats.probplot(std_resid, dist='norm', plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

# Scale-Location
axes[1, 0].scatter(results.fittedvalues, np.sqrt(np.abs(std_resid)), s=8)
axes[1, 0].set_xlabel('Fitted value')
axes[1, 0].set_ylabel(r'$\sqrt{|\mathrm{Standardized\ residuals}|}$')
axes[1, 0].set_title('Scale-Location')

# Residuals vs Leverage
axes[1, 1].scatter(infl.hat_matrix_diag, std_resid, s=8)
axes[1, 1].axhline(0, c='k', ls='--')
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized residual')
axes[1, 1].set_title('Residuals vs Leverage')

fig.tight_layout()

np.argmax(infl.hat_matrix_diag)
```

* In the residual vs fitted plot (the so-called Tukey-Anscombe plot) there is evidence of non-linearity.
* Observation $14$ has an unusually high leverage. This does not necessarily need to be a problem, but it would be wise to double-check that this observation is not an outlier.

## f)

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


```python
X1 = MS(['displacement', 'weight', 'year', 'origin', ('year', 'origin')]).fit_transform(Auto)
results1 = sm.OLS(y, X1).fit()
summarize(results1)
```

Again, if we want to find evidence whether the interaction term between `year` and `origin` is needed, we are actually testing whether two slope coefficients are zero at the same time ($\beta_\text{year:origin1} = \beta_\text{year:origin2} = 0$). Consequently, we need the $F$-test, comparing the model with and without the interaction:

```python
X1_main = MS(['displacement', 'weight', 'year', 'origin']).fit_transform(Auto)
results1_main = sm.OLS(y, X1_main).fit()
anova_lm(results1_main, results1)
```

There is very strong evidence that the year-effect depends on the origin of the car, as can be seen by the $F$-test that is given by the `anova_lm()` table ($p = 0.0001057$). For European (`2`) and Japanese (`3`) cars, it seems that the fuel consumption (`mpg`) has a steeper slope for `year`: $\hat{\beta}_{year} = 0.615$ for the reference category `1` (American), whereas $\hat{\beta}_\text{year} = 0.615 + 0.519$ and $\hat{\beta}_\text{year} = 0.615 + 0.356$ for category `2` (European) and `3` (Japanese), respectively. This means that the fuel consumption is reduced faster by cars produced outside America (note that `mgp` is "miles per gallon", thus a higher value means that the car consumes less fuel, as it can drive further per gallon).

Note: For a full understanding of interaction terms, you really do need both the `summarize()` and the `anova_lm()` tables.


## h)

We try three different transformations and look at the residual plots:

```python
# try 3 predictor transformations

Auto = Auto.assign(sqrtmpg=np.sqrt(Auto['mpg']),
                   logweight=np.log(Auto['weight']),
                   weight2=Auto['weight'] ** 2)

def resid_plot(res, title):
    fig, ax = subplots(figsize=(5, 4))
    ax.scatter(res.fittedvalues, res.resid, s=8)
    ax.axhline(0, c='k', ls='--')
    ax.set_xlabel('Fitted value'); ax.set_ylabel('Residual')
    ax.set_title(title)
    fig.tight_layout()

X3 = MS(['displacement', 'weight', 'year', 'origin']).fit_transform(Auto)
results3 = sm.OLS(Auto['sqrtmpg'], X3).fit()
resid_plot(results3, 'sqrt(mpg) ~ displacement + weight + year + origin')

X4 = MS(['displacement', 'logweight', 'year', 'origin']).fit_transform(Auto)
results4 = sm.OLS(Auto['mpg'], X4).fit()
resid_plot(results4, 'mpg ~ displacement + log(weight) + year + origin')

X5 = MS(['displacement', 'weight2', 'year', 'origin']).fit_transform(Auto)
results5 = sm.OLS(Auto['mpg'], X5).fit()
resid_plot(results5, 'mpg ~ displacement + weight^2 + year + origin')
```



# Problem 2

## a)

$$
\begin{aligned}
E(\hat{\boldsymbol\beta}) &= E((X^T X)^{-1}X^T Y) = (X^TX)^{-1}X^T E(Y) = (X^TX)^{-1}X^T E(X \boldsymbol\beta + \varepsilon) \\
&= (X^TX)^{-1}X^T (X \boldsymbol\beta + 0) = (X^TX)^{-1}(X^T X) \boldsymbol\beta = I \boldsymbol\beta = \boldsymbol\beta
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{Cov}(\hat{\boldsymbol\beta}) &= \mathrm{Cov}((X^T X)^{-1}X^T Y) = (X^TX)^{-1}X^T \, \mathrm{Cov}(Y) \, ((X^TX)^{-1}X^T)^T \\
&= (X^TX)^{-1}X^T \sigma^2 I ((X^TX)^{-1}X^T)^T \\
&= \sigma^2 (X^TX)^{-1}
\end{aligned}
$$

We need to assume that $Y$ is multivariate normal. As $\hat{\boldsymbol\beta}$ is a linear transformation of a multivariate normal vector $Y$, $\hat{\boldsymbol\beta}$ is also multivariate normal.

All components of a multivariate normal vector are themselves univariate normal. This means that $\hat{\beta}_j$ is normally distributed with expected value given by the $\beta_j$ and the variance given by the $j$'th diagonal element of $\sigma^2 (X^T X)^{-1}$.

## b)

Fix covariates `X` and repeat the following procedure many times (`nsim = 1000`)

* collect $Y$

* create CI using $\hat{\beta}$ and $\hat{\sigma}$

95 % of the times the CI contains the true $\beta.$

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS

# CI for beta_j
beta0 = 1
beta1 = 3
true_beta = np.array([beta0, beta1])  # vector of model coefficients
true_sd = 1  # choosing true sd
nobs = 100

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=nobs)
df = pd.DataFrame({'x': X})
Xmat = MS(['x']).fit_transform(df)  # design matrix

# Count how many times the true value is within the confidence interval
nsim = 1000
ci_int = np.zeros(nsim)
ci_x = np.zeros(nsim)
for i in range(nsim):
    y = rng.normal(loc=Xmat.values @ true_beta, scale=true_sd, size=nobs)
    mod = sm.OLS(y, Xmat).fit()
    ci = mod.conf_int()  # rows = (intercept, x), cols = (low, high)
    ci_int[i] = 1 if (true_beta[0] >= ci[0, 0] and true_beta[0] <= ci[0, 1]) else 0
    ci_x[i] = 1 if (true_beta[1] >= ci[1, 0] and true_beta[1] <= ci[1, 1]) else 0

np.array([ci_int.mean(), ci_x.mean()])
```

## c)

We apply the same idea from b), but now we fix X and $x_0$ and

* collect $Y$

* create PI using $\hat{\beta}$ and $\hat{\sigma}$

* simulate $Y_0$

95 % of the times the PI contains $Y_0.$

```python
# PI for Y_0
beta0 = 1
beta1 = 3
true_beta = np.array([beta0, beta1])  # vector of model coefficients
true_sd = 1  # choosing true sd

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, size=100)
df = pd.DataFrame({'x': X})
design = MS(['x']).fit(df)
Xmat = design.transform(df)  # design matrix

x0 = np.array([1.0, 0.4])
new_df = pd.DataFrame({'x': [x0[1]]})
newX = design.transform(new_df)

# simulating and fitting models many times
nsim = 1000
pi_y0 = np.zeros(nsim)
for i in range(nsim):
    y = rng.normal(loc=Xmat.values @ true_beta, scale=true_sd, size=100)
    mod = sm.OLS(y, Xmat).fit()
    y0 = rng.normal(loc=x0 @ true_beta, scale=true_sd)
    pi = mod.get_prediction(newX).conf_int(obs=True, alpha=0.05)[0]
    pi_y0[i] = 1 if (y0 >= pi[0] and y0 <= pi[1]) else 0

pi_y0.mean()
```

## d)


### Confidence Interval for ${\boldsymbol x}_0^T\boldsymbol{\beta}$

This corresponds to a CI for $\mu_0 = E(y_0) = \boldsymbol{x}_0^T\boldsymbol{\beta}$ of an observation $y_0$ at $\boldsymbol{x}_0.$

First, by using a) we have that

$E(\boldsymbol{x}_0^T\hat{\boldsymbol{\beta}}) = \boldsymbol{x}_0^T E(\hat{\boldsymbol{\beta}}) = \boldsymbol{x}_0^T \boldsymbol{\beta}$

and

$\operatorname{Var}(\boldsymbol{x}_0^T\hat{\boldsymbol{\beta}}) = \boldsymbol{x}_0^T \operatorname{Var}(\hat{\boldsymbol{\beta}}) \boldsymbol{x}_0 = \sigma^2\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0.$

Consequently,

$$
\boldsymbol {x}_0^T\hat{\boldsymbol{\beta}} \sim N(\boldsymbol{x}_0^T\boldsymbol{\beta}, \sigma^2\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0)
$$
or equivalently

$$
\frac{\boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}-\mu_0}{\sigma\sqrt{\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0}} \sim N(0, 1).
$$

By substituting $\sigma^2$ with an estimator $\hat{\sigma ^2},$ the last expression follows a $t-$distribution with $n-p$ degrees of freedom.
Now we can construct a confidence interval with $1-\alpha$ level (in our case $\alpha = 0.05$) as follows

$$
P\Big( -t_{n-p}(1-\alpha/2)\le
\frac{\boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}-\mu_0}{\hat{\sigma}\sqrt{\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0}}
\le t_{n-p}(1-\alpha/2) \Big)
= 1-\alpha
$$
or

$$
P\Big( \boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}-t_{n-p}(1-\alpha/2)\hat{\sigma}\sqrt{\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0}\le
\mu_0
\le \boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}+t_{n-p}(1-\alpha/2)\hat{\sigma}\sqrt{\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0} \Big)
= 1-\alpha
$$

For $\alpha = 0.05$ we have the following $95\%$ CI

$$
\Big[ \boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}-t_{n-p}(1-0.05/2)\hat{\sigma}\sqrt{\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0},
\boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}+t_{n-p}(1-0.05/2)\hat{\sigma}\sqrt{\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0} \Big]
$$

### Prediction interval

To construct the PI we can look at the distribution of the prediction error $\hat{\varepsilon}_0 = y_0-\hat{y}_0 = y_0 - {\boldsymbol x}_0^T\boldsymbol{\beta}.$

We have that
$$
E(\hat{\varepsilon}_0) =y_0 - E(\hat{y}_0) = y_0 - E(\boldsymbol{x}_0^T\hat{\boldsymbol{\beta}}) = y_0-y_0 = 0
$$

and by assuming that $y_0$ and $\hat{y_0}$ are independent and $y_0\sim N(\boldsymbol{x}_0^T \boldsymbol{\beta},\sigma^2)$
we have that

$$
\operatorname{Var}(\hat{\varepsilon}_0) = \operatorname{Var}(y_0)  + \operatorname{Var}(\hat{y}_0)
=  \sigma^2 + \sigma^2\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0
= \sigma^2(1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0)
$$

and

$$
\hat{\varepsilon}_0 = y_0 - \boldsymbol{x}_0^T\hat{\boldsymbol{\beta}}\sim N(0, \sigma^2(1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0)).
$$

Following the same logic as for the before and by substituting  the estimate of variance $\hat{\sigma ^2}$ we can construct a prediction interval with $1-\alpha$ level as follows

$$
P\Big( -t_{n-p}(1-\alpha/2)\le
\frac{y_0-\boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}}{\hat{\sigma}\sqrt{1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0}}
\le t_{n-p}(1-\alpha/2) \Big)
= 1-\alpha
$$

or

$$
P\Big( \boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}-t_{n-p}(1-\alpha/2)\hat{\sigma}\sqrt{1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0}\le
y_0
\le \boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}+t_{n-p}(1-\alpha/2)\hat{\sigma}\sqrt{1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0} \Big)
= 1-\alpha
$$

For $\alpha = 0.05$ we have the following $95\%$ PI

$$
\Big[ \boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}-t_{n-p}(1-0.05/2)\hat{\sigma}\sqrt{1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0},
\boldsymbol {x}_0^T\hat{\boldsymbol{\beta}}+t_{n-p}(1-0.05/2)\hat{\sigma}\sqrt{1+\boldsymbol{x}_0^T(X^TX)^{-1} \boldsymbol{x}_0} \Big]
$$

Observe now that the two intervals are similar, but the prediction interval is by construction wider. Specifically, it can be significantly wider in applications where the error variance $\hat{\sigma ^2}$ is large.

The connection between CI for $\boldsymbol{\beta}$, ${\boldsymbol x}_0^T\boldsymbol{\beta}$ and PI for $y$ at ${\boldsymbol x}_0$: The first is CI for a parameter, the second is CI for the expected regression line at the point $\boldsymbol{x}_0$ (when you only have one covariate, this may be more intuitive), and the last is the PI for the response $y_0$. The difference between the two latter is that $y$ are the observations, and ${\boldsymbol x}_0^T\boldsymbol{\beta}$ is the expected value of the observations and hence a function of the model parameters (NOT an observation).

## e)

We have a model on the form $Y=X \beta + \varepsilon$ where $\varepsilon$ is the error. The error of the model is unknown and unobserved, but we can estimate it by what we call the residuals. The residuals are given by the difference between the true response and the predicted value
$$\hat{\varepsilon}=Y-\hat{Y}=(I-\underbrace{X(X^TX)^{-1}X^\top}_{=H})Y ,$$
where $H$ is called the "hat matrix".

Properties of raw residuals: Normally distributed with mean 0 and covariance $Cov(\hat{\varepsilon})=\sigma^2 (I-H).$ This means that the residuals may have different variance (depending on $X$) and may also be correlated.

In a model check, we want to check that our errors are independent, homoscedastic (same variance for each observation) and not dependent on the covariates. As we don't know the true error, we use the residuals as predictors, but as mentioned, the residuals may have different variances and may be correlated. This is why we don't want to use the raw residuals for model check.

To amend our problem we need to try to fix the residuals so that they at least have equal variances. We do that by working with standardized or studentized residuals.
