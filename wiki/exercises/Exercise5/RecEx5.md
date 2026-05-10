---
title: "Module 5: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "February 12, 2025"
---

---

**We strongly recommend you to work through Section 5.3 in the course book (Lab: Cross-Validation and the Bootstrap)**

---

## Problem 1

Explain how $k$-fold cross-validation is implemented.

a) Draw a figure.

b) Specify algorithmically what is done, and in particular how the results from each fold are aggregated.

c) Relate $k$-fold cross-validation to an example from regression. Ideas are the complexity with regards to polynomials of increasing degree in multiple linear regression, or $K$ in KNN-regression.

d) Relate $k$-fold cross-validation to an example from classification. Ideas are the complexity with regards to polynomials of increasing degree in logistic regression, or $K$ in KNN-classification.

Hint: the words "loss function," "fold," "training," and "validation" are central.

## Problem 2

- a) What are the advantages and disadvantages of $k$-fold cross-validation relative to
  - i) The validation set approach
  - ii) Leave one out cross-validation (LOOCV)
- b) What are recommended values for $k$, and why?

Hint: the words "bias," "variance" and "computational complexity" should be included.


## Problem 3

Topic: Selection bias and the "wrong way to do CV."

The task here is to devise an algorithm to show that the wrong way is wrong and that the right way is right.

 a) What are the steps of such an algorithm? Write down a suggestion. Hint: How do you generate data for predictors and class labels, how do you do the classification task, where is the CV in the correct way and wrong way inserted into your algorithm? Can you make a schematic drawing of the right and the wrong way?

  b) We are now doing a simulation to illustrate the selection bias problem in CV, when it is applied the wrong way. Here is what we are (conceptually) going to do:

  Generate data

* Simulate high dimensional data ($p = 5000$ predictors) from independent or correlated normal variables, but with few samples ($n = 100$).

* Randomly assign class labels (here only $2$). This means that the "truth" is that the misclassification rate can not get very small. What is the expected misclassification rate (for this random set)?

Classification task:

* We choose a few (for example $d = 10$) of the predictors (those with the highest correlation to the outcome).
* Perform a classification rule (here: logistic empirical Bayes) on these predictors.
* Then we run CV ($k = 5$) on either only the $d$ (the wrong way), or on all $c + d$ predictors (the right way).
* Report misclassification errors for both situations.

One possible version of this is presented in the Python code below. Go through the code and explain what is done in each step, then run the code and observe if the results are in agreement with what you expected. Make changes to the code if you want to test out different strategies.

We start by generating data for $n=100$ observations.

```python
import numpy as np
import pandas as pd

# GENERATE DATA; use a seed for reproducibility
rng = np.random.default_rng(4268)
n = 100   # Number of observations
p = 5000  # Number of predictors
d = 10    # Top correlated predictors chosen

# Generating predictor data
xs = rng.normal(0, 1, size=(n, p))  # uncorrelated predictors
print(xs.shape)  # n times p
print(xs[:10, :10])

# Generate class labels independent of predictors
# - so if all classifies as class 1 we expect 50% errors in general
ys = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)  # Now really 50% of each
print(pd.Series(ys).value_counts())
```

**WRONG CV**: Select the 10 most correlated predictors outside the CV.

```python
# Correlation of each predictor with ys
corrs = np.array([np.corrcoef(xs[:, j], ys)[0, 1] for j in range(p)])

import matplotlib.pyplot as plt
plt.hist(corrs, bins=30)
plt.xlabel("correlation")
plt.show()

selected = np.argsort(-(corrs ** 2))[:d]

data = pd.DataFrame(xs[:, selected],
                    columns=[f"X{j}" for j in selected])
data["ys"] = ys
```

Then run CV around the fitting of the classifier. We use logistic regression and a `KFold` with the `cross_validate` helper, scoring by misclassification rate.

```python
import statsmodels.api as sm
from ISLP.models import sklearn_sm
from sklearn.model_selection import KFold, cross_validate

kfold = 10
X = data.drop(columns=["ys"]).to_numpy()
X = sm.add_constant(X)
Y = data["ys"].to_numpy()

logfit = sklearn_sm(sm.GLM,
                    model_args={"family": sm.families.Binomial()})

cv = KFold(n_splits=kfold, shuffle=True, random_state=0)
cv_results = cross_validate(logfit, X, Y,
                            cv=cv,
                            scoring="accuracy")
1 - cv_results["test_score"].mean()  # misclassification rate
```

Observe a misclassification rate of about 20%.

**CORRECT CV**: Do not pre-select predictors outside the CV, but as part of the CV. In other words, the entire analysis is done within the CV. We need to code this ourselves:

```python
from sklearn.model_selection import KFold

reorder = rng.permutation(n)
xs_r = xs[reorder]
ys_r = ys[reorder]

validclass = np.empty(n, dtype=int)
cv = KFold(n_splits=kfold, shuffle=False)
for train_idx, valid_idx in cv.split(xs_r):
    Xtr, Xva = xs_r[train_idx], xs_r[valid_idx]
    ytr, yva = ys_r[train_idx], ys_r[valid_idx]

    # Compute correlations only on the training fold
    foldcorrs = np.array([np.corrcoef(Xtr[:, j], ytr)[0, 1]
                          for j in range(p)])
    # Select top d correlated predictors
    selected = np.argsort(-(foldcorrs ** 2))[:d]

    Xtr_sel = sm.add_constant(Xtr[:, selected])
    Xva_sel = sm.add_constant(Xva[:, selected], has_constant="add")

    trainlogfit = sm.GLM(ytr, Xtr_sel,
                         family=sm.families.Binomial()).fit()
    pred = trainlogfit.predict(Xva_sel)
    validclass[valid_idx] = (pred > 0.5).astype(int)

tab = pd.crosstab(ys_r, validclass)
print(tab)
1 - np.trace(tab.values) / n
```

## Problem 4

We will calculate the probability that a given observation in our original sample is part of a bootstrap sample. This is useful for us to know in Module 8.

Our sample size is $n$.

a. We draw one observation from our sample. What is the probability of drawing observation $i$ (i.e., $x_i$)? And of not drawing observation $i$?
b. We make $n$ independent draws (with replacement). What is the probability of not drawing observation $i$ in any of the $n$ draws? What is then the probability that data point $i$ is in our bootstrap sample (that is, more than $0$ times)?
c. When $n$ is large $(1-\frac{1}{n})^n \approx \frac{1}{e}$. Use this to give a numerical value for the probability that a specific observation $i$ is in our bootstrap sample.
d. Write a short Python code chunk to check your result. (Hint: An example on how to do this is on page 219 in the ISLP book — Exercise 2(h) of Chapter 5.) You may also study the result in c. How good is the approximation as a function of $n$?

```python
import numpy as np

rng = np.random.default_rng(0)
n = 100
B = 10000
j = 0  # observation index (0-based)
res = np.empty(B)
for b in range(B):
    res[b] = np.sum(rng.choice(n, size=n, replace=True) == j) > 0
res.mean()
```

## Problem 5

Explain with words and an algorithm how you would proceed to use bootstrapping to estimate the standard deviation and the $95\%$ confidence interval of one of the regression parameters in multiple linear regression. Comment on which assumptions you make for your regression model.

## Problem 6

Implement your algorithm from 5 both using a `for`-loop and using the `boot_SE()` helper from the ISLP lab. Hint: See section 5.3.4 (the Lab on the bootstrap) in our ISLP book. Use the SLID data set and provide standard errors for the coefficient for `age`. Compare with the theoretical value $({\bf X}^\top{\bf X})^{-1}\hat{\sigma}^2$ that you find in the output from the regression model.

The SLID data set is from the `carData` R package; for this Python translation we load it directly from the `carData` mirror on the `vincentarelbundock/Rdatasets` repo.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS, summarize

SLID = pd.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/carData/SLID.csv",
    index_col=0,
).dropna()
n = SLID.shape[0]

design = MS(["education", "age", "sex", "language"])
X = design.fit_transform(SLID)
y = SLID["wages"]
SLID_lm = sm.OLS(y, X).fit()
summarize(SLID_lm).loc["age"]
```

Now go ahead and use bootstrap to estimate the $95\%$ CI. Compare your result to the confidence intervals from the linear model fit:

```python
SLID_lm.conf_int()
```
