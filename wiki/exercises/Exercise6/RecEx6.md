---
title: "TMA4268 Statistical Learning"
subtitle: "Module 6: Recommended exercises"
authors:
  - Sara Martino
  - Stefanie Muff
  - Kenneth Aase
affiliation: Department of Mathematical Sciences, NTNU
date: 2025-02-19
---

# 1

a. Show that the least square estimator of a multiple linear regression model is given by
$$
\hat{\boldsymbol \beta} =(\boldsymbol X^T \boldsymbol X)^{-1} \boldsymbol X^T \boldsymbol Y
$$
b. Show that the maximum likelihood estimator is equal to the least square estimator for the multiple linear regression model.

# 2

Write Python code to create a similar representation of the `Credit` data set in the `ISLP` package, as in the figure shown below.

![Credit data figure](credit_card_data.png){width=50%}

# 3

a. For the Credit dataset, pick the best model using Best Subset Selection according to $C_p$, $BIC$ and Adjusted $R^2$.
    + Hint: ISLP does not ship a single `regsubsets`-style function. Implement best subset selection manually using `itertools.combinations` over the columns of the design matrix and `sklearn.linear_model.LinearRegression` (or `statsmodels.OLS` if you want $C_p$/$BIC$/Adjusted $R^2$ for free).
b. For the Credit dataset, pick the best model using Best Subset Selection according to a $10$-fold CV.
    + Hint: Reuse the per-size best subsets from the previous step and write your own CV loop using `sklearn.model_selection.KFold`.
c. Compare the result obtained in Step 1 and Step 2.

# 4

a. Select the best model for the Credit Data using Forward and Backward Stepwise Selection.
    + Hint: Use `Stepwise` from `ISLP.models` together with `sklearn_selected` (see ISLP ch.6 lab).
b. Compare with the results obtained with Best Subset Selection.

# 5

a. Apply Ridge regression to the Credit dataset.
b. Compare the results with the standard linear regression.

# 6

a. Apply Lasso regression to the Credit dataset.
b. Compare the results with the standard linear regression and the Ridge regression.

# 7

How many principal components should we use for the Credit dataset? Justify.

# 8

Apply PCR on the Credit dataset and compare the results with the previous methods used in this module.

# 9

Apply PLS on the Credit dataset and compare the results with the previous methods used in this module.


### Acknowledgements

This document was originally adapted from the R-based recommended exercises by Sara Martino, Stefanie Muff and Kenneth Aase (Department of Mathematical Sciences, NTNU).
