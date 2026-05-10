---
title: "TMA4268 Statistical Learning"
subtitle: "Module 6: Solution sketches"
authors:
  - Sara Martino
  - Stefanie Muff
  - Kenneth Aase
affiliation: Department of Mathematical Sciences, NTNU
date: 2025-02-19
---

# 1

For the least square estimator, the solution can be found [here](https://en.wikipedia.org/wiki/Proofs_involving_ordinary_least_squares#Least_squares_estimator_for_%CE%B2).

For the maximum likelihood estimator, the solution can be found [here](https://en.wikipedia.org/wiki/Proofs_involving_ordinary_least_squares#Maximum_likelihood_approach).

# 2

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import seaborn as sns
from ISLP import load_data
```

```python
# Load Credit dataset
Credit = load_data('Credit')

# Check column names
Credit.columns

# Check dataset shape
Credit.shape
```

```python
Credit.head()
```

```python
# Select variables to plot
vars_ = ['Balance', 'Age', 'Cards', 'Education', 'Income', 'Limit', 'Rating']
pairwise_scatter_data = Credit[vars_]
```

```python
# Pairwise scatter plot
sns.pairplot(pairwise_scatter_data);
```

# 3

```python
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib.pyplot import subplots
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.api import OLS
from ISLP import load_data
from ISLP.models import ModelSpec as MS
```

```python
# Drop the 'ID' column (the ISLP version of Credit does not include ID by default,
# but we drop it if present to mirror the original exercise).
cred_data = Credit.drop(columns=['ID'], errors='ignore')

# Counting the dummy variables as well
cred_data_number_predictors = 11

# Take a look at the data
cred_data.head()

# Summary statistics
cred_data.describe(include='all')

# Build the design matrix (with intercept) and the response.
design = MS(cred_data.columns.drop('Balance')).fit(cred_data)
X_full = design.transform(cred_data)
Y = np.array(cred_data['Balance'])

# Train/test split (75/25), matching set.seed(1) idiomatically with random_state.
train_idx, test_idx = next(
    skm.ShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(cred_data)
)
cred_train = cred_data.iloc[train_idx]
cred_test = cred_data.iloc[test_idx]
X_train = X_full.iloc[train_idx]
X_test = X_full.iloc[test_idx]
Y_train = Y[train_idx]
Y_test = Y[test_idx]
```

```python
# Best subset selection: for each subset size k, fit OLS on every k-subset of
# predictors (excluding the intercept column) and keep the one with smallest RSS.
predictor_cols = [c for c in X_train.columns if c != 'intercept']
n_train = X_train.shape[0]

# Fit the full-model OLS once to estimate sigma^2 for C_p.
full_fit = OLS(Y_train, X_train).fit()
sigma2_hat = full_fit.scale  # MSE of the full model

best_by_size = {}  # k -> (rss, adj_r2, cp, bic, vars, coef, intercept)
for k in range(1, cred_data_number_predictors + 1):
    best_rss = np.inf
    best_entry = None
    for combo in combinations(predictor_cols, k):
        cols = ['intercept'] + list(combo)
        Xk = X_train[cols].values
        fit = OLS(Y_train, Xk).fit()
        rss = float(np.sum(fit.resid ** 2))
        if rss < best_rss:
            best_rss = rss
            adj_r2 = float(fit.rsquared_adj)
            # C_p = (RSS + 2 * d * sigma2_hat) / n   (d = number of predictors)
            cp = (rss + 2 * k * sigma2_hat) / n_train
            bic = float(fit.bic)
            best_entry = dict(rss=rss, adj_r2=adj_r2, cp=cp, bic=bic,
                              vars=combo, coef=fit.params, intercept=None)
    best_by_size[k] = best_entry

rss_arr = np.array([best_by_size[k]['rss'] for k in range(1, cred_data_number_predictors + 1)])
adjr2_arr = np.array([best_by_size[k]['adj_r2'] for k in range(1, cred_data_number_predictors + 1)])
cp_arr = np.array([best_by_size[k]['cp'] for k in range(1, cred_data_number_predictors + 1)])
bic_arr = np.array([best_by_size[k]['bic'] for k in range(1, cred_data_number_predictors + 1)])
```

```python
# Plot RSS, Adjusted R^2, C_p and BIC.
fig, axes = subplots(2, 2, figsize=(10, 8))
sizes = np.arange(1, cred_data_number_predictors + 1)

axes[0, 0].plot(sizes, rss_arr, '-')
axes[0, 0].set_xlabel('Number of Variables')
axes[0, 0].set_ylabel('RSS')

bsm_best_adjr2 = int(np.argmax(adjr2_arr)) + 1
axes[0, 1].plot(sizes, adjr2_arr, '-')
axes[0, 1].plot(bsm_best_adjr2, adjr2_arr[bsm_best_adjr2 - 1], 'ro', markersize=10)
axes[0, 1].set_xlabel('Number of Variables')
axes[0, 1].set_ylabel('Adjusted R^2')

bsm_best_cp = int(np.argmin(cp_arr)) + 1
axes[1, 0].plot(sizes, cp_arr, '-')
axes[1, 0].plot(bsm_best_cp, cp_arr[bsm_best_cp - 1], 'ro', markersize=10)
axes[1, 0].set_xlabel('Number of Variables')
axes[1, 0].set_ylabel('Cp')

bsm_best_bic = int(np.argmin(bic_arr)) + 1
axes[1, 1].plot(sizes, bic_arr, '-')
axes[1, 1].plot(bsm_best_bic, bic_arr[bsm_best_bic - 1], 'ro', markersize=10)
axes[1, 1].set_xlabel('Number of Variables')
axes[1, 1].set_ylabel('BIC');
```

```python
# 10-fold CV across model sizes.
# For each fold, redo best-subset selection on the training portion of the fold,
# then evaluate the size-k winner on the held-out fold.
K = 10
kfold = skm.KFold(n_splits=K, random_state=1, shuffle=True)

cv_errors = np.full((K, cred_data_number_predictors), np.nan)

for j, (tr_in, te_in) in enumerate(kfold.split(X_train)):
    X_tr = X_train.iloc[tr_in]
    Y_tr = Y_train[tr_in]
    X_te = X_train.iloc[te_in]
    Y_te = Y_train[te_in]

    for k in range(1, cred_data_number_predictors + 1):
        best_rss = np.inf
        best_combo = None
        best_coef = None
        for combo in combinations(predictor_cols, k):
            cols = ['intercept'] + list(combo)
            fit = OLS(Y_tr, X_tr[cols].values).fit()
            rss = float(np.sum(fit.resid ** 2))
            if rss < best_rss:
                best_rss = rss
                best_combo = combo
                best_coef = fit.params

        cols = ['intercept'] + list(best_combo)
        pred = X_te[cols].values @ best_coef
        cv_errors[j, k - 1] = float(np.mean((Y_te - pred) ** 2))

# Mean CV error per size
mean_cv_errors = cv_errors.mean(axis=0)
mean_cv_errors

# Plot the mean CV errors
fig, ax = subplots(figsize=(8, 5))
ax.plot(np.arange(1, cred_data_number_predictors + 1), mean_cv_errors, marker='o')
ax.set_xlabel('Number of Variables')
ax.set_ylabel('Mean CV error');
```

```python
# Fit the selected model on all training data and compute test error.
number_predictors_selected = 4

# Refit best-subset of size `number_predictors_selected` on the full training set
best_combo = best_by_size[number_predictors_selected]['vars']
cols = ['intercept'] + list(best_combo)
bsm_fit = OLS(Y_train, X_train[cols].values).fit()
print(bsm_fit.summary())

# Predict on the test set
bsm_preds = X_test[cols].values @ bsm_fit.params

# Test squared errors
bsm_squared_errors = (Y_test - bsm_preds) ** 2
squared_errors = pd.DataFrame({'bsm_squared_errors': bsm_squared_errors})

# Test MSE
bsm_squared_errors.mean()
```

# 4

The same pipeline as in Exercise 3 applies, only the search strategy changes. With ISLP we use `Stepwise` from `ISLP.models` together with `sklearn_selected` instead of a manual best-subset loop.

```python
from ISLP.models import Stepwise, sklearn_selected
from functools import partial

# Negative C_p scorer (sklearn maximises, so we negate).
def nCp(sigma2, estimator, X, Y):
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat) ** 2)
    return -(RSS + 2 * p * sigma2) / n

design = MS(cred_train.columns.drop('Balance')).fit(cred_train)
neg_Cp = partial(nCp, sigma2_hat)

# Forward stepwise.
strategy_fwd = Stepwise.first_peak(design,
                                   direction='forward',
                                   max_terms=len(design.terms))
hitters_fwd = sklearn_selected(OLS, strategy_fwd, scoring=neg_Cp)
hitters_fwd.fit(cred_train, Y_train)
hitters_fwd.selected_state_

# Backward stepwise.
strategy_bwd = Stepwise.first_peak(design,
                                   direction='backward',
                                   max_terms=len(design.terms))
hitters_bwd = sklearn_selected(OLS, strategy_bwd, scoring=neg_Cp)
hitters_bwd.fit(cred_train, Y_train)
hitters_bwd.selected_state_
```

Note: ISLP's `Stepwise` does not expose a "hybrid / sequential replacement" option equivalent to `regsubsets(..., method = "seqrep")` from R's `leaps`. Forward and backward as above cover the methods used in the lab; comparing their selected variable sets to those from best subset selection in Exercise 3 typically shows agreement on the smaller models and divergence on larger ones — best subset is allowed to swap variables in and out at each size, while forward/backward stepwise are constrained to grow/shrink monotonically.

# 5

```python
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.model_selection as skm

# Build numeric design matrices (drop the intercept column; the pipeline adds one).
D_train = X_train.drop(columns=['intercept']).values
D_test = X_test.drop(columns=['intercept']).values

# Grid of lambdas (sklearn calls them `alphas`).
lambdas = 10 ** np.linspace(8, -2, 100) / Y_train.std()

# 10-fold CV ridge via ElasticNetCV with l1_ratio=0, inside a StandardScaler pipeline.
kfold = skm.KFold(n_splits=10, random_state=1, shuffle=True)
scaler = StandardScaler(with_mean=True, with_std=True)
ridgeCV = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('ridge', ridgeCV)])
pipeCV.fit(D_train, Y_train)

tuned_ridge = pipeCV.named_steps['ridge']
best_lambda_ridge = tuned_ridge.alpha_
best_lambda_ridge

# CV-error plot.
ridge_fig, ax = subplots(figsize=(8, 5))
K = 10
ax.errorbar(-np.log(lambdas),
            tuned_ridge.mse_path_.mean(1),
            yerr=tuned_ridge.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_ridge.alpha_), c='k', ls='--')
ax.set_xlabel(r'$-\log(\lambda)$')
ax.set_ylabel('Cross-validated MSE');

# Predict on test set and store squared errors.
ridge_preds = pipeCV.predict(D_test)
ridge_square_errors = (ridge_preds - Y_test) ** 2
squared_errors['ridge_square_errors'] = ridge_square_errors
```

# 6

```python
# Lasso: same pipeline, l1_ratio=1.
lassoCV = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=kfold)
pipeCV = Pipeline(steps=[('scaler', scaler), ('lasso', lassoCV)])
pipeCV.fit(D_train, Y_train)

tuned_lasso = pipeCV.named_steps['lasso']
best_lambda_lasso = tuned_lasso.alpha_
best_lambda_lasso

# CV-error plot.
lasso_fig, ax = subplots(figsize=(8, 5))
ax.errorbar(-np.log(tuned_lasso.alphas_),
            tuned_lasso.mse_path_.mean(1),
            yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(K))
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
ax.set_xlabel(r'$-\log(\lambda)$')
ax.set_ylabel('Cross-validated MSE');

# Predict on test set and store squared errors.
lasso_preds = pipeCV.predict(D_test)
lasso_square_errors = (lasso_preds - Y_test) ** 2
squared_errors['lasso_square_errors'] = lasso_square_errors
```

# 7

```python
from sklearn.decomposition import PCA

# Use the full design matrix (no intercept) for PCA.
X_all = X_full.drop(columns=['intercept']).values
X_all_scaled = StandardScaler().fit_transform(X_all)

cred_pca = PCA().fit(X_all_scaled)

# Standard deviations of each PC and proportion of variance explained.
pc_sd = np.sqrt(cred_pca.explained_variance_)
pve = cred_pca.explained_variance_ratio_
cum_pve = np.cumsum(pve)
pd.DataFrame({'sd': pc_sd, 'PVE': pve, 'cumPVE': cum_pve},
             index=[f'PC{i}' for i in range(1, len(pve) + 1)])

# Scree plot.
fig, ax = subplots(figsize=(8, 5))
ax.plot(np.arange(1, len(pve) + 1), pve, marker='o')
ax.set_xlabel('Principal component')
ax.set_ylabel('Proportion of variance explained');
```

The first PC explains about $25\%$ of the variability in the data. Then the second PC explains an extra $15\%$ of the variability in the data. From the third PC until $8$th PC the extra variability explained per PC varies between $7.5\%$ to $10\%$, dropping to $3.6\%$ on the $9$th PC. So I would likely use $8$ PCs for the `Credit` dataset.

# 8

```python
# PCR via a Pipeline of StandardScaler -> PCA -> LinearRegression,
# with the number of components chosen by 10-fold CV.
pca = PCA()
linreg = skl.LinearRegression()
pipe = Pipeline([('scaler', scaler), ('pca', pca), ('linreg', linreg)])

param_grid = {'pca__n_components': range(1, cred_data_number_predictors + 1)}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(D_train, Y_train)

# Validation plot (CV MSE vs # components).
pcr_fig, ax = subplots(figsize=(8, 5))
n_comp = list(param_grid['pca__n_components'])
ax.errorbar(n_comp,
            -grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_xlabel('# principal components')
ax.set_ylabel('Cross-validated MSE');
```

```python
# Refit at M = 10 components (matching the R sketch) and evaluate on the test set.
pcr_best = Pipeline([('scaler', scaler),
                     ('pca', PCA(n_components=10)),
                     ('linreg', skl.LinearRegression())])
pcr_best.fit(D_train, Y_train)
pcr_preds = pcr_best.predict(D_test)
pcr_square_errors = (pcr_preds - Y_test) ** 2
squared_errors['pcr_square_errors'] = pcr_square_errors
pcr_square_errors.mean()
```

```python
# Boxplot of squared errors across methods so far.
fig, ax = subplots(figsize=(8, 5))
squared_errors.boxplot(ax=ax)
ax.set_ylabel('Squared error');
```

# 9

```python
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(scale=True)
param_grid = {'n_components': range(1, cred_data_number_predictors + 1)}
grid = skm.GridSearchCV(pls,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(D_train, Y_train)

pls_fig, ax = subplots(figsize=(8, 5))
n_comp = list(param_grid['n_components'])
ax.errorbar(n_comp,
            -grid.cv_results_['mean_test_score'],
            yerr=grid.cv_results_['std_test_score'] / np.sqrt(K))
ax.set_xlabel('# components')
ax.set_ylabel('Cross-validated MSE');
```

```python
# Refit at M = 3 components (matching the R sketch) and evaluate on the test set.
plsr_best = PLSRegression(n_components=3, scale=True)
plsr_best.fit(D_train, Y_train)
plsr_preds = plsr_best.predict(D_test).ravel()
plsr_square_errors = (plsr_preds - Y_test) ** 2
squared_errors['plsr_square_errors'] = plsr_square_errors
plsr_square_errors.mean()
```

```python
# Final boxplot and column means of squared errors.
fig, ax = subplots(figsize=(8, 5))
squared_errors.boxplot(ax=ax)
ax.set_ylabel('Squared error');

squared_errors.mean()
```


### Acknowledgements

This document was originally adapted from the R-based recommended exercises by Sara Martino, Stefanie Muff and Kenneth Aase (Department of Mathematical Sciences, NTNU).
