---
title: "Module 8: Solutions to Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "March 12, 2025"
---

---


## Problem 1 -- Theoretical

a)

1. _Recursive binary splitting_: We find the best single partitioning of the data such that the reduction of RSS is the greatest. This process is applied sequencially to each of the split parts until a predefined minimum number of leave observation is reached.

2. _Cost complexity pruning_ of the large tree from previoius step, in order to obtain a sequence of best trees as a function of a parameter $\alpha$. Each value of $\alpha$ corresponds to a subtree that minimize the following equation (several $\alpha$s for the same tree):

$$
\sum_{m=1}^{|T|}\sum_{i:x_i\in R_m}(y_i - \hat y_{R_m})^2 + \alpha |T|,
$$

where $|T|$ is the number of terminal nodes.

3. _$K$-fold cross-validation_ to choose $\alpha$. For each fold:

* Repeat Steps 1 and 2 on all but the kth folds of the training data.
* Evaluate the mean squared prediction on the data in the left-out kth fold, as a function of $\alpha$.
* Average the results for each value of $\alpha$ and choose $\alpha$ to minimize the average error.

4. Return the subtree from Step 2 that corresponds to the chosen value of $\alpha.$

For a **classification** tree, we replace RSS with Gini index or cross entropy.

b)

$\color{green}{\text{Advantages}}$

* Very easy to explain

* Can be displayed graphically

* Can handle both quantitative and qualitative predictors without the need to create dummy variables

$\color{red}{\text{Disadvantages}}$

<!-- * The predictive accuracy is usually not very high -->
* The predictive accuracy is usually not very high. This limitation often leads practitioners to use ensemble methods like Random Forests or Boosting to improve accuracy.

* They are non-robust. That is, a small change in the data can cause a large change in the estimated tree.


c)

*Limitation of decision trees*:
Decision trees suffer from high variance. Recall that if we have $B$ $i.i.d$ observations of a random variable $X$ with the same mean and variance $\sigma^2.$
We calculate the mean $\bar{X} = \frac{1}{B} \sum_{b=1}^B X_b,$ and the variance of the mean is $\text{Var}(\bar{X}) = \frac{\sigma^2}{B}.$
That is by averaging we get reduced variance.

*Idea behind bagging*:
For decision trees, if we have $B$ training  sets, we could estimate $\hat{f}_1({\boldsymbol x}),\hat{f}_2({\boldsymbol x}),\ldots, \hat{f}_B({\boldsymbol x})$ and average them as
$$
\hat{f}_{avg}({\boldsymbol x})=\frac{1}{B}\sum_{b=1}^B \hat{f}_b({\boldsymbol x}) \ .
$$
However we do not have many data independent data sets, and we bootstraping to create $B$ datasets. These datasets are however not completely independent and the reduction in variance is therefore not as large as for independent training sets.

*How do random forests improve that idea?*:
To make the different trees that are built from each bootstrapped dataset more different, random forests use a random subset of the predictors to split the tree into new branches at each step. This decorrelates the different trees that are built from the $B$ bootstrapped datasets, and consequently reduces variance.

d) An OOB is the set of observations that were not chosen to be in a specific bootstrap sample. From RecEx5-Problem 4c we have that on average $1-0.632 = 0.368$ are included in the OOB sample.

e)

**Variable importance based on node impurity**

- *Regression Trees*: The total amount that the RSS is decreased due to splits of each predictor, averaged over the B trees.
- *Classification Trees*: The importance is the mean decrease (over all B trees) in impurity (often measured by the Gini index) by splits of a predictor. A higher decrease indicates a more significant role of the predictor in partitioning the data into homogeneous groups, thereby enhancing classification accuracy.

**Variable importance based on randomization**

- This measure is based on how much the predictive accuracy (MSE or gini indiex) is decreased when the variable is replaced by a permuted version of it. You find a drawing [here](https://github.com/stefaniemuff/statlearning/blob/master/8Trees/M8_variableImportanceRandomization.pdf).

## Problem 2 -- Regression (Book Ex. 8)


a)


```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from ISLP import load_data
from ISLP.models import ModelSpec as MS
import sklearn.model_selection as skm

Carseats = load_data('Carseats')
model = MS(Carseats.columns.drop('Sales'), intercept=False)
D = model.fit_transform(Carseats)
feature_names = list(D.columns)
X = np.asarray(D)
y = Carseats['Sales']

(X_train,
 X_test,
 y_train,
 y_test) = skm.train_test_split(X,
                                y,
                                test_size=0.3,
                                random_state=4268)
```

b)

```python
from sklearn.tree import DecisionTreeRegressor as DTR, plot_tree

reg = DTR(random_state=4268)
reg.fit(X_train, y_train)
print("n_leaves:", reg.tree_.n_leaves)
print("training MSE:", np.mean((y_train - reg.predict(X_train))**2))
```

```python
ax = subplots(figsize=(12, 10))[1]
plot_tree(reg,
          feature_names=feature_names,
          ax=ax);
```

- Shelf location ('ShelveLoc') is should be one of the important predictors, with bad/medium locations correlating with lower sales while good location leads to higher sales in general.
<!-- - Price is the primary splitting criterion, indicating it's a significant predictor. -->
<!-- - Lower prices generally lead to lower predicted values, implying a potential increase in car seat sales or a related outcome at lower prices. -->
- Age criteria suggests that younger customers tend to contribue higher sales.
- For higher prices, Advertising becomes relevant, indicating that higher marketing efforts result in higher sales.
- etc.

```python
yhat = reg.predict(X_test)
mse = np.mean((y_test - yhat)**2)
print("test MSE:", mse)
```

c)

```python
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(10, shuffle=True, random_state=4268)
grid = skm.GridSearchCV(reg,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)

cv_results = pd.DataFrame({
    'ccp_alpha': G.cv_results_['param_ccp_alpha'].data.astype(float),
    'mean_neg_mse': G.cv_results_['mean_test_score']
}).sort_values('ccp_alpha')

ax = subplots(figsize=(8, 5))[1]
ax.plot(cv_results['ccp_alpha'], -cv_results['mean_neg_mse'], marker='o')
ax.set_xlabel(r'$\alpha$ (ccp_alpha)')
ax.set_ylabel('CV MSE')
ax.axvline(G.best_params_['ccp_alpha'], color='red', linestyle='--');
```

The cross-validation curve picks an $\alpha$ that prunes the full tree to a moderately sized subtree. We refit and inspect the pruned tree.

```python
best_ = G.best_estimator_
print("n_leaves of pruned tree:", best_.tree_.n_leaves)

ax = subplots(figsize=(12, 10))[1]
plot_tree(best_,
          feature_names=feature_names,
          ax=ax);
```


```python
yhat = best_.predict(X_test)
mse = np.mean((y_test - yhat)**2)
mse
```

There is a slight reduction in MSE for the pruned tree.

d)

```python
from sklearn.ensemble import RandomForestRegressor as RF

print(Carseats.shape)
bag_Carseats = RF(max_features=X_train.shape[1],
                  n_estimators=500,
                  random_state=4268).fit(X_train, y_train)
yhat_bag = bag_Carseats.predict(X_test)
mse_bag = np.mean((y_test - yhat_bag)**2)
mse_bag
```

Bagging decreases the test MSE significantly. From the importance plot we might conclude that `Price` and `ShelveLoc` are the most important variables.

```python
feature_imp = pd.DataFrame(
    {'importance': bag_Carseats.feature_importances_},
    index=feature_names).sort_values(by='importance', ascending=False)
print(feature_imp)

ax = subplots(figsize=(8, 5))[1]
feature_imp.sort_values(by='importance').plot.barh(ax=ax, legend=False)
ax.set_xlabel('Feature importance');
```



e)

```python
rf_Carseats = RF(max_features=3,
                 n_estimators=500,
                 random_state=4268).fit(X_train, y_train)
yhat_rf = rf_Carseats.predict(X_test)
mse_forest = np.mean((y_test - yhat_rf)**2)
mse_forest
```

We use $p/3 = 10/3 \approx 3$ predictors at each split, and we obtain an MSE that is comparable to or slightly larger than the bagging MSE. The two most important variables are again `Price` and `ShelveLoc`.

```python
feature_imp = pd.DataFrame(
    {'importance': rf_Carseats.feature_importances_},
    index=feature_names).sort_values(by='importance', ascending=False)
print(feature_imp)

ax = subplots(figsize=(8, 5))[1]
feature_imp.sort_values(by='importance').plot.barh(ax=ax, legend=False)
ax.set_xlabel('Feature importance');
```

f)

```python
np.random.seed(123)

n_trees = 500
bag_Car = RF(max_features=X_train.shape[1],
             n_estimators=n_trees,
             oob_score=False,
             random_state=123).fit(X_train, y_train)
rf_Car = RF(max_features=3,
            n_estimators=n_trees,
            oob_score=False,
            random_state=123).fit(X_train, y_train)

# Compute the test MSE as a function of the number of trees by averaging
# predictions from the first k trees for k = 1, ..., n_trees.
def staged_test_mse(forest, X_test, y_test):
    preds = np.array([est.predict(X_test) for est in forest.estimators_])
    cum_mean = np.cumsum(preds, axis=0) / np.arange(1, len(preds) + 1)[:, None]
    return np.mean((cum_mean - y_test.values)**2, axis=1)

mse_bag_path = staged_test_mse(bag_Car, X_test, y_test)
mse_rf_path = staged_test_mse(rf_Car, X_test, y_test)

ax = subplots(figsize=(7, 5))[1]
ax.plot(np.arange(1, n_trees + 1), mse_bag_path, color='blue', label='m = p')
ax.plot(np.arange(1, n_trees + 1), mse_rf_path, color='green', label='m = p/3')
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Test MSE')
ax.set_ylim(2, 2.8)
ax.legend();
```

NB! Typically, RandomForest models outperform Bagging models due to their decorrealtion property stemming from a limited selection of predictors.


## Problem 3 -- Classification

```python
from sklearn.datasets import fetch_openml
spam = fetch_openml('spambase', version=1, as_frame=True)
X_full = spam.data
y_full = spam.target  # '1' = spam, '0' = nonspam
feature_names = list(X_full.columns)
```


a)

```python
print(spam.DESCR)
```

b)

```python
import sklearn.model_selection as skm

(X_train,
 X_test,
 y_train,
 y_test) = skm.train_test_split(X_full,
                                y_full,
                                test_size=0.3,
                                random_state=4268)
```

c)

```python
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree
from matplotlib.pyplot import subplots

spam_tree = DTC(criterion='entropy', random_state=4268)
spam_tree.fit(X_train, y_train)

ax = subplots(figsize=(14, 10))[1]
plot_tree(spam_tree,
          feature_names=feature_names,
          class_names=['nonspam', 'spam'],
          ax=ax);

print("n_leaves:", spam_tree.tree_.n_leaves)
print("training accuracy:", spam_tree.score(X_train, y_train))
```

d)

```python
from sklearn.metrics import accuracy_score
from ISLP import confusion_table

yhat = spam_tree.predict(X_test)
print(confusion_table(yhat, y_test))

misclass_rate = 1 - accuracy_score(y_test, yhat)
misclass_rate
```

e)

```python
ccp_path = spam_tree.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(10, shuffle=True, random_state=4268)
grid = skm.GridSearchCV(spam_tree,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='accuracy')
G = grid.fit(X_train, y_train)

cv_results = pd.DataFrame({
    'ccp_alpha': G.cv_results_['param_ccp_alpha'].data.astype(float),
    'mean_acc': G.cv_results_['mean_test_score']
}).sort_values('ccp_alpha')

ax = subplots(figsize=(8, 5))[1]
ax.plot(cv_results['ccp_alpha'], 1 - cv_results['mean_acc'], marker='o')
ax.set_xlabel(r'$\alpha$ (ccp_alpha)')
ax.set_ylabel('CV misclassification rate')
ax.axvline(G.best_params_['ccp_alpha'], color='red', linestyle='--');
```

According to the cross-validation curve we pick the $\alpha$ that minimises the misclassification rate; this corresponds to a moderately small tree. We refit and plot the pruned tree.

```python
prune_spam = G.best_estimator_
print("n_leaves of pruned tree:", prune_spam.tree_.n_leaves)

ax = subplots(figsize=(12, 8))[1]
plot_tree(prune_spam,
          feature_names=feature_names,
          class_names=['nonspam', 'spam'],
          ax=ax);
```

We predict the response for the test data:

```python
yhat_prune = prune_spam.predict(X_test)
print(confusion_table(yhat_prune, y_test))
```

The misclassification rate is

```python
1 - accuracy_score(y_test, yhat_prune)
```

f)

```python
from sklearn.ensemble import RandomForestClassifier as RFC

bag_spam = RFC(max_features=X_train.shape[1],
               n_estimators=500,
               random_state=4268).fit(X_train, y_train)
```


We predict the response for the test data as before:

```python
yhat_bag = bag_spam.predict(X_test)
print(confusion_table(yhat_bag, y_test))
```

The misclassification rate is

```python
1 - accuracy_score(y_test, yhat_bag)
```

g)

We now use the random forest-algorithm and consider only $\sqrt{57}\approx 8$ of the predictors at each split. This is specified through `max_features`.

```python
rf_spam = RFC(max_features=int(round(np.sqrt(X_train.shape[1]))),
              n_estimators=500,
              random_state=4268).fit(X_train, y_train)
```

We study the importance of each variable

```python
feature_imp = pd.DataFrame(
    {'importance': rf_spam.feature_importances_},
    index=feature_names).sort_values(by='importance', ascending=False)
feature_imp.head(10)
```

If the importance is large, the corresponding covariate is important.

```python
ax = subplots(figsize=(8, 10))[1]
feature_imp.sort_values(by='importance').plot.barh(ax=ax, legend=False)
ax.set_xlabel('Feature importance');
```

In this plot we see that `char_freq_!` (the frequency of the `!` character) is among the most important covariates, together with `char_freq_$` and `word_freq_remove`. This is as expected as these variables are used in the top splits in the classification trees we have seen so far.

We now predict the response for the test data.

```python
yhat_rf = rf_spam.predict(X_test)
print(confusion_table(yhat_rf, y_test))
1 - accuracy_score(y_test, yhat_rf)
```

The misclassification rate is given by

```python
1 - accuracy_score(y_test, yhat_rf)
```

h)
The ranks of misclassification rates: 1) random forest, 2) bagging, 3) decision tree, with a higher rank indicating a lower misclassification rate.
This order is what we expected.


### Acknowledgements

This document was originally adapted from the R-based recommended exercises by Sara Martino, Stefanie Muff and Kenneth Aase (Department of Mathematical Sciences, NTNU).
