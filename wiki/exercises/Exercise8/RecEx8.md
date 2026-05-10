---
title: "Module 8: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Sara Martino, Stefanie Muff, Kenneth Aase — Department of Mathematical Sciences, NTNU"
date: "March 12, 2025"
---

---

## Problem 1 -- Theoretical

a) Provide a detailed explanation of the algorithm that is used to fit a regression tree. What is different for a classification tree?

b) What are the advantages and disadvantages of regression and classification trees?

c) What is the idea behind bagging and what is the role of bootstap? How do random forests improve that idea?

d) What is an out-of bag (OOB) error estimator and what percentage of observations are included in an OOB sample? (Hint: The result from RecEx5-Problem 4c can be used)

e) Bagging and Random Forests typically improve the prediction accuracy of a single tree, but it can be difficult to interpret, for example in terms of understanding which predictors are how relevant.
How can we evaluate the importance of the different predictors for these methods?



## Problem 2 -- Regression (Book Ex. 8)

In the lab (8.3 Lab: Decision Trees), the Carseats dataset is used.
This dataset comprises sales data for child car seats at 400 different stores. The dataset includes a mix of quantitative and qualitative predictors.

Key attributes of the dataset include:

- Sales: The number of car seats sold at each location, serving as the response variable for regression analyses.
- CompPrice: The price charged by competitors at each location.
- Income: The community income level where the store is located.
- Advertising: The budget for advertising in each location.
- Population: The population size in the region around the store.
- Price: The price of the car seats at each store.
- ShelveLoc: A qualitative variable indicating the quality of the shelving location at the store (Good, Medium, Bad).
- Age: The average age of the local population.
- Education: The education level at each location.
- Urban: A qualitative variable indicating whether the store is in an urban or rural location.
- US: A qualitative variable indicating whether the store is in the US or not.

In the lab, a classification tree was applied to the Carseats dataset after converting the variable `Sales` into a qualitative response variable.
Now we will seek to predict `Sales` using regression trees and related approaches, treating the response as a quantitative variable.

a) Split the data set into a training set and a test set. (Hint: Use 70% of the data as training set and the rest 30% as testing set)

**Python-hints**
```python
import numpy as np
import pandas as pd
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

b) Fit a regression tree to the training set. Plot the tree, and interpret the results. What test MSE do you obtain?

**Python-hints**

```python
from sklearn.tree import DecisionTreeRegressor as DTR, plot_tree
reg = DTR(random_state=4268)
reg.fit(X_train, y_train)
```

c) Use cross-validation in order to determine the optimal level of tree complexity. Does pruning the tree improve the test MSE?

**Python-hints**

```python
ccp_path = reg.cost_complexity_pruning_path(X_train, y_train)
kfold = skm.KFold(10, shuffle=True, random_state=4268)
grid = skm.GridSearchCV(reg,
                        {'ccp_alpha': ccp_path.ccp_alphas},
                        refit=True,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
G = grid.fit(X_train, y_train)
```

d) Use the bagging approach with 500 trees in order to analyze the data. What test MSE do you obtain? Use `feature_importances_` to determine which variables are most important.

**Python-hints**

```python
from sklearn.ensemble import RandomForestRegressor as RF
bag_Carseats = RF(max_features=X_train.shape[1],
                  n_estimators=500,
                  random_state=4268).fit(X_train, y_train)
```

e) Use random forests and to analyze the data. Include 500 trees and select 3 variables for each split. What test MSE do you obtain? Use `feature_importances_` to determine which variables are most important. Describe the effect of m, the number of variables considered at each split, on the error rate obtained.

**Python-hints**

```python
rf_Carseats = RF(max_features=3,
                 n_estimators=500,
                 random_state=4268).fit(X_train, y_train)
```


f) What is the effect of the number of trees (`n_estimators`) on the test error? Plot the test MSE as a function of `n_estimators` for both the bagging and the random forest method.


## Problem 3 -- Classification

In this exercise you are going to implement a spam filter for e-mails by using tree-based methods. Data from 4601 e-mails are collected and can be loaded from OpenML as follows:

```python
from sklearn.datasets import fetch_openml
spam = fetch_openml('spambase', version=1, as_frame=True)
X_full = spam.data
y_full = spam.target  # '1' = spam, '0' = nonspam
```

Each e-mail is classified by `type` (`spam` or `nonspam`), and this will be the response in our model. In addition there are 57 predictors in the dataset. The predictors describe the frequency of different words in the e-mails and orthography (capitalization, spelling, punctuation and so on).

a) Study the dataset (e.g. `print(spam.DESCR)` or look it up at the UCI Machine Learning Repository).

b) Create a training set and a test set for the dataset. (Hint: Use 70% of the data as training set and the rest 30% as testing set)

c) Fit a tree to the training data with `type` as the response and the rest of the variables as predictors. Inspect the result (e.g. number of leaves and training accuracy). Also create a plot of the tree. How many terminal nodes does it have?

d) Predict the response on the test data. What is the misclassification rate?

e) Use cost-complexity pruning together with cross-validation to find the optimal tree size. Prune the tree according to the optimal tree size and plot the result. Predict the response on the test data by using the pruned tree. What is the misclassification rate in this case?

f) Create a decision tree by using the bagging approach with $B=500$. Use `RandomForestClassifier()` and consider all of the predictors in each split. Predict the response on the test data and report the misclassification rate.

g) Apply `RandomForestClassifier()` again with 500 trees, but this time consider only a subset of the predictors in each split. This corresponds to the random forest-algorithm. Study the importance of each variable by using `feature_importances_`. Are the results as expected based on earlier results? Again, predict the response for the test data and report the misclassification rate.

<!-- h) Use `GradientBoostingClassifier()` to construct a boosted classification tree using 5000 trees, an interaction depth of $d=3$ and a shrinkage parameter of $\lambda=0.001$. Predict the response for the test data and report the misclassification rate. -->

h) Compare the misclassification rates in d-g.
Which method gives the lowest misclassification rate for the test data?
Are the results as expected?


### Acknowledgements

This document was originally adapted from the R-based recommended exercises by Sara Martino, Stefanie Muff and Kenneth Aase (Department of Mathematical Sciences, NTNU).
