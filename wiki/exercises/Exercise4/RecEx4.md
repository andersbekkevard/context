---
title: "Module 4: Recommended Exercises"
subtitle: TMA4268 Statistical Learning V2025
authors:
  - Sara Martino
  - Stefanie Muff
  - Kenneth Aase
affiliation: Department of Mathematical Sciences, NTNU
date: February 5, 2025
---

# Problem 1: KNN (Exercise 2.4.7 in ISL textbook, slightly modified)

The table and plot below provides a training data set consisting of seven observations, two predictors and one qualitative response variable.

```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

knnframe = pd.DataFrame({
    'x1': [3, 2, 1, 0, -1, 2, 1],
    'x2': [3, 0, 1, 1, 0, 1, 0],
    'y': pd.Categorical(['A', 'A', 'A', 'B', 'B', 'B', 'B'])
})
print(knnframe)

fig, ax = subplots(figsize=(5, 5))
for level, marker in zip(['A', 'B'], ['o', 's']):
    sub = knnframe[knnframe['y'] == level]
    ax.scatter(sub['x1'], sub['x2'], label=level, marker=marker, s=60)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(title='y');
```

We want to use this data set to make a prediction for $Y$ when $X_1=1, X_2=2$ using the $K$-nearest neighbors classification method.

## a)

Calculate the Euclidean distance between each observation and the test point,  $X_1=1,X_2=2$.

## b)

Use $P\left(Y=j\mid X=x_0\right) = \frac{1}{K}\sum_{I \in \mathcal{N}_0}I\left(Y=j\right)$ to predict the class of $Y$ when $K=1$, $K=4$ and $K=7$. Why is $K=7$ a bad choice?

## c)

If the Bayes decision boundary in this problem is highly non-linear, would we expect the best value for $K$ to be large or small? Why?

# Problem 2: Bank notes and LDA (with calculations)

To distinguish between genuine and fake bank notes, measurements of length and diagonal of part of the bank notes have been made. For $1000$ bank notes ($500$ genuine and $500$ false) this gave the following values for the mean and the covariance matrix (using unbiased estimators), where the first value is the length of the bank note, and the second is the diagonal.

Genuine bank notes:
$$
\bar{\bf x}_G=\left[     \begin{array}{c} 214.97 \\ 141.52  \end{array} \right]
\text{ and }
\hat{\boldsymbol \Sigma}_G=\left[     \begin{array}{cc} 0.1502 & 0.0055 \\ 0.0055 & 0.1998 
\end{array} \right]
$$

Fake bank notes:
$$
\bar{\bf x}_F= \left[     \begin{array}{c} 214.82 \\ 139.45  \end{array} \right]
\text{ and }
\hat{\boldsymbol \Sigma}_F= \left[     \begin{array}{cc} 0.1240 & 0.0116 \\ 0.0116 & 0.3112 
\end{array} \right]
$$


## a)

Assume the true covariance matrix for the genuine and fake bank notes are the same. How would you estimate the common covariance matrix?

## b)

Explain the assumptions made to use linear discriminant analysis to classify a new observation to be a genuine or a fake bank note. Write down the classification rule for a new observation (make any assumptions you need to make).

## c)

Use the method in b) to determine if a bank note with length $214.0$ and diagonal $140.4$ is genuine or fake. You can use Python to perform the matrix calculations.

**Python-hints**:
```python
import numpy as np
# inv(A)
np.linalg.inv(A)
# transpose of vector
v.T
# determinant of A
np.linalg.det(A)
# multiply vector and matrix / matrix and matrix
v @ A
B @ A
```

## d)

What is the difference between LDA and QDA? Use the classification rule for QDA to determine the bank note from c). Do you obtain the same result? You can use Python to perform the matrix calculations.

Hint: the following formulas might be useful.
$$
A^{-1} = \left[
\begin{array}{cc} a & b \\ c & d \end{array} 
\right]^{-1} = \frac{1}{ad-bc}
\left[
\begin{array}{cc} d & -b \\ -c & a \end{array} 
\right]
$$

$$
|A| = \det(A) = \left|\begin{array}{cc} a & b \\ c & d \end{array}\right| = ad - bc
$$



# Problem 3: Odds (Exercise 4.7.9 in ISL textbook)

This problem is about *odds*.

## a)

On average, what fraction of people with an odds of $0.37$ of defaulting on their credit card payment will in fact default?

## b)

Suppose that an individual has a $16\%$ chance of defaulting on her credit card payment. What are the odds that she will default?

# Problem 4: Logistic regression (Exercise 4.7.6 in ISL textbook)

Suppose we collect data for a group of students in a statistics class with variables $x_1$ = hours studied, $x_2$ = undergrad grade point average (GPA), and $Y = I(\text{student gets an A})$. We fit a logistic regression and produce estimated coefficient, $\hat{\beta}_0 = -6, \hat{\beta}_1 = 0.05, \hat{\beta}_2 = 1$. 

## a)

Estimate the probability that a student who studies for $40$ hours and has an undergrad GPA of $3.5$ gets an A in the class.

## b)

How many hours would the student in part a) need to study to have an estimated $50\%$ probability of getting an A in the class?

# Problem 5: Sensitivity, specificity, ROC and AUC

We have a two-class problem, with classes $0=$non-disease and $1=$disease, and a method $p(x)$ that produces probability of disease depending on a covariate $x$. In a population we have investigated $n$ individuals and know the predicted probability of disease $p(x)$ and true disease status for these $n$.

## a)

We choose the rule $p(x)>0.5$ to classify to disease. Define the sensitivity and the specificity of the test.

## b)

Explain how you can construct a receiver operator curve (ROC) for your setting, and why that is a useful thing to do. In particular, why do we want to investigate different cut-offs of the probability of disease?

## c)

Assume that we have a competing method $q(x)$ that also produces probability of disease for a covariate $x$. We get the information that the AUC of the $p(x)$-method is $0.6$ and the AUC of the $q(x)$-method is $0.7$ on independent validation sets. What is the definition and interpretation of the AUC? Would you prefer the $p(x)$ or the $q(x)$ method for classification?

---

# Data analysis with Python

For the following problems, you should check out and learn how to use the following Python tools: `sm.GLM(..., family=sm.families.Binomial())` from `statsmodels` and `LogisticRegression` from `sklearn.linear_model`, `LinearDiscriminantAnalysis` and `QuadraticDiscriminantAnalysis` from `sklearn.discriminant_analysis`, `KNeighborsClassifier` from `sklearn.neighbors`, and `roc_curve`, `auc` from `sklearn.metrics`.

# Problem 6 (Exercise 4.7.10 in ISL textbook - modified)

This question should be answered using the `Weekly` data set, which is part of the `ISLP` package. This data is similar to the `Smarket` data from this chapter's lab, except that it contains $1,089$ weekly returns for $21$ years, from the beginning of 1990 to the end of 2010.

## a)

Produce numerical and graphical summaries of the `Weekly` data. Do there appear to be any patterns?
**Python-hint**: Load the data as follows:
```python
from ISLP import load_data
Weekly = load_data('Weekly')
```

## b)

Use the full data set to perform a logistic regression with `Direction` as the response and the five lag variables plus `Volume` as predictors. Print the summary of the fitted model. Which of the predictors appears to be associated with `Direction`?
**Python-hints:** Use `sm.GLM(..., family=sm.families.Binomial())` from `statsmodels` to make a logistic regression model, following the ISLP ch.4 lab pattern with `ModelSpec` to build the design matrix.

## c)

Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by your logistic regression model.
**Python-hints:** insert the name of your fitted results object for `yourGlmResults` in the code below to get the predicted probabilities for "Up", the classified direction and the confusion matrix.
```python
from ISLP import confusion_table
import numpy as np
glm_probs_Weekly = yourGlmResults.predict()
glm_preds_Weekly = np.where(glm_probs_Weekly > 0.5, 'Up', 'Down')
confusion_table(glm_preds_Weekly, Weekly.Direction)
```

## d)

Now fit the logistic regression model using a training data period from 1990 to 2008, with `Lag2` as the only predictor. Compute the confusion matrix and the overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).
**Python-hints:** use the following code to divide into test and train set. For predicting the direction of the test set, pass the test design matrix as `exog=X_test` to the `predict()` method.
```python
train = (Weekly.Year < 2009)
Weekly_train = Weekly.loc[train]
Weekly_test = Weekly.loc[~train]
```

## e)
Repeat d) using LDA.

## f)
Repeat d) using QDA.

**Python-hints:** plug in your variables in the following pattern to perform LDA (and similarly for QDA).
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(store_covariance=True)
lda.fit(X_train, L_train)
lda_pred = lda.predict(X_test)
lda_prob = lda.predict_proba(X_test)
confusion_table(lda_pred, L_test)
```

## g)

Repeat d) using KNN with $K=1$.

**Python-hints:** plug in your variables in the following code to perform KNN. The `predict_proba()` method gives the class probabilities, which you will need later. We set the random state for reproducibility (KNN ties are broken in a deterministic order, but we keep the convention).
```python
from sklearn.neighbors import KNeighborsClassifier as KNN
knn_train = np.asarray(Weekly_train['Lag2']).reshape(-1, 1)
knn_test = np.asarray(Weekly_test['Lag2']).reshape(-1, 1)

knn1 = KNN(n_neighbors=YourValueOfK)
knn1.fit(knn_train, Weekly_train.Direction)
knn1_pred = knn1.predict(knn_test)
confusion_table(knn1_pred, Weekly_test.Direction)
```

## h)

Use the following code to find the best value of $K$. Report the confusion matrix and overall fraction of correct predictions for this value of $K$.
```python
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots

K = 30
knn_error = np.empty(K)

for k in range(1, K + 1):
    knn_k = KNN(n_neighbors=k)
    knn_k.fit(knn_train, Weekly_train.Direction)
    knn_pred = knn_k.predict(knn_test)
    knn_error[k - 1] = np.mean(knn_pred != Weekly_test.Direction)

knn_error_df = pd.DataFrame({'k': np.arange(1, K + 1), 'error': knn_error})
fig, ax = subplots()
ax.plot(knn_error_df['k'], knn_error_df['error'],
        linestyle=':', marker='o', color='blue')
ax.set_xlabel('k')
ax.set_ylabel('error');
```

## i)

Which of these methods appear to provide the best results on this data?

## j)

Plot the ROC curves and calculate the AUC for the four methods (using your the best choice for KNN). What can you say about the fit of these models?

**Python-hints**:

* For KNN you can use the `predict_proba()` method to get the class probabilities. Note that we want $P(\text{Direction} = \text{Up})$ when plotting the ROC-curve, so we select the column of `predict_proba` corresponding to the class `Up`.
```python
# get the probability for "Up" — this is the column whose label in
# knn_model.classes_ is 'Up'
up_idx = list(knn_model.classes_).index('Up')
knn_probs_Up = knn_model.predict_proba(knn_test)[:, up_idx]
```

* Use the following code to produce ROC-curves and AUCs:
```python
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import subplots

fig, ax = subplots()
for name, probs_Up in [('glm', glm_probs_Up),
                       ('lda', lda_probs_Up),
                       ('qda', qda_probs_Up),
                       ('knn', knn_probs_Up)]:
    fpr, tpr, _ = roc_curve(Weekly_test.Direction, probs_Up,
                            pos_label='Up')
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
ax.set_xlabel('1-Specificity')
ax.set_ylabel('Sensitivity')
ax.legend();
# glm is very similar to lda, so the ROC-curve for glm is not visible
# behind the lda curve.
```
