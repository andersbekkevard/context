---
title: "Module 4: Solutions to Recommended Exercises"
subtitle: TMA4268 Statistical Learning V2025
authors:
  - Sara Martino
  - Stefanie Muff
  - Kenneth Aase
affiliation: Department of Mathematical Sciences, NTNU
date: February 5, 2025
---

# Problem 1: KNN

## a)

Observation 1: $\sqrt{(3-1)^2+(3-2)^2}= \sqrt{5} \approx 2.24$

Observation 2: $\sqrt{(2-1)^2+(0-2)^2}= \sqrt{5} \approx 2.24$

Observation 3: $\sqrt{(1-1)^2+(1-2)^2} = 1$

Observation 4: $\sqrt{(0-1)^2+(1-2)^2}= \sqrt{2} \approx 1.41$

Observation 5: $\sqrt{(-1-1)^2+(0-2)^2}= \sqrt{8} \approx 2.83$

Observation 6: $\sqrt{(2-1)^2+(1-2)^2}= \sqrt{2} \approx 1.41$

Observation 7: $\sqrt{(1-1)^2+(0-2)^2} = 2$

## b)

When $K=1$ the nearest neighbor is observation 3 $(1,1)$ which is class A. The predicted probabilities for class A and B are then,
$$
P\left(Y=A|X=x_0\right) = \frac{1}{1}\sum_{i \in \mathcal{N}_0}I\left(y_i=A\right) = 1
$$
$$
P\left(Y=B|X=x_0\right) = \frac{1}{1}\sum_{i \in \mathcal{N}_0}I\left(y_i=B\right) = 0,
$$
and we classify the point as A.
For $K=4$ the nearest neighbors are observations 3, 4, 6 and 7 with corresponding classes \{A,B,B,B\} and the predicted probabilities for class A and B for point $(1,2)$ are then
$$P(Y=A|X=x_0) = \frac{1}{4}\sum_{i \in \mathcal{N}_0}I(y_i=A) = \frac{1}{4}(1+0+0+0) = 0.25$$
$$P(Y=B|X=x_0) = \frac{1}{4}\sum_{i \in \mathcal{N}_0}I(y_i=B) = \frac{1}{4}(0+1+1+1) = 0.75,$$
and we classify the point as B.
For $K=7$ we use all our observations to predict the probabilties,
$$P(Y=A|X=x_0) = \frac{1}{7}\sum_{i \in \mathcal{N}_0}I(y_i=A) = \frac{1}{7}(1+1+1+0+0+0+0) = \frac{3}{7} = 0.429,$$
$$P(Y=B|X=x_0) = \frac{1}{7}\sum_{i \in \mathcal{N}_0}I(y_i=B) = \frac{1}{7}(0+0+0+1+1+1+1) = \frac{4}{7} = 0.571,$$
and we classify the point as B.

Using all observations for classification with KNN, we will always choose the class that are most represented in the data (B in this case), and the predicted class will be B regardless of the position of the new point.

## c)
If the Bayes decision boundary is highly non-linear, then the optimal $K$ would be small, as the decision boundary becomes approximately linear when $K$ tends to infinity.

# Problem 2: Bank notes and LDA

## a)

Assuming the observations $x_G$ and $x_F$ are independent observations from normal distribution with the same covariance matrix $\boldsymbol \Sigma = \boldsymbol \Sigma_G = \boldsymbol \Sigma_F$, and all observations are independent of each other, we can find the estimated pooled covariance matrix as
$$\begin{aligned}
\hat{\boldsymbol \Sigma} &= \frac{(n_G - 1)\hat{\boldsymbol \Sigma}_G + (n_F - 1)\hat{\boldsymbol \Sigma}_F}{n_G + n_F - 2} \\
&= \left[     \begin{array}{cc} 0.13710 & 0.00855 \\ 0.00855 & 0.25550 
\end{array} \right]
\end{aligned}$$

## b)

For LDA we assume that the class conditional distributions are normal (Gaussian) and that all of the classes have the same covariance matrix.
Assuming the same covariance matrix for both classes (G and F), we classify the new observation, $x_0$ based on which of the discriminant functions that are the largest,
$$\delta_k(\bf{x}) = {\bf x}^T \boldsymbol{\Sigma}^{-1}\boldsymbol\mu_k - \frac{1}{2}\boldsymbol\mu_k^T \boldsymbol{\Sigma}^{-1}\boldsymbol\mu_k + \log \pi_k.$$

We have not been given any information of the prior probabilities - but we would believe that the probability of a fake bank note is much smaller than the probability of a genuine bank note.

However, since our training data is 50% fake and genuine, one could use this to estimate prior probabilities, $\hat{\pi}_G =\hat{\pi}_F =\frac{n_F}{n} = 0.5.$ Inserting the pooled covariance matrix and the estimated mean values, we have that
$$
\delta_G({\bf x_0}) = {\bf x_0}^T \hat{\boldsymbol{\Sigma}}^{-1}\boldsymbol\mu_G - \frac{1}{2}\boldsymbol\mu_G^T \hat{\boldsymbol{\Sigma}}^{-1}\boldsymbol\mu_G + \log \pi_G.
$$
and
$$
\delta_F({\bf x_0}) = {\bf x_0}^T \hat{\boldsymbol{\Sigma}}^{-1}\boldsymbol\mu_F - \frac{1}{2}\boldsymbol\mu_F^T \hat{\boldsymbol{\Sigma}}^{-1}\boldsymbol\mu_F + \log \pi_F.
$$

Alternatively: The rule would be to classify to $G$ if $\delta_G({\bf x})-\delta_F({\bf x})>0$, which can be written
$$
{\bf x_0}^T \hat{\boldsymbol{\Sigma}}^{-1}(\boldsymbol\mu_G -\boldsymbol\mu_F)- \frac{1}{2}\boldsymbol\mu_G^T \hat{\boldsymbol{\Sigma}}^{-1}\boldsymbol\mu_G +\frac{1}{2}\boldsymbol\mu_F^T \hat{\boldsymbol{\Sigma}}^{-1}\boldsymbol\mu_F+ (\log \pi_G -\log \pi_F)>0
$$


## c)

With ${\bf x_0} = [214.0, 140.4]^T$ and using the formula given in the exercise, $\hat{\boldsymbol{\Sigma}}^{-1} = \left[ \begin{array}{cc} 7.31 & -0.24 \\ -0.24 & 3.92  \end{array} \right]$. Inserting these into the formulas, we have that

$$
\begin{aligned}
\delta_G({\bf x_0}) &=  \left[ 214 \text{  } 140.4  \right]
\left[ \begin{array}{cc} 7.31 & -0.24 \\ -0.24 & 3.92 
\end{array} \right]
\left[     \begin{array}{c} 214.97 \\ 141.52  \end{array} \right] - \frac{1}{2}
\left[ 214.97 \text{  } 141.52 \right] \
\left[ \begin{array}{cc} 7.31 & -0.24 \\ -0.24 & 3.92 
\end{array} \right]
\left[     \begin{array}{c} 214.97 \\ 141.52  \end{array} \right] \ + \log 0.5 \\
&= 198667.1
\end{aligned}
$$

and
$$
\begin{aligned}
\delta_F({\bf x_0}) &= \left[ 214 \text{  } 140.4  \right]
\left[ \begin{array}{cc} 7.31 & -0.24 \\ -0.24 & 3.92 
\end{array} \right]
\left[     \begin{array}{c} 214.82 \\ 139.42  \end{array} \right] - \frac{1}{2}
\left[ 214.82 \text{  } 139.42 \right] \
\left[ \begin{array}{cc} 7.31 & -0.24 \\ -0.24 & 3.92 
\end{array} \right]
\left[     \begin{array}{c} 214.82 \\ 139.42  \end{array} \right] \ + \log 0.5 \\
&= 198668.3
\end{aligned}
$$
Since $\delta_F({\bf x_0})$ is larger than $\delta_G({\bf x_0})$, we classify the bank note as fake.

Doing all these calculations in Python:
```python
import numpy as np

xg = np.array([214.97, 141.52])
xf = np.array([214.82, 139.45])
sigmaG = np.array([[0.1502, 0.0055],
                   [0.0055, 0.1998]])
sigmaF = np.array([[0.1240, 0.0116],
                   [0.0116, 0.3112]])
sigma = (499 * sigmaG + 499 * sigmaF) / 998
x = np.array([214.0, 140.4])
sigma_inv = np.linalg.inv(sigma)
# lda discriminant function
a = x @ sigma_inv @ xg - 0.5 * xg @ sigma_inv @ xg - np.log(0.5)
b = x @ sigma_inv @ xf - 0.5 * xf @ sigma_inv @ xf - np.log(0.5)
```


## d)

The difference between LDA and QDA is the QDA assumes that each class has its own covariance matrix, such that an observation from class $k$ is on the form $X \sim N(\mu_k, \boldsymbol{\Sigma_k})$ where $\boldsymbol{\Sigma_k}$ is the covariance matrix. The discriminant function for class $k$ is
$$
\delta_k(\bf{x}) = -\frac{1}{2}\bf{x}^T\boldsymbol{\Sigma_k}^{-1}\bf{x} + {\bf x}^T \boldsymbol{\Sigma_k}^{-1}\boldsymbol\mu_k - \frac{1}{2}\boldsymbol\mu_k^T \boldsymbol{\Sigma_k}^{-1}\boldsymbol\mu_k - \frac{1}{2}\log|\boldsymbol{\Sigma_k}| + \log \pi_k,
$$
which unlike the disscriminant function for LDA is a quadratic function.

Using the QDA for classification of the bank note from c., we use that $\boldsymbol{\hat\Sigma_G}^{-1} = \left[ \begin{array}{cc} 6.66 & -0.18 \\ -0.18 & 5.01 \end{array} \right]$ and $\boldsymbol{\hat\Sigma_F}^{-1} = \left[ \begin{array}{cc} 8.09 & -0.30 \\ -0.30 & 3.22 \end{array} \right]$. Inserting the estimated values into the discriminant function, we have that

$$
\begin{aligned}
\delta_G({\bf x_0}) = & -\frac{1}{2}\left[ 214 \text{  } 140.4  \right]
\left[ \begin{array}{cc} 6.66 & -0.18 \\ -0.18 & 5.01 
\end{array} \right]
\left[     \begin{array}{c} 214 \\ 140.4  \end{array} \right] 
\\
& + \left[ 214 \text{  } 140.4  \right]
\left[ \begin{array}{cc} 6.66 & -0.18 \\ -0.18 & 5.01 
\end{array} \right]
\left[     \begin{array}{c} 214.97 \\ 141.52  \end{array} \right] 
\\
& - \frac{1}{2}
\left[ 214.97 \text{  } 141.52 \right] \
\left[ \begin{array}{cc} 6.66 & -0.18 \\ -0.18 & 5.01 
\end{array} \right]
\left[     \begin{array}{c} 214.97 \\ 141.52  \end{array} \right] \  
\\ 
& - \frac{1}{2}\log(6.66\cdot5.01 - 0.18^2) + \log(0.5) \\
=& -5.018
\end{aligned}
$$

and

$$
\begin{aligned}
\delta_F({\bf x_0}) = & -\frac{1}{2}\left[ 214 \text{  } 140.4  \right]
\left[ \begin{array}{cc} 8.09 & -0.30 \\ -0.30 & 3.22 
\end{array} \right]
\left[     \begin{array}{c} 214 \\ 140.4  \end{array} \right] 
\\
& +\left[ 214 \text{  } 140.4  \right]
\left[ \begin{array}{cc} 8.09 & -0.30 \\ -0.30 & 3.22 
\end{array} \right]
\left[     \begin{array}{c} 214.82 \\ 139.45  \end{array} \right] 
\\
&- \frac{1}{2} \left[ 214.82 \text{  } 139.45 \right] \
\left[ \begin{array}{cc} 8.09 & -0.30 \\ -0.30 & 3.22 
\end{array} \right]
\left[     \begin{array}{c} 214.82 \\ 139.45  \end{array} \right] \  
\\
&-\frac{1}{2}\log(8.09\cdot3.22 - 0.30^2) + \log(0.5) \\
= & -3.47
\end{aligned}
$$

Since $\delta_F({\bf x_0})$ is larger than $\delta_G({\bf x_0})$ we classify the bank note as fake using QDA, the same result as for LDA.

Python code to get to these results:

```python
# qda discriminant function
sigmaG_inv = np.linalg.inv(sigmaG)
sigmaF_inv = np.linalg.inv(sigmaF)
a = (-0.5 * x @ sigmaG_inv @ x + x @ sigmaG_inv @ xg
     - 0.5 * xg @ sigmaG_inv @ xg
     - 0.5 * np.log(np.linalg.det(sigmaG)) - np.log(2))
b = (-0.5 * x @ sigmaF_inv @ x + x @ sigmaF_inv @ xf
     - 0.5 * xf @ sigmaF_inv @ xf
     - 0.5 * np.log(np.linalg.det(sigmaF)) - np.log(2))
```


# Problem 3: Odds

## a)

Recall that the odds ratio is defined as $$\text{odds} = \frac{p}{1-p},$$ where $p$ is the probability of some event. We know that the odds ratio is equal to 0.37. We thus solve for $p$:
$$\begin{aligned} \frac{p}{1-p} &= 0.37 \\ p &= 0.37 (1-p) \\ p &= \frac{0.37}{1.37} = 0.270 \end{aligned} $$

## b)

For an individual we are given that $p = 0.16$. We are asked to find the odds that she will default. We can calculate this by inserting the probability of default into the formula for the odds ratio: $$\frac{p}{1-p} = \frac{0.16}{1-0.16} = 0.19$$


# Problem 4: Logistic regression

## a)

We have that $X_1$ = hours studied = 40 and $X_2$ = undergrad GPA = 3.5. The formula for the predicted probability is
$$
\begin{aligned}P(Y = A \mid X_1 = 40, X_2 = 3.5) &= \frac{\exp(\hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2)}{1 + \exp(\hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2)} \\ 
&= \frac{\exp(-6 + 0.05 \cdot 40 + 1 \cdot 3.5)}{1 + \exp(-6 + 0.05 \cdot 40 + 1 \cdot 3.5)}\\ 
&\approx 0.378 \end{aligned}
$$

## b)

We know that $x_2$ = 3.5, and need to solve $\hat{P}(Y = A \mid x_1, x_2 = 3.5) = 0.5$ for $x_1$.
$$
\begin{aligned} 0.5 &= \frac{\exp(-6+0.05 x_1 + 3.5)}{1+\exp(-6+0.05 x_1 + 3.5)} \\ 
0.5 (1+\exp(-2.5+0.05x_1)) &=\exp(-2.5+0.05x_1) \\ 
0.5 &= (1-0.5) \exp(-2.5+0.05x_1)\\ 
\log(1) &= -2.5+ 0.05x_1 \\ 
x_1 &= 50  \end{aligned}
$$

# Problem 5: Sensitivity, specificity, ROC and AUC

## a)

We start by denoting the number of true diseased as $P$ and the number of true non-diseased as $N$, and we have that $n = P+N$. We count the ones with predicted probability of disease $p(x)>0.5$, and denote the count $P^*$ and the count for the ones with $p(x)\leq0.5$ as $N^*$. Then, we can make the confusion table

|                       | Predicted non-diseased $-$ | Predicted diseased $+$ | Total |
| --------------------- | -------------------------- | ---------------------- | ----- |
| True non-diseased $-$ | TN                         | FP                     | N     |
| True diseased $+$     | FN                         | TP                     | P     |
| Total                 | $N^*$                      | $P^*$                  | n     |

where TN (true negative) is the number of predicted non-diseased that are actually non-diseased, FP (false positive) is the number of predicted diseased that are actually non-diseased, FN (false negative) is the number of predicted non-diseased that are actually diseased and TP (true positive) is the number of predicted diseased that are actually diseased. Using the confusion table, we can calculate the sensitivity and specificity where
$$
Sens = \frac{\text{True positive}}{\text{Actual positive}} = \frac{TP}{P}
$$
and
$$
Spes = \frac{\text{True negative}}{\text{Actual negative}} = \frac{TN}{N}
$$

## b)

In the ROC-curve we plot the sensitivity against 1-specificity for all possible thresholds of the probability. To construct the ROC-curve we would have to calculate the sensitivity and specificity for different values of the cutoff $p(x)>cut$. Using a threshold of $0.5$, you say that if a new person has a probability $> 0.5$ of having the disease, they are classified as diseased. Another person with a probability of $\leq 0.5$ would then be classified as non-diseased. However, using another cutoff could easily lead to a better balance between sensitivity and specificity (i.e., the point could lie closer to the top-left corner of the ROC-plot than for a $0.5$ cut-off). Moreover, depending on the cost of a false positive or a false negative, the cut-off could be adjusted. It is for example often less severe to generate a false positive (i.e., a positive test result although a person is healthy) than a false negative (i.e., you overlook a disease), in which case the sensitivity is increased at a cost of a lower specificity.

## c)

The AUC is the area under the ROC-curve and gives the overall performance of the test for all possible thresholds. An AUC value of $1$ means a perfect fit for all possible thresholds, while a AUC of $0.5$ corresponds to the classifier that performs no better than chance. Hence, a classification method $p(x)$ giving $0.6$ and another $q(x)$ giving $0.7$, we would probably prefer $q(x)$.

---
# Data analysis in Python

# Problem 6

## a)
We import the necessary libraries and load the `Weekly` data set from `ISLP`.
```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data, confusion_table
from ISLP.models import ModelSpec as MS, summarize
```
Now make a numerical summary of the `Weekly` data set using the `describe` method and pairwise plots of the variables using `seaborn`'s `pairplot`:
```python
Weekly = load_data('Weekly')
Weekly.describe()
sns.pairplot(Weekly, hue='Direction', plot_kws={'alpha': 0.3, 's': 2});
```

We can observe that the variables `Year` and `Volume` are highly correlated and it might look like the `Volume` increases quadratically with increasing `Year`. No other clear patterns are observed.

## b)
We fit a logistic regression model using `sm.GLM()` with `family=sm.families.Binomial()`, following the ISLP ch.4 lab pattern.
```python
allvars = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
design = MS(allvars)
X = design.fit_transform(Weekly)
y = Weekly.Direction == 'Up'
glm_Weekly = sm.GLM(y, X, family=sm.families.Binomial())
results_Weekly = glm_Weekly.fit()
summarize(results_Weekly)
```

`Lag2` has a relatively low $p$-value, so we find evidence that it is associated with `Direction`. However, remember that `Lag2` is not necessarily causally related to `Direction`, and it could also well be that the $p$-value is small just by chance.

## c)

We use our fitted model to calculate the probabilities for `Direction="Up"` for the response variable and compare the predictions with the true classes of the response variable. To find the confusion matrix, we use `confusion_table` from `ISLP`.
```python
glm_probs_Weekly = results_Weekly.predict()
glm_preds_Weekly = np.where(glm_probs_Weekly > 0.5, 'Up', 'Down')
confusion_table(glm_preds_Weekly, Weekly.Direction)
```

The fraction of correct predictions is $$\frac{54+557}{1089} = 0.561 \ ,$$ which may seem like a bad performance, but in the context of stock market predictions, it is actually quite good. From the confusion matrix we also see that the classifier does a better job predicting when the market goes `Up`, but a rather poor job predicting the market goes `Down`.

## d)
We start by dividing the `Weekly` data set into a train and a test set, where the training set consists of all observations in the period from 1990 to 2008, while the test set consists of the observations from the period 2009 to 2010. We then fit a logistic regression model to the training data set, make predictions for the test set, and calculate the confusion matrix.
```python
train = (Weekly.Year < 2009)
Weekly_train = Weekly.loc[train]
Weekly_test = Weekly.loc[~train]

design2 = MS(['Lag2'])
X2 = design2.fit_transform(Weekly)
X2_train, X2_test = X2.loc[train], X2.loc[~train]
y_train = (Weekly_train.Direction == 'Up')

glm_Weekly2 = sm.GLM(y_train, X2_train, family=sm.families.Binomial())
results_Weekly2 = glm_Weekly2.fit()
glm_Weekly2_prob = results_Weekly2.predict(exog=X2_test)
glm_Weekly2_pred = np.where(glm_Weekly2_prob > 0.5, 'Up', 'Down')
confusion_table(glm_Weekly2_pred, Weekly_test.Direction)
```

The fraction of correct predictions on the test set is
$$
\frac{9+56}{104} = 0.625.
$$

## e)
We import `LinearDiscriminantAnalysis` (abbreviated `LDA`) from `sklearn.discriminant_analysis`. We fit an LDA model to the training set, then use it to make predictions for the test set and compare the predicted and true classes using the confusion matrix.
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

L_train = Weekly_train.Direction
L_test = Weekly_test.Direction
# LDA adds an intercept automatically — drop it from the design matrix
X2_train_nc = X2_train.drop(columns=['intercept'])
X2_test_nc = X2_test.drop(columns=['intercept'])

lda_Weekly = LDA(store_covariance=True)
lda_Weekly.fit(X2_train_nc, L_train)
lda_Weekly_pred = lda_Weekly.predict(X2_test_nc)
lda_Weekly_prob = lda_Weekly.predict_proba(X2_test_nc)
confusion_table(lda_Weekly_pred, L_test)
```

The fraction of correct classifications is
$$
\frac{9+56}{104} = 0.625.
$$

## f)
We follow the same procedure and test a QDA classifier on the data:
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

qda_Weekly = QDA(store_covariance=True)
qda_Weekly.fit(X2_train_nc, L_train)
qda_Weekly_pred = qda_Weekly.predict(X2_test_nc)
qda_Weekly_prob = qda_Weekly.predict_proba(X2_test_nc)
confusion_table(qda_Weekly_pred, L_test)
```

The fraction of correct classifications is now
$$
\frac{0+61}{104} = 0.587.
$$

_However_, this QDA classifier is very naive and not really informative, because it just predicts an "Up" trend all the time. So it has a 100\% error rate for the true "Down" trends, and it is thus a useless classifier.

## g)
The KNN classifier is implemented in `KNeighborsClassifier` from `sklearn.neighbors`. The estimator follows the standard `sklearn` `fit`/`predict` pattern.

```python
from sklearn.neighbors import KNeighborsClassifier as KNN

knn_train = np.asarray(Weekly_train['Lag2']).reshape(-1, 1)
knn_test = np.asarray(Weekly_test['Lag2']).reshape(-1, 1)

knn1_Weekly = KNN(n_neighbors=1)
knn1_Weekly.fit(knn_train, L_train)
knn1_pred = knn1_Weekly.predict(knn_test)
confusion_table(knn1_pred, L_test)
```

The fraction of correct classifications for the KNN classifier with $K=1$ is
$$
\frac{21+32}{104}=0.5096
$$

## h)
To find the value of $K$ giving the smallest error, we make a for-loop where we store the error for each iteration and plot the error against the value of $K$.
```python
K = 30
knn_error = np.empty(K)
for k in range(1, K + 1):
    knn_k = KNN(n_neighbors=k)
    knn_k.fit(knn_train, L_train)
    knn_pred = knn_k.predict(knn_test)
    knn_error[k - 1] = np.mean(knn_pred != L_test)

knn_error_df = pd.DataFrame({'k': np.arange(1, K + 1), 'error': knn_error})
fig, ax = subplots()
ax.plot(knn_error_df['k'], knn_error_df['error'],
        linestyle=':', marker='o', color='blue')
ax.set_xlabel('k')
ax.set_ylabel('error');
```

From the plot, we see that $K=4$ gives the smallest error. We create a confusion matrix and store the probabilities of `Direction="Up"` for this model.
```python
knn4_Weekly = KNN(n_neighbors=4)
knn4_Weekly.fit(knn_train, L_train)
knn4_pred = knn4_Weekly.predict(knn_test)
confusion_table(knn4_pred, L_test)
# probability of Direction="Up": pick the column whose label is 'Up'
up_idx = list(knn4_Weekly.classes_).index('Up')
knn4_Weekly_prob = knn4_Weekly.predict_proba(knn_test)[:, up_idx]
```

The fraction of correct classifications for the KNN classifier is
$$
\frac{19+37}{104}=0.538
$$

## i)
The logistic regression model and the LDA classifier provided the highest fractions of correct classifications on this data. Note that QDA does not classify any of the observations as "Up".

## j)
To make ROC-curves and calculate AUC we use `roc_curve` and `auc` from `sklearn.metrics`.

```python
from sklearn.metrics import roc_curve, auc

# probability of "Up" for each method
up_idx_lda = list(lda_Weekly.classes_).index('Up')
up_idx_qda = list(qda_Weekly.classes_).index('Up')
glm_probs_Up = glm_Weekly2_prob
lda_probs_Up = lda_Weekly_prob[:, up_idx_lda]
qda_probs_Up = qda_Weekly_prob[:, up_idx_qda]
knn_probs_Up = knn4_Weekly_prob

fig, ax = subplots()
for name, probs_Up in [('glm', glm_probs_Up),
                       ('lda', lda_probs_Up),
                       ('qda', qda_probs_Up),
                       ('knn', knn_probs_Up)]:
    fpr, tpr, _ = roc_curve(L_test, probs_Up, pos_label='Up')
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
ax.set_xlabel('1-Specificity')
ax.set_ylabel('Sensitivity')
ax.legend();
# glm is very similar to lda, so the glm-ROC-curve is not visible behind the
# lda curve.
```

Note first that the ROC-curve for glm is very similar to lda, so the glm-ROC-curve is not visible behind the lda curve. The ROC-curves with the corresponding AUCs tells us that none of our models perform well when classifying the test set. The ROC-curves are very close to the diagonal line, representing the "no information" classifier, thus the AUC values are very close to $0.5$. A good classifier would hug the left upper corner in the ROC-plot and have a AUC value close to $1$.
