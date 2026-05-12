---
title: Math exercises
---

Every "mathy derivation" problem found across the weekly exercises, the compulsory exercises, and the past exams (2023–2025). The prof committed to *"at least one question that's more mathy where you should derive something"* on the 2026 exam (Apr 28); these are the historical instances of that question type, sorted by module.

Verbatim quotes only. Source links go to the line range in the bronze file.

---

## Module 2 — Statistical learning (bias-variance, foundations)

### M2.1 — Correlation from a covariance matrix
**Source:** [RecEx2](/exercises/Exercise2/RecEx2) (Problem 3 g)

The correlation of two variables $X$ and $Y$ are defined as
$$
\text{cor}(X,Y) = \frac{\text{cov}(X,Y)}{\sigma_X\sigma_Y}.
$$
The correlation matrix and covariance matrix can be easily found in `R` with the `cor()` and `cov()` functions, respectively. Use only the covariance matrix (as shown below) to find the correlation between `mpg` and `displacement`, `mpg` and `horsepower`, and `mpg` and `weight`. Do your results coincide with the correlation matrix you find using `cor(Auto[, quant])`?

### M2.2 — Bias-variance decomposition (simulation form)
**Source:** [RecEx2](/exercises/Exercise2/RecEx2) (Problem 5 c)

**c) Bias and variance - we use the truth!**

Finally, we want to see how the expected quadratic loss can be decomposed into

* irreducible error: $\text{Var}(\varepsilon)=4$
* squared bias: difference between mean of estimated parametric model chosen and the true underlying curve (`truefunc`)
* variance: variance of the estimated parametric model

Notice that the test data is not used -- only predicted values in each x grid point.

Study and run the code. Explain the plots produced.

[…]

Study the final plot you produced: when the flexibility increases (poly increase), what happens with
i) the squared bias,
ii) the variance,
iii) the irreducible error?

### M2.3 — Bias-variance decomposition (clean derivation)
**Source:** [Compulsory 1](/pdfs/compulsory-1.pdf) (Problem 1 a–c)

We have a univariate continuous random variable $Y$ and a covariate $x$. Further, we have observed a training set of independent observation pairs $\{x_i, y_i\}$ for $i = 1, \ldots, n$. Assume a regression model

$$Y_i = f(x_i) + \varepsilon_i ,$$

where $f$ is the true regression function, and $\varepsilon_i$ is an unobserved random variable with mean zero and constant variance $\sigma^2$ (not dependent on the covariate). Using the training set we can find an estimate of the regression function $f$, and we denote this estimate by $\hat{f}$. We want to use $\hat{f}$ to make a prediction for an independent new observation (not included in the training set) at a covariate value $x_0$. The predicted response value is then $\hat{f}(x_0)$. We are interested in the error associated with this prediction.

**a) (1P)**
Write down the definition of the expected test mean squared error (MSE) at $x_0$.

**b) (2P)**
Derive the decomposition of the expected test MSE into three terms.

**c) (1P)**
Explain with words how we can interpret the three terms.

---

## Module 3 — Linear regression

### M3.1 — Distribution of the OLS estimator
**Source:** [RecEx3](/exercises/Exercise3/RecEx3) (Problem 2 a)

**a)**
A core finding for the least-squares estimator $\hat{\boldsymbol\beta}$ of linear regression models is
$$ \hat{\boldsymbol\beta} = ({\bf X}^T{\bf X})^{-1} {\bf X}^T {\bf Y} \ , $$
with $\hat{\boldsymbol\beta}\sim N_{p}(\boldsymbol\beta,\sigma^2({\bf X}^T{\bf X})^{-1})$.

* Show that $\hat{\boldsymbol\beta}$ has this distribution with the given mean and covariance matrix.
* What do you need to assume to get to this result?
* What does this imply for the distribution of the $j$th element of $\hat{\boldsymbol\beta}$?
* In particular, how can we calculate the variance of $\hat{\beta}_j$?

### M3.2 — Connection between CI for β, CI for x₀ᵀβ, and PI for Y
**Source:** [RecEx3](/exercises/Exercise3/RecEx3) (Problem 2 d)

**d)**
Construct a 95% CI for ${\boldsymbol x}_0^T \beta$. Explain the connections between a CI for $\beta_j$, a CI for ${\boldsymbol x}_0^T \beta$ and a PI for $Y$ at ${\boldsymbol x}_0$.

### M3.3 — Derive β̂_OLS, then show MLE = OLS
**Source:** [RecEx6](/exercises/Exercise6/RecEx6) (Problem 1)

**1**

a. Show that the least square estimator of a multiple linear regression model is given by
$$
\hat{\boldsymbol \beta} =(\boldsymbol X^T \boldsymbol X)^{-1} \boldsymbol X^T \boldsymbol Y
$$
b. Show that the maximum likelihood estimator is equal to the least square estimator for the multiple linear regression model.

### M3.4 — OLS ≡ MLE under Gaussian errors (worked-out version)
**Source:** [Exam 2024](/pdfs/exam-2024.pdf) (Problem 3 b)

**b) (4P)**

Consider a regression setting and assume an additive error model:

$$
Y_i = f_{\theta}(X_i) + \epsilon_i,\qquad i = 1,\dots,N
$$

where $\theta$ denotes the model parameters defining the regression function $f_{\theta}$, and $\epsilon$ is the error term.

i) (2P) Describe the least squares method and the maximum likelihood method and how they are used to estimate $\theta$. It is enough to state the optimization problems, you do not need to solve them. State also the assumption that are made for each estimation method

ii) (2P) Show that, if you assume a Gaussian distribution for the error term, then the two methods are equivalent with respect to the estimate of $\theta$.

---

## Module 4 — Classification

### M4.1 — LDA assumptions and classification rule
**Source:** [RecEx4](/exercises/Exercise4/RecEx4) (Problem 2 b)

**b)**

Explain the assumptions made to use linear discriminant analysis to classify a new observation to be a genuine or a fake bank note. Write down the classification rule for a new observation (make any assumptions you need to make).

### M4.2 — LDA vs QDA
**Source:** [RecEx4](/exercises/Exercise4/RecEx4) (Problem 2 d)

**d)**

What is the difference between LDA and QDA? Use the classification rule for QDA to determine the bank note from c). Do you obtain the same result? You can use R to perform the matrix calculations.

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

### M4.3 — Logit is linear in covariates
**Source:** [Compulsory 1](/pdfs/compulsory-1.pdf) (Problem 3 a)

We will first create a logistic regression model where the probability to win for player 1 has the form

$$P(Y_i = 1 | \mathbf{X} = \boldsymbol{x}_i) = p_i = \frac{e^{\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_4 x_{i4}}}{1 + e^{\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \beta_3 x_{i3} + \beta_4 x_{i4}}} ,$$

where $x_{i1}$ is the number of aces for player 1 in match $i$, $x_{i2}$ is the number of aces for player 2 in match $i$, and $x_{i3}$ and $x_{i4}$ are the number of unforced errors committed by player 1 and 2 in match $i$.

**a) (1P)**

Use the above expression to show that $\text{logit}(p_i) = \log\left(\frac{p_i}{1-p_i}\right)$ is a linear function of the covariates.

### M4.4 — LDA decision boundary (δ₀ = δ₁, then ax₁+bx₂+c = 0)
**Source:** [Compulsory 1](/pdfs/compulsory-1.pdf) (Problem 3 e)

**e) (3P)**

In a two class problem ($K = 2$) the decision boundary for LDA between class 0 and class 1 is where $x$ satisfies

$$P(Y = 0 | \mathbf{X} = \boldsymbol{x}) = P(Y = 1 | \mathbf{X} = \boldsymbol{x}).$$

- (1P) Show that we can express this as

  $$\delta_0(\boldsymbol{x}) = \delta_1(\boldsymbol{x}),$$

  where

  $$\delta_k(\boldsymbol{x}) = \boldsymbol{x}^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log \pi_k; \quad k \in \{0, 1\}.$$

- (1P) We use the rule to classify an observation with covariates $\boldsymbol{x}$ to class 1 if $\hat{P}(Y = 1 | \boldsymbol{x}) > 0.5$. Write down the formula for the class boundary. Hint: formulate it as $ax_1 + bx_2 + c = 0$ and solve for $x_2$. Use R for the calculations.

### M4.5 — Bayes decision boundary (mixture-Gaussian classes)
**Source:** [Exam 2023](/pdfs/exam-2023.pdf) (Problem 2, second "a" (5P))

**a) (5P)**

In this problem, we consider a simulated data set with two classes (labelled 0 and 1) and two numerical covariates $x_1$ and $x_2$. Let $x = (x_1 , x_2)$ be a column vector with the two covariates. A training set with 500 observations of each class is available, and a scatter plot is given below. We simulate a data set as follows:

* Prior class probabilities: $\pi_0 = P(Y=0) = 0.5$ and $\pi_1= P(Y = 1) = 0.5$.
* Class-specific probabilities

\begin{align*}
P(\bm{x}|y=0) = f_0(\bm{x})=
0.5 \cdot \frac{1}{2\pi |\Sigma|} \cdot \exp\left( -\frac{1}{2} (\bm{x}-\bm{\mu}_{01})^\top \Sigma^{-1} (\bm{x}- \bm{\mu}_{01})\right) + \\
 0.5 \cdot \frac{1}{2\pi |\Sigma|} \cdot \exp\left(- \frac{1}{2} (\bm{x}-\bm{\mu}_{02})^\top \Sigma^{-1} (\bm{x}- \bm{\mu}_{02})\right)
\end{align*}

$$
P(\bm{x}|y=1) = f_1(\bm{x})=
\frac{1}{2\pi |\Sigma|} \cdot \exp\left(-\frac{1}{2} (\bm{x}-\bm{\mu}_{2})^\top\Sigma^{-1}(\bm{x}-\bm{\mu}_{2})\right)
$$

with $\bm{\mu}_{01}=\begin{pmatrix}2 \\ 2 \end{pmatrix}$, $\bm{\mu}_{02}=\begin{pmatrix}4 \\ 2 \end{pmatrix}$, $\bm{\mu}_{2}=\begin{pmatrix}3 \\ 5 \end{pmatrix}$ and $\Sigma = \begin{pmatrix} 1 & 0.5 \\0.5 & 1\end{pmatrix}$.

Given that $\pi_0=\pi_1=0.5$, and the knowledge about the class-specific distributions $f_0(x)$ and $f_1(x)$ given above, the aim is to derive the equation for the Bayes decision boundary to find the Bayes classifier.

(i) (1P) Explain what the Bayes decision boundary actually is.
(ii) (3P) Write down the equation to be solved with the actual values (you are not supposed/asked to solve the equation), and explain what the unknowns are.
(iii) (1P) Is the resulting boundary linear in the covariates? Why?

---

## Module 5 — Resampling

### M5.1 — k-fold CV algorithmic specification
**Source:** [RecEx5](/exercises/Exercise5/RecEx5) (Problem 1)

**Problem 1**

Explain how $k$-fold cross-validation is implemented.

a) Draw a figure.

b) Specify algorithmically what is done, and in particular how the results from each fold are aggregated.

c) Relate $k$-fold cross-validation to an example from regression. Ideas are the complexity with regards to polynomials of increasing degree in multiple linear regression, or $K$ in KNN-regression.

d) Relate $k$-fold cross-validation to an example from classification. Ideas are the complexity with regards to polynomials of increasing degree in logistic regression, or $K$ in KNN-classification.

Hint: the words "loss function," "fold," "training," and "validation" are central.

### M5.2 — Bootstrap inclusion probability → 1 − 1/e
**Source:** [RecEx5](/exercises/Exercise5/RecEx5) (Problem 4)

**Problem 4**

We will calculate the probability that a given observation in our original sample is part of a bootstrap sample. This is useful for us to know in Module 8.

Our sample size is $n$.

a. We draw one observation from our sample. What is the probability of drawing observation $i$ (i.e., $x_i$)? And of not drawing observation $i$?
b. We make $n$ independent draws (with replacement). What is the probability of not drawing observation $i$ in any of the $n$ draws? What is then the probability that data point $i$ is in our bootstrap sample (that is, more than $0$ times)?
c. When $n$ is large $(1-\frac{1}{n})^n \approx \frac{1}{e}$. Use this to give a numerical value for the probability that a specific observation $i$ is in our bootstrap sample.
d. Write a short `R` code chunk to check your result. (Hint: An example on how to this is on page 198 in our ISLR book.) You may also study the result in c. How good is the approximation as a function of $n$?

### M5.3 — Bootstrap SE and 95% CI procedure
**Source:** [RecEx5](/exercises/Exercise5/RecEx5) (Problem 5)

**Problem 5**

Explain with words and an algorithm how you would proceed to use bootstrapping to estimate the standard deviation and the $95\%$ confidence interval of one of the regression parameters in multiple linear regression. Comment on which assumptions you make for your regression model.

### M5.4 — k-fold CV vs LOOCV (write-up + trade-off)
**Source:** [Exam 2024](/pdfs/exam-2024.pdf) (Problem 3 a)

**Problem 3 (theory, 8P)**

**a) (4P)**
We learned about $k$-fold cross-validation (CV) as a way of doing model selection.

(i) (2P) Explain how $k$-fold CV is implemented and how the MSE is computed.
(ii) (2P) State what the advantages and disadvantages of $k$-fold cross-validation (CV) are with respect to the Leave-one-out cross-validation (LOOCV) approach.

---

## Module 7 — Beyond linearity (splines / GAMs)

### M7.1 — Natural cubic spline design matrix from basis
**Source:** [RecEx7](/exercises/Exercise7/RecEx7) (Problem 3)

**Problem 3**

Now, let us look at the `Wage` data set. The section on Additive Models [(in this week's slides)](https://github.com/stefaniemuff/statlearning2/blob/master/7BeyondLinear/7BeyondLinear_2024.pdf) explains how we can create an additive model by adding components together. One type of component we saw is natural cubic splines. Derive the design matrix $\mathbf{X}$ for a natural cubic spline with one internal knot at `year` $= 2006$, from the natural cubic spline basis:

$$
b_1(x_i) = x_i, \quad b_{k+2}(x_i) = d_k(x_i)-d_K(x_i),\; k = 0, \ldots, K - 1,\\
$$

$$
d_k(x_i) = \frac{(x_i-c_k)^3_+-(x_i-c_{K+1})^3_+}{c_{K+1}-c_k}.
$$

### M7.2 — Cubic spline DF count
**Source:** [Exam 2024](/pdfs/exam-2024.pdf) (Problem 2 c)

**c) (1P) Numeric answer**

A covariate is included in a regression model as a natural cubic spline with three cut points. How many degrees of freedom does this spline term consume?

---

## Module 8 — Trees

### M8.1 — Recursive binary splitting + cost-complexity pruning algorithm
**Source:** [RecEx8](/exercises/Exercise8/RecEx8) (Problem 1 a–e)

**Problem 1 -- Theoretical**

a) Provide a detailed explanation of the algorithm that is used to fit a regression tree. What is different for a classification tree?

b) What are the advantages and disadvantages of regression and classification trees?

c) What is the idea behind bagging and what is the role of bootstap? How do random forests improve that idea?

d) What is an out-of bag (OOB) error estimator and what percentage of observations are included in an OOB sample? (Hint: The result from RecEx5-Problem 4c can be used)

e) Bagging and Random Forests typically improve the prediction accuracy of a single tree, but it can be difficult to interpret, for example in terms of understanding which predictors are how relevant. How can we evaluate the importance of the different predictors for these methods?

---

## Module 10 — Unsupervised

### M10.1 — Prove k-means objective is monotone decreasing
**Source:** [RecEx10](/exercises/Exercise10/RecEx10) (Problem 2)

**Problem 2**

Show that the algorithm below is guaranteed to decrease the value of the objective

$$
\underset{C_1, \ldots, C_k}{\text{minimize}}\left\{\sum_{k=1}^{K} \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^{p}(x_{ij} - x_{i'j})^2 \right\}
$$

at each step.

### M10.2 — PCA: PVE, loadings, retain how many components
**Source:** [Exam 2025](/pdfs/exam-2025.pdf) (Problem 2 f)

**f) (3P)**
You perform PCA on a dataset with 5 **standardized** variables: $X_1,X_2,X_3,X_4,X_5$.

You obtain the following results:

Principal Component | Eigenvalue | Proportion of Variance Explained (PVE)
| -------- | ------- |------- |
PC1 | 2.7 | 0.54
PC2 | 1.5 | 0.30
PC3 | 0.5 | 0.10
PC4 | 0.2 | 0.04
PC5 | 0.1 | 0.02

You also know the loadings for the first principal component (PC1):
$$
\text{PC1}=0.85X_1+0.2X_2+0.2X_3+0.1X_4+0.05X_5
$$

You have an observation with the following standardized variable values:
$$
X_1=1,\ X_2=0.5,\ X_3=-0.5,\ X_4=-1,\ X_5=0
$$

**Questions**

   1. What proportion of the total variance is explained by the first four principal components combined?

   2. If you want to retain at least 90% of the total variance, how many principal components do you have to keep?

   3. Calculate the value of PC1 for the observation above, using the given loadings and standardized values.

   4. If you plotted the observations using only PC1 and PC2, what percentage of the original dataset's variability would be represented in the plot?

   5. Which variable contribute most strongly to PC1, based on the loadings?

   6. What is the total variance in the original data?

### M10.3 — Hierarchical clustering dendrogram by hand (complete linkage)
**Source:** [Exam 2025](/pdfs/exam-2025.pdf) (Problem 3 b)

**b) (2P)**
We have four observations for which we know the distance matrix in Euclidean space:

$$
\begin{bmatrix}
0 & 6.5 & 5 & 7 \\
6.5 & 0 & 6 & 4 \\
5 & 6 & 0 & 2 \\
7 & 4 & 2 & 0
\end{bmatrix}
$$

Based on this dissimilarity matrix, sketch the dendogram that results from hierarchical clustering using complete linkage. On the plot, indicate the height where each fusion occurs, as well as the observastions that correspond to the leafs in the dendogram (enumerated as 1, 2, 3, 4).

---

## Module 11 — Neural networks

### M11.1 — Write the feed-forward NN equation from an architecture diagram
**Source:** [RecEx11](/exercises/Exercise11/RecEx11) (Problem 1 a)

**a)**

Write down the equation that describes and input is related to output in this network, using general activation functions $\phi_o$, $\phi_h$ and $\phi_{h^\star}$ and bias nodes in all layers. What would you call such a network?

### M11.2 — Identify architecture & count parameters (ReLU, one hidden layer)
**Source:** [RecEx11](/exercises/Exercise11/RecEx11) (Problem 2 a)

**a)**

Which network architecture and activation functions does this formula correspond to?
$$ \hat{y}_1({\bf x})=\beta_{01}+\sum_{m=1}^5 \beta_{m1}\cdot \max(\alpha_{0m}+\sum_{j=1}^{10} \alpha_{jm}x_j,0)$$
How many parameters are estimated in this network?

### M11.3 — Identify architecture & count parameters (deep, ReLU+sigmoid)
**Source:** [RecEx11](/exercises/Exercise11/RecEx11) (Problem 2 b)

**b)**
Which network architecture and activation functions does this formula give?

$$ \hat{y}_1({\bf x})=(1+\exp(-\beta_{01}-\sum_{m=1}^5 \beta_{m1}\max(\gamma_{0m}+\sum_{l=1}^{10} \gamma_{lm}\max(\sum_{j=1}^{4}\alpha_{jl}x_j,0),0))^{-1}$$

How many parameters are estimated in this network?

### M11.4 — NN parameter count (3 → 4 → 1) + ReLU evaluation
**Source:** [Exam 2025](/pdfs/exam-2025.pdf) (Problem 2 c)

**c) (2P)**

Suppose you build a neural network with:

  * 3 input variables
  * 1 hidden layer with 4 neurons
  * 1 output neuron

Assume also that all units are fully connected.

  (i)  How many weights (parameters) are there in total, including biases?
  (ii) Suppose a hidden layer neuron has:

  - Input values: $x_1 = -1, x_2 = 2, x_3 = 0$
  - Weights: $w_1 = 0.5, w_2 = 0, w_3 = 1$
  - Bias: $b = 0$

    Assume a ReLU activation function.
   Question: What is the output of the neuron?
