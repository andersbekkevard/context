---
title: Math exercises (ISLP)
---

Every "mathy derivation" problem found in the ISLP textbook end-of-chapter exercises, restricted to chapters that are in scope per `docs/scope.md`. Same filter as `math-exercises.md`: derivations, proofs, algebraic show-that questions, and hand-calculations whose algebraic structure is the point. Conceptual prose, R/Python coding, and pure plot-interpretation problems are excluded.

Source links point to the ISLP chapter markdown in `wiki/book/`. Borderline / scope-edge problems carry a brief *Scope note*.

ISLP chapters that map to in-scope modules:

| Module | ISLP ch | File |
|---|---|---|
| 2 — Stat learning | 2 | [02-statlearn.md](../book/02-statlearn.md) |
| 3 — Linear regression | 3 | [03-linreg.md](../book/03-linreg.md) |
| 4 — Classification | 4 | [04-classif.md](../book/04-classif.md) |
| 5 — Resampling | 5 | [05-resample.md](../book/05-resample.md) |
| 6 — Model selection | 6 | [06-modelsel.md](../book/06-modelsel.md) |
| 7 — Beyond linear | 7 | [07-beyondlinear.md](../book/07-beyondlinear.md) |
| 8 — Trees, 9 — Boosting | 8 | [08-trees.md](../book/08-trees.md) |
| 10 — Unsupervised | 12 | [12-unsupervised.md](../book/12-unsupervised.md) |
| 11 — Neural networks | 10 | [10-deeplearning.md](../book/10-deeplearning.md) |

ISLP ch 9 (SVM), ch 11 (survival), and ch 13 (multiple testing) are excluded — prof took them out of scope.

---

## Module 2 — Statistical learning

Chapter 2's conceptual exercises (2.4.1–2.4.7) are all verbal/intuition prompts (KNN by hand, bias-variance reasoning) — no algebraic derivations. **Nothing in scope from this chapter.** The bias-variance derivation lives in the lecture notes and Compulsory 1, not ISLP exercises.

---

## Module 3 — Linear regression

### Exercise 3.5 — Fitted values as linear combinations of responses
**Source:** [03-linreg.md:1470-1490](../book/03-linreg.md) (Conceptual 5)

Consider the fitted values that result from performing linear regression without an intercept. In this setting, the $i$th fitted value takes the form

$$\hat{y}_i = x_i \hat\beta,$$

where

$$\hat\beta = \frac{\sum_{i=1}^{n} x_i y_i}{\sum_{i'=1}^{n} x_{i'}^2}. \quad (3.38)$$

Show that we can write

$$\hat{y}_i = \sum_{i'=1}^{n} a_{i'} y_{i'}.$$

What is $a_{i'}$?

*Note: We interpret this result by saying that the fitted values from linear regression are linear combinations of the response values.*

### Exercise 3.6 — Regression line passes through the means
**Source:** [03-linreg.md:1492](../book/03-linreg.md) (Conceptual 6)

Using (3.4), argue that in the case of simple linear regression, the least squares line always passes through the point $(\bar{x}, \bar{y})$.

### Exercise 3.7 — R² equals squared correlation
**Source:** [03-linreg.md:1494](../book/03-linreg.md) (Conceptual 7)

It is claimed in the text that in the case of simple linear regression of $Y$ onto $X$, the $R^2$ statistic (3.17) is equal to the square of the correlation between $X$ and $Y$ (3.18). Prove that this is the case. For simplicity, you may assume that $\bar{x} = \bar{y} = 0$.

### Exercise 3.11 (d, e) — Algebraic form of the t-statistic; symmetry under x ↔ y swap
**Source:** [03-linreg.md:1542-1554](../book/03-linreg.md) (Applied 11, parts d–e)

For the regression of $Y$ onto $X$ without an intercept, the $t$-statistic for $H_0: \beta = 0$ takes the form $\hat\beta / \mathrm{SE}(\hat\beta)$, where $\hat\beta$ is given by (3.38), and where

$$\mathrm{SE}(\hat\beta) = \sqrt{\frac{\sum_{i=1}^{n}(y_i - x_i \hat\beta)^2}{(n-1) \sum_{i'=1}^{n} x_{i'}^2}}.$$

(These formulas are slightly different from those given in Sections 3.1.1 and 3.1.2, since here we are performing regression without an intercept.) Show algebraically, and confirm numerically in `R`, that the $t$-statistic can be written as

$$\frac{(\sqrt{n - 1}) \sum_{i=1}^{n} x_i y_i}{\sqrt{\left(\sum_{i=1}^{n} x_i^2\right)\!\left(\sum_{i'=1}^{n} y_{i'}^2\right) - \left(\sum_{i'=1}^{n} x_{i'} y_{i'}\right)^2}}.$$

**(e)** Using the results from (d), argue that the $t$-statistic for the regression of `y` onto `x` is the same as the $t$-statistic for the regression of `x` onto `y`.

---

## Module 4 — Classification

### Exercise 4.8.1 — Logistic ⇔ logit equivalence
**Source:** [04-classif.md:1861](../book/04-classif.md) (Conceptual 1)

Using a little bit of algebra, prove that (4.2) is equivalent to (4.3). In other words, the logistic function representation and logit representation for the logistic regression model are equivalent.

### Exercise 4.8.2 — Bayes classifier reduces to LDA discriminant
**Source:** [04-classif.md:1863](../book/04-classif.md) (Conceptual 2)

It was stated in the text that classifying an observation to the class for which (4.17) is largest is equivalent to classifying an observation to the class for which (4.18) is largest. Prove that this is the case. In other words, under the assumption that the observations in the $k$th class are drawn from a $\mathcal{N}(\mu_k, \sigma^2)$ distribution, the Bayes classifier assigns an observation to the class for which the discriminant function is maximized.

### Exercise 4.8.3 — Bayes classifier is quadratic when σ²_k differs
**Source:** [04-classif.md:1865-1869](../book/04-classif.md) (Conceptual 3)

This problem relates to the QDA model, in which the observations within each class are drawn from a normal distribution with a class-specific mean vector and a class specific covariance matrix. We consider the simple case where $p = 1$; i.e. there is only one feature.

Suppose that we have $K$ classes, and that if an observation belongs to the $k$th class then $X$ comes from a one-dimensional normal distribution, $X \sim \mathcal{N}(\mu_k, \sigma_k^2)$. Recall that the density function for the one-dimensional normal distribution is given in (4.16). Prove that in this case, the Bayes classifier is *not* linear. Argue that it is in fact quadratic.

*Hint:* For this problem, you should follow the arguments laid out in Section 4.4.1, but without making the assumption that $\sigma_1^2 = \cdots = \sigma_K^2$.

### Exercise 4.8.7 — Predicted probability from Bayes theorem with Gaussian classes
**Source:** [04-classif.md:1901-1903](../book/04-classif.md) (Conceptual 7)

Suppose that we wish to predict whether a given stock will issue a dividend this year ("Yes" or "No") based on $X$, last year's percent profit. We examine a large number of companies and discover that the mean value of $X$ for companies that issued a dividend was $\bar{X} = 10$, while the mean for those that didn't was $\bar{X} = 0$. In addition, the variance of $X$ for these two sets of companies was $\hat{\sigma}^2 = 36$. Finally, 80% of companies issued dividends. Assuming that $X$ follows a normal distribution, predict the probability that a company will issue a dividend this year given that its percentage profit was $X = 4$ last year.

*Hint:* Recall that the density function for a normal random variable is $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x - \mu)^2 / 2\sigma^2}$. You will need to use Bayes' theorem.

### Exercise 4.8.9 — Odds ↔ probability conversion
**Source:** [04-classif.md:1907-1911](../book/04-classif.md) (Conceptual 9)

This problem has to do with *odds*.

**(a)** On average, what fraction of people with an odds of 0.37 of defaulting on their credit card payment will in fact default?

**(b)** Suppose that an individual has a 16% chance of defaulting on her credit card payment. What are the odds that she will default?

### Exercise 4.8.10 — Univariate LDA log-odds expressions (a_k, b_kj)
**Source:** [04-classif.md:1913](../book/04-classif.md) (Conceptual 10)

Equation 4.32 derived an expression for $\log\!\left(\frac{\Pr(Y = k \mid X = x)}{\Pr(Y = K \mid X = x)}\right)$ in the setting where $p > 1$, so that the mean for the $k$th class, $\mu_k$, is a $p$-dimensional vector, and the shared covariance $\boldsymbol{\Sigma}$ is a $p \times p$ matrix. However, in the setting with $p = 1$, (4.32) takes a simpler form, since the means $\mu_1, \ldots, \mu_K$ and the variance $\sigma^2$ are scalars. In this simpler setting, repeat the calculation in (4.32), and provide expressions for $a_k$ and $b_{kj}$ in terms of $\pi_k$, $\pi_K$, $\mu_k$, $\mu_K$, and $\sigma^2$.

*Scope note: invokes multi-class LDA framing; the multi-class generalization of logistic regression is OUT of scope (prof Jan 27), but LDA discriminant derivation itself is IN.*

### Exercise 4.8.11 — Detailed QDA discriminant coefficients
**Source:** [04-classif.md:1915](../book/04-classif.md) (Conceptual 11)

Work out the detailed forms of $a_k$, $b_{kj}$, and $b_{kjl}$ in (4.33). Your answer should involve $\pi_k$, $\pi_K$, $\mu_k$, $\mu_K$, $\boldsymbol{\Sigma}_k$, and $\boldsymbol{\Sigma}_K$.

### Exercise 4.8.12 — Binary logistic ↔ softmax parameterization
**Source:** [04-classif.md:1917-1937](../book/04-classif.md) (Conceptual 12)

Suppose that you wish to classify an observation $X \in \mathbb{R}$ into `apples` and `oranges`. You fit a logistic regression model and find that

$$\widehat{\Pr}(Y = \texttt{orange} \mid X = x) = \frac{\exp(\hat{\beta}_0 + \hat{\beta}_1 x)}{1 + \exp(\hat{\beta}_0 + \hat{\beta}_1 x)}.$$

Your friend fits a logistic regression model to the same data using the *softmax* formulation in (4.13), and finds that

$$\widehat{\Pr}(Y = \texttt{orange} \mid X = x) = \frac{\exp(\hat{\alpha}_{\texttt{orange}0} + \hat{\alpha}_{\texttt{orange}1} x)}{\exp(\hat{\alpha}_{\texttt{orange}0} + \hat{\alpha}_{\texttt{orange}1} x) + \exp(\hat{\alpha}_{\texttt{apple}0} + \hat{\alpha}_{\texttt{apple}1} x)}.$$

**(a)** What is the log odds of `orange` versus `apple` in your model?

**(b)** What is the log odds of `orange` versus `apple` in your friend's model?

**(c)** Suppose that in your model, $\hat{\beta}_0 = 2$ and $\hat{\beta}_1 = -1$. What are the coefficient estimates in your friend's model? Be as specific as possible.

**(d)** Now suppose that you and your friend fit the same two models on a different data set. This time, your friend gets the coefficient estimates $\hat{\alpha}_{\texttt{orange}0} = 1.2$, $\hat{\alpha}_{\texttt{orange}1} = -2$, $\hat{\alpha}_{\texttt{apple}0} = 3$, $\hat{\alpha}_{\texttt{apple}1} = 0.6$. What are the coefficient estimates in your model?

**(e)** Finally, suppose you apply both models from (d) to a data set with 2,000 test observations. What fraction of the time do you expect the predicted class labels from your model to agree with those from your friend's model? Explain your answer.

*Scope note: softmax over-parameterization. Useful framing for NN output layer, but the multi-class logistic specifics are OUT of scope per prof. Treat as illustrative for the binary ↔ softmax algebra.*

---

## Module 5 — Resampling

### Exercise 5.4.1 — Derive minimum-variance portfolio weight
**Source:** [05-resample.md:633](../book/05-resample.md) (Conceptual 1)

Using basic statistical properties of the variance, as well as single-variable calculus, derive (5.6). In other words, prove that $\alpha$ given by (5.6) does indeed minimize $\text{Var}(\alpha X + (1 - \alpha)Y)$.

### Exercise 5.4.2 — Bootstrap inclusion probability
**Source:** [05-resample.md:635-653](../book/05-resample.md) (Conceptual 2)

We will now derive the probability that a given observation is part of a bootstrap sample. Suppose that we obtain a bootstrap sample from a set of $n$ observations.

**(a)** What is the probability that the first bootstrap observation is *not* the $j$th observation from the original sample? Justify your answer.

**(b)** What is the probability that the second bootstrap observation is *not* the $j$th observation from the original sample?

**(c)** Argue that the probability that the $j$th observation is *not* in the bootstrap sample is $(1 - 1/n)^n$.

**(d)** When $n = 5$, what is the probability that the $j$th observation is in the bootstrap sample?

**(e)** When $n = 100$, what is the probability that the $j$th observation is in the bootstrap sample?

**(f)** When $n = 10{,}000$, what is the probability that the $j$th observation is in the bootstrap sample?

**(g)** Create a plot that displays, for each integer value of $n$ from 1 to 100,000, the probability that the $j$th observation is in the bootstrap sample. Comment on what you observe.

---

## Module 6 — Model selection / regularization

### Exercise 6.6.5 — Ridge vs lasso geometry under correlated predictors
**Source:** [06-modelsel.md:1416-1422](../book/06-modelsel.md) (Conceptual 5)

It is well-known that ridge regression tends to give similar coefficient values to correlated variables, whereas the lasso may give quite different coefficient values to correlated variables. We will now explore this property in a very simple setting.

Suppose that $n = 2$, $p = 2$, $x_{11} = x_{12}$, $x_{21} = x_{22}$. Furthermore, suppose that $y_1 + y_2 = 0$ and $x_{11} + x_{21} = 0$ and $x_{12} + x_{22} = 0$, so that the estimate for the intercept in a least squares, ridge regression, or lasso model is zero: $\hat{\beta}_0 = 0$.

**(a)** Write out the ridge regression optimization problem in this setting.

**(b)** Argue that in this setting, the ridge coefficient estimates satisfy $\hat{\beta}_1 = \hat{\beta}_2$.

**(c)** Write out the lasso optimization problem in this setting.

**(d)** Argue that in this setting, the lasso coefficients $\hat{\beta}_1$ and $\hat{\beta}_2$ are not unique—in other words, there are many possible solutions to the optimization problem in (c). Describe these solutions.

*ISLP 6.6.7 (Bayesian connection to ridge/lasso via Gaussian / Laplace priors) is omitted — prof Mar 2: "I really don't think I'd put this on the test… kind of assumes a lot of knowledge that maybe you don't have."*

---

## Module 7 — Beyond linearity (splines / GAMs)

### Exercise 7.9.1 — Cubic spline basis: continuity of f, f′, f″ at the knot
**Source:** [07-beyondlinear.md:1124-1156](../book/07-beyondlinear.md) (Conceptual 1)

It was mentioned in this chapter that a cubic regression spline with one knot at $\xi$ can be obtained using a basis of the form $x, x^2, x^3, (x - \xi)_+^3$, where $(x - \xi)_+^3 = (x - \xi)^3$ if $x > \xi$ and equals 0 otherwise. We will now show that a function of the form

$$f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \beta_4 (x - \xi)_+^3$$

is indeed a cubic regression spline, regardless of the values of $\beta_0, \beta_1, \beta_2, \beta_3, \beta_4$.

**(a)** Find a cubic polynomial

$$f_1(x) = a_1 + b_1 x + c_1 x^2 + d_1 x^3$$

such that $f(x) = f_1(x)$ for all $x \le \xi$. Express $a_1, b_1, c_1, d_1$ in terms of $\beta_0, \beta_1, \beta_2, \beta_3, \beta_4$.

**(b)** Find a cubic polynomial

$$f_2(x) = a_2 + b_2 x + c_2 x^2 + d_2 x^3$$

such that $f(x) = f_2(x)$ for all $x > \xi$. Express $a_2, b_2, c_2, d_2$ in terms of $\beta_0, \beta_1, \beta_2, \beta_3, \beta_4$. We have now established that $f(x)$ is a piecewise polynomial.

**(c)** Show that $f_1(\xi) = f_2(\xi)$. That is, $f(x)$ is continuous at $\xi$.

**(d)** Show that $f_1'(\xi) = f_2'(\xi)$. That is, $f'(x)$ is continuous at $\xi$.

**(e)** Show that $f_1''(\xi) = f_2''(\xi)$. That is, $f''(x)$ is continuous at $\xi$.

Therefore, $f(x)$ is indeed a cubic spline.

*Scope note: prof said Mar 9 he won't derive natural-spline basis math; this is the **truncated-power** cubic-spline basis, which is foundational. Borderline.*

### Exercise 7.9.5 — Comparing smoothing-spline penalty orders
**Source:** [07-beyondlinear.md:1182-1192](../book/07-beyondlinear.md) (Conceptual 5)

Consider two curves, $\hat{g}_1$ and $\hat{g}_2$, defined by

$$\hat{g}_1 = \arg\min_g \left( \sum_{i=1}^{n} (y_i - g(x_i))^2 + \lambda \int \left[ g^{(3)}(x) \right]^2 dx \right),$$

$$\hat{g}_2 = \arg\min_g \left( \sum_{i=1}^{n} (y_i - g(x_i))^2 + \lambda \int \left[ g^{(4)}(x) \right]^2 dx \right),$$

where $g^{(m)}$ represents the $m$th derivative of $g$.

**(a)** As $\lambda \to \infty$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller training RSS?

**(b)** As $\lambda \to \infty$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller test RSS?

**(c)** For $\lambda = 0$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller training and test RSS?

---

## Modules 8 & 9 — Trees & boosting

### Exercise 8.4.2 — Why boosting stumps gives an additive model
**Source:** [08-trees.md:895-901](../book/08-trees.md) (Conceptual 2)

It is mentioned in Section 8.2.3 that boosting using depth-one trees (or *stumps*) leads to an *additive* model: that is, a model of the form

$$f(X) = \sum_{j=1}^{p} f_j(X_j).$$

Explain why this is the case. You can begin with (8.12) in Algorithm 8.2.

### Exercise 8.4.3 — Gini, classification error, entropy as functions of p̂
**Source:** [08-trees.md:903-905](../book/08-trees.md) (Conceptual 3)

Consider the Gini index, classification error, and entropy in a simple classification setting with two classes. Create a single plot that displays each of these quantities as a function of $\hat p_{m1}$. The $x$-axis should display $\hat p_{m1}$, ranging from 0 to 1, and the $y$-axis should display the value of the Gini index, classification error, and entropy.

*Hint: In a setting with two classes, $\hat p_{m1} = 1 - \hat p_{m2}$. You could make this plot by hand, but it will be much easier to make in `R`.*

*Scope note: the by-hand version (write each formula in p̂, evaluate at a few points) is the exam-shaped version; the R-plot framing is for the lab.*

### Exercise 8.4.4 — Tree ↔ partition reconstruction
**Source:** [08-trees.md:907-912](../book/08-trees.md) (Conceptual 4)

This question relates to the plots in Figure 8.14.

> **Figure 8.14.** Left: A partition of the predictor space corresponding to Exercise 4a. Right: A tree corresponding to Exercise 4b.

**(a)** Sketch the tree corresponding to the partition of the predictor space illustrated in the left-hand panel of Figure 8.14. The numbers inside the boxes indicate the mean of $Y$ within each region.

**(b)** Create a diagram similar to the left-hand panel of Figure 8.14, using the tree illustrated in the right-hand panel of the same figure. You should divide up the predictor space into the correct regions, and indicate the mean for each region.

### Exercise 8.4.5 — Bagging: majority vote vs average probability
**Source:** [08-trees.md:914-920](../book/08-trees.md) (Conceptual 5)

Suppose we produce ten bootstrapped samples from a data set containing red and green classes. We then apply a classification tree to each bootstrapped sample and, for a specific value of $X$, produce 10 estimates of $P(\text{Class is Red} \mid X)$:

$$0.1,\ 0.15,\ 0.2,\ 0.2,\ 0.55,\ 0.6,\ 0.6,\ 0.65,\ 0.7,\ \text{and } 0.75.$$

There are two common ways to combine these results together into a single class prediction. One is the majority vote approach discussed in this chapter. The second approach is to classify based on the average probability. In this example, what is the final classification under each of these two approaches?

### Exercise 8.4.6 — Regression-tree fitting algorithm
**Source:** [08-trees.md:922](../book/08-trees.md) (Conceptual 6)

Provide a detailed explanation of the algorithm that is used to fit a regression tree.

---

## Module 10 — Unsupervised (ISLP ch 12)

### Exercise 12.6.1 — Prove K-means monotone decrease
**Source:** [12-unsupervised.md:1281-1283](../book/12-unsupervised.md) (Conceptual 1)

This problem involves the $K$-means clustering algorithm.

**(a)** Prove (12.18).

**(b)** On the basis of this identity, argue that the $K$-means clustering algorithm (Algorithm 12.2) decreases the objective (12.17) at each iteration.

### Exercise 12.6.2 — Hand-build dendrograms (complete & single linkage)
**Source:** [12-unsupervised.md:1285-1299](../book/12-unsupervised.md) (Conceptual 2)

Suppose that we have four observations, for which we compute a dissimilarity matrix, given by

$$
\begin{bmatrix}
    & 0.3 & 0.4 & 0.7 \\
0.3 &     & 0.5 & 0.8 \\
0.4 & 0.5 &     & 0.45 \\
0.7 & 0.8 & 0.45 &
\end{bmatrix}.
$$

For instance, the dissimilarity between the first and second observations is 0.3, and the dissimilarity between the second and fourth observations is 0.8.

**(a)** On the basis of this dissimilarity matrix, sketch the dendrogram that results from hierarchically clustering these four observations using complete linkage. Be sure to indicate on the plot the height at which each fusion occurs, as well as the observations corresponding to each leaf in the dendrogram.

**(b)** Repeat (a), this time using single linkage clustering.

**(c)** Suppose that we cut the dendrogram obtained in (a) such that two clusters result. Which observations are in each cluster?

**(d)** Suppose that we cut the dendrogram obtained in (b) such that two clusters result. Which observations are in each cluster?

**(e)** It is mentioned in this chapter that at each fusion in the dendrogram, the position of the two clusters being fused can be swapped without changing the meaning of the dendrogram. Draw a dendrogram that is equivalent to the dendrogram in (a), for which two or more of the leaves are repositioned, but for which the meaning of the dendrogram is the same.

### Exercise 12.6.3 — Hand-execute K-means with K=2, n=6, p=2
**Source:** [12-unsupervised.md:1301-1317](../book/12-unsupervised.md) (Conceptual 3)

In this problem, you will perform $K$-means clustering manually, with $K = 2$, on a small example with $n = 6$ observations and $p = 2$ features. The observations are as follows.

| Obs. | $X_1$ | $X_2$ |
|:----:|:-----:|:-----:|
|  1   |   1   |   4   |
|  2   |   1   |   3   |
|  3   |   0   |   4   |
|  4   |   5   |   1   |
|  5   |   6   |   2   |
|  6   |   4   |   0   |

**(a)** Plot the observations.

**(b)** Randomly assign a cluster label to each observation. You can use the `np.random.choice()` function to do this. Report the cluster labels for each observation.

**(c)** Compute the centroid for each cluster.

**(d)** Assign each observation to the centroid to which it is closest, in terms of Euclidean distance. Report the cluster labels for each observation.

**(e)** Repeat (c) and (d) until the answers obtained stop changing.

**(f)** In your plot from (a), color the observations according to the cluster labels obtained.

### Exercise 12.6.4 — Single vs complete linkage fusion heights
**Source:** [12-unsupervised.md:1319-1321](../book/12-unsupervised.md) (Conceptual 4)

Suppose that for a particular data set, we perform hierarchical clustering using single linkage and using complete linkage. We obtain two dendrograms.

**(a)** At a certain point on the single linkage dendrogram, the clusters $\{1, 2, 3\}$ and $\{4, 5\}$ fuse. On the complete linkage dendrogram, the clusters $\{1, 2, 3\}$ and $\{4, 5\}$ also fuse at a certain point. Which fusion will occur higher on the tree, or will they fuse at the same height, or is there not enough information to tell?

**(b)** At a certain point on the single linkage dendrogram, the clusters $\{5\}$ and $\{6\}$ fuse. On the complete linkage dendrogram, the clusters $\{5\}$ and $\{6\}$ also fuse at a certain point. Which fusion will occur higher on the tree, or will they fuse at the same height, or is there not enough information to tell?

### Exercise 12.6.6 — PCA loadings via p separate least-squares regressions
**Source:** [12-unsupervised.md:1325-1327](../book/12-unsupervised.md) (Conceptual 6)

We saw in Section 12.2.2 that the principal component loading and score vectors provide an approximation to a matrix, in the sense of (12.5). Specifically, the principal component score and loading vectors solve the optimization problem given in (12.6).

Now, suppose that the $M$ principal component score vectors $z_{im}$, $m = 1, \ldots, M$, are known. Using (12.6), explain that each of the first $M$ principal component loading vectors $\phi_{jm}$, $m = 1, \ldots, M$, can be obtained by performing $p$ separate least squares linear regressions. In each regression, the principal component score vectors are the predictors, and one of the features of the data matrix is the response.

---

## Module 11 — Neural networks (ISLP ch 10)

### Exercise 10.10.1 — Two-hidden-layer NN: write f(X), count parameters
**Source:** [10-deeplearning.md:2379-2383](../book/10-deeplearning.md) (Conceptual 1)

Consider a neural network with two hidden layers: $p = 4$ input units, 2 units in the first hidden layer, 3 units in the second hidden layer, and a single output.

**(a)** Draw a picture of the network, similar to Figures 10.1 or 10.4.

**(b)** Write out an expression for $f(X)$, assuming ReLU activation functions. Be as explicit as you can!

**(c)** Now plug in some values for the coefficients and write out the value of $f(X)$.

**(d)** How many parameters are there?

### Exercise 10.10.2 — Softmax invariance under additive shifts
**Source:** [10-deeplearning.md:2385-2389](../book/10-deeplearning.md) (Conceptual 2)

Consider the softmax function in (10.13) (see also (4.13) on page 145) for modeling multinomial probabilities.

**(a)** In (10.13), show that if we add a constant $c$ to each of the $z_\ell$, then the probability is unchanged.

**(b)** In (4.13), show that if we add constants $c_j$, $j = 0, 1, \ldots, p$, to each of the corresponding coefficients for each of the classes, then the predictions at any new point $x$ are unchanged.

This shows that the softmax function is *over-parametrized*. However, regularization and SGD typically constrain the solutions so that this is not a problem.

*Scope note: softmax is the NN multi-class output activation. Useful invariance proof but multi-class logistic specifics are OUT of scope per prof.*

### Exercise 10.10.3 — Multinomial log-likelihood reduces to binomial for M=2
**Source:** [10-deeplearning.md:2391](../book/10-deeplearning.md) (Conceptual 3)

Show that the negative multinomial log-likelihood (10.14) is equivalent to the negative log of the likelihood expression (4.5) when there are $M = 2$ classes.

*Scope note: same as above — useful for understanding the binary case from the multinomial form, but the multinomial framing itself is OUT.*

*ISLP 10.10.4 (CNN parameter counting + filter weight constraints) is omitted — prof Apr 28 said detailed CNN math is out of scope.*
