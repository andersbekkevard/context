---
lecture: 19
date: 2026-03-17
module: 09-boosting
title: Boosting and Additive Trees 1 (Trees wrap-up)
slides: modules/9TreeBoosting/9TreeBoosting.md
topics:
  - bagging
  - out-of-bag-error
  - variable-importance
  - random-forest
  - decorrelation
  - boosting
  - adaboost
  - gradient-boosting
  - weak-learner-and-learning-rate
tags:
  - lecture
  - module/08-trees
  - module/09-boosting
aliases:
  - Lecture 19
---

# L19: Boosting and Additive Trees 1 (Trees wrap-up)

The prof closes module 8 by walking through the ensemble-of-trees pipeline (bagging, [[out-of-bag-error|OOB]] error, variable-importance plots with impurity vs. randomization, then [[random-forest|random forests]] with a derivation of the correlated-trees variance formula motivating decorrelation). After the break he opens module 9 with [[boosting]]: AdaBoost (sequential weak classifiers with re-weighted data) and gradient boosting on regression trees (sequential fitting to residuals), ending with the punchline that fitting residuals **is** taking a gradient step on squared-error loss, which is why the method generalizes to other loss functions.

## Key takeaways

- **Individual trees suck; ensembles fix them.** A single tree is interpretable and powerful but high-variance and sensitive to data. The fix is averaging many trees. Three flavors covered today: **bagging**, **random forests**, **boosting** (Adaboost + gradient boosting). The prof numbered them 1, 2, 3, 4 explicitly.
- **OOB error is free test error.** Each bootstrap sample uses ~2/3 of the data; the remaining ~1/3 ("out of bag") gives an honest validation set per tree. "It's approximately B divided by 3."
- **Two variable-importance flavors, both useful.** **Impurity**-based: total decrease in RSS / Gini across splits, averaged over trees. **Randomization**-based: permute one predictor in the OOB samples, measure performance drop. Prof: *"I would generally go with randomization one because it makes more sense, but both seem reasonable."*
- **Why random forests > bagging: the correlated-trees formula.** With correlated predictions, $\operatorname{Var}\!\bigl(\bar X\bigr) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. The first term **does not shrink with B**. So just adding more bagged trees has a floor; you have to **decorrelate** them. RF does that by giving each split only a random subset of $m$ predictors. Defaults: $m=\sqrt{p}$ for classification, $m=p/3$ for regression.
- **B is "use enough"; M for boosting is a real tuning parameter.** For bagging/RF, number of trees $B$ "just has to be enough," not really tuned, no CV. For boosting, the trees build on each other, so the count matters and you don't blast 1000 of them.
- **AdaBoost** sequentially refits weak classifiers, **re-weighting samples** so points the previous model got wrong dominate the next fit; final prediction is the sign of the $\alpha$-weighted vote. Trees here are **stumps** (very small).
- **Gradient boosting on regression trees** sequentially fits a small tree to the **residuals**, scaled by a **learning rate** $\eta$. Punchline: this is steepest descent. For squared-error loss, the residual *is* the (negative) gradient. Generalizes to any differentiable loss, hence "gradient" boosting.

## Recap: from a single tree to averaging

The prof opens by recapping where we ended last time. Trees are an algorithmic model that recursively splits the predictor space into regions and predicts within each, for regression or classification.

> "The individual trees are not particularly good models. They're sensitive to which data you use. They're very powerful, very cool (they're interpretable and you get regions and everything) but they're sensitive to things."

So the trick is to use *more than one*. In the ideal world of infinite IID datasets, if each model has variance $\sigma^2$, then the average across $B$ independent trees has variance $\sigma^2 / B$, so variance just shrinks with $B$. That's the motivation for ensembling.

> "It's this idea of model averaging, ensembling. We've heard of it before… we're going to talk about how we can make these ensembles better."

We don't have infinite data, so we **bootstrap** to fake it.

## Bagging

### What it is

**Bagging = bootstrap aggregating.** Resample the data with replacement $B$ times, fit a tree on each bootstrap sample, average the predictions. For classification, each tree gets one equal vote, majority wins.

> "It's as simple as it sounds."

### Out-of-bag error

For each bootstrap sample, only ~2/3 of the original data ends up in it. The leftover ~1/3 (the **out-of-bag (OOB)** observations) can be used as a free test set per tree.

> "It's approximately B divided by 3."

(Said quickly; the meaning is that ~1/3 of samples are out-of-bag for any given tree, giving an OOB error estimate without needing a separate held-out set.)

### Worked example: head injury data

500 bagged trees on the head-injury classification, fit with the `randomForest` R function but with `mtry = 10` (i.e. all predictors available at every split, which turns it into bagging, not RF).

Confusion matrix on the test set: 63 true negatives, 83 true positives. Total error 0.15. **Worse than the single-tree result** (~0.145 / ~0.141 from yesterday).

> "It didn't perform as well as we hoped. And considering that we made 500 trees, it's maybe a little disappointing. But it at least gets the idea across. We'll get to better versions."

### The interpretability cost

Single trees are easy to read: "this region predicts here, that region predicts there." Aggregated ensembles destroy that. The fix is **variable importance plots**, a way to recover *which predictors matter* even when you can't read the trees individually.

## Variable importance plots

Two general flavors. Both apply to bagging, RF, and (some forms of) boosting.

### Node-impurity-based

For each predictor, sum up the total decrease in node impurity it produces across all splits, averaged over all trees.

- **Regression**: impurity = RSS.
- **Classification**: impurity = Gini index (or cross-entropy).

It tells you how much each predictor is "doing the work" of fitting.

### Randomization-based (uses OOB)

Mechanically:

1. Run the OOB samples through the trained ensemble and record performance.
2. Take **one predictor at a time**, randomly permute its values across the OOB samples (whole column gets shuffled), keep the trees fixed.
3. Re-evaluate on the corrupted OOB set. Compare to step 1.
4. Average the difference across trees. Optionally normalize by the SD of the differences.

> "If X is important, permuting the observations will decrease the performance a lot. If it doesn't matter, then it won't matter."

### Comparing the two on real data

**Fuel consumption data** (mpg, weight, horsepower, displacement, cylinders, ...): both methods identify weight (`WT`) and horsepower as the dominant predictors. Order is largely preserved between the two. Gap sizes differ: one has the top two close, the other has them far apart. "They do measure slightly different things, so you don't expect them to be the same."

> "I don't have a strong preference here. I think I would generally go with randomization one because it makes more sense. But both seem reasonable."

**Head injury data**: top-4 the same in both rankings, but order shuffles slightly within those four. Bottom predictor (`skull` something) is the same in both, clearly the least useful.

> "Anyways, they do capture different aspects of what's being measured."

## Random forests

### Motivation: bagged trees are correlated

If there's a strong predictor in the data, every bootstrap sample will pick it first → every tree starts with roughly the same root split → ensemble lacks diversity. With the head-injury data, "the first predictor it chose was always the same" across all bagged trees.

> "If you want to have a lot of opinions, don't clone the same person 50 times. They're just going to say the same thing. Figure out a way to add diversity into your pool."

### Variance for correlated predictors: the on-board derivation

The prof did this on the board (this is the slide-deck "Recall: variance for independent / correlated datasets" part). Setup: $B$ predictors with identical variance $\sigma^2$ and pairwise correlation $\rho$, so $\operatorname{Cov}(X_i, X_j) = \rho\sigma^2$ for $i \neq j$.

Starting from $\operatorname{Var}\!\bigl(\frac{1}{B}\sum_i X_i\bigr) = \frac{1}{B^2}\operatorname{Var}\!\bigl(\sum_i X_i\bigr)$, expand the variance into a covariance double-sum and split into diagonal + off-diagonal:

$$
\operatorname{Var}\!\Big(\tfrac{1}{B}\textstyle\sum_i X_i\Big)
\;=\; \tfrac{1}{B^2}\Big[\, B\sigma^2 \;+\; 2\sum_{i<j} \operatorname{Cov}(X_i,X_j) \,\Big]
$$

The off-diagonal sum has $\frac{1}{2}B(B-1)$ terms each equal to $\rho\sigma^2$, giving:

$$
\operatorname{Var}\!\Big(\tfrac{1}{B}\textstyle\sum_i X_i\Big)
\;=\; \tfrac{\sigma^2}{B} \;+\; \tfrac{B-1}{B}\,\rho\sigma^2
\;=\; \rho\sigma^2 \;+\; \tfrac{1-\rho}{B}\sigma^2
$$

> [!important] The point of the derivation
> "This thing does not shrink with respect to $B$. No matter how many times we make different trees, if they're all correlated the same way, then there's a portion that just never improves."

Intuition: if all your models share the same bias ("they all say potato"), throwing in more of them doesn't help. They all say potato.

> "We want our individual models to not have these annoying correlations. We want to add diversity into our models."

### How RF decorrelates: random subset of predictors per split

Force diversity not by resampling the data more aggressively but by **restricting the predictor pool at each split**. Every split, the algorithm gets a fresh random subset of $m$ predictors and can only choose from those.

This breaks the dominance of strong predictors at the root: in some fraction of trees the strong one isn't even on the menu, so the tree starts with a different split → diverges thereafter.

**Defaults:**
- Classification: $m = \sqrt{p}$
- Regression: $m = p/3$

(Bagging is the special case $m = p$.)

### Number of trees: not a tuning parameter

> "It just has to be enough… as long as it's enough of them, you're fine. You don't typically estimate this, you don't typically run cross-validation — you can use the OOB error if you want, but really, it's just use enough of them."

Slide example (gene expression, classification): test error vs. $B$ saturates well before $B = 500$. So pick a big number and move on. The hyperparameter that *does* matter is $m$.

The same plot also showed that $m = \sqrt{p}$ (the convention for classification) substantially outperformed $m = p$ and $m = p/2$, concrete evidence that decorrelation pays.

### Worked example: head injury, $\sqrt{p}$

10 predictors → $m = 3$. 500 trees. Confusion matrix: fewer off-diagonal errors than bagging. Total error: **the lowest seen so far**, better than bagging, better than the single tree.

> "This is currently our winner."

Variable-importance plots are roughly similar to the bagging version: top-4 mostly preserved, bottom predictor still last.

### Worked example: Boston housing, regression

Median home value, ~13 predictors. Pipeline:

- **Single regression tree**: test MSE = 35. Predictions vs. truth fall on a line, modest spread.
- **Bagging** (`mtry = 13`, all predictors at every split): test MSE = 23.67. Visible improvement.
- **Random forest** ($m = p/3 \approx 4$): test MSE = 18.9. Best yet.

> "It's improved with respect to simple bagging, it's likely because of the more diversity in our trees. And now we are talking about a forest and no longer a tree, and we have trees, trees live in a forest, and we're adding more diversity to our trees so that it gets better."

(The prof eventually annotates the variables, post-break: `RM` = rooms, `LSTAT` = % low-socioeconomic-status population, `CRIM` = crime rate, `NOX` = nitric oxides / pollution, `AGE` = pre-1940 housing, `DIS` = distance to jobs, `PTRATIO` = pupil/teacher ratio, `B` (the variable named `Black`) = troubling demographic encoding from the original dataset. `INDUS` = industrial-area share. The pupil/teacher and SES/crime variables are heavily correlated via local school funding. "This is why you don't want to live in the U.S.")

## Module 8 wrap

> "We like trees. They're simple but they're not particularly good by themselves. We can improve them by averaging. Bagging is just averaging over the bootstrap data sets, and random forest is the same idea but now restricting the parameters. And we can improve our interpretation with this variable-importance measure."

That closes Module 8. Now: module 9.

## Boosting: opening framing

> "So now we're going to go into boosting."

Slide refs the textbook (ISL ch. 8) plus a piece by another math-department person on boosting. Multiple variants will be covered across the module:

- #3 = AdaBoost (today)
- #4 = gradient boosting (today)
- #5+ = XGBoost, LightGBM, etc. (later)

The big-picture sales pitch:

> "Boosting is very successful. As a general approach, if you look on Kaggle… boosting tends to be higher performing than even the neural networks."

Specifically: standard time-series statistical models lose; fancy ML models do better; **gradient-boosted trees (XGBoost, LightGBM) tend to win at the top.** Caveat: "I would say this is generally true when you don't have a ton of data… I don't think you'd want to build a large language model with trees."

### Sequential trees, not parallel ones

The new idea (vs. bagging / RF):

> "Trees are actually grown sequentially. You build one predictor, and then using that predictor you build a refinement predictor, so you're always trying to improve over the last one… a simple version, then a simple version to correct the simple version, then a simple version to correct the simple version of the simple version."

Key consequence: number of trees becomes a real tuning parameter:

> "In bagging and random forests you just pick a whole bunch of them. In this case, the number of models is going to be smaller."

Why simple trees specifically: variance reduction was the bagging/RF angle. Boosting takes the **bias-reduction** angle, starting with biased weak learners and combining many of them. "Rather than performing variance reduction on complex trees, can we decrease the bias by only using simple trees?"

The general recipe: build $f_1(x)$. Build $f_2(x)$ to *improve* $f_1$. Build $f_3(x)$ to improve the sum so far. One naive way: define residuals $r = y - f_1(x) - f_2(x) - \dots$, fit a new tree to those, add it on. ("This is not such a good idea to do it directly, but we'll go through what's a better idea.")

## AdaBoost

Historically the first version. Setting: binary classification with $y \in \{-1, +1\}$. Each weak classifier $G(x)$ is a small tree, often a **stump** (essentially a bias term, "very, very small tree"). The recipe is not exclusive to trees, any classifier works, hence the abstract notation $G$.

### The mechanism

Sequential, with **sample re-weighting**:

1. Build $G_1$ on the training data with uniform sample weights.
2. See where it fails. Re-weight the data so misclassified points get **more** weight in the next fit.
3. Build $G_2$ on the re-weighted data → it tries hardest to get the points $G_1$ missed.
4. Re-weight again. Build $G_3$. Etc.
5. Final prediction = sign of an $\alpha$-weighted sum of all $G_m$.

> "Each of these little bad predictors is biased towards whatever the worst shit is. Wherever the model sucks the most, it says, OK, I'm going to try there."

### The algorithm (textbook form)

Initialize observation weights $w_i = 1/N$. For $m = 1, \dots, M$:

1. Fit classifier $G_m(x)$ to the training data with weights $w_i$.
2. Compute weighted error rate
   $$\text{err}_m = \frac{\sum_i w_i \cdot \mathbb{1}[y_i \neq G_m(x_i)]}{\sum_i w_i}$$
3. Compute classifier weight
   $$\alpha_m = \log\!\frac{1 - \text{err}_m}{\text{err}_m}$$
4. Update sample weights:
   $$w_i \leftarrow w_i \cdot \exp\!\big(\alpha_m \cdot \mathbb{1}[y_i \neq G_m(x_i)]\big)$$

Final classifier: $G(x) = \operatorname{sign}\!\big(\sum_m \alpha_m G_m(x)\big)$.

### Why the exponential update

> "If it was just the indicator function, it would just be a step function — that's boring. So this gives us nice smooth things."

Trace through it:
- Misclassified ($\mathbb{1}=1$): $w_i \leftarrow w_i \cdot e^{\alpha_m}$, weight goes up by factor $e^{\alpha_m}$.
- Correct ($\mathbb{1}=0$): $w_i \leftarrow w_i \cdot e^0 = w_i$, unchanged.

So persistently-missed points keep accumulating weight, pulling later classifiers' attention onto them.

### Why it works: counterintuitive but real

> "It's very counterintuitive. Instead of finding the perfect model ever to fit your data, you find a whole bunch of shitty models that build on each other so they're really diverse and different, and then average them. And then they work well. Super weird, but it does work."

Mechanically: simple trees → low variance individually (no overfitting room). Sample re-weighting → forced diversity. Sum + sign → ensemble that covers different parts of the input space.

### Worked example: synthetic 10-D Gaussian

Data generated from a 10-dim Gaussian, labels from a sum-of-squared-values rule. Adaboost with weak-tree base learners. Test error stays bad through ~early iterations (only 4–5 stumps in the average), then drops steeply, plateauing around iteration 90. Slide uses a large test set so the error curve is smooth.

### Worked example: Boston data, classification

> "Random forest on the Boston data: error rate 0.17. AdaBoost: 0.1. Quite a striking difference."

So Adaboost beats the previous winner (RF) by a meaningful margin. The narrative arc:

> "Adaboost used the idea of bagging trees and the benefit of using multiple trees and taking these things and averaging them together, but now improving on the randomization that random forest tries to do, to generate a more diverse set of models that will generalize better because they're individually very simple."

## Gradient boosting on regression trees

Variant #4. Same sequential-improvement skeleton as AdaBoost, but now for **regression**, and instead of re-weighting samples, you **fit the next tree to the residuals**.

### The algorithm

Initialize $\hat f(x) = 0$. So initial residuals $r_i = y_i$. For $b = 1, \dots, B$:

1. Fit a tree $\hat f^b(x)$ with $d$ splits to the training data $(x_i, r_i)$, i.e., predict the *residuals*, not the raw $y$.
2. Update the model:
   $$\hat f(x) \leftarrow \hat f(x) + \eta \cdot \hat f^b(x)$$
   where $\eta$ is the **learning rate / shrinkage** (Greek letter eta on the slides).
3. Update the residuals:
   $$r_i \leftarrow r_i - \eta \cdot \hat f^b(x_i)$$
4. Repeat.

### Three hyperparameters

- $B$: number of trees. **Real tuning parameter** here (unlike RF). Too many → overfit.
- $\eta$: learning rate / shrinkage. Small $\eta$ "doesn't let the model jump right to a good solution," keeps each tree from voting too strongly. Forces weak learning by construction.
- $d$: depth (or number of splits) per tree. Controls how complex each weak learner is allowed to be.

> "We want weak learners, and this way we're kind of forcing them to be weak by not letting them have a strong vote."

### The punchline: boosting residuals = gradient descent

The closing payoff. Look at the update rule:
$$\hat f(x) \leftarrow \hat f(x) + \eta \cdot \hat f^b(x)$$

That's the same shape as a gradient-descent step $f(x) \leftarrow f(x) + \eta \cdot \nabla$. So what is $\hat f^b(x)$ approximating? Take the squared-error loss:
$$L = \sum_i \big(y_i - f(x_i)\big)^2$$
Differentiate w.r.t. $f(x_i)$:
$$\frac{\partial L}{\partial f(x_i)} = -2 \big(y_i - f(x_i)\big) = -2 r_i$$

> [!important] The whole point
> "By updating according to the residuals, what really we were doing is actually fitting the gradient of the function that we're optimizing. We're estimating the direction we have to go in. We're updating the gradient that will get us into a better direction."

So boosting regression trees on residuals is **steepest descent in function space** for squared-error loss. The residual *is* (proportional to) the negative gradient.

### Why this matters: generalization

Once you see boosting as gradient steps on a loss, the framework extends to **any differentiable loss**: logistic loss for classification, Huber for robust regression, ranking losses, etc. That's the root of "gradient boosting" as the umbrella name, and why XGBoost / LightGBM / etc. (to be covered Monday) exist as a family.

> "Now, with the notion that what we're boosting is the gradient of an improvement of the overall fit… that generalizes the idea of this boosting."

## Where we stop

Time runs out mid-module-9. Next session is delayed a week; the Mar 23/24 slot is for project supervision (the prof is away; Siemens covers it). Module 9 picks back up after that and finishes the rest of boosting (XGBoost, practical considerations, regularization variants).
