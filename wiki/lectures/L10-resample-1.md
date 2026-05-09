---
lecture: 10
date: 2026-02-09
module: 05-resample
title: Resampling 1
slides: modules/5Resample/5Resample.md
topics:
  - sensitivity-specificity
  - roc-auc
  - confusion-matrix
  - bias-variance-tradeoff
  - training-validation-test-split
  - cross-validation
  - validation-set-approach
  - leave-one-out-cv
  - k-fold-cv
  - one-standard-error-rule
  - aic-bic-conceptual
  - data-reuse
  - gaussian-error-assumptions
  - design-matrix-and-hat-matrix
tags:
  - lecture
  - module/05-resample
aliases:
  - Lecture 10
---

# L10: Resampling 1

The prof closes module 4 with a recap of [[sensitivity-specificity]], the [[confusion-matrix]], and [[roc-auc|ROC/AUC]] curves, then opens module 5 on resampling. He motivates why we need [[cross-validation]] (assumptions of [[aic-bic-conceptual|AIC/BIC]] are typically wrong), partitions data into training/validation/test, and walks through three CV schemes: the [[validation-set-approach]], [[leave-one-out-cv|LOOCV]], and [[k-fold-cv|k-fold CV]], with heavy emphasis on the [[gaussian-error-assumptions|independence trap]] in spatial/temporal data. Ends mid-classification-CV, just before model assessment vs. model selection.

## Key takeaways

- **Sensitivity = TPR; specificity = TNR.** Where you place the threshold depends on the cost of FP vs. FN. ROC sweeps the threshold; AUC summarizes it. Diagonal = chance; below diagonal means "your model is really stupid" - invert it.
- **AIC/BIC penalize complexity, but lean on assumptions** (correct distribution, IID, well-behaved models) that *"are not typically right."* Resampling gets at the same thing with **fewer assumptions**.
- **Three data partitions, three jobs**: training (fit), validation (model selection), test (assessment). Reusing the test set for selection makes you "too optimistic." Don't do data reuse.
- **Validation set approach**: simple 50/50 split. High variance across splits, smaller training set → upward-biased error estimate. Conservative and easy to explain.
- **LOOCV**: train on $n-1$, test on the held-out 1, repeat $n$ times. No randomness, low bias, but **expensive** ($n$ refits) and high variance because the $n$ training sets are nearly identical (highly correlated). For OLS there's a hat-matrix shortcut: $\text{CV}_n = \frac{1}{n}\sum \big(\frac{y_i - \hat y_i}{1 - h_{ii}}\big)^2$, only one fit needed.
- **k-fold CV** (k=5 or 10) is the workable compromise: less variance than the validation set, less compute and less correlation than LOOCV. Still suffers some upward bias because each training set is only $(k-1)/k$ of the data.
- **Verbatim trap**: with spatial or temporal correlation, naïvely random splitting is broken: *"two points right next to each other, one in your training, one in your validation, it's the same damn thing."* Chunk by the dependency dimension before splitting.
- **One-standard-error rule**: pick the simplest model whose CV error is within one SE of the minimum. *"Not quite valid"* (and you should think about why).

## Wrap-up of module 4: classification recap

Quick run through what last week covered: [[logistic-regression]] for binary (and multiclass) classification, [[knn-classification|k-nearest neighbors]] as a non-parametric alternative. Same problem, very different mechanism. Both need a **threshold** decision; both need a way to measure how good the classifier is.

> "Even with just these two, you already see like rather different mechanisms by which you can build a classification, right? So it makes sense you'd want to understand how good is my classifier."

### Sensitivity and specificity

Four cells in the truth × prediction grid: TP, FP, FN, TN. The prof flubbed the chart's labels live and corrected as he went, but the message stuck:

- **[[sensitivity-specificity|Sensitivity]]** = TPR = TP / P. *"If you maximize for this, then you would make sure that you get all the sick people and you're okay with having a few false positives."*
- **Specificity** = TNR = TN / N. Maximize this to be conservative about false alarms.

The right trade-off is application-dependent. Putting people in jail → bias toward fewer false positives (high specificity). Screening for disease → bias toward catching every case (high sensitivity), even at the cost of some false alarms.

### Confusion matrix

The same four cells, drawn as a 2×2 table. Compute sensitivity and specificity from it directly.

### ROC and AUC

[[roc-auc|ROC curve]] = sweep the threshold of your classifier and plot $(1 - \text{specificity},\ \text{sensitivity})$ at each cutoff. The extremes are degenerate:

> "If you were to pick a point on the extreme, then why did you build a classifier in the first place?"

Bottom-left = classify everyone negative (no FPs but no TPs). Top-right = classify everyone positive (catch all sick people but lock up the healthy ones too). The interesting region is between.

You compare classifiers by **AUC** (area under the curve). The prof's example shape: a "blacker" curve with AUC = 0.776 vs. a hypothetical greener curve closer to the upper-left corner with AUC ≈ 0.9.

> "If you had to decide between these two models, the green one is obviously a better option, because it's performing better in all regards, regardless of where you put the threshold."

The diagonal (AUC = 0.5) is **chance level**, area of a triangle in the unit square is half. If your ROC bows *below* the diagonal, *"either you made it wrong or your model is really stupid"* - flip the labels.

> "It should not really go worse than chance, because then if that's really the way you're doing it, then don't do that at all and just randomly assign labels and you'll do better."

## Bridge into module 5

The first project is posted (homework-style; groups of 1–3). The remaining material needed to complete it is today and tomorrow. Next week is project work, no slides, no lecture. Brief admin aside: do the work even though "you could probably find the solutions if you wanted to" - *"everything I've learned is by like through sweat… you'll learn more."*

Module 5 is **bootstrap and cross-validation** (ISLR ch. 5; some additional material in Elements of Statistical Learning). Two goals: **model assessment** (how good is the final model?) and **model selection** (which of these candidates do I pick?).

### Why not just AIC/BIC?

Both [[aic-bic-conceptual|AIC and BIC]] are penalties for model complexity, derived under nice assumptions:

- AIC from information theory.
- BIC from Bayesian arguments.

> "If all your assumptions are true and other situations… you can use these criteria, which are essentially penalties, for adding additional parameters to your model. And then those alone can give you a nice way to evaluate your model."

The catch:

> "Your assumptions have to be right. And they're not. They're not typically right."

Common breakdowns: distribution wasn't what you thought, samples are correlated in time/space/relationships, "people you check are not independent because they're all related or they're all white or they're all whatever." Truly IID data with all model assumptions met is rare. Resampling lets you ask the same questions with **fewer assumptions** and less sensitivity to violations.

## The bias-variance setup recap

Same picture as in earlier modules: as model complexity grows, **training error always drops** (more parameters can't make in-sample fit worse), but **test error makes a U**: first drops as bias falls, then climbs again as variance / overfitting takes over. The valley is where bias and variance balance.

> "It's a nice balance between when you've removed the bias, but you still have a good variance, and the reason this can test your variance is because it's a different data set than you've used to fit the data."

Anders flagged the standard concern that picking the model at the test-set minimum is itself a kind of fit-to-test. Prof acknowledged: yes, that's real, and there are corrections: instead of using the lowest point, *"you move over a bit"* (foreshadowing the [[one-standard-error-rule]] later in lecture).

### KNN regression as the running model-selection example

Recall [[knn-regression|k-nearest neighbors]] regression: $\hat f(x_0) = \frac{1}{K} \sum_{i \in \mathcal N_0} y_i$ where $\mathcal N_0$ is the K nearest training points. K small → high complexity (jagged, K=1 hits every training point); K large → low complexity (smooth, K=number-of-points = horizontal mean).

Slide example: true curve $f(x) = -x + x^2 + x^3$, $x \in [-3, 3]$, $n=61$ training points, K swept from 1 to 25, experiment repeated $M=1000$ times.

> "1 is going to suck - it's going to jump to every single point - and then 25 is going to be a lot, because there's 61 points… you're using like almost more than a third of the data, so that's going to be way too smooth."

The bias-variance plot from this experiment: training (red) error is low at K=1 and grows with K; variance is high at K=1 and shrinks; irreducible error is flat. Optimal K balances them.

> "Where the total error is minimized. But again, not exactly where."

One subtlety the prof flagged: in this slide the "bias" is computed on the **fit data**, not from the true model. *"In other parts, we talk about the bias as how biased it is from the true model, not from the fit data."* In real settings you don't know the truth, so you have to **estimate** these from data; that's why we need resampling.

## Training, validation, test: three partitions

So far we've only used training and test. Now add a **validation set** in the middle.

- **Training**: fit the model.
- **Validation**: select among candidate models / pick hyperparameters.
- **Test**: report final performance to the world. This is the *"assessment"* you want to showcase.

> "Yeah, it makes sense though, that we would actually need three because these are different goals - to be able to select the model and then also say how good it is."

### Why not reuse the test set for selection?

Slide Q & A, verbatim:

> [!important] The data-reuse principle
> "We will be too optimistic if we report the error on the test set when we have already used it to choose the best model… Don't do that. I have lots of examples of where people have done that. It's very sad. It's very common. It kind of sucks."

The prof followed with a long anecdote about a paper where the authors had committed exactly this kind of statistical sin, refused to fix it, ended up rejected by the journal, and then (annoyingly) they put his mother's name on the paper. Moral: *"don't make the dumb mistakes. Because it's embarrassing."*

### "If you have a lot of data, you don't need module 5"

Slide makes the point: with a *truly* large data set you can blindly partition into thirds (training/validation/test) and skip resampling tricks. In practice this is often unrealistic, and people psychologically resist tossing 1/3 of their data because *"somehow the extra 20% is going to tell us more or something. It generally doesn't."*

> "Module 5 is about how to do this efficiently, how to create a test and validation set efficiently."

## Cross-validation: the three flavors

Three approaches we'll cover:

1. The **[[validation-set-approach]]** (not strictly *cross*-validation).
2. **[[leave-one-out-cv|LOOCV]]**.
3. **[[k-fold-cv|k-fold CV]]**, typically $k=5$ or $10$.

All three are about resampling within the data you have, not about the test set (which is held out separately for assessment).

### Validation set approach

Random 50/50 split into training and validation. (Or with the test held out separately, more like 33/33/33, but the most common version of this idea is 50/50 between train and validation.)

Slide example: Auto data, $n=392$, predict `mpg` from a polynomial in `horsepower`, polynomial degree 1–10. Single split → an apparent best degree. Repeat the split 10 times → curves disagree:

> "No consensus which model really gives the lowest validation set MSE."

Why? Two drawbacks:

- **High variability** of the validation error from one split to the next.
- **Smaller training sample** (only half the data fits the model) → tends to **overestimate** the test error you'd get from a model fit on all the data.

> "If you have a lot of data… screw it, just do this - because this is going to be more conservative than the other approaches. And it's also very easy to explain."

### The independence trap (verbatim warning)

The prof's longest tangent of the lecture, and one he flagged as a **very common mistake**:

> "Be careful with independence - independence of points. For example, let's say your data is spatial in nature. There's a natural correlation between things spatially, and this also happens temporally in time."

If your data is correlated (in space, in time, by family relationships, etc.), random partitioning into training and validation gives you two sets that **leak into each other**:

> "If you just randomly sort them into the test and fit and validation without any concern of these things, it's going to suck. Because you're going to take two points that are right next to each other, but one in your training data, one in your validation data - it's the same damn thing. You haven't created two sets of data. You've just partitioned the same data twice, basically."

Mitigation: chunk by the relevant dimension first, then split chunks. The prof's neuro-data trick:

> "Often as a first pass, if I want to understand something, I'll throw away 80% or 90% of my data by taking a time bin and then jumping ahead another 10 time bins away and throwing all the stuff away in the middle. Because by throwing away everything in the middle, I'm confident that these two time bins are now independent."

This concern recurs in LOOCV and k-fold below; it's the prof's headline pitfall for the whole module.

### Leave-one-out cross-validation (LOOCV)

The other extreme. For $n$ data points:

1. Hold out point $i$.
2. Fit the model on the remaining $n-1$.
3. Predict $\hat y_i$ and compute $\text{MSE}_i = (y_i - \hat y_i)^2$.
4. Repeat for $i = 1, \ldots, n$.

$$\text{CV}_n = \frac{1}{n} \sum_{i=1}^n \text{MSE}_i$$

For classification, swap MSE for misclassification: $\text{Err}_i = I(y_i \neq \hat y_i)$.

**Pros**:

- No randomness in the splits, fully deterministic.
- Low bias: you train on almost all the data each time.

**Cons**:

- Expensive: $n$ refits.
- High variance in the average: training sets differ by only one observation, so the per-fold estimates are **highly correlated**, and correlated estimates have high variance when you average them. (Slide spells out the variance-of-sum identity:
$\text{Var}(\sum a_i X_i) = \sum a_i^2 \text{Var}(X_i) + 2 \sum_{i>j} a_i a_j \text{Cov}(X_i, X_j)$, and the cross-covariance terms blow up.)
- **Outlier sensitivity**: an extreme point hits one fold with a huge held-out error. The prof flagged this as a kind of feature (it tells you which points are problematic) but mostly as a property to be aware of.
- **Independence trap (extra-bad version)**: if your data is dependent in time/space, LOOCV is *terrible* because the model can predict the held-out point from its neighbor; fold error is artificially tiny, and CV will tell you to use the most complex model possible. *"It will be basically identical to just using the likelihood without any penalization."*

### LOOCV shortcut for OLS

For linear regression there's a beautiful closed form:

$$\text{CV}_n = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat y_i}{1 - h_{ii}} \right)^2$$

where $h_{ii}$ is the $i$-th diagonal of the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top$ and $\hat y_i$ is the original full-data fitted value. **Only one fit needed.** Compulsory exercise 1 walks through this.

> "We have a nice shortcut for linear regression and in some other settings, but not all."

### k-fold cross-validation

Compromise between the two extremes. Split data into $k$ (roughly equal) folds. For $j = 1, \ldots, k$: train on the $k-1$ other folds, validate on fold $j$. Average the $k$ MSEs.

$$\text{MSE}_j = \frac{1}{n_j} \sum_{i \in C_j} (y_i - \hat y_i)^2 \qquad \text{CV}_k = \frac{1}{n} \sum_{j=1}^k n_j\, \text{MSE}_j$$

Setting $k = n$ recovers LOOCV.

Why it helps:

- **Less correlation between folds** than LOOCV: in 5-fold, two training sets share 60% of the data, not 99.9%. So the $k$ estimates are less correlated → averaging them gives lower variance.
- **Less compute**: 5 or 10 fits instead of $n$.
- **More observations per validation fold** → individual fold errors are themselves lower-variance.

Slide example, Auto data again: 5-fold and 10-fold curves look very similar to each other and to LOOCV, but rerunning 10-fold with 10 different random splits shows **much less variability** than rerunning the validation set approach 10 times.

> "By doing the repeated folds, we get a much lower variability across reruns. So there is variability, but much less than the other approach, which is nice."

The independence concern from earlier still applies, but it's **easier to handle in k-fold** than in LOOCV because you can deliberately construct folds that respect the dependency structure (e.g., put a whole spatial block in one fold).

### Issues with k-fold CV

Slide enumeration:

1. Result depends on how the folds are made, but variation is lower than the validation-set approach.
2. Computationally cheaper than LOOCV (without the OLS hack).
3. Training set is $(k-1)/k$ of the data → estimate of prediction error is **biased upwards**.
4. This bias is smallest for $k = n$ (LOOCV), but LOOCV has the variance problem.
5. By the [[bias-variance-tradeoff]], $k=5$ or $10$ is the **standard compromise**.

## Choosing the best model: and the one-standard-error rule

You have a tunable parameter $\theta$ (e.g., K in KNN, polynomial degree). Plot $\text{CV}_k$ as a function of $\theta$ over the candidate range. The "best" model is the one minimizing $\text{CV}_k$.

After picking $\theta$, **refit on the entire non-test data** and report performance on the held-out test set as your final assessment.

### Standard error of the CV estimate

$$\widehat{\text{SE}}(\text{CV}_k(\theta)) = \sqrt{\sum_{j=1}^k (\text{MSE}_j(\theta) - \overline{\text{MSE}}(\theta))^2 / (k-1)}$$

(the sample standard deviation of the per-fold MSEs).

### One-standard-error rule

Instead of picking the $\theta$ that **minimizes** CV, pick the **simplest** model whose CV is within one SE of the minimum:

$$\text{CV}(\theta) \leq \text{CV}(\hat\theta) + \widehat{\text{SE}}(\text{CV}_k(\hat\theta))$$

Walk $\theta$ in the direction of "simpler" until that bound stops holding. The simplest model still within one SE wins.

> [!note] Footnote on the slide
> "Strictly speaking, this estimate is not quite valid. Why?"

The prof leaves it as a thought question, connected to the same point Anders raised earlier: you're already using the validation data to *select*, so the SE you compute on it isn't a clean independent SE.

## CV for classification

Same machinery, different loss. Replace MSE with the misclassification indicator:

$$\text{CV}_n = \frac{1}{n} \sum_{i=1}^n I(y_i \neq \hat y_i)$$

Slide example: 2D classification with [[logistic-regression]] of varying polynomial degree. Three curves:

- **Orange**: true test error (since this is a simulated example with the truth known).
- **Black**: 10-fold CV error.
- **Blue**: training error.

The training-error curve shown actually trends **upward** with complexity, which doesn't make sense. The prof called this out live:

> "I don't know why it's going up. That shouldn't happen. Maybe it's how they did it… The training error shouldn't go up."

He drew the corrected shape on the board. The black (10-fold CV) closely tracks the orange (true test) curve; CV is a good proxy for the true test error. The minimum of the CV curve is slightly more complex than the true optimum, but using the **one-standard-error rule** would push you toward a simpler (and arguably better) model.

(Bayes error rate for this problem ≈ 0.133; logistic regression with polynomial degrees 1–4 gives error rates 0.201, 0.197, 0.160, 0.162.)

## Stopping point and what's next

Stopped before the model-assessment use of CV, the right-vs-wrong-way-to-CV cautionary slide (the $p=5000, n=50$ feature-selection trap), and the bootstrap.

> "We're over halfway, so I think it is fine to stop here, and then we'll pick this up tomorrow… Tomorrow we're going to start off with model assessment, do a quick recap, and then we'll continue with cross-validation and bootstrapping and these other sampling methods."

Closing emphasis:

> "These things are used a lot and they're nice because they don't make so many assumptions."
