---
lecture: 11
date: 2026-02-10
module: 05-resample
title: Resampling 2
slides: modules/5Resample/5Resample.md
topics:
  - cross-validation
  - validation-set-approach
  - leave-one-out-cv
  - k-fold-cv
  - bias-variance-tradeoff
  - nested-cv-and-cv-pitfalls
  - bootstrap
  - bagging
  - standard-error
tags:
  - lecture
  - module/05-resample
aliases:
  - Lecture 11
---

# L11 — Resampling 2

The prof finishes module 5: a tight recap of validation set / [[leave-one-out-cv|LOOCV]] / [[k-fold-cv|k-fold CV]] through the bias-variance lens, an explanation of last lecture's mysterious upturn in the classification CV plot, [[nested-cv-and-cv-pitfalls|nested CV]] for combined model selection + assessment, the right vs wrong way to do CV (the "filter predictors first" trap), then the [[bootstrap]] in detail with the histogram-of-the-median framing and a sneak preview of [[bagging]] as a bias/variance-reducing precursor to module 8.

## Key takeaways

- **CV recap as bias-variance**: validation set = high bias, low variance; LOOCV = low bias, high variance (training sets nearly identical → estimates correlated); $k$-fold (5 or 10) = the practical sweet spot. *"Typically in this setting, you're winning by having less variance"* because what you really want is a model that generalizes across data sets.
- **Why the classification CV plot ticks back up at higher polynomial degree**: not just bias-variance — *"the thing that they're using to evaluate the model is different than the thing that they're fitting with the model."* Logistic regression maximizes likelihood, not [[misclassification-rate]], so adding parameters can technically make misclassification worse.
- **Use [[nested-cv-and-cv-pitfalls|nested CV]] when you need both model selection and model assessment.** Outer fold = assessment, inner folds = selection. Reusing the same held-out data for both *"tends to overfit the test data, and the bias will be underestimated."*
- **Right vs wrong CV — verbatim trap**: if you preselect predictors using $y$ (e.g. correlation filter), step 1 is part of training and **must** be inside the CV loop. Doing it outside can give misclassification ≈ 0 on pure noise.
- **The bootstrap's central idea**: *"Your best model for the real world… is the data itself."* Resample with replacement to get the empirical sampling distribution of any statistic. Picture: a histogram of medians, not a derived formula.
- **Sample with replacement.** Without replacement just permutes the data → useless.
- **[[bagging]] preview**: bootstrap aggregation. Average $B$ models fit on $B$ bootstrap samples → reduces variance. *"It's actually using this bagging trick implicitly"* when an over-parameterized single model performs unexpectedly well — preview of module 8.

## Recap of cross-validation as a bias-variance argument

The whole point of resampling: build a validation set so we can do the three things that have to happen with any model — *train*, *validate / select* (which model? which hyperparameter?), and *test / assess* (how good is it?). You can't reuse the same data for both selection and assessment: *"you've either fit the model or you've selected different parameters using that data, so you don't want to reuse that data to also report an evaluation."*

In the model-complexity picture (low-complexity → high-complexity model on the x-axis), training error always drops because *"each model assumes also the previous one — so as you go to a more complex model, your training error will always go down."* Test error eventually rises again — the classic bias-variance picture. *"In the beginning you have a high bias and a high variance because you don't have a lot of data or you don't have a lot of complexity."* The variance is across realizations of the data; the bias is how close your estimates are to the truth.

Three approaches, mapped onto the [[bias-variance-tradeoff]]:

- **[[validation-set-approach]]** — split once into two halves. Simple, but uses less data for training (high bias) and the answer depends a lot on which split you got. Probably low variance only because the test half is large; *"you're kind of unnecessarily hurting yourself."*
- **[[leave-one-out-cv|LOOCV]]** — leave one sample out, repeat $n$ times. Very low bias (train on $n-1$ each time), but **high variance**: *"you're almost always using the same amount of data to train, so you're going to get a lot more variance between the test error."* Across realizations of the data, the per-fold estimates are highly correlated.
- **[[k-fold-cv|k-fold CV]]** — partition into $k$ blocks, hold each out once. Slightly more bias than LOOCV (train on $(k-1)/k$ of the data instead of $n-1$), but **much better variance**. With 5-fold, only 60% of training data overlaps between any two folds. The non-overlapping test set is what matters for the variance reduction.

> "Leave one out will have a better bias. But the K-fold will likely have a much better variance. And typically in this setting, you're winning by having less variance — because you want to know, you want to pick a model that will do well if you have another data set, and it's that variance across data sets that you really want to reduce."

So training error is useless for model selection. Only test error eventually goes back up, and we estimate it via CV.

## CV in classification — and revisiting last lecture's confusion

CV doesn't only apply to regression. For classification we just swap MSE for [[misclassification-rate]] (0/1 loss): each fold contributes $\text{Err}_j = \frac{1}{n_j}\sum_{i \in C_j} \mathbb{1}(y_i \neq \hat y_i)$.

The prof recaps the ISL classification example. The **purple dashed [[discriminant-score-and-decision-boundary|Bayes decision boundary]]** is *"the optimal decision boundary"* — what you'd use if you knew all the distributions. We approximate it with [[logistic-regression]] and let the boundary depend on polynomial features:

- degree 1: just $\beta_0 + \beta_1 x + \beta_2 y$ in the [[linear-regression|linear predictor]] — a straight-line boundary;
- degree 2: add $\beta_3 x^2 + \beta_4 y^2$ → curved boundary, can produce circles/ellipses depending on coefficients.

> "I didn't ask for a circle. The curve you're making, remember, it has y squared and x squared, so you could get a circle, of course. It all just depends on how these parameters come out to be when you train it."

[[misclassification-rate|Misclassification rate]] vs polynomial degree: it drops, then **rises** at the highest degree.

Last lecture the prof was confused by the rise. He went home, played with it, and gave the explanation today:

> "The error rate that they use on the y is not actually computed the same way as the log likelihood or the likelihood of the model. So you're not actually fitting the error rate directly when you're fitting logistic regression. So even though you're adding more parameters to your model, so you're getting what should be a better fit in your logistic regression, it's not actually minimizing the same thing as the misclassification rate."

Logistic regression maximizes likelihood; we evaluate with misclassification. **They're related but not the same**, so monotone improvement on the loss doesn't have to translate to monotone improvement on the metric. He also speculates the book may have *dropped* the linear terms when adding quadratic ones, just to dramatize: *"they're using cartoon examples, so they might make it just harder."*

> [!note] The general lesson
> When the loss you optimize differs from the metric you report, "more flexibility" no longer guarantees "lower reported error." That's what's happening in Fig 5.7. *"So we're all fine — it's okay that the training error goes up, even though it's strange."*

The CV plot from the slide:
- **Blue** = training error (rises with degree, weirdly);
- **Orange** = true error on a 10,000-sample held-out set (the ground truth);
- **Black** = 10-fold CV error.

CV picks roughly the right complexity. *"In the cross[-validation] data, that would be terrible"* if you went with the highest-flexibility option — black points correctly to a lower-degree model. Following training error here would lead you astray.

## Nested CV — model selection *and* assessment

Once you've used CV to **select** a model (tune hyperparameters, pick polynomial degree, pick $K$, etc.), you can't reuse that same CV error to **assess** the chosen model — you'd be reporting a number you already optimized against. Bias goes the wrong way.

> "Using the test set for both model selection and estimation tends to overfit the test data, and the bias will be underestimated."

Solution: two layers of CV, aka [[nested-cv-and-cv-pitfalls|nested CV]]. The prof drew it on the board:

- **Outer split** — partition the data into folds for *assessment*. In each outer iteration, hold one fold out as a true held-out test set (the "pink/purple" stuff in his drawing).
- **Inner CV on the rest** — within the remaining outer-training data, run another CV (e.g. 5-fold) to do *selection*. This inner CV picks the model / hyperparameter that the outer iteration will use.
- **Score** the chosen model on the held-out outer fold → contributes to the assessment estimate.
- Repeat across outer folds.

So:
- inner folds → "selection" (which polynomial degree, which $k$, which $\lambda$);
- outer folds → "assessment" (honest generalization estimate).

> "You actually would be getting a different selection for each fold, but it tends to give you the same selection anyway, so then you could just look at which ones are popping up."

If different outer folds keep picking the same hyperparameter, you can confidently fit the final model on **all** the data with that choice and report the outer CV's assessment as your honest test error.

## The right and the wrong way to do CV

ISL's worked example — **wide data**: many more predictors than samples ($p \gg n$). Common in genetics / protein data, sometimes in finance — *"you can imagine you only have a year's worth of stock-trading data but you have a million stocks."* The setting amplifies a problem that exists more generally.

A two-step pipeline:

1. **Filter predictors** — compute the correlation between the class label and each of the $p$ predictors, keep the top $d=25$ with the strongest correlation.
2. **Fit** logistic regression on those 25.

> "How can we use cross-validation to produce an estimate of our performance? Can we apply cross-validation only to step 2?"
>
> **No, you can't.**

Why: step 1 already used the labels. Correlation-with-the-label *is* a statistical model — *"the correlation is already doing a bit of the work for you. It's already a statistical model. You already are selecting parameters based on this for this specific data."* If you do step 1 once on the full data and then wrap CV only around step 2, you're estimating performance on held-out folds the filter has *already* peeked at. *"You're using the same data to pick your variables and then seeing how well they work. So of course it's going to work well."*

The whole pipeline — selection + fitting — has to live **inside** the CV loop, so the predictor selection is redone on each training fold *without* the held-out fold.

> [!warning] How bad it can get
> *"In the recommended exercises, one of the exercises is basically create fake data using data where there should be no relationship at all, but by pre-selecting which variables you use, you actually get a misclassification error of zero — suggesting like, this is an excellent model, when in reality we know that it's crap."*

The prof's broader frame: *"This is an example of how to lie with — we'll say bad statistics."* Hastie and Tibshirani devote a full subsection to it because they've seen genomics studies make this exact mistake repeatedly.

> "If you look from this year, you'll find at least one article that's made this mistake within genome stuff. And if you look at other fields like neuroscience you'll find it there, you'll find it basically everywhere. People make the same mistakes all the time, especially because they don't take that many statistics classes."

> [!warning] Why the mistake is sneaky
> *"You maybe make a selection process and then you forget about it. You're like, oh, I did that already, and then you move on and you're looking at performances and you forget that it's no longer valid. I'm sure if I really looked into it, I probably made that mistake at least once. I don't know if I should admit that."*

## The bootstrap

Pivot from cross-validation to a different resampling philosophy. *"In all these cross-validation approaches — leave one out, K-fold, whatever — we were always partitioning the data so that you took each time point and either put it in a training, a test, or a validation set. Always each data point goes somewhere, and only once. In the bootstrap we're going to do something different."* The [[bootstrap]] **resamples with replacement at the same length as the original**, treating each resample as a fresh draw from the population.

Invented by Efron in 1979. *"It's funny because it's kind of the obvious thing you would do instead of all this fancy statistics that we've been developing for like the last hundred years. So I'm pretty sure Fisher, if he had a better computer, would have done bootstrap instead of all the distribution stuff."*

### Why "bootstrap"

The bootstrap is the loop on the back of a boot. Pulling yourself up by it is famously impossible — *"a long time ago they used to have in the standard elementary school textbook the example of proving using physics that you can't pull yourself up by your own bootstrap."* The phrase comes from *The Surprising Adventures of Baron von Munchausen*, where the Baron, having fallen into a lake, pulls himself out by his own bootstrap (in the movie it's his own hair).

Why Efron picked the name:

> "It feels like magic. It feels like you shouldn't be able to do it. It doesn't… it's too fancy, it's too cool. Like, how can you reuse the data and somehow make a better model? It's weird, right?"

### The central idea

> "Your best model for the real world — like for the real data, not just the sample that you have but the bigger sample everywhere — your best model for that is the data itself. And so if you want to look at different realizations of the data, you resample from that same data with replacement, because it's always going to be the best model for the world."

The [[bootstrap|empirical distribution]] $\hat f$ puts mass $1/n$ on each observed point; **sampling from $\hat f$ with replacement is the bootstrap**. We can't sample more from the unknown $f$, but we can sample as much as we want from $\hat f$ — and $\hat f$ is the best estimate of $f$ we have. *"Just keep it as it is, repick points, take the same number of points, and now you have a new data set. It's different from the original one, and it's as close to what you can expect as possible because you're just resampling your best estimate of the true process."*

> "It's like taking a picture and then taking a picture of the picture 100 times and then averaging them, and somehow it's better than the original picture. It doesn't sound like it makes sense, but it does."

### Standard deviation of the sample median

Worked example, set up the way the prof likes it: imagine a population (the heights of everyone in the country). We have a sample, we compute the sample median $\tilde X$. We don't just want the value — *"we want to understand what the distribution is of the median, because it's going to shift a little bit depending on which sample you get. We want to understand how variable that is."*

For the *mean* we have a derived formula ($\text{Var}(\bar X) = \sigma^2/n$), distributional theory, $t$-tests, etc. — that's what Fisher and Gauss did, *"because they didn't have computers, so they said okay, well, if I want to know what the distribution is of my estimator, I have to prove it. I have to figure out what it should be. And then I understand that distribution, and then I can fit that distribution and use those. That's where we get all these funky things like F-tests and t-tests and all these things."* For the *median*, no clean closed form. The bootstrap sidesteps the whole distributional-theory machinery: we want the distribution **directly**, in the form of a histogram.

Bootstrap recipe:

1. Take your data $X_1, \ldots, X_n$.
2. Sample $n$ values **with replacement** from it → bootstrap sample $X^*_1, \ldots, X^*_n$. *"Maybe at one point [is] sampled five times, one is three, one is two, one doesn't get sampled at all"* — so it's a smaller effective sample, but the same length as the original.
3. Compute the median $\tilde X^*_b$ of that bootstrap sample.
4. Repeat $B$ times. *"You typically want to use a big number like a thousand or ten thousand, and that's because you want to have a nice shape to the distribution."*
5. The empirical SD of $\tilde X^*_1, \ldots, \tilde X^*_B$ is your estimate of $\text{SD}(\tilde X)$. The histogram of $\tilde X^*_b$'s is your estimate of the sampling distribution.

> "That's our bootstrap estimate of the underlying distribution of that estimator — of the median in this case."

The pedagogical win:

> "I always thought that was weird to assume that somehow people will just understand that — especially in a first statistics course where you barely understand the idea of a distribution and then suddenly have to think about the distribution of an estimator, of something you estimated from data. It's not obvious what that means. But with this bootstrap, it's really nice because you can see the shape of the distribution. You can compute it and you can see intuitively what it is."

In the slide's R example, the "true" SD (sampling fresh data each time from the known $N(0,1)$) was about 0.125; the bootstrap from a single $n=101$ sample gave 0.1365. *"Surprisingly close to the truth, especially considering it seems like magic."* (The slide's first code snippet *isn't* bootstrap — it's resampling fresh from the known distribution to demonstrate what the true sampling distribution looks like; *"in this case they just want to show what it would be if the data was always independent."* The bootstrap version follows immediately after.)

Once you have the histogram you can do anything you'd do with a sampling distribution: confidence interval (e.g. percentile method), hypothesis test, ask whether some new value plausibly came from this distribution. *"It's much more meaningful than just saying this point is far away from that point."*

The prof's heights example to make it concrete: imagine you've bootstrapped the median height a thousand times and your histogram fluctuates from, say, 1.6 m to 2.1 m. Now if someone tells you a person is 0.5 m tall, you can ask — *"what's the likelihood that this person… is part of this distribution?"* The histogram literally gives you a place to put the new value relative to the sampled medians.

> [!note] One bootstrap sample is meaningless
> A *single* bootstrap resample gives you a different median from the original — *"because we have slightly different data, even though it's generated from that original."* That alone tells you nothing. The whole point is repeating $B$ times and looking at the **distribution** of those $B$ medians.

### Why with replacement

> "Could we do it without replacement? I mean, I guess you could, but it's not really the same model. The reason you do replacement is that you want to have the same length of data every single time, so it's not a data-length problem, and you want your best estimate of the true distribution, which is your data. That's the whole idea. If you do it without replacement, then either you're always getting the same data every single time, just reordered — so that's useless."

Without replacement at full size: just a permutation. With replacement: genuinely different draws from the empirical distribution. (Could you sample, say, only 10% without replacement? *"It would be different"* — but that's not bootstrap.)

### Use cases

> "Very popular for obtaining standard errors, confidence intervals, doing tests."

Bootstrapped distribution → confidence interval (e.g. percentile method), standard error, hypothesis test.

**Regression example:** for [[linear-regression|multiple linear regression]] we already know the closed form $\text{Cov}(\hat\beta) = \sigma^2(X^TX)^{-1}$ — but only **under** the standard distributional assumptions on the residuals. *"All these distributional assumptions are assumptions, and for those to be right your data has to be of a certain type, and maybe they're not. So often using a bootstrap will be better — will be making fewer assumptions and can give you a different result."* You can also use the bootstrap to build confidence intervals for the $\beta_j$'s and prediction intervals for new $y$'s. (Worked in the recommended exercises — *"OK to look at an example where we know the truth."*)

In R there's a built-in `boot::boot()` function (and a `cv.glm()` for CV), but the prof emphasizes:

> "It's just really very easy to make yourself."

A `for`-loop over `sample(x, size=n, replace=TRUE)` and a few lines around it is enough.

### A connection from intro stats

Aside the prof drew at the board: when you're first taught the **two-sample $t$-test** (compare two means), some courses also show you the **bootstrap version** — resample the labels, recompute the difference of means many times, see how often it exceeds the observed difference. Same logic, no distributional assumption.

> "In my mind this is much easier to understand than the notion that a given estimator, like a mean or a median, has a distribution."

He polled the room about whether anyone learned that version in their basic statistics course. Mixed results. The point: the bootstrap framing of "what's the distribution of a statistic" is more concrete than the textbook framing — *"with this bootstrap, it's really nice because you can see the shape of the distribution."*

## Bagging — preview of module 8

[[bagging]] = **B**ootstrap **AGG**regat**ING**. Same resampling trick, used not to estimate uncertainty but to *build a better model*. *"There's a similar magic"* to the bootstrap, the prof says, *"in that you can use this trick in a different way to basically make an ensemble of different models of your data and then average across those."*

Recipe:
1. Draw $B$ bootstrap samples from your training data.
2. Fit your model on each → $\hat f^{*1}, \hat f^{*2}, \ldots, \hat f^{*B}$.
3. Predict by averaging:
$$\hat f_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat f^{*b}(x)$$

> "You bootstrap your data, you fit a model, and then you average across all those models, and then magically that's a better model."

The compute argument runs through the whole module: validation set, LOOCV, $k$-fold, bootstrap, bagging — all are *"in the age of compute"* methods. *"It's really not a problem"* to refit the model many times because hardware is cheap. *"I like it because it's super easy. And it just takes more compute, and compute's cheap, so who cares?"*

The variance argument: if the $B$ samples were genuinely independent, taking their average would reduce variance by $1/B$ — same as $\text{Var}(\bar X) = \sigma^2/n$. The bootstrap samples *aren't* independent (all drawn from the same data), so the actual reduction is less; for pairwise correlation $\rho$ between bootstrap fits:
$$\text{Var}_{\text{bag}} = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
*"It's not as good of a reduction as if it were independently sampled data, but it's still pretty good."*

The prof flagged that bagging can also remove **bias**, not just variance — *"by doing that, even though you're always just resampling the same data, you can actually remove bias from your model"* — though he's clear the main and easiest argument is the variance one. Bagging shines for **high-variance models with poor prediction ability** — particularly [[regression-tree|trees]], which we'll cover in module 8 ("bagging trees"). Variance reduction here is variance *across realizations of the data*, same notion that's been driving the whole module.

> [!note] A surprising aside
> *"It's actually using this bagging trick implicitly. So you end up with so many different parameters that actually the collection of parameters finds different models and different parts of the parameters and then essentially averages them together."* The prof's intuition: very large-parameter single models — the regime where the bias-variance curve goes back *down* on the far right — end up implicitly bagging. *"It's super weird, but it's interesting."* Comes back in the [[double-descent]] discussion much later.

## Closing — admin and what's next

The remainder of the slide deck is on **R Markdown / `knitr`** — how to render documents, set up code chunks, the YAML header, troubleshooting `pdf_document` failures. The prof skips it: *"It's very boring. And like I said, R is optional this year. So you can use it, or you can use LaTeX and Python. Either one is fine for the projects."*

Practical reminders for compulsory exercise 1:

- Next week → in-class work on the compulsory exercise (no formal lecture).
- Due date is on Blackboard ("the 18th, I think — but don't quote me on that, look on Blackboard").
- Required for exam eligibility.
- Sign up for a group on Blackboard; email the PhD student "Seaman" if you don't have one.
- Hand in a PDF (and the Rmd or zipped LaTeX scripts).
- Scores and comments come back via Blackboard.

Module 6 (Model Selection and Regularization) starts the week after.

> "We did cover the material that is included in the exercises so you should have everything you need for that."
