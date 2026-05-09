---
lecture: 9
date: 2026-02-03
module: 04-classif
title: Classification 3
slides: modules/4Classif/4Classif.md
topics:
  - linear-discriminant-analysis
  - quadratic-discriminant-analysis
  - classification-setup
  - discriminant-score-and-decision-boundary
  - confusion-matrix
  - multivariate-normal
  - pooled-covariance
  - naive-bayes
  - sensitivity-specificity
  - roc-auc
  - logistic-regression
  - knn-classification
  - bias-variance-tradeoff
tags:
  - lecture
  - module/04-classif
aliases:
  - Lecture 9
---

# L09 — Classification 3

The prof wraps up module 4 by finishing [[linear-discriminant-analysis]] (1D recap, then multivariate), introducing [[quadratic-discriminant-analysis]] as "what happens when you stop pooling the covariance," then comparing all five classifiers (logistic, KNN, LDA, QDA, naive Bayes). He flags multiple "good exam questions" along the way — most of them about deriving the [[discriminant-score-and-decision-boundary|discriminant score]] or solving for the [[discriminant-score-and-decision-boundary|decision boundary]]. Ends with a first pass at [[sensitivity-specificity]] / confusion matrices, which spills into next week.

## Key takeaways

- **LDA recap**: model $P(X|Y)$ as Gaussian with class-specific means but **pooled** variance, plus class priors $\pi_k$. Apply Bayes. Take logs and drop terms without $k$ → linear [[discriminant-score-and-decision-boundary|discriminant score]] $\delta_k(x)$. Decision boundary: equate $\delta_k(x) = \delta_\ell(x)$ and solve for $x$.
- **"That would be a typical exam question"** — given $\pi_k$, $\mu_k$, $\sigma$ for an LDA setting, **solve for the decision boundary**. The prof said this twice. Also good: derive the discriminant score from the Gaussian (show where the $x^2$ drops out in LDA, where it survives in QDA).
- **QDA = LDA without pooling.** Sigma becomes class-specific → the $x^\top \Sigma_k^{-1} x$ term no longer cancels across $k$ → discriminant is **quadratic in $x$** → boundaries are curves, not lines. More flexible, more parameters, more prone to overfit.
- **Confusion matrix**: in-sample (training) error is misleadingly low because of overfit. **Bayes-optimal does worse than fitted LDA on training data, but better out of sample.** Bias–variance trade-off again.
- **Why ever use LDA over QDA?** Same reason linear regression often beats quadratic — when you don't have enough data to estimate the extra parameters reliably, the simpler model wins on test error.
- **LDA is also a dimensionality-reduction method**: $K$ classes → $K-1$ discriminant scores. Maps high-D $X$ down to a representation specifically built to linearly separate the classes.
- **Comparison closing**: pick by assumptions. Gaussian + small $n$ → LDA. Gaussian + unequal $\Sigma_k$ → QDA. Large $p$ → naive Bayes (drop off-diagonals). Wonky boundary → KNN. Two classes, no distributional commitment → logistic regression.

## Recap: where we are in module 4

Three methods so far: [[logistic-regression]] (model $P(Y|X)$ directly via the sigmoid; Bernoulli likelihood), [[knn-classification|k-nearest neighbors]] (non-parametric, very flexible boundary, but suffers in high $D$), and [[linear-discriminant-analysis]] which uses Bayes' theorem to flip the problem:

$$P(Y|X) = \frac{P(X|Y) \, P(Y)}{P(X)}$$

So instead of modeling $P(Y|X)$ directly, model the **prior** $P(Y) = \pi_k$ and the **class-conditional density** $P(X|Y)$ (assumed Gaussian), then apply Bayes. This is the third approach.

> "We're flipping it around. So now, instead of modeling this probability of Y given X directly, we want to make a model of what is the prior distribution of Y, like how likely is this class A or class B, and the probability of X given Y."

## LDA in 1D — recap and example

Single predictor $X$, $K$ classes. Assume $X | Y=k \sim N(\mu_k, \sigma^2)$. Note the **shared variance** $\sigma^2$ (not $\sigma_k^2$) — this is the "L" in LDA, and we will relax it later for QDA.

In the running 2-class example: green class centered at $\mu = -2$, orange at $\mu = +2$, equal $\sigma$, equal priors $\pi = 0.5$. The product $\pi_k f_k(x)$ gives two Gaussians; the **decision point** is where they cross — in the middle, by symmetry. The classification error equals the tails on the wrong side, ~0.09 in this setup.

> "It's just going to be whatever the tail is going to be… it's about as good as you can get."

### What changes when priors aren't equal

If you increase $\pi_{\text{orange}}$, more samples are orange in expectation. The orange Gaussian gets multiplied by a larger factor, the curves shift, and the **decision point shifts** away from the midpoint toward the rarer class.

> "The cool thing about this decision point is that it becomes a consequence of the equations. So in this model, you're finding equations… and the decision points or boundaries will then be solutions of where those equations meet."

Contrast with logistic regression, where the boundary's slope is *itself* a fitted parameter. In LDA, the boundary **falls out** of equating the per-class equations.

> [!important] Exam-flag, verbatim
> "And that's often an exam question. Or that would be a typical exam question. So you would, given an LDA setting and here are the values for the parameters — you have the pies, you have the mu's, and you'd have a value for the standard deviation — and then you would solve for where is the decision point."

### Where LDA's assumptions break

Two natural failure modes the prof called out:

1. **The Gaussian assumption on $X$.** If your covariates aren't well modeled by a Gaussian, the model breaks. *"Maybe it's not a good idea to pretend that the X's are well modeled by a Gaussian — that's a good way to break a model."*
2. **Pooled variance.** All classes share one $\sigma$. If they shouldn't, LDA suffers. (This is exactly the relaxation that gives QDA later.)

### Estimating priors

Estimate $\hat\pi_k$ directly from class frequencies in the training data: 80 of 100 samples are class A → $\hat\pi_A = 0.8$. Class probabilities sum to 1.

## Discriminant score derivation

We want to find $k$ that maximizes

$$P(Y=k | X=x) = \frac{\pi_k f_k(x)}{\sum_\ell \pi_\ell f_\ell(x)}$$

Three observations:

1. The denominator doesn't depend on $k$ — drop it.
2. We only need $\arg\max_k$, not the actual probability.
3. $\log$ is monotone — $\arg\max$ unchanged.

Take the log of $\pi_k f_k(x)$ with $f_k$ Gaussian. Throw away **everything that doesn't depend on $k$**: the $x^2$ term has no $k$, gone; constants vanish. What survives:

$$\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log \pi_k$$

This is the [[discriminant-score-and-decision-boundary|discriminant score]]. **Linear in $x$**. Bigger $\delta_k$ wins.

> "That would be another question one could ask if I was so inspired — show that this leads to this thing. I don't know if that's a very interesting question to ask, but one could ask it."

> [!note] Why we throw the $x^2$ term away
> The Gaussian density has an $x^2$ inside the exponent. In LDA, the coefficient on $x^2$ is $-1/(2\sigma^2)$ — **same for every $k$** because we pooled the variance. So the $x^2$ contribution is identical across classes and drops out of the $\arg\max$. **In QDA we don't pool**, so the coefficient becomes $k$-dependent, the $x^2$ term survives, and the discriminant becomes quadratic.

The discriminant score is **not a probability, not a likelihood** — but it gives the same classification.

### Decision boundary from discriminants

Equivalent to equating $\pi_k f_k(x) = \pi_\ell f_\ell(x)$, you can equate

$$\delta_k(x) = \delta_\ell(x)$$

and solve for $x$. Same answer, much less algebra. In 1D you get a point; in 2D you get a line; in higher dimensions a hyperplane. The boundary is **linear in $x$** — that's the "L" doing work.

> "Same exact trick as finding the point that divides the classes in two. You do the same thing, only now in a higher dimension and then you get a line out of it, or a plane, whatever it is."

## Estimating LDA from data

In practice you don't know $\pi_k$, $\mu_k$, $\sigma^2$ — estimate them from the training set:

- $\hat\pi_k = n_k / n$ (class frequency).
- $\hat\mu_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i$ (within-class mean).
- $\hat\sigma^2$ = **pooled** variance: subtract each point's class-mean, sum squared deviations across **all** classes, divide by appropriate denominator. Everyone shares the same variance estimate.

Plug into the $\delta_k(x)$ formulas, classify by max.

In a simulated 1D example with the assumed model truly generating the data, the empirical error rate (~0.05–0.115 across runs) sits right around the theoretical 0.09 — exactly because all assumptions are met.

## Confusion matrix

Standard tool for visualizing how a classifier does. Rows = true class, columns = predicted class (or vice versa — pick a convention). Diagonal = correct, off-diagonal = mistakes.

> "What can happen, this happens a lot, is the true class was K but a bunch got misclassified as 1. This can happen because maybe the two classes are very similar, have high overlap. The confusion matrix is a way to visualize **how the model is getting confused**."

The prof returned to the matrix later in the lecture to make a separate point: **in-sample (training) confusion matrices flatter the model**. They reflect overfit, not true generalization. We'll see this hit hard in the QDA-vs-LDA comparison.

## Multivariate LDA

Now $X \in \mathbb{R}^p$ — many predictors. Replace the 1D Gaussian with a **multivariate Gaussian**:

$$f_k(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2} (x - \mu_k)^\top \Sigma^{-1} (x - \mu_k)\right)$$

Same idea, lowercase $\sigma$ → uppercase $\Sigma$ ([[random-vector-and-covariance|covariance matrix]]), and $\mu_k$ becomes a $p$-vector. Still pooled across classes.

Plug into Bayes, take logs, kill anything without $k$, and the discriminant becomes:

$$\delta_k(x) = x^\top \Sigma^{-1} \mu_k - \tfrac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \log \pi_k$$

Three terms, same shape as 1D: linear cross term, quadratic-in-$\mu$ term, log-prior. **Still linear in $x$.**

### LDA as dimensionality reduction

KNN is "cursed by dimensionality." LDA actually does well in high $p$. Why?

> "It's mapping it down to a lower dimension. It's mapping it down to a dimension specifically to separate out these categories. So if your X's are very high dimensional, you're basically taking a big matrix… and transforming it into a new matrix, now of discriminant scores, that if you have fewer categories than you did originally, then actually you have a new transformation of the matrix specifically designed such that these things are best separated with a line."

**$K$ classes → $K-1$ discriminant scores** (the $K$th is determined by the others summing to 1). This is why LDA is often listed as a [[dimensionality-reduction]] method.

### Worked 2D example — solving for the boundary

Generate data with $\mu_A = (1,1)$, $\mu_B = (3,3)$, $\Sigma = 2 I$ (so $\Sigma^{-1} = \tfrac{1}{2} I$), equal priors. Equate $\delta_A(x) = \delta_B(x)$, gather terms in $x$. The cross term gives an $x$-coefficient from $(\mu_A - \mu_B)^\top \Sigma^{-1}$; the $\mu^\top \Sigma^{-1} \mu$ terms become numbers; equal priors cancel the $\log \pi$ contribution.

Solving (the prof flubbed this live before the break and corrected after):

$$x_1 + x_2 = 4 \quad\Longleftrightarrow\quad x_2 = 4 - x_1$$

A line. He emphasized he did **not** ask for a line — it just falls out:

> "I didn't ask for a line. I just equated the two things and then I solved for them and then it became a line. I didn't tell the math to give me a line."

> [!important] Exam pattern
> "This would be another kind of question that you could ask on an exam — like find the equation for the decision boundary between these two categories. Then you solve for the equation for the line or the plane or whatever it happens to be depending on the dimension of the X's."

The pre-break attempt failed because the prof claimed two terms cancelled when they didn't. After the break:

> "I stupidly said these two would cancel. Obviously that's not true… When I was doing my PhD, one of my co-authors said he doesn't make a habit of doing algebra in public. I've heard his voice in my head a million times. It's a bit of a fool to do public algebra — your brain shuts off, you look like an idiot."

### In-sample vs. Bayes-optimal — the overfitting confusion

After estimating from data, the prof initially couldn't explain why fitted LDA had **lower** error on the training data than the **Bayes-optimal** classifier (which uses the true parameters). Came back after the break with the answer: it's an in-sample comparison.

> "You're not looking at how well you classified out of sample. This is in-sample, on the data you actually train on. So you're going to do really well because you have all the noise in the data. That doesn't mean you're going to do well out of sample."

Generalization point: **the Bayes-optimal classifier was optimal because it used the true values**, but on a finite sample, a fitted classifier can shave error by chasing noise. Out of sample, LDA is worse.

> "It's very much along the lines of the [[bias-variance-tradeoff]] — you're going to have a reduced bias on the data you train on if you're only looking at how well it performs on the data you fit on."

### Recovering actual class probabilities

If you want $P(Y=k|X=x)$ (not just the predicted class), you can either compute the full Bayes denominator — painful — or, more cleanly, **softmax the discriminant scores**:

$$P(Y=k|X=x) = \frac{e^{\delta_k(x)}}{\sum_\ell e^{\delta_\ell(x)}}$$

This works because total probability has to equal 1.

## Quadratic Discriminant Analysis (QDA)

Drop the pooled-variance assumption. Each class gets its own [[random-vector-and-covariance|covariance matrix]] $\Sigma_k$. Everything else identical: still multivariate Gaussian for $f_k$, still estimate $\pi_k$ from frequencies, still apply Bayes.

The consequence is in the discriminant. The Gaussian's quadratic term is $-\tfrac{1}{2} x^\top \Sigma_k^{-1} x$. Before, $\Sigma$ had no $k$, so this term was constant across classes and we threw it away. **Now it depends on $k$ and we have to keep it.** The discriminant becomes:

$$\delta_k(x) = -\tfrac{1}{2} x^\top \Sigma_k^{-1} x + x^\top \Sigma_k^{-1} \mu_k - \tfrac{1}{2} \mu_k^\top \Sigma_k^{-1} \mu_k - \tfrac{1}{2} \log |\Sigma_k| + \log \pi_k$$

**Quadratic in $x$.** Decision boundaries are no longer lines — they're curves (conic sections).

> [!important] Exam-flag
> "That's another good exam question — where does the quadratic come from in QDA? Or show that, yeah… it's an interesting point that simply by making the sigma $k$-dependent, we introduced a new term, and that term is quadratic. So now the decision boundary is no longer going to be a line — it's going to be quadratic."

### LDA vs QDA trade-off

QDA is more flexible. Why ever use LDA?

Same reason linear regression often beats quadratic regression: **flexibility costs you parameters, and parameters cost you variance**. Count the parameters in the covariance matrices alone:

- **LDA**: one shared $\Sigma$ → $p(p+1)/2$ parameters.
- **QDA**: $K$ class-specific $\Sigma_k$'s → $K \cdot p(p+1)/2$ parameters.

For $p = 100$, $K = 5$: LDA needs ~5,050; QDA needs ~25,250. The difference matters when $n$ is small.

> "If P is 100, that's a very big number. It's going to be K times almost 10,000. So it can be a very big difference. And also if you're in 100-dimensional space, then probably a plane is just fine to separate things — you don't need some giant curvy thing."

This is a [[bias-variance-tradeoff]] argument: simpler model → higher bias, lower variance; QDA → lower bias, higher variance. With limited data, the variance penalty wins.

### Visual comparison and the iris example

Slide example: purple Bayes-optimal, black dotted LDA, green QDA. QDA gives nice curves; LDA gives the straight-line approximation. *"In this case, I don't know what's actually best — it looks like the green one might be winning."*

Then Fisher's iris data ([[fisher]]: *"this crazy data by Fisher, was trying to prove race stuff"* — historical aside about Fisher's eugenics work). Sepal length × width, three species. LDA gives **two lines** (because $K-1 = 2$); QDA gives curves.

R results on this exact split:

| Method | Train error | Test error |
|--------|------------|-----------|
| LDA    | 0.19       | 0.17      |
| QDA    | 0.17       | 0.32      |

QDA fits training data slightly better (as it must — strictly more flexible) but **doubles its error on test data**. Classic overfit.

> "If this was my situation… I would go with LDA. The argument would be it's doing better on held-out data. We would maybe try this multiple times — break it into threes and try it three times — and then compare them more rigorously."

(Foreshadowing module 5: [[cross-validation]].)

## Naive Bayes and other variants

The same Bayesian-discriminant machinery generalizes:

- **Naive Bayes**: assume the [[random-vector-and-covariance|covariance matrix]] is **diagonal** — the predictors are conditionally independent given the class. Drops all off-diagonal covariance parameters. *"You toss all the diagonal things, but you could still keep K"* — meaning you keep class-specific diagonal variances. Many fewer parameters.
- **Other density choices**: nothing forces $f_k$ to be Gaussian. You could substitute a Student-$t$ or any density you have prior reason to use. The Bayes math still works.

Naive Bayes is *"optimal or popular when $p$ is large"* because of the parameter count — fewer things to estimate, more robust. Same bias-variance argument.

## Comparing all five classifiers

The lecture's closing synthesis. Five methods, three direct ($P(Y|X)$ explicit) and two indirect (via Bayes):

- **Direct**: [[logistic-regression]], [[knn-classification|k-nearest neighbors]].
- **Indirect (via Bayes)**: LDA, QDA, naive Bayes.

When to use what — picked by **which assumptions are met**:

- **LDA**: small $n$, classes well separated, Gaussian assumption holds, lots of classes. *"More stable than logistic when classes are well separated."*
- **QDA**: Gaussian holds but $\Sigma_k$ unequal. Need enough data.
- **Naive Bayes**: large $p$. Skip the cross-covariance.
- **Logistic**: two classes, no distributional commitment on $X$, want interpretability. *"Logistic regression would be easier to explain than LDA, especially to doctors."*
- **KNN**: wonky non-parametric boundary. Bad for large $p$.

> "The annoying thing about statistics is you don't really know when to use what."

The escape: estimate generalization performance directly from data instead of arguing from assumptions — exactly the QDA-vs-LDA test-error comparison we just did. **That's the bridge into module 5 next week** ([[cross-validation]] and resampling methods).

## Sensitivity, specificity, ROC — first pass

Started but didn't finish — one slide left for next time. The setup:

- **Sensitivity** = true positive rate = TP / (TP + FN) = TP / P.
- **Specificity** = true negative rate = TN / (TN + FP) = TN / N.

Different applications want different trade-offs. The prof's analogy:

> "Typically in the justice system, when we're convicting people, we want a bias towards not putting them in jail. So we're okay letting a few people who committed the crime walk free, because then it keeps the innocent people from going to jail. So if you want to put a lot of people in jail, you want a very sensitive way — your jury reacts in a very sensitive manner, very easily convinced. Whereas if you want to bias towards more specificity, you want them to be more likely to just get all the negative ones right."

Each classifier (logistic, LDA, QDA, …) gives a slightly different sensitivity-specificity trade-off, controlled by the threshold (default 0.5 for binary logistic). Sweeping the threshold traces out the **[[roc-auc|ROC curve]]**, summarized by **AUC**. To be picked up next class.

## Closing — admin

First project is on the wiki for the course. PhD student "Seaman" available Mondays for help. Group sign-up on Blackboard. Project completion at a passing grade is required to take the exam. Module 5 ([[cross-validation]]) starts next week.

> "In general, in many problems, you want to use the simplest possible thing that works."
