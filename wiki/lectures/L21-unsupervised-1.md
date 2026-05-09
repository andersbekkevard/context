---
lecture: 21
date: 2026-04-13
module: 10-unsuper
title: Unsupervised Learning 1 (PCA)
slides: modules/10Unsuper/10Unsuper.md
topics:
  - xgboost
  - nn-regularization
  - partial-dependence-plots
  - interpretability
  - supervised-vs-unsupervised
  - principal-component-analysis
  - explained-variance-and-scree-plot
  - standardization
  - eigenfaces
  - clustering
  - dimensionality-reduction
tags:
  - lecture
  - module/10-unsuper
aliases:
  - Lecture 21
---

# L21: Unsupervised Learning 1 (PCA)

The prof closes module 9 with [[xgboost]] (second-order gradients, parallelization, regularization, [[nn-regularization|dropout]]), recent variants (LightGBM, CatBoost), and [[partial-dependence-plots]] as a way to claw some interpretability back from tree ensembles. After the break he opens module 10 on [[supervised-vs-unsupervised|unsupervised learning]] (flagging how easily it leads to dishonest statistics) and walks through [[principal-component-analysis|PCA]] from scratch: the optimization view (find rotations of maximal variance), the linear-algebra view ([[principal-component-analysis|SVD]] of the covariance matrix), [[principal-component-analysis|loadings]], [[explained-variance-and-scree-plot|explained variance]] / [[explained-variance-and-scree-plot|scree plot]], the *USArrests* and *eigenfaces* examples, why [[standardization]] matters, and ends by sketching where PCA fails (curved data) as motivation for [[clustering]] tomorrow.

## Key takeaways

- **XGBoost wrap-up**: it's still the practical workhorse (second-order gradients for more accuracy with fewer trees, parallelization, greedy approximations, and L1/L2 regularization plus a pruning rate $\gamma$ and per-leaf $\lambda, \alpha$). **Dropout** = randomly remove trees during training so no single tree carries all the weight; came from Hinton, motivated by brain robustness.
- **Tree ensembles kill interpretability**: "this is really like the first case in this course where we have a model that we can't understand." [[partial-dependence-plots]] are the patch: marginalize the model over all variables except $X_j$ by **averaging the prediction across the data with $X_j$ fixed**.
- **Unsupervised learning is dangerous scientifically**: no labels, no goal, no ground truth, and "inevitably you will find something." If you do enough exploring you can lie your way to "statistical significance." Defensible only when tied to a downstream supervised task you can validate.
- **PCA = find rotations of maximal variance**, subject to $\|\phi\| = 1$ (so you rotate, not rescale). The first PC is the direction of largest variance; each subsequent PC is orthogonal to the previous and has the largest remaining variance.
- **The PCA optimization is exactly the eigenvalue/eigenvector problem of the covariance matrix** ([[principal-component-analysis|SVD]]). $\phi_1$ = eigenvector of largest eigenvalue, etc., and the algorithm just hands you a sorted decomposition.
- **Loadings** ($\phi_{jm}$) tell you how much variable $j$ participates in PC $m$, letting you interpret PCs in terms of original variables.
- **Standardize before PCA.** PCA chases total variance, so a variable measured in km vs. cm will dominate purely because of units. Mean-center and scale to unit variance.
- **Scree plot / cumulative variance explained** tells you where to chop. Often the first 1–2 PCs explain 60–90% of the variance; the tail is "probably noise anyway."
- **PCA is linear**: for curvy/circular structure it fails. Clustering (next lecture) is one alternative; non-linear factorizations are another.

## XGBoost wrap-up

The prof reframes module 9 as "many ways of ensembling, clever averaging over different models." The same bagging idea could be applied to a linear model, nothing stops you. He lands back on [[xgboost]] as the modern boosting variant.

### From residuals to gradients

The boosting story he told earlier started intuitively (fit each new tree to the residuals), then abstracted to "follow the gradient":

> "Every new tree is like a correction on the gradient... if the model itself is the least squares problem, then the derivative... is just the data minus the prediction, right? So it ends up being the residuals."

So the "fit-to-residuals" intuition and the "gradient descent in function space" formulation are the same algorithm in the squared-error case, but the gradient framing generalizes the algorithm to any loss / model.

### What XGBoost adds

- **Second-order gradients**: Taylor expansion to second order, not just first. "Second order is better than first." More accurate with fewer trees.
- **Parallelization**: "in the adoption of AI over the last five years, probably the first thing that changed for me in terms of my work was I basically started using the GPU for everything." Easy to verify because parallel versions still compute the same thing.
- **Greedy approximations** for the greedy algorithm.
- **Regularization**: L1 ($\alpha$) and L2 ($\lambda$) on leaf values, plus pruning rate $\gamma$. Same penalization story from the regularization module: pushes parameters toward zero, generalizes better.
- **XGBoost is still the most popular** in practice.

### Dropout

> "It was a clever idea by, I believe it was Jeffrey Hinton, like 15 years ago or so, where basically the idea is to remove parts of the network or parts of the algorithm... in this case, it's trees."

Mechanism: during training, randomly remove some trees. The model's prediction degrades because the removed trees were carrying weight. The next tree fitted then **fills the void**, but in a different way than the original trees did. Effect: redundancy, robustness, less sensitivity to any single tree.

> "Actually I believe it came from the idea that if you smack someone upside the head they won't just die, right? Because we have so much redundancy built into our brains."

Will resurface in the neural network module.

### LightGBM, CatBoost

> "LightGBM uses some other tricks basically to speed it up. So smart subsampling."

CatBoost is for categorical variables. "It's not obvious how you would incorporate those. But they figured it out. Wrote a paper. Everyone uses it."

The prof half-jokes that for time-series competitions, "if you really want to win... you use this stuff" rather than ARIMA: XGBoost will beat the standard time-series model, but you lose interpretability. "Having played a lot with time series models I would argue maybe you can't [interpret them either], but you have the illusion that you can."

### Hyperparameters suck

> "Tuning model sucks. Hyperparameter tuning is very boring. It often matters a lot... a lot of machine learning is engineering the hyperparameters of the model."

XGBoost is "probably the first model in the course where we have a number of hyperparameters" with weird interactions (regularization vs. pruning vs. learning rate).

## Interpretability and partial dependence plots

> "This is really like the first case in this course where we have a model that we can't understand."

A single tree is fine — you read it off. An ensemble of trees built by gradient boosting is opaque: you can't visualize how each tree contributes, what to do when they disagree, etc.

Some people don't care; others do. "If you have a model that is predicting your behavior, wouldn't you want to know how it's doing that?" Tracing why an LLM produced an output is the extreme version. [[interpretability]] is a big topic in modern ML.

### Partial dependence plots

One way to claw interpretability back: collapse the model onto **one predictor at a time** by marginalizing out the others.

The mathematical ideal:
$$f_j(x_j) = \mathbb{E}_{X_{-j}}\!\left[ f(x_j, X_{-j}) \right]$$

The book's empirical estimator: average the model's prediction over all the data points, with $X_j$ fixed at the value of interest:
$$\bar f_j(x_j) = \frac{1}{n} \sum_{i=1}^n f(x_j, x_{i,-j}).$$

> "You're essentially integrating out all the other things... the suggestion is just averaging over all the data points and letting all the other parameters be whatever they were in that data point and fixing just the feature that you care about."

Why use the data and not the population: same bootstrapping idea. "You don't have the underlying distribution, so you use the data to approximate that distribution."

> "The partial dependence function represents the effects of $X_j$ on $f(X)$ after accounting for the effects of all the other variables. And that's of course different than... what's the effect of $X$ if everything else was not there. That's a totally different question."

Pitfall the prof flags: averaging over the data can put you in nonsense regions. "You might be looking at a tiny, tiny house with like 50 bedrooms, which doesn't make any sense."

The Boston housing example: PDPs for `RM` (number of rooms) and `LSTAT`. Computed locally each time; the result can wobble, but if there's overall structure that's a sign of believability.

## Unsupervised learning: framing

After the break the prof goes "straight into unsupervised learning." The rest of the lecture is module 10.

### Supervised vs. unsupervised

- **Supervised**: labels to train against.
- **Unsupervised**: no labels, no goal.

> "This can be dangerous, especially scientifically, because unsupervised learning, you have no goal."

He contrasts two collaborators: one with strong hypotheses ahead of time, one who lets "the data speak for itself." He prefers the former. Letting data speak risks a moving target. "Like if you think of like the ancient explorers... when are you ever done? You're not."

### The dishonesty pitfall

> "If you explore and explore and explore, eventually you will find something no matter what. But then at that point is your finding interesting or significant because you know it was inevitable that you find something?"

The honest answer is "interesting observation, not statistically sound." What many people do instead:

> "What many people do, some at least, is that they kind of lie and they say, oh, this is statistically significant. And they do some tests, ignoring the fact that they've just explored the data... All of the statistical arguments for your finding tend to fall apart after you've done all this subjective exploring. But people don't know that and they just lie."

> [!warning] Verbatim signal: assessing unsupervised methods
> "Unsupervised methods are going to be more subjective because you're not training for a specific goal... whoever does is going to get a different result. And it's very hard to assess results. I think this is undersold here... and has led to many issues."

### When unsupervised is OK

When it's tied to a bigger goal you *can* validate. Example: cluster shoppers, then evaluate whether the clusters lead to better recommendations. "This brings us back to cross-validation and evaluation of a model on a validation set."

### Why we need it

When data has many dimensions, scatter plots and regression curves don't work. "Who can think in eight dimensions?" We need ways to **visualize** and find patterns in high-dimensional data. PCA is the first tool.

## PCA

> "PCA is the thing that I always start with and we often move away from it because it's too simple but everyone uses it. It's the most common way of trying to reduce the dimensionality of your data, and the main reason it's common is the same reason linear regression is common. It doesn't work very well, but you understand it. It's easy. It's fast. It's simple. It's based on a lot of very nice math. It's old. It's not going to do anything too weird."

### Worked-on example: USArrests

The book's `USArrests` data: arrests per 100 000 inhabitants by US state (murder, assault, urban population, rape). The prof spends a lot of breath complaining about the dataset choice ("why are all these data sets so miserable... move away from these miserable data sets"). The point that survives the rant: variables in such matrices are correlated (murder and assault track together), suggesting **underlying factors** (he riffs on "evilness" and then "temperature") that you'd like to recover **without naming them in advance**.

### The optimization view

You have data $X \in \mathbb{R}^{n \times p}$. Find a unit-norm linear combination $\phi_1 \in \mathbb{R}^p$ such that the new variable
$$z_{i1} = \phi_{11} x_{i1} + \phi_{21} x_{i2} + \ldots + \phi_{p1} x_{ip}$$
has **maximum sample variance**, subject to $\sum_j \phi_{j1}^2 = 1$.

> "Find the best way of rotating our shit such that now it has a maximum variance."

Why the unit-norm constraint:

> "If you didn't subject that constraint, you could just make it really... arbitrarily create high numbers, but also because we don't want to rescale the data."

It's a rotation, not a rescaling.

The second principal component is the unit vector $\phi_2$ orthogonal to $\phi_1$ that maximizes variance. Equivalently, $z_{i2}$ should be uncorrelated with $z_{i1}$, hence the orthogonality.

> "Z2 should be uncorrelated with Z1, so therefore we want it to be orthogonal or perpendicular."

And so on up to $p$ principal components.

### The linear-algebra view

> "The really nice thing about this is that... this problem is very related to the stuff we do in linear algebra. So specifically the SVD or singular value decomposition of $X$."

Form the (sample) covariance matrix $\Sigma = \tfrac{1}{n-1} X^\top X$ (after centering). Its eigendecomposition gives **eigenvectors = principal component directions** $\phi_m$ and **eigenvalues = variances** $\lambda_m$ of those PCs.

> "It's not obvious that it's the same problem... I think the way that it was developed, because it was written by Pearson, the Pearson correlation guy like 100 years ago... I think he came to it through this idea like, oh, eigenvalues and eigenvectors, they're amazing, and look, they're actually the directions of maximal variance."

Practical note: "And often it will spit it out ordered, so you're all good to go." Also: SVD lets you exploit the smaller dimension if $n \ll p$ or vice versa. "You only need to do it in the shorter dimension."

### The decorrelation property and PC regression

> "Each subsequent dimension is the direction of maximal variance... you're reducing it into directions that are no longer correlated with each other. You're removing all these annoying correlations in your data."

The prof flags **principal component regression** as a common trick: when your predictors are heavily correlated, PCA them first, regress on the orthogonal PCs.

> "That way the new set of variables of your PCA components, they're not correlated with each other at all... that's what your model wanted to assume in the first place. Then it might be difficult to interpret, but whatever, at least it runs quickly and everyone's happy."

### Loadings

The entries of $\phi_m$ are the **loadings** of PC $m$: how much each original variable participates in that PC.

> "If the direction of maximal variance was murder, and let's say that actually was the thing underlying everything... then the $\phi$ for murder would be one. It would be very strong."

Loadings let you sort of interpret each PC as a function of the original variables. On USArrests: PC1 picks up an "underlying-criminality" axis (multiple variables loading strongly), PC2 picks up something different (urban population, basically).

### Variance explained and the scree plot

Each PC explains less variance than the last. Define the **proportion of variance explained** by PC $m$ as
$$\text{PVE}_m = \frac{\lambda_m}{\sum_{k=1}^p \lambda_k}.$$

The **scree plot** plots PVE (or cumulative PVE) against $m$.

The prof reads the USArrests scree plot:

> "Here you can see what the first component actually explains... more than 50%. Yeah, 60-something percent of the variance... And now with two components, you're almost to 90% of the variance explained... that last bit going from the third to the fourth, that was small. So probably you could chop that away."

The recipe: chop where the marginal PVE flattens out. Useful when you have ~100 dimensions and want a manageable / interpretable subset.

### Standardization is mandatory

> [!important] Verbatim signal
> "It really depends [on] the scaling of the variables, right? Because it's trying to capture the total variance. So let's say one aspect was the area of the thing — if you measure it in terms of meters or kilometers, you would get a very different scale of the variable. So then the variable would appear to have a higher variance along that dimension just because that has bigger numbers."

You want to **z-score each variable** (subtract mean, divide by standard deviation) before PCA, so unit choice doesn't drive the algorithm.

> "So you want to meaningfully normalize each variable because otherwise the scales screw it all up."

(Slide deck has a whole demo: `prcomp(..., scale = TRUE, center = TRUE)` vs. without, illustrating exactly this.)

### Uniqueness up to a sign flip

> "The principal component vector is unique. So there's only going to be one. Of course, the sign flip is boring. It just means which direction it is."

### Picking $M < p$ to actually reduce dimension

After computing all $p$ PCs (no dimensionality reduction yet, you've just rotated coordinates), you keep the first $M$ where the variance is concentrated. The directions of small variance "are probably noise anyway. So who cares? Just throw away the noise. It's not always true... but it's still often a good idea."

### Eigenfaces

A second worked example. Take a face image dataset (400 faces, 10 different people, zoomed in on the face, purely linear so no rotation modeling). Stack each image as a column; the matrix is `pixels × faces`. Run PCA along the faces dimension.

> "By the time we get to 1, 2, 3, 4, 5, 6, 7 PCs, you already have 60% of the variance explained. So somehow with just seven principal components, you already have 60% of all the variability that you're going to see in these 400 images of faces."

The principal components themselves can be rendered back as images (the **eigenfaces**). They look creepy but show what the algorithm picked up about variance between people and expressions.

> "It's just simply doing it based off of the data, just trying to explain what other transformation in the space of the images can I do to try to capture more of the variance in the data?"

### What PCA isn't

The prof points to an alternative on the same data: factorize $X$ into individual additive parts (NMF-style). On faces, instead of "creepy whole faces" (eigenfaces) you get parts (a nose, an eye, a mouth) that sum to the image. Different assumption, same dimensionality-reduction goal. He won't go into details, but flags that **PCA is one choice among many**.

## Bridge to clustering

> "There's a bunch of dimensionality reduction methods. This is by far the oldest and the simplest and often the most useful."

He sketches PCA's failure mode on the board: data lying along a curved 1-D manifold (like a circle or arc). PCA's first PC will cut through the middle and the second perpendicular. Neither captures the actual structure.

> "PCA is kind of stuck just making linear stuff. It can't do anything non-linear. Whereas clustering would try to group these into distinct things."

So **clustering** is another way to summarize high-dimensional data:

> "The goal of clustering is to partition the data into different groups. You want the points that are within that group to be close together or similar... and you want the different groups to be different."

The key knob: how you **define distance** (Euclidean, cosine, something else), as different distance functions give different clusterings.

Used for: cancer subtypes, market segmentation, search engines. The prof's own current example: segmenting animal behaviors by clustering features of head and back angles plus their frequency decompositions, recovering discrete behaviors like "rearing," "running," "meandering."

> "Both dimensionality reduction and clustering approaches are trying to simplify the data, to give simple summaries. Dimensionality reduction looks for low-dimensional ones... clustering tries to find groups. So it's more of a discretized thing. But in both cases, you're looking at a short description of high-dimensional data."

Tomorrow: k-means and hierarchical clustering.
