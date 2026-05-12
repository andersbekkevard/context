---
lecture: 15
date: 2026-03-03
module: 06-modelsel
title: "L15: Model Selection and Regularization 4 (PCR wrap)"
slides: modules/6ModelSel/selection_regularization_presentation_lecture2.md
topics:
  - principal-component-analysis
  - principal-component-regression
  - partial-least-squares
  - dimensionality-reduction
  - ridge-regression
  - lasso
  - high-dimensional-regression
  - curse-of-dimensionality
  - collinearity
  - bias-variance-tradeoff
tags:
  - lecture
  - module/06-modelsel
aliases:
  - Lecture 15
---

# L15: Model Selection and Regularization 4 (PCR wrap)

The prof closes module 6 by finishing [[principal-component-regression]] (PCA recap, PCR on the credit data, why it looks like a discretized [[ridge-regression]]), introducing [[partial-least-squares]] as "PCR but with a $y$ in the covariance," and then a closing motivational segment on **why** all this regularization machinery exists: the high-dimensional setting where $p > n$ is now everywhere and standard regression breaks. Lots of intuition, no new formal derivations; treat the PCR↔ridge analogy and the high-dim cautions as the load-bearing content.

## Key takeaways

- **PCR pipeline.** Standardize $X$ → run [[principal-component-analysis|PCA]] to get components $Z_1, \dots, Z_M$ (orthogonal, ordered by variance) → regress $Y$ on the first $M$ of them. Choose $M$ from a CV-MSE curve. PCs are linear combos of the original $X$'s, so PCR (like ridge) does **not** select variables.
- **PCA: "not scale invariant, so you have to standardize."** Otherwise the largest-unit predictor dominates the first PC.
- **Eigenvalue = variance of the corresponding PC.** Explained-variance plot stacks $\sum_{i=1}^M \lambda_i / \sum_{i=1}^p \lambda_i$.
- **Two big PCR assumptions** (both can fail): (1) the response actually lives in the directions of largest $X$-variance, *"no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response"*; (2) the relationships are linear. PCA is unsupervised; you "make an assumption, hit go."
- **PCR ≈ a discretized [[ridge-regression]].** Both shrink the small-eigenvalue directions; ridge does it smoothly via $\lambda$, PCR does it abruptly by truncating components. *"PCR is doing it more abruptly because it simply says okay, direction, direction, direction, new axes, new data, everything that's shared go here."*
- **PLS = PCR but using $\text{Cov}(X, Y)$ instead of $\text{Var}(X)$** as the maximization target: "the same idea as PCA, only now you're finding the principal components not as the directions of maximal variance of $X$, but the maximal covariance of $X$ and $Y$." Algorithm: regress each $X_j$ on $Y$ to get the $\phi_j$'s, deflate, repeat.
- **PLS verdict (verbatim, prof's editorial):** *"PLS often performs no better than ridge regression or PCR but it's Swedish."* Common in chemometrics; he's never really used it.
- **High-dim is the whole reason this module exists.** When $p > n$, OLS fits training data perfectly even when nothing predicts anything; CP/AIC/BIC are unreliable because $\hat\sigma^2$ is hard; the true coefficients are essentially unrecoverable due to extreme [[collinearity|multicollinearity]]; *"we can never know exactly which variables, if any, truly are predictive of the outcome."*
- **Adding noise features always hurts** test error eventually, even with regularization. "Noise" means actual noise OR features that are real but irrelevant.
- **Aside on science vs. fishing.** This entire module is *exploratory*: finding which relationships are strongest. Real science formulates a hypothesis first and then tests it on a different sample. Don't confuse the two.

## Recap: where we are in module 6

Three families of tools so far for taming variance with too many predictors:

1. **Subset selection**: best subset, [[subset-selection|forward selection]], [[subset-selection|backward selection]]. Pick the variables.
2. **Shrinkage / regularization**: [[ridge-regression]], [[lasso]], [[elastic-net]]. Penalize the coefficients.
3. **Dimensionality reduction**: *"a way of kind of cheating where you would actually reduce the number of variables you have before you even go to regression."* You go from $X$ ($n \times p$) to $Z$ ($n \times M$, $M < p$), throw stuff away, then do standard regression on $Z$.

This session finishes (3), PCR and PLS, and then steps back to the high-dimensional motivation for the whole module.

## PCA recap

The prof re-summarized PCA before launching into PCR proper:

- Take a linear combination of the columns of $X$ to maximize variance, **subject to** $\sum_j \phi_j^2 = 1$. That's the first principal component.
- The next PC is the linear combination orthogonal to the first with maximum variance under the same constraint. Then the next one orthogonal to both, etc.
- *"There's a lot of shit that's orthogonal to the first. Remember that you're in high dimension, so it's okay. It's not so restrictive."*
- You can do this $p$ times (assuming $n > p$), and the components come out **ordered by variance**: PC1 is the strongest, PC$_p$ the weakest.

Stack the variances → the explained-variance curve climbs from 0 toward 1. As a scientist you draw a threshold somewhere and decide *"I don't give a shit about the weak shit, the stuff that doesn't vary much."* Justification:

> "If you, in a regression model, adding something that's constant is the same as just adding a bias term. You need for the thing to actually matter in the model, it should vary somewhat."

Why care about PCA at all? Two reasons:

1. **Visualization**: projecting high-D data into 2 PCs is the standard visualization trick.
2. **Decorrelation for downstream models.** *"Many models, basically all models, will suffer when there's a high correlation between variables. I can't think of a model that really benefits from it."* Orthogonalizing also makes fitting fast, no trade-off between correlated coefficients.

> [!important] PCA is **not** scale-invariant
> "If you don't standardize, then it'll just be dominated by the strong one." Imagine one $X$ in km and another in mm. Standardize so all variables share the same scale (in standard-deviation units).

### How "explained variance" is actually computed

A student asked yesterday and the prof didn't remember. Today's answer:

- The **eigenvalues** of the covariance matrix of $X$ are equal to the variances of the corresponding principal components. *"This is not obvious, but the eigenvalue is equal to the variance of the principal component."*
- So fraction of variance explained by the first $M$ PCs = $\sum_{i=1}^M \lambda_i \,/\, \sum_{i=1}^p \lambda_i$.
- That's what the explained-variance plot is showing: running normalized cumulative sum of eigenvalues.

## Principal Components Regression (PCR)

The procedure:

1. Standardize $X$, run [[principal-component-analysis|PCA]] → get $Z_1, \dots, Z_p$.
2. Fit a standard linear regression of $Y$ on $Z_1, \dots, Z_M$, where $M$ is a tuning parameter.
3. Sweep $M$ from $1$ up to $p$. Pick the $M$ that minimizes CV-MSE.

Two motivations: fewer effective variables, and the hope that PCA's compression generalizes better.

### When PCR helps and when it doesn't

The whole bet is captured here:

> "A key assumption of the data is that it actually does have strong components where things vary. So like these figures are often very good to make because you can see, in this case, quite a lot was explained by just one PC, but others will look more like that, in which case the variables were already orthogonal and then adding more PCs is just the same as adding another variable. In that case, PCA isn't doing nothing for you."

Plus the linearity caveat: PCA is linear, so nonlinear $X$-relationships won't be captured.

### Bias / variance / test-error decomposition for PCR

Standard plot: $x$-axis = number of components $M$, $y$-axis = MSE, curves for bias², variance, and test MSE. The lecture's reading:

- **Bias** drops fast at small $M$, then plateaus. (More PCs $\Rightarrow$ more parameters, but bias was already low.)
- **Variance** rises with $M$: more parameters, more noise fitting.
- **Test MSE** is U-shaped: drops with the first few useful PCs, then climbs as you start adding directions that explain $X$ but not $Y$.

The simulated example shown is a friendly one: a few PCs do most of the work.

> "Of the many solutions that have a low bias, the mean squared error on the held out data is in a similar location to where the variance is minimized."

### PCR vs. ridge vs. lasso comparison

Same family of curves shown for ridge and lasso, plotted against a *shrinkage factor* (not $\lambda$ directly, *"it gets rid of the scaling, the relative scaling of the different models"*). Shape is similar to PCR's: U-shaped test error with a minimum aligned with the variance-minimum.

Slide summary on the simulated data: *"PCR needed five components. The results are only slightly better than Lasso and very similar to Ridge."*

> [!important] PCR is **not** a variable-selection method
> "Just like Ridge, it doesn't actually select the parameters — it gives you components, and each component is a combination of the original axes." Use PCR/ridge when you want predictive power and don't care which raw variables drive it. Use [[lasso]] or [[subset-selection|subset selection]] when you need *"this or that one."*

### PCR as a discretized ridge regression

The most important conceptual point of the segment:

> "PCR can be seen as a discretized version of ridge regression. Ridge regression encourages ties — if two things explain the same thing, then have them share the weight. The point where they share is actually very similar to getting a principal component that captures the part that both share. So PCR is doing it more abruptly because it simply says okay, direction, direction, direction, new axes, new data, everything that's shared go here. Ridge regression does it more continuously."

Both methods place pressure on the **least important** principal directions:

- **PCR**: drop them outright (those past the cutoff $M$).
- **Ridge**: shrink them most, because the least-important directions correspond to the smallest eigenvalues, and ridge's effective shrinkage on each PC direction is heavier when the eigenvalue is smaller.

The slide showing this with $p = 8$ components had broken axes: *"this is a confusing figure, I'll try to make another one for next time."*

### PCR on the credit data

Run PCR on the standard credit example. Solve regression with $M$ components, then *"back out what the coefficients are"* in the original $X$-space via the inverse map from $Z$ to $X$. Read the standardized-coefficient plot:

- With $M = 1$ the original-variable betas are tiny: PC1 didn't load heavily on the variables that drive $Y$.
- A big jump at $M = 9$: that's where **income** finally enters strongly. *"Income just happened not to vary as much or wasn't as correlated as other things."*
- Other variables that were correlated with **student** came in earlier, since student varied a lot in this dataset.

CV-MSE was decreasing slowly, then dropped massively from $M=9$ to $M=10$. Settle on $M = 10$.

> "Even though the things that you care about were just ended up being four variables, primarily — it took you 10 PCs to get there, probably because other stuff was in there and that stuff just didn't matter. The nice thing is that 10 is still smaller than the total number of variables, so it was helpful, but maybe not maybe the other approaches that we talked about earlier with respect to this particular data would be more useful."

Recommended exercise: redo this on credit and compare against ridge / lasso / subset selection.

### Drawbacks of PCR

Two real ones:

1. **PCA is unsupervised.** It uses only $X$, not $Y$. So the directions of largest $X$-variance might be unrelated to $Y$. *"You're saying, I'm assuming that the things that Y cares about are the strongest directions of variance in my data, X."* Not always true.
2. **No interpretation in terms of original variables**, just like ridge.

> [!quote] Verbatim summary line
> "There's no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response."

## Partial Least Squares (PLS)

Developed in the 70s by [[wold|Herman Wold]] (Swedish) for **chemometrics**: they had lots of variables, needed fast linear methods, computers were terrible, and PCA wasn't doing it for them.

### The setup vs. PCA

Same shape: a [[dimensionality-reduction]] step, $Z_m = \sum_j \phi_{jm} X_j$, then regress $Y$ on $Z_1, \dots, Z_M$.

Difference is the objective:

- **PCA**: choose $\phi$ to maximize $\text{Var}(Z)$ subject to $\sum_j \phi_j^2 = 1$.
- **PLS**: choose $\phi$ to maximize $\text{Cov}(X\phi, Y)$ subject to the same constraint.

> "It's the same idea as the principal component analysis, only now you're finding the principal components not as the directions of maximal variance of $X$, but the maximal covariance of $X$ and $Y$."

It can sound like cheating, *"it kind of sounds like you're doing regression twice"*, but it works.

### Algorithm

To get the first PLS component:

1. Regress $Y$ on each $X_j$ separately. The simple-regression coefficient $\hat\beta_j$ is **proportional to** $\text{Corr}(Y, X_j)$: that's your $\phi_{j1}$. So $Z_1$ weights each $X_j$ by how well it (alone) explains $Y$.
2. To get $Z_2$: orthogonalize $X$ with respect to $Z_1$ (work with the residuals after regressing each $X_j$ on $Z_1$), then repeat on the residuals.
3. Iterate to get as many components as you want.

So the new direction at each step is the maximum-covariance direction in the part of $X$-space not yet explained by previous components.

### Verdict

Same kind of curves as PCR. From the slide summary:

> "PLS often performs no better than ridge regression or PCR but it's Swedish, so it's like they're meatballs — they're not better but they sound good."

It can reduce bias but can also increase variance. The prof's bottom line: niche, common in chemometrics, *"there's actually a guy here who made a whole company about it,"* but he's never really used it.

> "All of these tend to behave similarly. A lot of things come down to looking like ridge regression, surprisingly. And conveniently, ridge regression behaves smoothly instead of this kind of discrete thing which PCR and PLS both have."

### Where the prof lands on the four families

Editorialized ranking from the prof's mouth:

> "Lasso falls somewhere between ridge and best subset regression and has some nice properties of each. For me, definitely the most interesting things are lasso and also implicit lasso. Many different ways you can get lasso and ridge regression. Those are, I think, the most interesting part of this module. But being exposed to the ideas of partial least squares regression and PCR are good because they're fairly common — certainly PCR is very common."

### Generalization beyond linear PCR

The PCR pattern (compress $X$ down to fewer features with some method, then regress) generalizes far beyond PCA. Example: video frames as $X$. They're huge. Run them through a learned feature extractor (a neural net) to get a small vector per frame, then regress $Y$ on that compressed representation. You don't know what the network is doing, *"but either way you can get it compressed to something small, and then that you can use in something like a regression model."* Same idea, nonlinear compressor.

## High-dimensional regression: closing motivation

Final segment, after the break. Less algorithmic, more "why this whole module exists." Splines and nonlinear methods start next module.

### The setting

> "Most of statistics has really been built on the notion that you have more data points than your parameters. So a lot of what happens when the situation is reversed, it's just not well understood. And so it's very important. And now it's also very common."

Why $p > n$ became common: cheap sensors, cheap storage, cameras and sensors recording everything. Prof's own neuroscience example: went from 8–15 neurons recorded simultaneously to 6–7 thousand, plus eye cameras, whisker angles, head tilt, body posture, behavioral decomposition. Both $X$ and $Y$ exploded.

### What goes wrong with vanilla OLS in high dim

Replay of the Day 1 figure: as you add predictors,

- **Training $R^2 \to 1$** (training MSE drops to 0). Always: *"if you have enough of them, you will fit the training data perfectly."*
- **Test MSE blows up.**

> "If you have a lot of variables, then you're going to perfectly fit the data, even though the relationship might be completely meaningless."

Worse, sometimes you don't even have enough data to construct a test set.

### Why CP / AIC / BIC don't save you

The prof's complaint: *"They're problematic, partially because it's hard to estimate the $\sigma^2$. I would say they're more problematic because their assumptions are always wrong, and they're typically always wrong. So I wouldn't recommend using those."*

### The three-dataset comparison plot

Three simulated datasets, total $p = 20$, varying number of included predictors:

- One: error **drops** when you include more, adding predictors helps.
- Another ($p$ ≈ 50, models with up to ≈ 28 parameters): U-shape, drops, then climbs again. Don't include all of them.
- Third: monotonically bad, *"all of it sucks. You have a lot of parameters and it's hard to figure out which ones to use."*

> "These figures were made where there was zero regularization put in the model. This is just standard regression when you probably shouldn't use regression. Probably all of these would look better with some regularization."

### Adding noise features always (eventually) hurts

> "Adding more features can help, smaller can help if you have the right ones, but adding noise features that are not associated with the response increases test error. Noise features exasperates the risk of overfitting because you start fitting to the noise instead of the variables that matter."

"Noise" can mean either actual measurement noise or *real-but-irrelevant* features. Including details about Japan's weather in a model of what's happening in this classroom: not noise in the statistical sense, just irrelevant. The effect is the same.

> "Regularization does not entirely eliminate the problem. It certainly helps. And these things are actually ongoing and improving over the years."

### The exploratory-vs-confirmatory aside

A pause to flag something the prof felt wasn't being said often enough:

> "Right now I'm saying you're trying to explore for relationships in your data. That's not often what we want to do scientifically. Often we want to test relationships. So this would be more in the exploratory phase of trying to understand what relationships are there. That's not really typically how you want to do science. You want to do science in such a way that you make a hypothesis, formulate it as a null hypothesis, get the data, and then you test that one thing. Here, we're not testing things, right? We're just seeing what relationships are the strongest. So it's still called statistics, but we're not doing hypothesis testing. We're not doing good science in a way. We're just kind of fishing around. It would be like a first step — you fish, then maybe you have another set of data where you would test something."

### Multicollinearity in high dim

The slide author was *"Brazilian, so of course he's going to exaggerate everything"* but the prof endorsed the substance:

> "In the high-dimensional setting, the multicollinearity problem is extreme. We can never know exactly which variables, if any, truly are predictive of the outcome. We can never identify the best coefficients for use in the regression. At most, we can hope to assign the large regression coefficients to variables that are correlated with the variables that truly are predictive of the outcome. We will find one of possibly many suitable predictive models."

> "Sometimes you can say, oh actually everything can be reduced onto this one variable and it's super clear. I agree that's not the typical setting — the typical setting is it's just a big mess, but sometimes it works."

[[collinearity|Multicollinearity]] = linear correlation between predictors. In high dim it becomes pathological.

### Curse of dimensionality

The prof's add-on, not on the slide:

> "One of the key problems with high-dimensional stuff is that distances between points become less variable. So when you have a lot of dimensions, you have this [[curse-of-dimensionality]], so then it's harder to distinguish between things."

This is the same KNN-killing phenomenon from earlier modules: in high dim, *every* pair of points is roughly equidistant.

## Closing

That ends module 6. Next module: **splines and nonlinear methods**, *"a very nice trick"* for handling simple nonlinear stuff. Back to the Swiss slide deck.
