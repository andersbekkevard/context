---
lecture: 17
date: 2026-03-10
module: 08-trees
title: "L17: Tree-based Methods 1 (BeyondLinear wrap-up)"
slides: modules/8Trees/8Trees.md
topics:
  - generalized-additive-models
  - logistic-regression
  - polynomial-regression
  - regression-splines
  - local-regression
  - regression-tree
  - recursive-binary-splitting
  - cart
  - greedy-algorithm
  - tree-pruning
  - cost-complexity-pruning
  - cross-validation
  - bias-variance-tradeoff
  - data-models-vs-algorithmic-models
tags:
  - lecture
  - module/08-trees
aliases:
  - Lecture 17
---

# L17: Tree-based Methods 1 (BeyondLinear wrap-up)

A two-module session: the prof first finishes Module 7 by extending the [[generalized-additive-models]] machinery to the **logistic** (binary) case (polynomial logistic regression, splines inside a logistic regression, and finally a logistic GAM) using the same wage > $250k data. After the break he opens Module 8 with Breiman's "two cultures" framing of statistics, then walks through the **regression tree**: recursive binary splitting via CART, greedy minimisation of RSS, stopping criteria, and finally **cost-complexity pruning** chosen by CV.

## Key takeaways

- **All the BeyondLinear tricks lift to logistic regression unchanged**: replace $Y$ with the log-odds $\log(p/(1-p))$ and the same polynomial / cubic-spline / natural-spline / local-linear / GAM machinery applies. "Polynomial logistic regression extends the polynomial regression to logistic data. Just the exact same way."
- **GAMs let one variable explain another away**: when you add a better-suited predictor, the contribution of a previously-important one can flatten out. This is exactly the mechanism the prof used in his rat-postural-coding paper to overturn an earlier claim.
- **Breiman's two cultures (2001)**: *data-model* statistics (assume distribution, do MLE / inference) vs *algorithmic-model* ML (validate by prediction accuracy / CV). Trees are the prof's first **algorithmic** model, explicitly different in flavour from regression / splines / GLMs.
- **Regression tree = recursive binary splitting + region-mean prediction.** Predictor space is chopped into non-overlapping rectangles $R_1, \ldots, R_J$; the prediction in each region is the **mean of the training $y$'s** in that region. Loss is RSS (sum, not mean) over all regions.
- **Splitting is greedy and one-variable-at-a-time.** Exhaustive optimal partitioning is **NP-complete**, so at each node we pick the (variable $j$, threshold $s$) that minimises the two-region RSS, and once split we never go back. This is the **CART** algorithm (Breiman again).
- **Trees are highly interpretable.** Unlike GAMs which look at one variable at a time at the means of the others, a tree path directly couples conditions on multiple variables, giving "I want to be those guys who are older than X *and* hit more than Y."
- **Build deep, then prune.** Grow a maximal tree using lax stopping criteria, then apply **cost-complexity pruning**: minimise $\sum_{m=1}^{|T|} \sum_{x_i \in R_m}(y_i - \hat y_{R_m})^2 + \alpha |T|$. Walk back from the maximal tree, each step removing the leaf-merge that hurts least, generating a sequence of nested subtrees. Pick $\alpha$ (equivalently the tree size) by **K-fold CV**.

## Module 7 wrap-up: GAMs visualisation recap

The prof opens by reminding us we ended yesterday on [[generalized-additive-models]]. The model is

$$Y = \beta_0 + f_1(X_1) + f_2(X_2) + \cdots + f_p(X_p) + \varepsilon$$

where each $f_j$ can be **any** of the basis-expansion methods covered yesterday: piecewise constant intervals, cubic spline with chosen knots, [[regression-splines|natural spline]] (linear at the boundaries), smoothing spline (penalising $\int (f''(x))^2 dx$), or piecewise / [[local-regression|local linear]]. You pick the $f_j$ per variable based on the data:

> "Maybe one you just want to keep as intervals because you're worried it's going to be a very unruly function or there's not a lot of data."

### The wage GAM plot

Each panel is the **contribution** of one variable to wage, with the other variables held at their means.

- **age**: cubic spline with knots at two values; peaks around 47 ("which sucks because that's today for me").
- **year**: [[regression-splines|natural spline]] with a knot at 2006; basically linear, "makes sense, it's going to go up every time, just because of inflation."
- **education**: piecewise, broken into the four education intervals.

Crucially, each panel is a contribution, not a prediction:

> "It's not that you look at this and like, oh well, he's 47, so he makes eight whatever units. … this is just the contribution. … You also have to consider all these other ones."

A subtle assumption baked into this additive form: **education is independent of year**. Extrapolating to 2025 assumes the wage premium for a degree hasn't changed over time, which historically it has ("when I was very young there weren't that many people who had masters or PhDs … now people are arguing it's going to matter less and less"). Prof flags this as motivation for letting variables interact, which is what trees will give us next.

## Logistic versions of the same models

> "Sometimes your data is not distributed that way … the thing you actually care about is a decision or it's a binary thing."

Switch from continuous $Y$ to binary. Model the log-odds:

$$\log\frac{P(X)}{1 - P(X)} = \eta(X)$$

and let $\eta$ be **anything we built yesterday**. "This idea extends to lots of things. … like you could imagine it has three states or n states or whatever but in the binary case we're going to look at this."

### Polynomial logistic regression

Use `glm(... family=binomial)` (binomial = Bernoulli = logistic regression, all synonymous). Response: indicator wage > $250k. Predictor: age as a polynomial.

> "Polynomial logistic regression extends the polynomial regression to logistic data. Just the exact same way."

The fitted curve is now $P(\text{wage} > 250k \mid \text{age})$. Probability is near zero below 20, peaks around the prof's age, then drops, with **enormous confidence intervals at the tail** because almost no 80-year-olds in the data. The CI explosion is the takeaway, not the apparent uptick.

### Splines inside a logistic regression

Same model, replace the polynomial with a [[regression-splines|cubic spline]] basis on age:

$$\log\frac{P}{1-P} = b_0 + b_1 B_1(\text{age}) + b_2 B_2(\text{age}) + \cdots$$

Now the probability curve has a different shape: peaks around 60 instead. The prof half-jokes:

> "This is, of course, the model that I'm going to use because it tells the story I want to say, which is not how you do science at all, which is bad."

Still, real point: **different basis choices give meaningfully different fitted shapes** (here even bimodal-looking). All the methods from yesterday (local linear, natural splines, etc.) plug in identically.

### Logistic GAM

Combine into a multivariate logistic GAM:

$$\log\frac{P(X_1, X_2)}{1 - P(X_1, X_2)} = \beta_0 + f_1(X_1) + f_2(X_2)$$

In the demo: $f_{\text{age}}$ is a **local linear** (`lo`) fit, $f_{\text{year}}$ is just linear. What's plotted per panel is the **contribution to the log-odds**, with the other variable at its mean, same convention as the regression GAM.

> "These models will actually center all of the data and then when you predict for each value you're essentially having seeing the other variables set to their mean values."

### "Explaining away" via additive structure

Key conceptual payoff of GAMs (and a recurring theme): if year is **really** the variable carrying the signal, then including year alongside age can **flatten** age's contribution. Whichever variable is doing the actual explanatory work absorbs the signal; the other one decays.

The prof tells a long story from his postdoc with Jonathan Whitlock at the Kavli. Earlier neuroscience had claimed certain neurons "respond to direction changes" because researchers tracked only position and head direction. He added 3D body tracking:

> "What mattered is how the animals were postured, how their heads were relative to their bodies, those were the variables that mattered. And when we included those, then the increase that they saw, this trend that they would see, just flattened out. … we explained away the correlations that they saw by getting a better set of variables."

The same mechanism happens in any GLM/GAM: add the right covariate and a previously-significant one can drop out under model selection. **This is the closing lesson of Module 7.**

## Breiman's two cultures (Module 8 opens)

The prof goes back to a slide deck from earlier in the course to set up trees with **Leo Breiman's 2001 paper**. Breiman: statistician → industry → back to academia, instrumental in tree methods.

### Data models vs algorithmic models

- **Data-model culture (statistics)**: start with $Y = f(X) + \varepsilon$, declare a parametric form for $f$ (linear / logistic / etc.) and a noise distribution, do MLE, derive distributions of estimators, run inference. AIC, BIC, p-values, the whole scaffolding.
- **Algorithmic-model culture (ML)**: define a procedure (algorithm) that maps inputs to predictions. Validate by **prediction accuracy on held-out data** (i.e. cross-validation). Don't commit to a generative story for the data.

Breiman's argument:

> "Too much focus on the data models has led to irrelevant theory and questionable scientific conclusions … and it's kept statisticians from using more suitable algorithmic models like the trees."

Prof endorses Breiman's side, with characteristic edge:

> "I don't know why so many people spent so much time developing AIC and Bayesian information criteria and those methods for that stuff. It boggles my mind. Like, why do that? All your assumptions are wrong."

He references **Cox's published response** "pooping on everything Breiman said," recommends reading the back-and-forth as a meta-question for aspiring statisticians:

> "Do you want to restrict yourselves only to models where you can really lay out all your assumptions ahead of time in this very kind of data model perspective? Or are you willing and interested in understanding these more algorithmic models, which are very commonly used?"

> [!note] Not on the test
> The Breiman/Cox debate itself is "not going to be on the test." But it reframes what trees are: trees are the **first algorithmic model** of the course.

## Regression trees: setup

Slide deck attributed to Stephanie. Topics for Module 8: decision trees, regression trees, classification trees, pruning, **bagging**, with "trees and forests and bagging trees and pruning and leaves and branches" to come.

### The motivating picture: hitters data

Two predictors: **years** of playing experience and **hits**. Response $Y$: salary, shown as the colour of each point. The pattern is **interactive**: salary increases with years, but **only for players with high hit counts**. An additive model can't capture this: it would force years and hits each into a 1-D contribution and add them up.

Instead, just **chunk** the predictor space:

> "We're going to chunk up our X's with rectangles such that the mean of the points within that rectangle predicts the activity well."

Three-region partition example: a vertical split on years produces region 1 on the left; the right half is then split horizontally on hits into regions 2 and 3. Every point in $R_m$ gets the same prediction: the mean salary within $R_m$.

### Trees as a representation

This same partition is drawn as an upside-down tree: root at the top (the "tree trunk you would hug"), each internal node is a binary split on one variable at one threshold, each leaf is a region $R_m$ with its mean prediction.

Critical detail: when the right half of the years-split gets split on hits, that horizontal line **does not extend to the left half**. If we extended every cut across the whole space we'd be back to the additive grid of yesterday. **Trees split one current region at a time, on one variable at a time.**

## Building the tree: recursive binary splitting

### Why not exhaustive

The number of possible partitions is enormous; finding the optimal partition is **NP-complete**:

> "It's NP-complete, which is just computer science for wicked hard and not reasonable. … We're not going to do it unless we have quantum computers."

So we use a **greedy** approach (same word as in forward/backward model selection from Module 6): at each step, pick the locally best split and never revise.

### CART

The specific algorithm is **CART = Classification And Regression Trees**, by Breiman. Other partitioning approaches exist; "no reason to dwell on that."

### The split criterion

At each node, for each variable $X_j$ and each threshold $s$, define

$$R_1(j, s) = \{x : x_j < s\}, \qquad R_2(j, s) = \{x : x_j \ge s\}$$

and minimise

$$\sum_{i : x_i \in R_1(j, s)} (y_i - \hat y_{R_1})^2 + \sum_{i : x_i \in R_2(j, s)} (y_i - \hat y_{R_2})^2$$

over all $(j, s)$ pairs. Note these are **sums of squares** (RSS), not means. Explicitly:

> "We're summing up the squared difference between each point and the average of all the shit in its region. … I'm not taking any mean values here. If we did, we would have a different algorithm, and one that wouldn't behave very well."

### Recursive

After the first split, repeat the same procedure **inside each child region**. Each region is split or not based on the same RSS criterion. Hence "recursive binary splitting." Continue until a stopping rule fires.

> "Trees don't go and change their mind and reconnect. I don't think they do. I'm not a tree guy."

## Worked example: ozone data

Data: ozone concentration (response), with predictors **radiation** (Langleys), **temperature** (°F), **wind** (mph). First few rows shown on slide.

The fitted tree splits **on temperature first**, then on temperature again, then again ("I guess temperature really matters"), and finally on wind, never on radiation. So the tree's induced partition lives in (temperature, wind) only; radiation is implicitly squished. If radiation had been chosen, the regions would become **3-D cuboids**, but the tree representation absorbs that gracefully:

> "These things are easier to generalize to higher dimensions, right? … Whereas this, you can always just say what you're splitting on, right? And then you can work up the tree to figure out what your region is."

A leaf might say: "if temperature > 77.5 and < 84.5 and wind < 7.7 then expected ozone = …", directly readable.

## Stopping criteria

You **could** keep splitting until every leaf has one point (perfect training fit, zero variance within region). In practice the `tree` R package uses three stopping parameters (named approximately):

- `mincut`: controls tree depth
- `minsize`: minimum observations per leaf (e.g. don't split if either child would have < 10 points)
- `mindev`: minimum reduction in deviance, expressed as a fraction of the root deviance (e.g. stop if a candidate split would reduce RSS by less than 1% of the root RSS)

Prof is hazy on the exact `mindev` formula in the slide and promises to clarify Monday: it's the minimum proportional reduction in within-node sum-of-squares that justifies a further split. Below that, "you're just splitting noise."

> "More importantly, this isn't how we just decide our final model. Typically, this is more criteria to just stop splitting. We're then going to find another way of determining our tree."

So: **grow much bigger than you actually want**, stopping only at very lax criteria, then prune.

## Pruning

### Why prune

A maximal tree overfits: high variance, hard to interpret. A smaller tree:

- **Lower variance** → generalises better
- **More interpretable**
- **Higher bias** (bias-variance trade-off, again)

> "It's good to think about [bias-variance] … we want to keep making sure smaller trees, but of course we don't want to go too far."

### Cost-complexity pruning

Define the penalised objective:

$$\sum_{m=1}^{|T|} \sum_{x_i \in R_m} (y_i - \hat y_{R_m})^2 + \alpha |T|$$

where $|T|$ is the number of terminal nodes (leaves). Note: $|T|$ in the slide is written with a norm-style notation; the prof flags this is not absolute value but **cardinality** ("the zeroth norm"). The penalty $\alpha |T|$ is exactly analogous to the lasso / ridge complexity penalty from Module 6.

### The pruning algorithm

Solving the penalised problem exhaustively over all possible subtrees is again ugly. CART instead does **weakest-link pruning**: walk back from the maximal tree, at each step collapse the internal node whose removal **increases the training RSS the least**. This generates a **nested sequence of subtrees**:

> "Let's say when you first did the exhaustive one you ended up with 100 regions. As you go backwards you have one that has 99, then 98, then 97 … all the way to one. And then what that gives you is 100 different models."

These subtrees correspond to the optimal solutions of the penalised problem at increasing values of $\alpha$: small $\alpha$ → big tree; big $\alpha$ → small tree.

### Picking $\alpha$ via CV

Use **K-fold cross-validation**: hold out the $k$-th fold, grow + prune on the rest, evaluate the full sequence on the held-out fold, average across folds. Plot CV error against tree size. The minimum gives you the best $\alpha$ → apply it to the full-data tree.

### CV plot on the ozone data (5-fold)

Error is high at tree size 1 (just the global mean), drops, bottoms out at **tree size 5**, then rises again as overfitting kicks in. The chosen tree has 5 leaves, splits only on temperature (three times) and wind (once); radiation is dropped entirely. Matches the partition shown earlier.

> "Our best model didn't include that third variable at all, whatever that was. … In this case, at least that's what the data suggests."

## Closing

Next session (L18, Mar 16) continues trees: classification trees, then bagging / random forests and boosting later. Prof's parting recommendation:

> "Since this is kind of like a different topic, especially for people who have been mainly statisticians, I'd recommend playing with trees, maybe writing your own algorithm or something, and of course doing the exercises. But it's a different idea, right? We're doing algorithms now."
