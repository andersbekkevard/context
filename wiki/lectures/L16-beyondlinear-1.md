---
lecture: 16
date: 2026-03-09
module: 07-beyondlinear
title: Moving Beyond Linearity 1
slides: modules/7BeyondLinear/7BeyondLinear.md
topics:
  - basis-functions
  - polynomial-regression
  - step-functions
  - regression-splines
  - smoothing-splines
  - local-regression
  - generalized-additive-models
  - effective-degrees-of-freedom
  - cross-validation
tags:
  - lecture
  - module/07-beyondlinear
aliases:
  - Lecture 16
---

# L16: Moving Beyond Linearity 1

The prof opens module 7 and tries to "speed up a little bit" after admitting the course is "a little behind." He walks through the unifying idea of [[basis-functions]] (replace $X$ with $b_j(X)$ and you're still doing linear regression) and applies it to [[polynomial-regression]], [[step-functions]], [[regression-splines]] (cubic and natural), then drops the basis-function frame for [[smoothing-splines]] and [[local-regression]] (a second objective on the second derivative; locally weighted Gaussian fits), and closes with [[generalized-additive-models]] as the way to combine all of the above across multiple predictors.

## Key takeaways

- **The unifying trick of module 7**: replace $X$ with **basis functions** $b_j(X)$. Polynomials, indicators of intervals, and spline pieces are all just different choices of $b_j$. Fitting is still ordinary linear regression — "linear in the parameters $\beta$, but nonlinear in what you get."
- **Splines come from shipbuilding** — knots are the pegs, the wood is the polynomial pieces. Statisticians (incl. Hastie) picked them up after Renault engineers in the 50s/60s used them for computer-aided design (Bézier).
- **Cubic spline** = piecewise cubic with continuous 0th, 1st, 2nd derivatives at the knots. The book's basis: $X, X^2, X^3$, plus one truncated cubic $(X - c_j)^3_+$ per knot. **Natural spline** adds boundary knots that force the function to go linear at the ends, killing the wild tail behaviour of plain cubic splines.
- **Smoothing splines** drop the knot-and-basis story entirely. Add a second objective $\lambda \int g''(x)^2\,dx$ to the RSS. $\lambda = 0$ → arbitrarily wiggly fit; $\lambda \to \infty$ → straight line (second derivative forced to zero everywhere). Tune $\lambda$ by **leave-one-out CV**.
- **Effective degrees of freedom** = trace of the smoother matrix $S$ (where $\hat y = Sy$). Lets you specify smoothness as a non-integer "df" instead of an opaque $\lambda$.
- **Local regression (LOESS)** = "smooth k-nearest-neighbours". At every $x$, fit a line weighted by a Gaussian kernel around $x$. The Gaussian width plays the same role as $\lambda$.
- **GAMs** = additive combination $f_1(X_1) + f_2(X_2) + \ldots$, where each $f_j$ can be **any** of the above (polynomial, spline, natural spline, smoothing spline, LOESS, indicator). Assumes no interactions but lets each predictor have its own nonlinear shape.

## Where we are and where we're going

Module 7 is "relatively short." The prof opens by acknowledging the schedule slip:

> "I'm a little behind in the material. I'm going to try to not get further behind, maybe even speed up a little bit."

The plan: finish module 7 today/tomorrow and start module 8.

The motivation for the whole module: linear regression — really a Gaussian linear model — only does so much. We want to move beyond it. The route taken here is **not** "throw away linear regression"; it's "keep linear regression, but feed it transformed inputs."

## Basis functions: the unifying frame

The standard linear model
$$y = \beta_0 + \beta_1 x_1 + \ldots + \beta_k x_k + \varepsilon$$
becomes
$$y = \beta_0 + \beta_1 b_1(x) + \beta_2 b_2(x) + \ldots + \beta_k b_k(x) + \varepsilon.$$

> "Instead of looking at $x$ directly we're going through basis functions, and that's a very general term, a very powerful term that has many many versions."

The fit is still least squares. The design matrix is built from $b_j(x_i)$ values. Everything we know about OLS — the closed form, the variance, the inference — carries over because the model is **linear in $\beta$**. This is the through-line for the entire module.

## Polynomial regression

Simplest choice: $b_j(x) = x^j$. So
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \ldots + \beta_d x^d + \varepsilon.$$

Worked on the **wage-vs-age** data — "how much money people make as a function of age." It peaks around mid-career and drops off after 80 (in this dataset, "people over 80 don't typically make very much because... maybe they weren't so alive"). The prof also flagged the income-inequality outliers visible in the scatter ("we're looking at the wrong question") but shrugged it off and moved on.

A degree-4 polynomial fit looks reasonable. But push $d$ too high and you get the same wiggly-overfit story from day one of the course:

> "If you make $d$ very big, which intuitively gives you a very flexible model... if you make $d$ too big" you get the wiggly behavior. With degree 12 "it probably would have gotten a little bit wiggly, a little bit weird. There's enough data here that would probably keep it relatively contained."

This is the same flexibility/overfit warning from L02-L04, applied to a richer hypothesis class. Polynomial regression "looks kind of okay but we can do better, or we can do something more interesting which is particularly interesting for other kinds of data."

## Step functions

Now $b_j(x) = \mathbb{1}(c_{j-1} \leq x < c_j)$ — an indicator that $x$ falls in some interval. The cut points $c_j$ are picked manually (often equally spaced) but you can also move them around or pick them by hand. The fit is a piecewise-constant function:

> "I mean, this one is very stupid, but it's actually quite common, simply because you don't need that much information... so you don't have too many constraints that push things around. Even the step functions are actually pretty nice, even if they are a bit stupid looking."

The downside, mathematically: "you don't have derivatives here. They're not even... it's piecewise constant, but it's not connected — it can jump."

Used on the **wage-vs-education** data, broken up by years of schooling (high school, bachelor's, master's, "PhD something excessive"). Within each interval you fit a constant. Confidence intervals come for free "same as you would get the confidence intervals for the prediction in the linear model — because it's just a linear model only now on these basis functions, which in that case were just fixed intervals."

The connecting moral so far:

> "There's a commonality, right? We're talking about basis functions, but all you have to do is fit it with regression."

## Regression splines

The bridge from "fixed intervals" to "smooth pieces of polynomial joined together."

### Origin story

> "This idea of splines... comes from the idea of in shipbuilding where you have... pegs and then you take a piece of wood and... bend it down over these pegs."

The wood naturally maintains derivative continuity over the pegs. Splines really took off in the 50s–60s at Renault (the prof name-checks Bézier) for computer-aided design — before splines you got triangular airplanes because flat surfaces were all you could model. Statisticians (incl. Hastie) picked them up in the 70s.

The cut points are now called **knots**. The basic idea: piecewise polynomials joined at the knots, with derivative continuity enforced.

### Continuity hierarchy

The prof sketched the spectrum on the slide:

- **Piecewise polynomial** — pieces don't even meet at the knot.
- **Continuous piecewise polynomial** — pieces meet, but a kink at the knot ("kind of looks like a butt crack").
- **Cubic spline** — pieces are cubic, **second derivative is continuous**, third derivative not enforced. Smooth to the eye.
- **Linear spline** — same idea, but with linear pieces (and you can't enforce continuous first derivative without collapsing it to a single straight line).

> "You can define a spline with degree 10 or something. It wouldn't look very good, but you can do it."

### The cubic spline basis

For $K$ knots $c_1, \ldots, c_K$, the book's basis for a cubic spline is
$$x, \quad x^2, \quad x^3, \quad (x - c_j)^3_+ \text{ for each } j = 1, \ldots, K.$$

The **plus** notation means
$$(x - c_j)^3_+ = \begin{cases} (x-c_j)^3 & \text{if } x > c_j \\ 0 & \text{otherwise.} \end{cases}$$

So each truncated-cubic basis function "doesn't come in until you're beyond where the knot starts." This automatically gives continuity of value, first derivative, and second derivative at each knot. The prof:

> "I'm not gonna go through the math of this, they don't do it in the book either, so I'm not really justifying it but this gives you nice smooth functions."

Total parameter count: $K + 3$ basis terms, plus an intercept = $K + 4$. For $K = 3$ knots, "you have seven parameters that you're going to find."

### Why splines beat high-degree polynomials

> "If I was going to fit this with like a degree five polynomial it's going to get funky... but with a spline I can always just make a smooth function through it and it becomes very natural. Actually worked out. It's surprising."

### Natural splines

Plain cubic splines get wild near the boundaries (the polynomial pieces in the outermost regions can swing). **Natural splines** add **boundary knots** that force the function to **degenerate from cubic to linear** beyond the boundary. The result: linear extrapolation at both ends, smooth cubic in the middle.

> "Where the basis functions meet, and certainly at the boundaries, then... the spline degenerates from a cubic to just a linear function, and they do that by adding these boundary knots."

The math isn't in the book or slides, so the prof skips it: "in other courses they go through the math of what these natural splines are. The book doesn't, so I won't either."

### Recommendation

> "I recommend going into, if you've never played with splines before, find some visualization tool where they have knots with splines to play with and just start moving the points around."

You'll see exactly when and why they get wiggly — same intuition as bending wood weirdly over the pegs. ("If you put two points like this and then you pull this one down here and you pull that one like here, it just goes wiggly, wiggly, wiggly.")

### Note on naming

The R demo's spline output is labeled `bs` (B-spline). The prof: "I don't know why they call it BS. It's funny." Cosmetic; it's still a cubic spline fit by ordinary regression on the truncated-cubic basis.

### Restating the through-line before moving on

> "It's nonlinear, but linear. It's linear in the parameters $\beta$, but it's nonlinear in what you get."

This is the slogan for everything basis-function based in this module.

## Smoothing splines: a different objective

We've been adding flexibility by inflating the basis. Now switch frame: **add a regularizer** to the loss, just like in [[ridge-regression]]. The objective becomes
$$\sum_{i=1}^n (y_i - g(x_i))^2 + \lambda \int g''(t)^2 \, dt.$$

> "Now we're going to do something more than just finding betas."

The first term is fit; the second penalizes curvature (integrated squared second derivative). The optimization is over **functions** $g$, not over a finite parameter vector. The prof draws the analogy explicitly to ridge:

> "We've looked at situations where we have more than one objective before — we had regularizers... $Y - \beta X$ squared plus sum of $\beta$ squared, that was our ridge regression... a.k.a. $L_2$ norm or regularizer. Really what you're doing is you're adding another objective to your optimization."

Same "two objectives" structure. The first fits the data, the second imposes a property — here, smoothness via the curvature penalty.

### The two extremes

- $\lambda = 0$: regularizer vanishes; $g$ can be arbitrarily wiggly to fit the data. "If you zero out an objective then that thing doesn't do anything."
- $\lambda \to \infty$: smoothness term dominates; the solution forces the second derivative to be zero everywhere → **straight line**.

So $\lambda$ slides between "very flexible" and "completely straight."

### Effective degrees of freedom

R parameterizes smoothness by **effective df** instead of $\lambda$ — and lets you input either one. Construction:

1. Let $\hat y$ be the vector of fitted values from the smoothing spline.
2. Because the optimization is quadratic in $g$, $\hat y$ is linear in $y$: $\hat y = S y$ for some $n \times n$ smoother matrix $S$.
3. **Effective df** = $\text{tr}(S)$ — sum of the diagonal entries.

> "It's not obvious, but that's how they define it."

You can specify a non-integer df (e.g. 6.8) and the package back-solves $\lambda$ — "my guess is the way to make this work is that they would try a different value of $\lambda$ until you get the degree of freedom you want." High df → small $\lambda$ → wiggly. Low df → large $\lambda$ → close to straight.

This is also why "you can get non-integer values of degrees of freedom, which... typically we think of degrees of freedom as being integer values. But here it's an effective degree of freedom."

### Choosing $\lambda$ via leave-one-out CV

> "How do I choose $\lambda$? One could be just a decision you make. Or [[cross-validation]], which is a thing I think is particularly useful, because you're still letting the data tell you or give you indications as to what to do."

The book suggests **leave-one-out CV**: fit smoothing spline on $n-1$ points, predict the held-out one, sum squared errors over all $n$ choices of held-out point, minimize over $\lambda$. ("It's a computer, so it's fine.") This is how the prof speculates the demo's df ≈ 6.8 was chosen — it gives "arguably better" fit than the arbitrary high-df one.

## Local regression

Yet another approach. No basis functions, no global function. At every query point $x_0$, fit a **local linear regression** using only nearby points, weighted by a Gaussian centered at $x_0$.

> "It's basically the same idea as the nearest neighbor algorithm, only now we're going to have a smoothing version of it."

Equivalently: it's a "smooth k-nearest-neighbours," where the Gaussian width plays the role of $k$.

The fit is defined by
$$\hat\beta(x_0) = \arg\min_\beta \sum_{i=1}^n K_\lambda(x_0, x_i) (y_i - \beta_0 - \beta_1 x_i)^2$$
and the prediction at $x_0$ is the local linear value. Move $x_0$ along the axis and you get a smooth curve.

### Behaviour vs Gaussian width

- **Wide Gaussian** → all points contribute equally to every local fit → degenerates to a single straight line.
- **Narrow Gaussian** → only a couple of points contribute → choppy. The prof notes the wage data has integer ages (no months), so very narrow Gaussians give "choppy derivatives because it won't smoothly... see the points that are another fixed year away." You see the data sampling structure leak into the fit.

> "Only now you can see it gets wiggly in a rather different way... it can actually get kind of choppy because it's only looking at the first derivative and the Gaussian can get really tiny."

Same df-as-smoothness-knob idea applies: "you have the same notion with the local regression that now the Gaussian is related to the degrees of freedom."

### Where it sits

The prof groups smoothing splines and local regression as the two "fit-a-function-directly" methods, in contrast to the basis-function methods (polynomial, step, regression splines):

> "The local regression and the smoothing spline are both kind of functions that you have to fit, where it's not these knots and stuff."

## Generalized additive models (GAMs)

So far everything's been one $X$. With multiple predictors, we want each one to be modeled flexibly, but combined in a manageable way.

**Additive model**:
$$y = \beta_0 + f_1(x_1) + f_2(x_2) + \ldots + f_p(x_p) + \varepsilon.$$

Each $f_j$ can be any of the things from this lecture: polynomial, indicator (step), cubic spline, natural spline, smoothing spline, local regression. They're combined **additively** — no interactions across predictors.

> "You're assuming that they don't interact. Like it's not like you have to be educated and old, but rather the component that has to do with how old you are can be considered separate from education, right? They combine in how they predict, right? But they combine additively."

The assumption is "often not a terrible one, either because the assumption is maybe a good one or because adding more terms could be bad — they could get unwieldy."

### Worked example

Wage as function of:
- $f_1(\text{age})$ — cubic spline with knots at e.g. 40, 60.
- $f_2(\text{year})$ — natural spline with its own knots.
- $f_3(\text{education})$ — indicators (because education is discrete: high school, some college, college grad, advanced degree).

Different predictors can use different basis types, different numbers and locations of knots, different domains.

### Fitting and naming

When all the $f_j$ are basis-function expansions, you stack the basis matrices into one big $X$ and fit by OLS as usual. They are called **generalized additive models** — the prof:

> "Sounds better like GAMs, right? Because it sounds like you're, I don't know, like jelly or something."

You get the usual outputs: coefficient estimates, predictions, confidence intervals — and because the components are additive, you can plot each $f_j$ individually as a function of its predictor (the prof points to a partial-dependence-style plot showing each $f_j$ separately).

## Closing

> "That's as far as I wanted to go today. We're almost done with this section. Made a little bit faster progress than normal, so that's good. Tomorrow we're going to wrap up this module seven, and we're going to start the next module eight."
