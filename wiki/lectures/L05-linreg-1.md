---
lecture: 5
date: 2026-01-20
module: 03-linreg
title: "L05: Linear Regression 1"
slides: modules/3LinReg/3LinReg.md
topics:
  - linear-regression
  - least-squares-and-mle
  - gaussian-error-assumptions
  - residuals
  - standard-error
  - confidence-and-prediction-intervals
  - t-test-and-significance
  - r-squared
  - categorical-encoding-and-interactions
  - multivariate-normal
tags:
  - lecture
  - module/03-linreg
aliases:
  - Lecture 5
---

# L05: Linear Regression 1

The prof opens by finishing last lecture's [[multivariate-normal]] slides, then makes the conceptual switch from joint distribution of (X, Y) to the conditional model Y given X, which is what regression *is*. He motivates [[linear-regression|simple linear regression]] with a body-fat / BMI example, derives the model and its assumptions, ties [[least-squares-and-mle|least squares]] to MLE under Gaussian errors (Legendre vs. Gauss), then walks through standard errors, confidence intervals, the t-test, and finishes on R². The recurring theme he keeps returning to: **statistical vs. practical significance**, and the assumption you will break: independence.

## Key takeaways

- Regression is **Y given X**, not the joint distribution; different mindset from last lecture's multivariate normal.
- [[least-squares-and-mle|Least squares]] minimization is *equivalent* to MLE under the assumption that errors are i.i.d. N(0, σ²). Legendre got the minimization, Gauss got the distributional understanding.
- The two assumptions you will break and shouldn't: **independence of errors from other variables**, and **independence of errors from each other**. The Gaussian / zero-mean / common-variance assumptions don't screw things up nearly as badly.
- Standard error of β̂₁ = σ / √(Σ(xᵢ − x̄)²), so you reduce uncertainty by **more samples** and **wider spread of x**. This is how the equation tells you to design your experiment.
- Big n makes everything look significant: one of the reasons "significance is just sample size" is becoming a real problem.
- Residual standard error divides by **n − 2**, not n, because two degrees of freedom are eaten by β̂₀ and β̂₁.
- **Statistical significance ≠ practical significance.** Statistical = is it real; practical = is the slope big enough to matter. You want both, ideally.
- R² (coefficient of determination) = 1 − RSS/TSS: the first and crudest measure of overall model fit.

## Wrap-up of last lecture: multivariate normal

Before starting today's material the prof finished the leftover slides from last time on [[multivariate-normal]]. The covariance matrix is one of the things you can interpret from the multivariate distribution; visualize it as ellipses of different levels. Worked the matching exercise:

- Circular ellipse, no diagonal pull → correlation 0, equal variances.
- Diagonal pull going up → positive correlation.
- Diagonal pull going down → negative correlation.
- Stretched only along one axis → unequal variances, no correlation.

The point of revisiting it was to set up the contrast: **last time was the joint distribution of X and Y**; **today is Y given X**.

> [!important] The mindset shift
> "Today we will discuss Y given X. So not the joint distribution of them, but Y given X. So we're trying to essentially make a model of it… we will look at how things co-vary, but in the sense of how Y varies as a function of X."

This shift in mindset is the framing for the rest of the module. Material today is in **chapter 3 of ISL** ("It's a good book"). Slides will spill over into next week.

He skipped the embedded multivariate-normal exercise questions to avoid getting further behind, but suggested doing them alone or in the Monday exercise sessions.

## Why study a model this simple

The prof's defense of linear regression as worth the time:

- It's a [[parametric-vs-nonparametric|parametric model]]: you have parameters you fit to the data, and you can construct, modify, and reason about them.
- It still captures phenomena that more complicated models exhibit. His example: the second descent in the [[bias-variance-tradeoff]] when you scale up parameters (credited for much of deep learning's success) *can be seen and understood from the simple regression lens*. Complicated deep models are unreachable theoretically; regression is.
- It's interpretable. The slope has meaning. You don't get lost.
- Methodological minimalism: "I always aim to minimize the length of my method sections." If you add complexity you don't know whether the complexity gave you the answer or whether it's true.

> "Is linear regression too simple? Yeah, for some things… but for other things it can be quite useful. And the nice thing is that, like I said, it's one you can understand."

He pointed out next-word language modeling is structurally not so different (Y is the next token, X is the prior text), just much more complicated. Same kind of categorical encoding (tokenization) as below.

### Quantitative vs qualitative variables

Y is quantitative; X can be either. Quantitative = measurable like height/weight. Qualitative = categorical like red/green/blue, encoded numerically.

For a binary category (black/white) → 0/1. For three categories (black/white/blue) → use **two** dummy variables (e.g. 00, 01, 10), **never** 0/1/2 because that imposes an ordering between the categories. See [[categorical-encoding-and-interactions]]. He'll come back to this in later examples.

## Motivating example: body fat ~ BMI

Body fat is a useful indicator of health (fat in places you can't see is the dangerous part) but **hard to measure**: "you have to like stick people in water." So people use [[bmi]], which is just height + waist + "some other random shit," explicitly engineered to correlate with body fat.

Looking at scatter plots of body fat vs. (BMI, age, neck width, hip width), you can already eyeball that BMI has the **smallest width around the line**: best indicator. Multilinear regression coming later (think of a third axis sticking out of the page that would tighten the spread further).

### Interesting questions you can ask

- How good is BMI as a predictor of body fat?
- How strong is the relationship, and "strength" is **ambiguous**: it could mean how well the line fits the data, OR how steep the slope is. Two different things. The first is statistical reliability; the second is practical effect size. He flags this as a setup for [[t-test-and-significance|statistical vs practical significance]].
- Is the relationship linear? You can also still do "linear" regression with X² as your covariate, still linear in parameters.
- Are other variables associated with body fat? (Multivariate.)
- Can we predict an *individual's* body fat?

### Interesting questions you can't answer

> "Is this relationship causal, or better explained by something else?"

He hammers this: regression results are "fancy correlations." Example: intelligence vs. wealth would show a relationship, but giving someone money doesn't make them smarter. Don't read causation into the slope. It might be the wrong form, deceiving, or driven by a lurking variable.

## The model

Notation: response Y, covariate X.

$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$

i indexes the n samples (the prof switches between little and big n freely). β₀ is the **bias / intercept** (he prefers "bias"), β₁ is the **slope**. Y is the dependent variable / outcome / response; X is the independent variable / explanatory variable / regressor; he'll use all three names interchangeably and never remember which.

### Fitting it: least squares, and a little history

Many ways to fit a line. The classical one is [[least-squares-and-mle|least squares]]: minimize

$$\sum_i (y_i - \beta_0 - \beta_1 x_i)^2.$$

Origin story: both **Legendre and Gauss** independently derived this trying to fit telescope data of celestial orbits. Legendre published first (and was younger, 19). Gauss complained he'd thought of it first; "typically we give Gauss credit. I think it also has something to do with Germans." Compromise: credit Legendre for the **minimization**, Gauss for the connection to the **Gaussian distribution** and the deeper theoretical understanding.

You could just as well minimize the absolute value (least absolute deviations), also gives a line, also converges, harder to fit, and corresponds to assuming a different error distribution. Or you could minimize the fourth power: "would really, really penalize anything far away", but no one recommends it.

### Least squares ⇔ Gaussian MLE

Implicitly, least squares assumes εᵢ ~ N(0, σ²) i.i.d. Minimizing the sum of squared residuals is **equivalent** to maximizing the likelihood under that Gaussian assumption; that's the link Gauss is credited for. See [[least-squares-and-mle|maximum likelihood]].

> "If you minimize this least squares error, it's equivalent to minimizing this likelihood function… So this is why - Legendre, he figured that out, and this one was Gauss, and that was convenient because then he could say that I'm assuming that my epsilon are normally distributed with a zero mean and a fixed variance."

If instead you minimized |yᵢ − β₀ − β₁xᵢ|, you'd be implicitly assuming a **Laplace** distribution for ε: symmetric, peakier at zero, fatter tails than the Gaussian. Practical consequence: a Laplace fit is **more robust to outliers** because the cost grows linearly in the residual, not quadratically.

> "If we had our data was like this and then there was a point here, that point would have a stronger effect when fitting the model with a least squares fit, whereas a Laplace fit it wouldn't be pulling it as strongly."

(He'd forgotten the name "Laplace" mid-lecture and remembered it after the break.)

### Variability of the fit

If you got a fresh sample of data, you'd get a slightly different line. That variability is exactly what the **variance** in the [[bias-variance-tradeoff]] is referring to:

> "The variance refers to: depending on which sample you get, you're going to get different values, and how much they vary is the variance."

## Assumptions

For each (X, Y) pair, the error εᵢ is:

1. **Normally distributed**.
2. **Mean zero**. (Implied by 1 and the centering choice.)
3. **Common variance σ²**: homoscedastic, not depending on x.
4. **Independent of any other variable** ("independent of other shit").
5. **Independent of each other** εⱼ.

He flags **(4) and (5) as the dangerous ones**:

> "These two are the two ones that are so easily violated that they're independent of other things and they're independent of each other… violating these is super common and ruins everything."

Concrete failure mode: temperature samples over time → neighboring time bins are correlated, so sampling more finely makes the relationship *look* stronger because you're inflating effective sample size you don't actually have. He confessed his own third paper had a figure with wrong numbers because he assumed independence when he shouldn't have.

The other three (Gaussian / zero mean / common variance) "I don't think those are so bad. They're so common… I don't think that screws up so much." See [[gaussian-error-assumptions|independence assumption]].

## Geometric / graphical view

Two ways to picture least squares:

- **Vertical distances**: εᵢ is the vertical distance from point to line. Minimize the sum of squared distances.
- **Squared rectangles**: each εᵢ becomes a square of side εᵢ. The fit minimizes the **total area** of all those squares. This makes the outlier sensitivity geometric: a big residual contributes its area, which scales quadratically. With absolute deviation you'd be minimizing total *line lengths*, scaling linearly, with much less pull from a single far-away point.

Generative picture: imagine you draw a line then sample points around it with N(0, σ²) noise. Most points fall near the line; the histogram of vertical deviations approximates the Gaussian. You "never" get points far out, only with probability set by the area under the curve. When you violate this (an outlier appears), the model contorts itself trying to keep the outlier inside its assumed distribution, and the whole fit goes bad.

## Estimators and their distribution

True parameters β₀, β₁ are unknown; estimates from data get hats: β̂₀, β̂₁. Predictions ŷᵢ = β̂₀ + β̂₁xᵢ. The residuals are eᵢ = yᵢ − ŷᵢ ("residuals, other people would just write R, I don't know, because it sounds like epsilon or looks like epsilon").

> "The error terms are random variables and cannot be estimated. They can be predicted."

(He noted this is a real distinction even if it sounds pedantic, the residuals are predictions of the underlying error terms, not estimates.)

Equations for β̂₀, β̂₁ come from the least-squares solution; he'll derive the multivariate version next time and get this as a special case. For now: data in, β̂₁ first, then plug into β̂₀'s equation. "All of these data, data, data, data, data, data goes there. Data goes here."

### What R gives you

The `lm()` output gives, for each coefficient: estimate, **standard error**, t-value, and p-value. Same in Python. The standard error is the **standard deviation of the sampling distribution of the estimator**, analogous to the standard error of a sample mean from the first stats class. It tells you how much β̂ would jiggle across resampled datasets.

For the body-fat / BMI example, the slope estimate was about 1.8, i.e., a one-unit move in BMI moves the predicted body fat by ~1.8. That's the **practical** interpretation of the slope.

### Sampling distribution of the estimators

Under the assumptions, β̂₀ and β̂₁ are **normally distributed**, centered on the true β values, with variance equal to SE². The prof showed this with a simulation: generate data 1000× from a known model, fit each time, plot the histogram of β̂₀ and β̂₁; both look Gaussian, both centered at the true values.

This is what makes regression so theoretically tractable compared to large modern models:

> "If you have a billion parameters, what's the uncertainty of them, and they're all working against each other? It becomes very confusing. But in this case, you can do it very well."

## Standard error and experiment design

The slope's standard error has the form

$$\mathrm{SE}(\hat\beta_1)^2 = \frac{\sigma^2}{\sum_i (x_i - \bar x)^2}.$$

We can't change σ² (it's a property of the underlying noise), but we can change the denominator. Two practical levers:

- **Bigger n**: more samples.
- **Wider spread of x**: sample further apart in x.

This is the part where staring at the equation tells you how to design the experiment. "It is kind of weird to think that you can look at these equations and then from that gain an intuition of how you can do your experiment better. But you do."

> "If n is infinity… your standard [error] is going to be small as shit, which means it's going to look significant even if it isn't."

So in the limit of large n, *anything* becomes statistically significant, one reason modern big-data results often look more impressive than they should. "Significance is just a notion of how many samples you have."

## Residual standard error

[[least-squares-and-mle|Residual sum of squares]] RSS = Σeᵢ². The unbiased estimator of σ is

$$\hat\sigma = \sqrt{\frac{\mathrm{RSS}}{n - 2}}.$$

The **n − 2** divisor: two degrees of freedom are consumed by estimating β̂₀ and β̂₁. He says he'll "make a big deal out of it" but in practice if n = 300 it barely matters, and after taking a square root, even less. Still, "that's a 2, not an average."

## Confidence intervals and tests

With β̂ ~ N(β, SE²) and σ estimated, build CIs and test hypotheses with the **t-distribution** on n − 2 degrees of freedom.

CI: β̂₁ ± t · SE(β̂₁). For 95%, t ≈ 2 (close enough to the normal because we're past the small-sample regime usually).

Default null hypothesis test:

$$H_0: \beta_1 = 0, \quad t = \frac{\hat\beta_1 - 0}{\mathrm{SE}(\hat\beta_1)}, \quad \mathrm{df} = n - 2.$$

This is what generates the p-value in `lm()` output.

> "We assume the most boring thing and then we try to reject it being boring - you know, just like we do with people in jail. We assume they're innocent, which is boring, and then we try to argue that they're guilty, which is more interesting because everyone's innocent."

### Statistical vs practical significance

A recurring theme he kept circling back to. Worth the verbatim:

> "You could have a statistically very reliable, very confident, very, very confident that there is a relationship there, and it might not matter at all. The trend is like basically zero."

Slope size is **practical significance**; CI width and p-value tell you about **statistical significance**. You can have one without the other, and only "ideally both" is what you want.

He contrasts disciplines:

- **Engineering**: nobody cares about p-values. Effect size is everything. "If you need statistics to show a relationship is meaningful, you don't study it, you just ignore it."
- **Biology / softer sciences**: effects are squishy and small; you have to lean on statistics. So you care a lot about p-values, and the danger is conflating significance with meaning.

This is also where the assumption-violations bite: if you violated independence, your reported significance is "horseshit": your effective sample size is smaller than you think. The course will return to methods in **module 6** that don't make as strong an independence assumption (he'd rather use a less powerful test that doesn't assume things he can't verify).

> "We don't want to report shit, right? That would be embarrassing."

### Drawing the line by eye: and a trap

Quick aside while looking at the body-fat / BMI fit: if you tried to draw the best line by eye, you might draw something steeper, more like the major axis of the point cloud. That instinct corresponds to the **multivariate / correlation** view, *not* the conditional Y-given-X view. Regression's blue line is shallower because it minimizes vertical residuals, not the perpendicular distance.

> "The red thing is maybe what you would think you would draw, but you're implicitly assuming that you're looking at the joint distribution, but we're not. We're trying to model y with x."

A small but real trap that connects today's mindset shift back to the visual.

## R²: coefficient of determination

The first measure of *overall* model fit (not just one slope's reliability). Define:

- TSS = Σ(yᵢ − ȳ)²: total variability in Y.
- RSS = Σ(yᵢ − ŷᵢ)²: leftover after fit.

Then

$$R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} = \frac{\mathrm{TSS} - \mathrm{RSS}}{\mathrm{TSS}}.$$

Interpretation: fraction of variance in Y explained by the model.

> "How much does your shit vary in general versus how much can you actually explain of that."

He flags this as just the first of many model-accuracy measures, "kind of just the first one that they came up with. A lot of people don't like it. There's an adjusted version. There's other versions. I would always use the test error" - foreshadowing later modules. There's also the danger of confusing **training fit** with **test performance**, which links straight back to the [[bias-variance-tradeoff]] from L04.

## Closing

Out of time mid-slide-deck. Continues on Monday ([[L06-linreg-2]]) on the same module 3 material, picking up from R² and moving toward multivariate regression (where the matrix derivation lives).
