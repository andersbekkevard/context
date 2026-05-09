---
lecture: 6
date: 2026-01-26
module: 03-linreg
title: Linear Regression 2
slides: modules/3LinReg/3LinReg.md
topics:
  - r-squared
  - linear-regression
  - design-matrix-and-hat-matrix
  - least-squares-and-mle
  - sampling-distribution-of-beta
  - collinearity
  - f-test
  - subset-selection
  - confidence-and-prediction-intervals
  - categorical-encoding-and-interactions
  - polynomial-regression
  - residual-diagnostics
tags:
  - lecture
  - module/03-linreg
aliases:
  - Lecture 6
---

# L06 — Linear Regression 2

The prof picks up R² from L05, then commits the rest of the lecture to **multiple linear regression** in matrix form: derives the [[design-matrix-and-hat-matrix|normal equations]] and the [[design-matrix-and-hat-matrix|hat matrix]], derives the sampling distribution of β̂, and previews how the X'X factor blows up under [[collinearity]]. He surveys the four canonical "important questions" in multiple regression (using the [[f-test]] for the first, deferring [[subset-selection|model selection]] to module 6), then extends the linear model to **categorical predictors**, **interactions**, and **non-linear transformations** (still linear in the parameters). Closes with a fast tour of [[residual-diagnostics]] and the [[residual-diagnostics|QQ plot]]. Ends mid-list of broken assumptions; will resume next week (a guest covers classification tomorrow).

## Key takeaways

- Multiple regression = simple regression in matrix form. The closed form β̂ = (XᵀX)⁻¹Xᵀy is the **maximum likelihood estimator** under the same Gaussian-error assumption — and it's special: most ML problems don't have a one-shot closed form.
- The [[design-matrix-and-hat-matrix|hat matrix]] H = X(XᵀX)⁻¹Xᵀ exists because ŷ = Hy "puts the hats on" — the prof's pun, but the matrix recurs later in the course.
- β̂ is multivariate normal, centered on β, with covariance σ²(XᵀX)⁻¹. **The X'X factor is what explodes under [[collinearity]]** — variances trade off between near-duplicate predictors.
- Four "important questions" in multiple regression. Q1 (any predictor useful?) → [[f-test]] on H₀: all β = 0 (won't be on the exam to compute, but he might ask **why** you'd use it).
- "Only checking individual p-values is dangerous" — variables can be correlated so individual tests look insignificant while the joint F-test is highly significant. **Ask sequentially**: F first, then individual.
- For K-level categorical predictors: use **K − 1 dummies + a reference category**. If you include all K, the model is unidentifiable.
- **Interactions**: if you include the product term β·X·Z, you must also include the **main effects** X and Z. Non-negotiable.
- "Linear regression" is linear in the **parameters**, not in X. You can include x², log x, sin x, √x — all still linear regression.
- [[residual-diagnostics|QQ plot]] is the diagnostic worth learning: visual check that residuals are Gaussian. The prof said this is the kind of thing he might put on a test.

## Recap: R² and goodness of fit

Picks up where L05 ended. R² (coefficient of determination, "explained variance") is the classical goodness-of-fit measure for linear regression. Equation:

$$R^2 = 1 - \frac{\sum(y_i - \hat y_i)^2}{\sum(y_i - \bar y)^2}.$$

Numerator: residuals from the model. Denominator: deviations from the mean. "How much of the variance do you get from the model versus what is just in the data."

> "It seems easy — you're just like, 'just tell me the error.' Yes, but… should you look at the data that you fit the model on, should you look at held-out data, should you penalize it in some way? These are all questions that are interesting to ask."

Caveat that lands later in the lecture: R² **never gets worse when you add a parameter** (on training data), so it's a biased measure of model improvement. This sets up [[r-squared|adjusted R²]] later.

## Multiple linear regression in matrix form

The natural extension when you have more than one X. Body-fat example: age, weight, height, BMI, abdomen circumference, hip — many candidate predictors for one response (e.g. life expectancy, 100m time, ski performance, BMI itself). Rather than picking one, use them all.

### The model

Same model as before, more β's, more X's:

$$y_i = \beta_0 + \beta_1 x_{i,1} + \dots + \beta_p x_{i,p} + \varepsilon_i.$$

β₀ is still the **bias term**; the other β's are slopes, one per direction in covariate space.

In matrix form: y = Xβ + ε. The bias term **disappears from the equation** because it's hidden in the first column of X — that's why X has a column of ones.

> "Behind this beta is actually an X. It's just all the values of X are one. So you don't need to write it."

Dimensions: y is n×1, X is n×(p+1) (the +1 is the bias column), β is (p+1)×1, ε is n×1. The prof notes a notation gotcha:

> "You can define p as including or not the intercept or bias term. This is just a note for those who are taking both classes that the notation is different in the books."

### The classical assumptions (matrix form)

Same as L05, restated for the multivariate case:

- E[ε] = 0 (centered).
- Cov(ε) = σ² I — identity scaled. The off-diagonals are zero (independence) and the diagonals are all σ² (homoscedasticity). The prof draws the matrix mentally: σ² σ² σ² … on the diagonal, zeros everywhere else.
- "We always have this independence assumption — which is what we break all the time and don't understand."
- n ≫ p — more data points than parameters, classical setting. He acknowledges modern ML breaks this, and gene/protein matrices break it badly: "those matrices are very weird… they have a ton of genes and just a few samples, and they're trying to find something in there. Good luck." But for this part of the course, n ≫ p.

The error vector then has an n-dimensional [[multivariate-normal]] distribution — "no matter which direction you look in, has the same variance, and it's just kind of a big n-dimensional bell."

### "Design matrix" naming

X is conventionally called the **design matrix**. The prof complains about it:

> "It's often called the design matrix. The data. Never understood why. It's not really a design of any kind. But it's what people call it."

### Distribution of Y given X

Mirror of the univariate case. E[Y|X], Cov(Y|X), and the conditional distribution Y|X is again Gaussian. Not obvious that the **least squares fit equals MLE under a Gaussian likelihood**, but it's true.

Sketch of why (he'll re-derive properly in the GLM chapter): write the Gaussian likelihood, take log, the exponent collapses to (x − μ)² / σ² plus stuff. The product over samples becomes a sum, and "it becomes very obvious that maximizing the log likelihood is equivalent to minimizing the least squares problem."

> "For those who have taken GLMs or who understand what likelihood is — these are the same thing."

## Deriving β̂: the normal equations and the hat matrix

The estimator. Take the **residual sum of squares** in matrix form, differentiate w.r.t. β, set to zero, solve. The prof walks through the algebra at the board.

Start with

$$\mathrm{RSS}(\hat\beta) = (y - X\hat\beta)^\top (y - X\hat\beta).$$

Expand:

$$= y^\top y - 2\hat\beta^\top X^\top y + \hat\beta^\top X^\top X \hat\beta.$$

(The two cross terms combine because each is a scalar, and a scalar equals its transpose.)

Differentiate with respect to β̂:

$$\frac{\partial \mathrm{RSS}}{\partial \hat\beta} = -2 X^\top y + 2 X^\top X \hat\beta = 0.$$

The 2's cancel:

$$X^\top X \hat\beta = X^\top y \quad \text{(normal equations)}.$$

If XᵀX is **full rank** (invertible — which requires n > p plus no other pathologies), invert it:

$$\boxed{\hat\beta = (X^\top X)^{-1} X^\top y.}$$

**Uniqueness.** "I'm not showing, but you can prove if you want to take more derivatives, that this problem has a unique solution. There's only one solution." Conditional on n ≥ p and no degeneracies.

**Why this is special.** Most ML problems don't have a closed-form maximum. You typically iterate to climb the likelihood surface.

> "Most of the time, you actually have to go iteratively… most of the time when we're trying to find this peak, we have to like climb up and go there. In fact, I would argue most of machine learning is finding good tricks to get to that peak… But in the case of linear regression with full-rank X — just right to the top. Very convenient. Which is also why we're studying regression even though we want to study something more complicated. This one has all these nice properties where you can really get at things quickly."

### The hat matrix

Define

$$H = X(X^\top X)^{-1} X^\top.$$

Then **ŷ = Hy**. Why "hat"? Because H is the matrix that puts the hat on Y.

> "In math we call it a hat. It's a pointy hat. But it's a hat. And so this matrix H has all the shit you need to get your hats for your parameters. So it's called the hat matrix."

The [[design-matrix-and-hat-matrix|hat matrix]] recurs later in the course (leverage, diagnostics).

### LM in R, and reproducing the answer manually

R: `lm(...)`. Python: `statsmodels`. He notes that you can reproduce the same β̂ values by hand by computing H · y directly — ugly notation in R ("lots of dollar signs and percent signs are stupid… this notation sucks, but whatever, I hope you like it"), but the same numbers come out.

## Sampling distribution of β̂

Same logic as L05: get the distribution of the estimator so you can build CIs and tests.

> "A lot of the reasons we ask those questions is so we can make tests on them."

Result (proving this is **problem 2 of the recommended exercises**): β̂ has a (p+1)-dimensional multivariate normal distribution,

$$\hat\beta \sim N_{p+1}\bigl(\beta,\; \sigma^2 (X^\top X)^{-1}\bigr).$$

- Centered at the true β — i.e. β̂ is unbiased. "That's what we want. If it was biased then we'd be upset because then our model is not going to give us the right shit."
- Covariance has the constant-σ² assumption baked in, plus the data-dependent factor (XᵀX)⁻¹.

### Why X'X matters: collinearity

The (XᵀX)⁻¹ factor controls how wide each β̂'s distribution is — i.e. how confident you can be. It depends on **how related the X variables are to each other**.

> "This factor X transpose X comes into play in particular when two variables are basically the same — because then they can trade off each other and then this variance explodes. That's a thing we discuss at the very end of today. It's called collinearity."

> "I think this is really why statisticians love these distributions, because you can read out what's going to happen when you look at them. You can be like — ah, that X transpose X is going to screw us later."

He computes the covariance matrix on the worked example data and shows it has small values. Doesn't dwell on it; the [[collinearity]] discussion is teed up for later (didn't reach it today).

### Reduces to univariate

Recommended exercise: show that the matrix formula reduces to the simple-linear-regression formula for β̂₁ when p = 1. The setup is identical — same RSS, same derivative-and-solve — so the answer must agree.

## The four "important questions" in multiple regression

Lifted from ISL ch. 3 (he calls out that this is a slide structure he might exam on). The four:

1. **Is at least one of the predictors useful?**
2. **Are all of them, or only a subset?**
3. **How well does the model fit?**
4. **How accurate is a prediction?**

> "I might say in the multivariate linear regression case, how would I test if at least one of the predictors is useful in predicting the response, or I might ask: why would I want to know this, what's the point? I like to ask kind of these questions… where you have to just sort of reason through why you're doing something. I don't know if that's been in previous exams, but it's something I would ask."

So: knowing **why** each test exists matters more than the mechanics.

### Q1: F-test for "at least one predictor useful"

State a null hypothesis — the boring one — and try to reject it.

$$H_0: \beta_1 = \beta_2 = \dots = \beta_p = 0 \quad \text{vs.} \quad H_1: \text{at least one } \beta_j \neq 0.$$

> "It's like we're assuming innocence so that we can prove guilty… We don't prove anything in statistics… at least I've never proved anything with data."

The test statistic is the **F-statistic**. In the simple-regression case (p = 1), the equivalent test was the t-test; the F-statistic generalizes that and **degrades back to the t-statistic when p = 1**, so it's consistent with what was learned in the previous course.

> [!important] F-test scope flag — verbatim
> "I'm going to say right now I probably won't ask any questions about an F-test, which is what I'm going to show you in a minute. I think it's too boring for this class, to be honest… I don't think I'm going to say anything about an F-test. I'm more interested in things that can generalize to other distributions. This thing's rather specific to the case of linear regression."

He shows the F-statistic anyway — degrees of freedom, table look-up "if you're in 1985" or computer otherwise — but reiterates: "I don't really care that much about it." The F-test material is "not really in our book" — better reference is **Walpole** ("pretty good for classic statistics"). It matters for **ANOVA** (analysis of variance), heavily used in design of experiments / engineering, but that's a different course.

> "If I was going to ask you something about this, it'd be more about the null hypothesis. And then you could say, 'I would use an F-test.' But I'm not going to make you compute it because I honestly don't care."

### Subset variant

Same F-statistic can test a **group** of predictors instead of all of them. E.g. "is the joint contribution of weight and height non-zero?" Use R's `anova()` to do this. ANOVA = **An**alysis **o**f **Va**riance, again more of a different-course topic but the machinery is the same.

### Why not just look at individual p-values?

Slide flag: **"only checking individual p-values is dangerous."** Why?

> "The variables can actually be correlated, and then none of them actually look significant, but overall the test is very significant."

So you ask **sequentially**: F-test first ("is anything good at all?"), then drill into individual p-values. Don't skip Q1.

## Confidence intervals on β̂

Same logic as univariate L05. From the sampling distribution above and the standard error read off (XᵀX)⁻¹·σ², build a CI using the **Student-t distribution** (heavy-tailed Gaussian; "looks like a Gaussian that someone lifted up a bit"). For 95%: two-sided, so 97.5 percentile each tail.

In a typical case with reasonable n the t looks essentially Gaussian. You get something like `0.078 < age coefficient < 0.186` — assuming the model is right ("big assumption — probably wrong, but still useful").

## Model selection and adjusted R²

Anders had asked about model choice. Pre-req for [[subset-selection|model selection]] (covered in module 6) is having a **good objective criterion** for "how good is this model." Candidates listed:

- AIC, BIC, Mallow's Cₚ — flagged but not covered today.
- [[r-squared|Adjusted R²]].
- R² (he'd add to the list, but with the caveat about it never decreasing).

### Why R² alone fails

Adding a parameter to the model (on training data) **cannot make the fit worse**. The new model still contains the old one as a special case (set the new β to zero). So R² monotonically improves with parameters, regardless of whether they actually help. Worked example: BMI alone gives R² ≈ 0.50, adding more pushes it to 0.54, 0.58, 0.72 — but is that improvement meaningful?

> "In my field, that's like unheard of. That's fine. It's also one of the annoying things about R² — like what is good is very different depending on the context of the data."

### Adjusted R²

Penalize for parameter count:

$$R^2_{\mathrm{adj}} = 1 - \frac{\mathrm{RSS}/(n - p - 1)}{\mathrm{TSS}/(n - 1)}.$$

The (n−1)/(n−p−1) factor penalizes adding parameters that don't pull their weight. With this, "adding more parameters can actually give you a smaller R squared if they don't do anything."

> "I have reported this in articles, but I would never really, if it was up to me, we wouldn't include it. Especially in the analysis I do, typically there's so much stuff that we've left out that this number doesn't mean anything. But it's still interesting."

The bigger point: any model-comparison metric must account for the parameter-count temptation.

## Confidence interval vs. prediction interval

Two different intervals you can place around ŷ.

- [[confidence-and-prediction-intervals|Confidence interval]]: where the **true mean** Y(x) lies, given the model. Narrow.
- [[confidence-and-prediction-intervals|Prediction interval]]: where a **future observation** y_new would land at this x. Wider, because it includes the irreducible noise σ².

Both come from the same machinery (problem 2 of the recommended exercises again — derive these). Plot interpretation: both intervals are narrowest where data is dense, fanning out at the extremes; the prediction interval is always wider than the confidence interval because it eats the noise too.

R can plot both directly; Python has the same with simple equations.

## "All models are wrong, some are useful"

Box, 1979 (the prof attributes via David Hand). Quoted because it organizes the rest:

> "When we're building a statistical model we must not forget that the aim is to understand something — not everything, just something. And it's about the real world, despite the fact that it's wrong."

Two purposes:
- **Understand relationships** — interpretive use.
- **Predict** — decisional use. "If you predict, it's often because you need to choose something."

> "In this course, I think the focus is more about prediction."

## Extending the linear model — the menu

Up to this point: continuous covariates with simple slopes. The model class is much broader. The menu the prof works through next:

- **Binary / categorical predictors.**
- **Interaction terms.**
- **Non-linear transformations** (polynomials, logs, etc.) — still "linear regression."

> "Right now we're going to talk about how can we extend linear regression to include a lot of other stuff."

### Binary predictors

Encode 0/1. Two equivalent ways to write it:

- Conditional form: y = β₀ if X=0, y = β₀ + β₁ if X=1.
- Indicator form: y = β₀ + β₁·X with X ∈ {0,1}. Same thing.

This is a **two-level** factor.

### K-level categorical: dummy encoding + reference category

For K > 2 levels, **K − 1 dummy variables**, plus one **reference category** baked into the intercept. Three colors {red, green, blue} → two dummies (e.g. is_red, is_blue). If both are 0, the level is **green** (the reference).

Why not K dummies? **Identifiability**:

> "If they're all no, you know — that doesn't happen, right? Then the world ends. The reality is if you have all of those things, then you have actually too many parameters that become too interrelated, and then actually the model is not identifiable. The reason being that you can determine the value of one of them from the other two."

Mathematical statement: the columns become linearly dependent, XᵀX is singular, no inverse. See [[categorical-encoding-and-interactions|categorical encoding]], [[categorical-encoding-and-interactions|reference category]].

### Worked example: credit-card data (with a digression)

ISL `Credit` dataset. Predict balance from ethnicity (Caucasian / Asian / African American). Worth quoting his discomfort:

> "I really hate this kind of data… they're specifically pointing out ethnicity, like to see if ethnicity leads to more debt and credit cards. I hate credit cards… Anyways, sorry, but they're looking at predicting the balance."

R picks the **alphabetically first** category as the reference — here, African American. Output then gives:

- Intercept (= prediction for African American) ≈ 531.
- Coefficient for Asian dummy ≈ −18.7.
- Coefficient for Caucasian dummy ≈ −12.5.

Predictions:
- African American: 531.
- Asian: 531 − 18.7 ≈ 512.
- Caucasian: 531 − 12.5 ≈ 518.

Interpretation of standard errors: the intercept is significant, but the dummy coefficients have weak p-values → **ethnicity is not informative for balance** in this fit. Disclaimer: the more relevant predictors in the actual model were things like income.

> "We don't see a result for African American here, and that's because that was our reference category."

You can still use the F-test (or its grouped variant) on the joint significance of all the ethnicity dummies — the standard hypothesis-test machinery carries over. (And again: F-test won't be on the exam.)

## Interactions: beyond additivity

You might want predictions to depend on **combinations** of variables — the income effect for students vs. non-students, etc.

Pure indicator addition (just add a "student" dummy) gives a **shifted intercept** for students with the same slope. Plot it: parallel lines.

> "Income is like a variable that changes in time. So you want to separate what's specific to…"

To allow students and non-students to have **different slopes**, add an interaction term — the product of income and the student dummy. Now students get both a different bias *and* a different income slope. Fit: in the worked example, the income×student interaction term came out **statistically weak** (the slope difference was suggestive but not significant); the bias jump from "student" alone *was* significant. Students owe more.

> "It makes sense — they owe more. Even if they make money, you're still going to owe more, so haha. Yeah, credit card debt's really bad, by the way."

(Personal aside: he had credit card debt once when his son was born, an unaffordable car — "the worst.")

### The main-effects rule

> [!important] Main-effects rule — verbatim
> "If you do include interactions… whenever you include an interaction, you want to include what is referred to as the main effects. So if you want to look at A times B, then you also want to include A and B."

Generalizes: this works for two continuous variables (product), one continuous × one categorical, more than two levels, etc. "Creativity is up to you. You can go crazy with this. It's good and bad — you have so many ways you can change this model that at the end you want to be careful what you're doing."

## Non-linear transformations of X — still "linear regression"

You're not restricted to using X raw. Polynomials, logs, square roots, sin/cos — all fair game.

Why "log of X" is so common: many real-world Y's depend on log(X) "just because of the nature of the thing — it depends on how this thing changes, and that changes exponentially."

The key conceptual point — and the one he wants you to internalize:

> [!important] Linear in parameters, not in X — verbatim
> "It's still called linear regression even though you're fitting y = β₀ + β₁ x + β₂ x². It's a quadratic function but it's still a linear model. And it's linear because it's linear in the coefficients. Linear in the parameters — the things that we actually find out, the things that we're fitting. Those all look like slope terms. So it's weird to think of it that way because the x is changing quadratically. But you have to remember it's still a line. It's just now it's a line in terms of x squared. Even though you might think about it in terms of x, in which case it's curvy — but you're doing everything as linear regression."

Connects back to the [[bias-variance-tradeoff]] discussion from L04 (the polynomial-fits demo with degrees 1, 5, 10, 25 — those were all linear-regression fits in the parameters).

> "That's why it's so powerful — you can expand it to all these different kinds of variables. Very convenient."

## Challenges: when the assumptions break

The list of common assumption violations (he doesn't get through all of them — picks up next week):

- Non-linearity in Y (not just in X — that one we already handled).
- **Correlation in error terms** — "a pain in the ass." Will revisit in another chapter.
- Non-constant variance (heteroscedasticity).
- Non-normal errors.
- Outliers.
- High-leverage points.
- [[collinearity]] — flagged earlier via X'X explosion, but didn't reach the formal treatment today.

> "These are all assumptions, or I guess we could call it broken assumptions, because we talked about ahead of time what we're assuming in our model… and now we can see that these are the common ways that we can break assumptions."

These are also called **diagnostics** ("another word for this would be a diagnostic"). They matter even if your downstream model isn't classical regression: "these issues will permeate any model you want to do."

## Residual diagnostics

### Residuals vs. fitted

Plot the residuals against the fitted ŷ. Tukey & Cook plot. Want **no structure**.

> "You want to see that there's no structure in it. If there is structure in this, then there's structure in your residuals, and that's bad. Because that's related to — yeah, correlation of errors. It's definitely breaking this independently distributed errors."

If you see a pattern (curvature, fanning, clusters), the model is missing something — likely a non-linearity in Y, or a violated independence assumption.

### QQ plot

The diagnostic the prof flags hardest. Plot **theoretical quantiles** of the assumed distribution (Gaussian by default) against the **standardized residuals**. If the residuals really are Gaussian, the points fall on a straight line.

How it works conceptually: each residual gets a quantile rank (smallest = lowest quantile, largest = highest, median = middle). For each rank, look up where a Gaussian would put that quantile (e.g. the most-extreme one ≈ ±3σ, far in the tail). Plot empirical against theoretical. Straight line ⇒ assumptions OK. Wonky line ⇒ residuals deviate from normality.

> "If everything works out great, then these QQ plots should look like a straight line. And if they're not — like if it looks kind of like this — then you're like, okay, shit's broken."

Useful because it's **visual** — outliers stick out, you can tell whether the tails or the bulk are misbehaving. And it's not Gaussian-specific: works for Poisson or any other assumed distribution.

> [!important] Exam flag — verbatim
> "This one I do recommend learning. This is the kind of thing I would put on a test. This one [residuals-vs-fitted] probably not — I don't want to say no, but I don't know. This [QQ plot] is super useful."

QQ stands for **quantile-quantile**. If you don't know what a quantile is: "Google it. It's very useful."

## Closing

Out of time before finishing the assumption-violation list. Tomorrow: a guest lecturer on classification and logistic regression (the prof is in Oslo). He'll resume the rest of module 3 (collinearity, the remaining diagnostics) next week.
