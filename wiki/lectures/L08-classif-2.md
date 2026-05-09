---
lecture: 8
date: 2026-02-02
module: 04-classif
title: Classification 2 (LinReg wrap-up + LDA)
slides: modules/4Classif/4Classif.md
topics:
  - residual-diagnostics
  - design-matrix-and-hat-matrix
  - collinearity
  - principal-component-regression
  - logistic-regression
  - classification-setup
  - knn-classification
  - curse-of-dimensionality
  - linear-discriminant-analysis
tags:
  - lecture
  - module/04-classif
aliases:
  - Lecture 8
---

# L08 — Classification 2 (LinReg wrap-up + LDA)

A two-module session: the prof first wraps up the leftover Module 3 diagnostics (residual plots, QQ plot, leverage / hat matrix, studentized residuals, collinearity), then gives a fast recap of what Valdemar covered last week in Module 4 (logistic regression, Bayes classifier, KNN) and finally introduces [[linear-discriminant-analysis]] via Bayes' rule. He flags the assumptions you actually break (independence, collinearity), the "fat-kid seesaw" intuition for leverage, and the curse of dimensionality as the death of KNN.

## Key takeaways

- **Residual plots, QQ plots, leverage and studentized residuals are diagnostics, not new fits** — they're how you check the linear-regression assumptions hold.
- **Leverage = the diagonal of the hat matrix H = X(XᵀX)⁻¹Xᵀ.** A point with high leverage *and* a large residual is the dangerous combination: "fat kid at the end of the seesaw." The exercise class verifies the formula.
- **Collinearity** lets the βs trade off against each other → unstable fits, blown-up p-values. Detect with [[collinearity|variance inflation factor]] (self-study). Fix with PCA / [[principal-component-regression]] or by dropping a variable; LDA also reduces dimension.
- **Logistic regression** is a Bernoulli GLM: linear η = β₀ + βᵀx fed through a logistic link to give P(Y=1|x), fit by maximum likelihood. Same assumptions as linear regression — collinearity wrecks it the same way.
- **Bayes classifier** is theoretically optimal (smallest possible test error, irreducible noise sets the floor) but assumes you know the true posterior, which you never do. Estimates are "probably bullshit" — still a useful benchmark.
- **KNN**: non-parametric majority vote of the K nearest Euclidean neighbours; can produce wildly complicated decision boundaries with islands. Choose K by test error (bias-variance again). Killed in high dimensions by the **curse of dimensionality** — distances become uniform, no neighbour is meaningfully "closest."
- **LDA flips the modelling**: instead of modelling P(Y|X) directly, model P(X|Y) (Gaussian in each class) and the priors P(Y), then combine via Bayes' rule. Yields the [[discriminant-score-and-decision-boundary|discriminant score]] you maximise over k. Decision boundary moves with the prior.

## Module 3 wrap-up: residual diagnostics

The prof opens by saying he wants to finish Module 3 before starting Module 4. The Monday before, Valdemar covered the start of Module 4; today the prof closes the loop on regression diagnostics, then re-recaps Valdemar's content "hopefully much shorter than him."

### The setup we're checking

Same model as last lecture: y = β₀ + β₁x₁ + … + ε, with εᵢ ~ N(0, σ²) i.i.d. We fit by minimising Σ(yᵢ − ŷᵢ)² and obtain residuals eᵢ = yᵢ − ŷᵢ.

> "We'd expect the residual to be centered at zero, kind of mean a zero and some nice distribution. If they don't, then we're like, uh-oh, shit broken."

So the diagnostics in this section all answer one question: **do the residuals look the way the assumptions require?**

### Residuals vs fitted values

Plot eᵢ either as a histogram or — more usefully — against ŷᵢ. You're looking for a centered cloud at zero with constant spread. If errors get big where fitted values are big, the constant-variance assumption is broken.

### QQ plot

Plot theoretical quantiles of N(0, σ²) against the empirical quantiles of the residuals. A straight line means your residuals match the assumed distribution.

> "A QQ plot is a visual illustration of how close your histogram is to your distribution… if the theoretical quantiles match those quantiles from the histogram then it's like, bink, and then you're very happy because your assumptions are good."

Deviations almost always show up in the **tails**. The classic "S-shape" (undercutting then overcutting) is the other characteristic failure pattern. Tests exist (Shapiro–Wilk etc.) — "we're not going to talk about it, but, you know, like people love to do tests." See [[residual-diagnostics|QQ plot]].

### Recap of the four assumptions

The diagnostics map to these:

1. E[εᵢ] = 0
2. Constant variance (homoscedasticity)
3. Normally distributed
4. Independent of each other

> "I think this is the assumption that more often we accidentally screw up and it screws up a lot of shit and people are like 'that's fine.' It's not."

(He's pointing at independence — same flag as L05.)

The residuals-vs-fitted plot checks (1), (2), and gives a hint about (4). The QQ plot checks (3).

## Leverage and the hat matrix

A point has high leverage when its **x-value** sits far from the bulk — note this is an x-only quantity, not involving y. Formally, for [[linear-regression|simple linear regression]],

$$h_{ii} = \frac{1}{n} + \frac{(x_i - \bar x)^2}{\sum_j (x_j - \bar x)^2}.$$

### The fat-kid-on-a-seesaw intuition

> "The further away the fat kid was from the center the more leverage he had to shoot the other kid off the seesaw and hurt them right and then you know they jump out and and then everyone cried."

> "They should have called this the fat kid seesaw effect. That would have been way better. … Yeah, that won't be on the test."

A single x-outlier far from the centre of mass pulls the regression line by a lot. The same y-outlier near the middle of x is much less harmful — there's no lever arm.

### The hat matrix

The same H that gave us leverage is the **hat matrix** from [[linear-regression|OLS]]:

$$\hat{\boldsymbol\beta} = (X^\top X)^{-1} X^\top y, \quad \hat{y} = H y, \quad H = X(X^\top X)^{-1}X^\top.$$

> "This matrix H is also known as the hat matrix. The reason we call it a hat matrix is that it's where all the hats come from."

Leverage hᵢᵢ is then literally the **diagonal of H**. He briefly mentions the Moore-Penrose pseudoinverse as the natural object when X isn't square — flagged for context, not for the exam.

> "One exercise you can do for your exercise class is figuring out / verify this formula comes from linear regression."

### Leverage-vs-residual plot

Plot leverage on one axis, standardised residual on the other. The dangerous corners are **top-right** and **bottom-right**: high leverage *and* large residual. A high-leverage point that lies on the line is fine ("it actually lines up in the same direction"); the problematic case is a high-leverage point the model can't fit despite being pulled toward it.

> "Whenever you see an outlier just figure out like did you screw something up. Why is it there? Don't just throw it away. Use it as a way to understand your data better."

## Studentized / standardized residuals

The raw residuals don't actually have variance σ². Their joint distribution is

$$\boldsymbol{e} \sim N\!\left(0,\; \sigma^2 (I - H)\right),$$

so they can be correlated and have unequal variances — annoying for diagnostics. To get something that behaves more like the εᵢ we *assumed*, **standardise**:

$$r_i = \frac{e_i}{\hat\sigma \sqrt{1 - h_{ii}}}.$$

**Studentized** residuals additionally drop point i when estimating σ̂² for that point — removes the circularity of using yᵢ both to fit and to evaluate.

> "My guess is, in practice, that these two numbers are going to look exactly the same if you have more than like 50 points or something. But still, it's still nice to be exact or correct."

Wherever you used raw residuals before, you can swap in studentized ones for cleaner-looking diagnostic plots. They are diagnostics, not new estimators — the βs are unchanged.

## Collinearity

A topic the prof flags as "really interesting" but covers in one slide. [[collinearity]] = some of the predictors x₁, x₂, … are themselves correlated.

### Why it breaks the fit

If x₁ and x₂ are strongly correlated, the least-squares objective doesn't change much when you make β₁ bigger and β₂ smaller in compensation:

> "We could trade between x1 and x2 — e.g. make β₁ bigger and β₂ smaller, while fit is similar."

Perfect collinearity → infinitely many solutions. Even mild collinearity → highly sensitive solution that swings around with tiny data perturbations: "going to go wah, wah." Predictions blow up out of sample because you end up with "a million minus a million."

### Diagnose and fix

- **Detect**: [[collinearity|variance inflation factor]] (VIF). The book covers it, prof says **read it as self-study**.
- **Fix**: drop the offending variable, or combine the collinear ones. Combining is most cleanly done with [[principal-component-analysis|PCA]] / [[principal-component-regression]]:

> "PCA is a way of compressing your variables into fewer variables with some loss. … in this case X1 and X2 would turn into… one would be this trend, basically, and then the other one would be the one that's moving around — the shit around it. … because then they all become orthogonal."

He also previews [[linear-discriminant-analysis]] as another dimensionality-reduction route — "hopefully we'll get to it today." See [[principal-component-regression]] and [[curse-of-dimensionality]] (foreshadowing).

### Aside: studentized residuals are diagnostics, not estimators

A student asks whether studentized residuals can be used for inference.

> "The idea is to make them look more like the true distribution of the underlying epsilons … but you wouldn't typically use them for inference. It's like a diagnostic."

> "Your betas stay the same. It's just a way to say, is my model any good?"

### Aside: history-of-stats

The prof riffs that statistics looks the way it does because computation was hard:

> "I always wonder how different statistics would be taught if we would have had computers before the theory. … We figured out a lot of stuff about distributions and stuff way before we could ever do anything on a computer."

A pre-flag for cross-validation (Module 5), which "wasn't taught historically because it was hard to think about and they didn't have computers."

## Module 4 begins: classification

Switch into Chapter 4 of ISL. Goal: predict a categorical Y (spam/ham, eye colour, disease vs. not). The output is a class assignment, plus ideally a probability/uncertainty.

> "It ends up being like a true/false kind of setting and you want to make a decision. Yes, no. And the question is, how do you draw this line? Should it be a line? Should it be a curve? … Should it be something very squiggly?"

Three methods to be covered today / tomorrow:

- [[logistic-regression]]
- [[knn-classification|KNN classifier]]
- [[linear-discriminant-analysis]] / [[quadratic-discriminant-analysis]]

He says he'll recap Valdemar's logistic regression material first, then move to KNN and LDA.

## Logistic regression (recap)

### Why not just linear regression?

Y is binary (0/1). Fitting it with an OLS line gives a "terrible fit": predictions go above 1 and below 0, no probabilistic meaning. We want a function bounded in [0, 1], smooth (so we can take derivatives and optimise), and parameterised so we can fit it. The **logistic / sigmoid curve** is exactly that.

### Bernoulli GLM

[[logistic-regression]] is the **Bernoulli GLM**. Setup:

- Response yᵢ ∈ {0, 1}.
- Linear predictor ηᵢ = β₀ + β₁xᵢ₁ + … + βₚxᵢₚ. (The prof writes ηᵢ — "I typically use the letter H because I can remember what that thing's called.")
- Link: pᵢ = 1 / (1 + exp(−ηᵢ)). η very negative → 0; η very positive → 1.
- Inverse / log-odds form: log(pᵢ / (1 − pᵢ)) = ηᵢ.

### Fitting by maximum likelihood

The likelihood factorises over data points (independence assumption again):

$$L(\boldsymbol\beta) = \prod_{i=1}^n p_i^{y_i}(1 - p_i)^{1 - y_i}.$$

> "The probability that my mother gives me an ice cream is that she's happy *and* we have ice cream. … This *and* condition, you end up multiplying those probabilities together. You have the same thing with the likelihoods."

Take log → sum → derivatives → unique maximum (when assumptions hold). Solve with Newton–Raphson or your favourite optimiser.

### Inference and the GLM table

You get the same R-style summary table as in linear regression: estimate, standard error, z-value, p-value. Same class of model (GLM), same machinery. He warns about absurdly small reported p-values:

> "These numbers are of course ridiculously small. Never write that in an article, people will laugh at you, because a probability of negative 200 is, you know, more likely we don't exist. … Some assumption is wrong regardless of what it is."

### Same assumptions, same failure modes

Logistic regression inherits linear regression's assumptions on the linear predictor, including independence. Collinearity here breaks the unique-maximum property the same way it breaks OLS:

> "The collinearity problem we talked about a minute ago, that can happen here, and then this thing's no longer having a single maximum and it gets weird."

Briefly mentioned: multi-class logistic regression (one-vs-rest binary trick or a different distribution); odds and log-odds as another lens on the same fit; you can predict on new x and read off a class probability — handy, but "if you're a doctor, don't really trust this."

## Bayes classifier

Restate logistic-regression's outputs as posterior probabilities P(Y | X). The [[classification-setup|Bayes classifier]] assigns the class with the highest posterior — for two classes, whichever side of 0.5.

### Why it's both beautiful and annoying

It is **provably optimal**: smallest possible test error, optimal decision boundary. But:

> "The annoying thing is, like, you're making more assumptions. They have all these prior distributions and those are always wrong. … It's optimal if you're right, but you're probably wrong."

The unavoidable error floor is the [[classification-setup|Bayes error rate]] — the analogue of irreducible error from regression. Same vocabulary you've seen since Module 2: training error, test error, loss function. Training error here is the **misclassification rate** — a 0/1 indicator of "did the prediction match the label," not a continuous residual.

> "Here this error rate, we don't use the value, the continuous value … we actually binarized the orange one to be whatever the Bayes classifier suggests, and then just look at this indicator: did it guess right or wrong? Whereas the likelihood would be the one that accounts for the continuous variable."

Test error is the one we'll really care about — to be picked up in Module 5.

## K-nearest neighbours

Totally different idea: no parametric model. To classify a new point, look at its **K nearest** training points (Euclidean distance) and take a **majority vote**. The class probability is just the fraction of the K neighbours of each class.

> "If of 10 neighbours, 9 of them are blue, then you can assume that it's a probability of 0.9 that you're blue too."

(Cute analogy: look around the classroom and your closest neighbour tells you your sex; bigger circle → more reliable estimate.)

### Decision boundary shape

KNN can produce extremely complicated decision regions — including disconnected "islands" — even though the model itself is trivial.

> "Complicated shape. Easy model. … You can have islands in the middle of nowhere."

### Choosing K → bias-variance

Small K (e.g. K=1) → super wiggly, overfits each training point. Large K → too smooth, eventually predicts the majority class everywhere. Pick K by minimising **test error**: same bias-variance argument used for choosing model complexity in regression.

### The curse of dimensionality

The biggest problem with [[knn-classification|KNN classifier]]:

> "If there's a number of dimensions and you look at the average distance between points, of course as you add more dimensions that average distance is going to be bigger. But the weird thing is that the variance — like how much those distances vary just by chance — gets smaller and smaller. … So that in this million-dimensional space, the points that are close together have almost the same distance value as points that are far away."

So "nearest" stops being meaningful in high dimensions. He recommends **simulating it yourself**: sample uniform points in d dimensions, plot the distribution of pairwise distances as d grows. The fix is dimensionality reduction (PCA again), but it's "a bit hacky still."

> "The dimensionality is a curse, and in this case, it's a curse for K-nearest neighbours. … makes it useless."

### Statisticians don't love KNN

Hard to analyse — no clean asymptotic distribution of anything, no parametric story. "Not super appealing from a statistical perspective."

## Linear discriminant analysis

Conceptual pivot. Logistic regression and KNN both model **P(Y | X)** — the posterior — directly. [[linear-discriminant-analysis]] models things the **other way around** and uses Bayes' theorem to flip:

$$P(Y = k \mid X = x) = \frac{f_k(x)\, \pi_k}{\sum_l f_l(x)\, \pi_l}$$

where f_k(x) = P(X | Y = k) is the **class-conditional density** and π_k = P(Y = k) is the **prior** for class k.

> "We're not trying to model P(Y | X) directly. We're trying to get the other distributions to then use Bayes to get the classification."

### Two-class Gaussian example

Two classes (orange, green), each with prior 0.5, each with a Gaussian density on x. The product π_k f_k(x) is the curve you visualise; the **decision boundary is where the two product-curves intersect** — i.e. where the posteriors are equal.

If you shift the priors to (0.3, 0.7) — orange more common — the orange product-curve grows, the green one shrinks, the intersection point **slides toward green**. So the prior literally moves the boundary.

> "More things are going to be labeled as orange because you have more mass over here, and that makes sense because the prior says the orange one has a higher prior probability."

### LDA assumption: Gaussian class densities

Assume f_k is Gaussian with mean μ_k and a **shared** covariance Σ. Estimate μ_k, Σ, and π_k from data. Plug into Bayes; take logs; throw away terms that don't depend on k (including the denominator, which is the same across classes).

> "Regardless of what the k is, this denominator is going to look the same. … you can compute it once and use it, or just skip it and say, well, they all have the same denominator."

What survives is the **discriminant score** δ_k(x). Classify x to whichever k maximises δ_k(x). For two classes you only need one δ; for K classes you need K of them.

### LDA as dimensionality reduction

> "You can also think of this as a way of reducing the dimensionalities of your data, where now instead of the axes being the X's, now it's actually the categories. … you can also use it as a way to transform your data into one that has different dimensions, which are these discriminant score things."

This is the second answer to the collinearity question from earlier ([[principal-component-analysis|PCA]] was the first).

## Closing

Out of time at the end of the LDA setup. Continues tomorrow (L09 — [[L09-classif-3]]) finishing LDA (and on into QDA / Naive Bayes / ROC). Reminder to attend the lab session — TA "is great."
