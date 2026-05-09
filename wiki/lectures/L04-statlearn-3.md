---
lecture: 4
date: 2026-01-19
module: 02-statlearn
title: Statistical Learning 3
slides: modules/2StatLearn/2StatLearn.2.md
topics:
  - bias-variance-tradeoff
  - benign-overfitting
  - double-descent
  - ridge-regression
  - random-vector-and-covariance
  - multivariate-normal
  - marginal-distribution
  - contrasts
  - pseudo-inverse
tags:
  - lecture
  - module/02-statlearn
aliases:
  - Lecture 4
---

# L04: Statistical Learning 3

The prof opens with a short recap of the [[bias-variance-tradeoff]] from last lecture, then spends about a third of the class on a **digression into the over-parameterized regime** (fitting a step function with a 100,000-degree polynomial via the [[pseudo-inverse]] to motivate **benign overfitting** / [[double-descent]] and to foreshadow [[ridge-regression]] and model averaging). He then transitions into Module 2 part 2 proper: [[random-vector-and-covariance|random vector]]s, expectation/[[random-vector-and-covariance|covariance matrix]] rules, the [[random-vector-and-covariance|correlation matrix]], [[contrasts]], and the [[multivariate-normal]] distribution. He runs out of time and cuts off in the middle of the multivariate-normal contour exercise; that's tomorrow.

## Key takeaways

- Reframing of "bias-variance tradeoff": the prof dislikes the word "tradeoff": "*it doesn't always have to be a tradeoff. In fact, you can reduce both.*" Pick a degree where both still drop.
- **Benign overfitting / double descent.** When p ≫ n and you fit with the [[pseudo-inverse]], the test MSE explodes near p = n but then *decreases again* and can beat the classical sweet spot. "It's overfitting, but it's okay." Doesn't break the math; explained by implicit [[ridge-regression]] and model averaging, both treated later in the course.
- The over-parameterized win **only happens when the true model is not in the assumed function class**. If the true f is a polynomial, fitting with a polynomial gives no benefit from going huge.
- Random vectors and matrices follow the same expectation rules as the univariate case: E(X+Y)=E(X)+E(Y), E(AXB)=A·E(X)·B. Proof of the second was done element-wise on the board.
- [[random-vector-and-covariance|Covariance matrix]] Σ has variances on the diagonal and covariances off; [[random-vector-and-covariance|correlation matrix]] is Σ rescaled by the standard-deviation diagonal V so the diagonal is 1. Cov measures linear co-variation: "*we're sort of assuming a linear line*."
- [[multivariate-normal]] generalizes the bell curve: x − μ becomes a vector, σ² becomes Σ, and σ² in the exponent becomes Σ⁻¹. det(Σ) = 0 is bad (singular).
- The connection that's the whole point: minimizing the negative log-likelihood of a normal model with mean parameterized by some f(x) **is** linear regression. The multivariate normal is the route into multiple regression.

## Recap of the bias-variance decomposition

Fast review of last lecture's polynomial-fit example: poly-1 is a line, poly-10 is wiggly, "*as you get more and more parameters the model essentially starts overfitting.*" MSE on **held-out test data** is U-shaped; that minimum is "the model that you'd pick." Decomposes into two pieces:

- **Irreducible error**: the "hopeless part," due to noise. "*You can improve on that if you have more data, or if you can get better samples without noise.*"
- **Reducible error** = bias² + variance.

He re-stated the bias term: it's the difference between the **true underlying function** (no noise) and the expectation over fits to many sampled datasets. So bias absorbs both "errors due to your sample" and "errors due to the model class you assume." Variance is then a separate object: the expected squared deviation of *one fit on one dataset* from *the expectation over many dataset realizations*. "*This is really telling you about how your fit will vary just because of the data itself - assuming the same model every time, assuming the same number of parameters every time.*"

In the U-shape picture: as p grows, **variance grows** (you're more sensitive to the particular sample) while **bias² shrinks** (you can express more functions). Cross at the minimum.

> [!quote] On the word "tradeoff"
> "The reason I don't really like the word tradeoff is that it doesn't always have to be a tradeoff. In fact, you can reduce both. For example, here it starts with degree 2 - but if instead you start with degree 1 then you'd see that both the bias and the variance are decreasing."

The "real" tradeoff intuition is local: "*maybe I can sacrifice a little bit of bias to get more variance - to reduce the variance.*" You're choosing which side to give up *near the minimum*.

## Digression: benign overfitting and double descent

The prof spent a chunk of the previous week running his own simulations because the textbook version of the example only goes up to degree 8 or 9, and he wanted to enter "*this ridiculous region of like a degree 50,000 or 100,000.*"

The motivation is the AI/scaling discourse: "*there's all these notions that we have now about scale … there was a news article yesterday - Sam Altman said we're still in the region of infinite scale, or like we're still - we can still scale our models bigger with more data, bigger computers, bigger models, and see an improvement in our AI models. And that seems to be really counterintuitive to what I just talked about, where I said if you go too big, it's bad.*" The whole digression is the prof confronting that apparent contradiction inside the course's own framework.

### The setup

- Truth: a **step function** (deliberately a terrible fit for any polynomial: "*it's like the worst case for a polynomial*"). He picked it on purpose: "*I purposely wanted to pick a model where the - it's not a good fit at all.*"
- Sample n = 100 noisy points from it.
- He cheated slightly on the train/test split: training data lives in [−1, 0] (or similar), test extends out to [−1, 1]: "*I purposely picked points to be a little bit less … just because I wanted to make sure we had that extrapolation regime which really sucks for polynomials, which I really wanted to screw up.*"
- Fit polynomials of various degrees by **mean-squared-error minimization with the [[pseudo-inverse]]**: "*we'll get to it tomorrow.*" The pseudo-inverse choice is the reason the high-p regime works at all (see below).

### What he saw

- For small p (degrees 1–10): test MSE low, classical regime.
- Around **p = n = 100**: test MSE *explodes*. This makes sense: same number of parameters as data points, "*you can almost fit everything perfectly*," and tiny noise blows up.
- Just past p = n: training error suddenly collapses to zero and test error starts dropping again.
- For p ≫ n (e.g. degree 2636, 100 000): test MSE **drops again** and the fit is **smoother than the small-p polynomials**. The training error is exactly zero (the curves go through every data point) but the interpolation between those points is gentle. The classical low-degree minimum sat around test-MSE ≈ 5; the high-degree solutions get down to ≈ 2 or 2.8.

This is the **double-descent curve**. It directly contradicts the standard intuition: "*we always learn that if you go to higher degree polynomial and you have too many parameters that it's going to get wiggly and scary and it's going to explode and it's going to be really bad. But actually when you go really really big - ridiculously large - then it does something else, right? It starts getting these solutions that are actually smooth.*" The prof's verbatim framing for the high-p regime:

> [!quote] Benign overfitting
> "What they call this is actually benign overfitting. Benign being like, you know, it won't hurt you. So you're overfitting, but it's okay."

He measured the bias and variance terms in this regime too: bias² grows slowly while variance shrinks as p increases (visible only on log scale). The decomposition still holds: sum of bias² + variance + irreducible matches the empirical test MSE. **"It doesn't break any of the math. It doesn't break any of the statistics."**

### Why the high-p smoothness happens

The prof flagged two mechanisms, both deferred to later in the course:

- **Implicit [[ridge-regression]]**: "*the parameters are being controlled in some way so that they don't get out of hand. … Ridge regression is what's going on here even though it's not actually put in there explicitly.*" The pseudo-inverse picks the minimum-norm solution among the infinite zero-training-error solutions.
- **Averaging across models**: also deferred.

### When does it actually help?

The prof switched the truth from a step function to **f(x) = x²** (something polynomials *can* fit) and re-ran:

- Low-p polynomial recovers the true model and extrapolates well.
- High-p regime still exists (train error → 0, then a second minimum) but **does not beat the right low-p model**.

> [!quote] When over-parameterization wins
> "If the underlying model, the f of x, exists as part of the functions that you're assuming in your model, then even if you increasingly add more and more degrees of flexibility, you're not going to improve over what you can get with a few parameters. … But in the real world, I don't know how often you really can assume that you have the right model."

His takeaway: most statisticians dismiss the high-p regime as "just overfitting"; he's not so sure. "*It's interesting, right? It's philosophically interesting, especially because all the things - this doesn't break any of the math. … It's just giving you a solution that we normally don't think about. We don't think about the solution where you can perfectly interpolate through all the points and the error - your fit error - is zero.*" Will return to it more than once in the course.

> [!note] Why this matters for what's coming
> Two ideas are pre-loaded for later modules: (1) [[ridge-regression]] and other regularizers as the explicit version of what the pseudo-inverse is doing implicitly; (2) the bias-variance decomposition as a tool that *keeps working* in regimes far past the classical sweet spot. Anders should expect both to be cited back to this lecture.

## Module 2 part 2: random vectors

> "Today we're going to talk about random vectors. We're going to talk about the covariance matrix, correlation matrix, what those are, and also the normal distribution - in particular the multivariate case. And this is setting us up to talk about regression."

The reason regression is the next thing: "*it's like the simplest data model we have, and yet, even though it's the simplest one, it already shows these kind of complex phenomenon.*"

### Definition and setup

A [[random-vector-and-covariance|random vector]] **X** is a p-dimensional vector of random variables. Examples:

- Cork-deposit weights in 4 directions (N, E, S, W): the running dataset for the lecture.
- Body-fat predictors: BMI, age, weight, hip circumference.

You stack n samples to get an n × p data matrix. There's a **joint distribution** f(x) over the whole vector, and you can **marginalize** by integrating over all but one coordinate to get f₁(x₁). "*The idea of integrating over the rest of the things and getting the marginal — it's very much calculus, simple calculus.*"

He's not going into much depth here, assumes you've seen this before in introductory stats.

### The cork data

Rao (1948) data, n = 28 cork trees, p = 4 holes drilled (N/E/S/W), measured weight of cork sample in centigrams from each. Used because the variables are *very* correlated within a tree (sun exposure aside, dense in one direction usually means dense in the others), making it a good toy multivariate example.

The prof showed `ggpairs` output (R / GGally: "*there's no equivalent in Python that I've ever seen, but you can make your own*"):

- Diagonal: kernel-density (smoothed histogram) of each variable's marginal.
- Lower triangle: pairwise scatter plots.
- Upper triangle: Pearson correlations with significance stars: "*these stars are this ridiculous notion of significance*" (visible disdain).

He gave the standard exploratory-data-analysis advice while showing this: "*the first thing you do is just to plot it as in as many interesting and informative ways as you can. … you want to see, are there any, is there anything weird here?*" Then on outliers: "*Throwing - just deleting data - is generally a really bad idea. But if you see points that are very out of distribution, very strange, look like outliers, then you might want to go and look like, well, what might be going on here?*"

(He briefly forgot what the data actually was, took the break to google it, and came back with the bore-hole story above.)

## Expectation and rules for random matrices

Expectation of a random vector is element-wise: E(X) = (E(X₁), …, E(Xₚ))ᵀ. Same idea for matrices.

**Rule I.** E(X + Y) = E(X) + E(Y): vector/matrix addition; trivially the same as the univariate version.

**Rule II.** For constant matrices A, B and random matrix X:

$$E(AXB) = A \cdot E(X) \cdot B$$

He **proved this on the board** by element. Let e_{ij} be the (i,j) element of AXB:

$$e_{ij} = \sum_k \sum_l a_{ik} \, X_{kl} \, b_{lj}$$

Each a_{ik}, b_{lj} is constant; X_{kl} is the only random thing. Take E and pull constants out, element-wise it's the (i,j) element of A·E(X)·B. He flagged the proof as basically obvious: "*I mean, I thought it was obvious, but I don't know - it said 'proof' on the board, so I felt like I had to do it.*"

Univariate analogue (he wrote it as the contrast): E(aX + b) = aE(X) + b.

### Apologetic aside on pace

> "I don't know how in the previous years they went through this so quickly. I think maybe I just talk really slowly or I get distracted. I will find a way to get through all the same material, but currently I'm going a lot slower than this year."

Worth noting because the prof has been noticeably behind the slide deck's intended pace in subsequent lectures too; this is the first explicit acknowledgement.

## Covariance and the covariance matrix

Definition (per pair):

$$\sigma_{ij} = \mathrm{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

Reading: "*if they have a high covariance then they're varying together; if they have a negative covariance then they're varying the opposite way; and if it's just all random then the covariance is going to be small in magnitude.*"

When i = j: it's the **variance**. He called this out as a quiz-style fact ("*what would be the covariance of X_i with X_i?*").

Stack into a [[random-vector-and-covariance|covariance matrix]] Σ: variances on the diagonal, covariances off-diagonal. Convenient matrix-algebra identity:

$$\Sigma = E(XX^T) - \mu\mu^T$$

> [!important] Covariance is a *linear* notion
> "We talk about things - I'm kind of always doing this [drawing a line] - because we're sort of assuming a linear line. We're assuming some kind of linear function. This covariance is really getting at this notion of a slope. It [could] co - and yet be - and it doesn't have to be a line."

So zero covariance does not mean "independent" except under specific conditions (e.g. joint normality, flagged later). It just means "no linear co-variation."

He suggested as an exercise: simulate data with prescribed positive / zero / negative correlation just to **see** what each looks like in a scatter plot.

## Correlation matrix

Same idea as the covariance matrix, but normalized:

- Diagonal becomes all 1 (correlation of a thing with itself).
- Off-diagonal entries are in [−1, 1] (Pearson).

In matrix form, if V is the diagonal matrix of standard deviations,

$$\mathrm{Corr}(X) = V^{-1} \, \Sigma \, V^{-1}$$

He worked the cork example by hand on the slide: a covariance matrix where all variances are 2, with off-diagonals 1, 0, ½, etc.; divide everything by the appropriate √(σ_i σ_j) = 2, get the correlation matrix.

## Contrasts (linear combinations)

A **contrast** is any linear combination of the variables you find interesting: e.g. N − S, E + W, (E + W) − (N + S). Once you've defined them as new variables, you can take their expectations and covariances using the same machinery (E(CX) = C·E(X), Cov(CX) = C·Σ·Cᵀ, implicit, not derived in detail). The prof said working out the cork contrasts is "*a good exercise to do in the exercise session.*"

He skipped a couple of slides at this point to make sure he'd reach the multivariate normal, with the offhand "*all this matrix algebra stuff - it really, really important.*"

## The multivariate normal distribution

The destination of the whole part-2 setup. Univariate Gaussian, just to anchor:

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Center μ, width σ², normalizer makes the integral 1. Generalize to p dimensions:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\!\left(-\tfrac12 (\mathbf{x}-\boldsymbol\mu)^T \Sigma^{-1} (\mathbf{x}-\boldsymbol\mu)\right)$$

Mapping the pieces:

- x − μ becomes a vector.
- (x − μ)² in the exponent becomes (x − μ)ᵀ Σ⁻¹ (x − μ): Σ⁻¹ plays the role of 1/σ².
- σ in the normalizer becomes |Σ|^{1/2}.
- The 2π gets raised to p/2 for dimensional consistency.

Reduce to p = 1 and you get the univariate density back; "*you have to go really, really far out, but you can see how it reduces down.*"

> [!warning] Singular Σ
> "If the determinant is zero, then that's typically bad, because you divide by zero and yuck. And I think that's the point they're trying to make."

### The key connection to regression

> [!quote] Why we're doing this
> "Do you guys know the relationship between the normal distribution and regression? … If you minimize the normal distribution — if you assume your data is normally distributed and you have it the mean parameterized by some model — then that's equivalent to linear regression. So this multivariate case is a way of understanding how we do regression in multiple variables."

This is the bridge to module 3. Everything set up today (vectors, Σ, multivariate normal) is the language for multiple linear regression.

### Useful properties (listed, not derived)

- Contours of the density are **ellipsoids**.
- Any **linear combination** of the components is normal.
- Any **subset** of the components is normal (marginals are normal).
- **Zero covariance ⇒ independence** (this is special to normality; not true in general).

For the spectral / geometric understanding of these contours, "*you take this Linear Statistical Models, which is not a course I teach, so have fun there.*"

## Cut-off: contour exercise

The prof started showing contour plots — pairs of variables with different Σ patterns, asking students to match contours to covariance matrices — and ran out of time mid-exercise:

> "We're going to call it a day here just because slides are different than I expected, so I'm a bit confused about what I'm saying. … We'll come back to this tomorrow."

The contour-matching is finished at the start of [[L05-linreg-1]].

## Course admin (brief)

- TA (Simon) running exercise sessions; not mandatory but suggested. Speaks Norwegian and English.
- Two compulsory **projects** dropping today/this week: start picking groups.
- Wiki page being expanded as a course resource; Simon driving that.
