---
lecture: 26
date: 2026-04-27
module: 11-nnet
title: Neural Networks 3 (RNNs and Double Descent)
slides: modules/11NNet/11Nnet.md
topics:
  - recurrent-neural-network
  - gaussian-error-assumptions
  - regularization
  - interpretability
  - bias-variance-tradeoff
  - double-descent
  - splines
  - polynomial-regression
  - lasso
tags:
  - lecture
  - module/11-nnet
aliases:
  - Lecture 26
---

# L26: Neural Networks 3 (RNNs and Double Descent)

The prof closed the neural-networks module with [[recurrent-neural-network|RNNs]] (motivation, architecture, weight sharing, the NYSE time-series example), a "when to use deep learning?" Hitters comparison (linear / lasso / unregularized NN), and finally [[double-descent]], the part he likes most, presented as the key intuition for *why* big modern ML models work without contradicting the [[bias-variance-tradeoff]]. Tomorrow (L27) is summary + exam Q&A.

## Key takeaways

- RNNs extend feed-forward nets to sequential data (time series, language) by carrying a hidden state $A_t$ across the sequence; the same weights $B, U, W$ are reused at every step (**weight sharing**), which is why training stays well-behaved.
- The output activation depends on the goal (linear for regression-style, softmax for next-token), exactly as for feed-forward nets. All outputs $O_t$ are used in training, even when only the last one is the prediction of interest.
- NYSE example: predict trading volume $V_t$ from lag-$L$ history of $(V, R, Z)$ (volume, returns, volatility). Time-series data has high autocorrelation → samples not independent. RNNs were "equally bad" as classical time-series models on cryptocurrencies, but that's a data problem.
- "When to use deep learning?" Hitters: linear ≈ 0.56 MAE, lasso-CV (12 vars) ≈ 0.50, unregularized 1049-param NN slightly worse. The prof flags the comparison as **contrived**: "you should never train a neural network without regularization." Real value of NN comes with very large data and when interpretability is not a priority.
- **Double descent**: as model complexity grows past the interpolation point (≈ #params = #samples), the test error can come down a *second* time. Does not violate bias-variance: bias and variance still always sum correctly; it's just no longer U-shaped.
- Mechanism: past the interpolation limit, the model has *infinitely many* perfect fits to the training data; training implicitly picks the minimum-variance (≈ minimum-norm) one. Optimization changes from "fit + penalty" to "min $\sum \beta^2$ subject to fitting the data exactly."
- Bias-variance is the running theme: "I think the first day they're definitely going to be a question about bias variance in the test."

## Recurrent neural networks: motivation

Recap of where the module has been: started with feed-forward nets (the workhorse, "the basis"), then CNNs as a clever extension via patch-weights (a *simplification* of the feed-forward paradigm with shared patches). Now another extension, **recurrence**, to push the model "beyond what you can do with just a feed-forward network."

The motivation is **sequential data**: time series, language. Sequential data has the notion of "what comes next." RNNs were the first models in machine learning to be used in language modeling, "the precursor to the language model that we know now." Many LLMs are trained for next-word / next-token prediction; this is where that paradigm started.

> [!note] Why ordering matters
> "If there was no order to it, if these were just samples, right. Let's say we just had, you know, images that were disconnected in time. Just someone's random or a bunch of pictures from the Internet. One picture doesn't necessarily lead to the next. So there would be no obvious ordering of the pictures. So then it wouldn't make any sense to carry over any information about the previous picture because it has nothing to do with it."

Versus a feed-forward net, the differences are: sequential structure, and that the sequence length is no longer a fixed input.

## RNN architecture

[Slide: §"Recurrent neural networks (RNNs)", figure 10.12]

Inputs $X_t$, hidden activations $A_t$, outputs $O_t$. Without the carry-over, this would just be a feed-forward net at each timestep. The new piece: **$A_t$ is carried into $A_{t+1}$**, so information propagates along the sequence dimension (time, position in sentence, etc.).

For text: tokenize words / word-parts, optionally one-hot encode, then embed into a smaller space ("no reason to have it so big"). Often you learn the embedding; sometimes you can fix it. Tokenizers often work on word-parts rather than full words.

Hidden activations:

$$A_t = \sigma\!\left( b + W X_t + U A_{t-1} \right)$$

Here: bias $b$, weighted sum of the current input, plus a weighted sum of the *previous hidden state*. The carry-over is how sequence information is propagated.

Output:

$$O_t = \beta_0 + \beta^\top A_t$$

In the figure-10.12 version this is **linear** (no activation), equivalent to a least-squares output. As with feed-forward nets, you could pick softmax, sigmoid, etc. depending on the task. "The activation function for the output would depend on what the goal is of the model."

Graphically, "it kind of looks like a Markov chain."

> [!important] Information from $X_1$ reaches $O_L$
> "The interesting thing is that this just keeps going, right? If $L$ is a billion, then essentially even the first thing in your sequence is affecting the billionth output because it's carried through the model. In theory the information will get lost because it just yeah a billion is pretty long but the idea is still nice."

(This is the vanishing-information problem that motivates LSTMs / attention, but in this course we stop at the basic RNN.)

### Weight sharing

You'd expect a model carrying state across $L$ steps to have a huge number of parameters, but:

> [!important] The same $B, U, W$ at every step
> "If you notice here we just say B and U and W every time it's because we don't have if it was like this that would be a very different model if the weights were changing every time but they don't in fact we assume the same weights B, U, and W every single time. … And that also why the gradients behave well why the training behaves well is because it not a model with so many parameters that it starts collapsing or getting weird."

A form of [[recurrent-neural-network|weight sharing]] (same flavor as the patch-weights of CNNs). $b, U, W, \beta$ are learned during training but do **not** change with position in the sequence.

### Loss and training

Same recipe as other neural networks: pick a loss, minimize via gradient descent. With the linear-output setup,

$$L = \sum_t \| Y_t - O_t \|^2$$

(softmax / sigmoid loss if those output activations are used).

A common confusion: the equation as written looks like only the *last* output is what we care about. But every $X_t$ has flowed forward into the prediction at time $L$ via the $A$'s, **and** every intermediate $O_t$ is also predicted and contributes to the loss: "you do train using every single output." The intermediate outputs "come for free" from the architecture; you don't have to do anything special, the model is constructed that way.

## RNN example: NYSE time series

[Slide: §"Example of an RNN: Time series forecasting"]

Predict next day's trading volume $V_t$ from lag-$L$ history of three time series:

- $V$: trading volume
- $R$: returns
- $Z$: volatility (think of as variance, "how much is changing each day")

Construct inputs by stacking the past $L$ time steps:

$$X_1 = (v_{t-L}, r_{t-L}, z_{t-L})^\top, \quad \dots, \quad X_L = (v_{t-1}, r_{t-1}, z_{t-1})^\top, \quad Y = v_t$$

"$L$ is very much like the token length in a large language model. You want it to know about the previous $L$ things for your prediction to work."

> [!note] Why time series is hard
> All three series have **high [[gaussian-error-assumptions|autocorrelation]]** — successive time points are not independent. Any model has to account for this dependency in both the fit and the uncertainty quantification. Stocks in particular: "often it's less about how the mean is changing and more about how the variance is changing. You have to consider both things."

Why use a black-box model at all? When the goal is purely forecasting, and a hand-built statistical model would be too delicate to construct, an expressive ML model that "you don't really care how it works" is attractive. The prof recalled an old time-series course where students compared RNNs to classical models on cryptocurrencies: "they were equally bad. But I think that was really just the data problem because no one should be using Bitcoin. It is stupid."

Trade-off: the RNN is much easier to train (just feed in variables and predict) than a classical time-series model (where you'd carefully pick parameters and structure). "If you don't care how it works and you just want to get the results, these models are pretty good."

The slide has more complex variants where outputs feed back into the hidden state, but the lecture stayed at the simplest version.

## When to use deep learning?

[Slide: §"When to use deep learning?", Hitters dataset]

Recurring question in the building: "How often should we just be using regression and how often should we turn to more advanced methods like the forest or in particular machine learning models like the neural networks, models with a lot of parameters?"

Compare three models on the Hitters salary prediction (predict baseball player salary, 263 observations, 20 predictors), "the least depressing data set" the course has touched. (Long aside about how baseball is boring; "Hockey is more exciting.")

| Model | Setup | Test MAE |
|---|---|---|
| Linear regression | 20 parameters | ~0.56 |
| Lasso + CV | 12 variables left | ~0.50 |
| Neural network | 1 hidden layer, 64 units, 1049 params, **no regularization** | slightly worse |

A thousand parameters "is peanuts" by modern ML standards (current LLMs are at trillions).

> [!important] Don't train an NN without regularization
> "I can't think of an example where you'd ever want to train a neural network without a regularization. … The whole point of regularization is to get you to generalize better."

Forms of regularization the course has covered:
- Explicit: L1 / L2 ([[lasso]], [[ridge-regression]])
- Implicit: mini-batch [[gradient-descent-and-sgd|stochastic gradient descent]] (from L24)
- Dropout (randomly remove nodes during training, "forces the model to not rely on any specific part")

The slide's setup is therefore **contrived** in the prof's view: "I think other people teaching the same course would not call this contrived but call it real and i would say this is if you constructed your model this way you doing it wrong."

The slide's stated takeaway, "NNs are attractive when the size of the training set is extremely large and when model interpretability is not a priority", gets the prof's qualified agreement:

- **Big data**: yes in the trivial sense; but the Hitters set is also tiny: "263 baseball players that's not much data I wouldn't have bothered with a neural network either just because what's the point you're not going to be able to identify very complicated functions because there is none."
- **Interpretability**: agrees strongly. "If you fit your model, if you have a data set and you fit it with a complicated neural network, in the end, you don't know what you have." Same problem as XGBoost (residuals of residuals of residuals).

### Aside: explainable / interpretable AI

The prof briefly described his own research direction: studying activation functions / learned representations in neural networks. The hope: identify what representations a big network has learned, then train a much smaller, interpretable model that reproduces just those representations. Mentioned [[shapley-values]] (Inga at IES works on this) as a parallel direction.

> [!quote] On model engineering vs. interpretability
> "If we did that and if that's what it came out to we could say well as fun as that was and as great as it was it actually did not really improve over just the lasso with the 12 terms so then we have an idea as to what is the the upper bound in terms of model fit like how well you can really do."

I.e. fit a giant NN to find the *upper bound* of achievable performance, then check whether a simple model already gets close. If it does, the simple model wins on interpretability for free.

## Double descent

[Slide: §"Double descent"]

After break. Prof's own framing:

> [!quote] Bias-variance is the running theme
> "I think the first day they're definitely going to be a question about bias variance in the test because i think that concept is the kind of the running theme through the course in all the models we looked at, we thought we talked about at least briefly or not. In what way is this affecting the trade-off between the bias and the variance? Why do we want to reduce the variance? Why is that such an important thing?"

Historical aside: the phenomenon was first observed in the 70s at Bell Labs ("this heavenly place for science … no one had to do paperwork. … Claude Shannon, lots of famous people"); popularized recently in the ML context "because it illustrates something that's really, I think, fundamental to how these big models are working so well."

### The classical regime

[Slide: bias_variance.png]

Standard story (statistics / optimization classes): as model complexity grows, training error keeps falling, but test error first falls, then rises (overfitting). The sweet spot is the test-error minimum.

How we usually find that sweet spot in this course:
- [[regularization]] pushes us toward it more stably
- [[cross-validation]] for hyperparameter selection: "we're essentially always going back to the test set to determine our model complexity"
- (Caveat: repeatedly using the test set to tune can itself overfit the test set; covered earlier in the course)

### The second descent

[Slide: double_descent.png]

If you keep going past where you'd normally stop:

- Training error keeps falling: "always just keeps, I mean, inevitably as you add more and more parameters, no matter how many more parameters you add, this shit will always go down."
- Test error climbs to a peak at the **interpolation point**, typically where #parameters ≈ #samples, "where the variance explodes."
- *Then* test error comes down again: a second descent.

> [!important] Does this break bias-variance?
> "It doesn't if you compute all of the things from the bias and variance trade off it all always adds up that the bias and the variance are always trading off of each other right because you can always just go up in one or down in the other. And remember, the variance is variance squared. … It's just the model at this complexity is where it blows up. But, you know, you just keep going."

Bias and variance still always sum to (test MSE − irreducible error). The U-shape just isn't the only possibility; sometimes you get double descent.

### Sine + splines example

[Slide: §"Double descent" with sine example, fig 10.21 / 10.20]

Generative model: $y_i = \sin(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, 0.3^2)$, 20 noisy observations. Fit with [[regression-splines|natural splines]] of varying degree:

- Degree 8: fits well, looks reasonable
- Degree 20: starts blowing up, classical overfit warning territory
- Degree 42, 80: gets weird in places, but degree 80 is **better than degree 20**

For sine specifically, degree 8 still wins (splines are good at cosine curves), but the test error went down a second time.

### Square-pulse + polynomial regression example

The prof's own contrived example (to make the second descent clearly *win*):

- True function: a square pulse (jumps from 0 to 1 to 0). **Discontinuous, not in the model class.**
- Add Gaussian noise.
- Fit with Legendre [[polynomial-regression]] (just $a + bx + cx^2 + \dots$ in a basis).

Trained at increasing degree $d$ with L2 regularization on 100 data points. Plotted training and test error vs. $d$:

- Training error always decreases (eventually to zero)
- Test error: first descent → minimum near $d = 24$ → blows up near $d = 100$ (= number of training points = interpolation point) → **second descent**, going *below* the first minimum

Visualizing the learned function:
- Best model in first descent ($d = 24$): smooth, OK fit
- Best model in second descent: spikes at every data point, but local mean elsewhere; fits **all** training points exactly and predicts something close to the mean in between

> [!important] What changes past the interpolation point
> "All of the models that we are considering there, all of the possible solutions all fit the training data perfectly. Whereas before we're trying to find the best fit to the training data with some penalty maybe. But now we have an infinite number of models that all fit the training data perfectly. And we're selecting from those models the one that gives us the best prediction out of sample. How do we do that? By minimizing the variance."

### The optimization changes

The prof's preferred framing of what's *really* going on past the interpolation point:

**First regime** (under-parameterized): minimize
$$L = \sum_i (y_i - \sum_j \beta_j x_{ij})^2 + \lambda \sum_j \beta_j^2$$
Normal regression loss + L2 penalty. The fit term is *not* zero.

**Second regime** (over-parameterized, post-interpolation): minimize
$$\sum_j \beta_j^2 \quad \text{subject to} \quad y_i = \sum_j \beta_j x_{ij} \text{ for all } i$$
The data-fit constraint is satisfied exactly (infinitely many ways), and you're picking the **minimum-norm** solution. The L2 term effectively becomes the *only* objective.

> [!quote] The mechanism
> "We've changed the optimization that we're doing we haven't really because we've still we're still constructing it the same way but since we have so many parameters that … the model is so flexible it has an infinite number of ways of fitting it and therefore it doesn't have to pick a model that fits the training data well, all of them do it's finding the one that has the best variance."

This minimum-variance / regression-to-the-mean instinct is why the spike-pattern solution emerges: the model doesn't want to be sensitive to any single noisy data point, so it shoots back to a local mean as fast as possible while still passing through every observation.

### Bias-variance still adds up

Computed bias term, variance term, and irreducible error for the polynomial example. Their sum matches the empirical test MSE across all degrees (**the trade-off identity is never violated**). Double descent is just a non-U shape of the same identity.

### Slide summary points

[Slide: §"Double descent: some considerations"]

- Does **not** contradict bias-variance. (The $x$-axis is "flexibility," which can grow past the interpolation point.)
- Achievable mostly in **high signal-to-noise** problems: natural image recognition, language translation. Prof: "I think this is likely better done in low noise situations but I mean clearly it works in high noise too but it makes more sense with high signal the noise."
- Most statistical learning methods covered in this course **do not** exhibit double descent. Regularized methods can give great results without ever interpolating the training data.

> [!important] Slide takeaway (verbatim from deck)
> "Though double descent can sometimes occur in neural networks, we typically do not want to rely on this behavior."

The prof's qualified agreement: "We don't want to rely on this maybe yeah again it depends on what you're trying to do."

### Why the prof emphasizes this

> [!quote] Why double descent matters
> "I think this is one of the main points of, or one of the main reasons machine learning works. … For me, the right way of looking at it is actually it's changing the optimization. Because you in this regime where you can interpolate between an infinite number of correct solutions … so that part of the magic of the model."

## Closing

Slide section "DL in Medicine / DL in Ecology" listed: pointers to applied papers, not lectured.

End of Module 11 and end of new content. Tomorrow (L27, Apr 28) is summary + exam Q&A: "please come with questions." See [[L27-summary]].
