---
concept: prediction-vs-inference
module: 01-intro
lectures: [L01, L02, L03]
isl-ref: 2.1.1
exercises:
  - Exercise2.1  -  describe a real classification and regression task, identify response/predictors, decide prediction or inference
related: [statistical-learning, supervised-vs-unsupervised, bias-variance-tradeoff, linear-regression, r-squared]
tags:
  - concept
  - module/01-intro
aliases:
  - inference vs prediction
---

# Prediction vs inference

The two reasons to estimate $f$ when $Y = f(X) + \varepsilon$. **Same model, two uses, very different design choices.** The prof returns to this distinction whenever the question of model design comes up, it's the lens that decides whether you should care about coefficients, R², test MSE, or interpretability.

## Definition (prof's framing)

Two reasons to estimate $f$:

- **Prediction**: care only about $\hat Y$. The shape of $f$ doesn't matter; the model is a black box.
- **Inference**: care about the *form* of $f$. Which $X$'s go in, what shape the relationship takes, which coefficients matter.

> "Prediction: you don't really care if the model is right. That's secondary. Being right is not as important as being able to predict well and predict with knowing your uncertainty of your prediction." - [[L01-intro]]

> "Inference is often the shorthand or the single word that we would use to describe trying to understand. We want to infer what parameters matter." - [[L02-statlearn-1]]

The same statistical setup ($Y = f(X)+\varepsilon$) and even the same fitted model can serve either goal, but the design choices look very different in practice ([[L03-statlearn-2]]).

## Notation & setup

No special notation beyond the standard supervised setup:

$$Y = f(X) + \varepsilon, \quad \hat Y = \hat f(X)$$

What changes per goal:

| | Prediction | Inference |
|---|---|---|
| Goal | $\hat Y \approx Y$ on unseen data | Understand $f$, which $X_j$ matter, sign / size of effect |
| Care about $\hat\beta$? | No (often). Black box OK | Yes, interpretability is the whole point |
| Care about test MSE / AUC? | Yes, primary metric | Mostly as a sanity check |
| Care about p-values / CIs? | Mostly no | Yes, they're how you assess "which $X_j$ matter" |
| Care about coefficient SEs? | No | Yes |
| Tolerance for flexible black-box models | High (NNs, boosting) | Low (favor linear, GAMs, simple trees) |

## Insights & mental models

### Finance vs science, the canonical contrast

> "Quants and econometricians modelling stocks/Bitcoin… they don't really care if the model is right. That's secondary. Being right is not as important as being able to predict well and predict with knowing your uncertainty of your prediction." - [[L01-intro]]

> "I'm often not really caring about predicting the neural activity. Right. It's useless. The animal's dead. The goal is an understanding or a development of the science of what's going on." - [[L01-intro]]

Finance: you don't care *why* Bitcoin moves, you care that you forecast tomorrow's price with calibrated uncertainty. Neuroscience / medicine: predicting future blood pressure of someone already dead is useless; you want to know which knobs (BMI, smoking, age) move it so people can act.

### The medicine example as a working illustration

Both lectures use the **Framingham SBP regression** as the inference exemplar ([[L01-intro]], [[L02-statlearn-1]]):

> "Like being fat, you die younger, right? Statistically, I think… So then don't do that, right? It's not that you want to predict your death. You're just trying to change it. You're trying to make decisions based off of a model that gives you an understanding." - [[L02-statlearn-1]]

The Framingham linear regression has terrible R² ("blob-grade correlation") but is a seminal paper precisely because it's an inference paper, doctors don't predict your death, they tell you to exercise more. **Low R² is acceptable for inference, unacceptable for prediction.** This is the prof's stock illustration of the prediction/inference split affecting how you read the same model output.

### Same model, different design choices

In modern ML the *exact same dataset* yields very different solutions depending on goal:

- **Prediction-first**: deep nets, boosting, ensembles. Throw model size at the problem; CV-tune hyperparameters; test MSE / AUC are the only currency.
- **Inference-first**: linear / logistic regression with a few well-chosen predictors, GAMs, single trees. Coefficients have to mean something; SEs / p-values matter; you keep the model simple enough that you can *explain it*.

> "The same problem can be approached either way and gives very different solutions often." - [[L03-statlearn-2]]

LLMs as the extreme prediction case ([[L03-statlearn-2]]): ~trillion parameters, "we don't really care what's in it", though you may still care about *controlling* outputs (no hallucinations) even when you don't care about parameters.

### Prediction is also useful for inference work

Even when inference is the real goal, prediction quality is a useful sanity check on the model:

> "Even when prediction isn't your real goal, it's often a good way of evaluating a model." - [[L02-statlearn-1]]

If your inference model can't predict at all, the inferred coefficients are probably noise too. So: inference goal doesn't mean ignore test MSE, it just means it isn't the headline.

## Exam signals

> "I like this kind of question … it's conceptual, but you also don't have to write a whole book, you just have to know which words to fill in correctly." - [[L27-summary]]

(Prof on the 2025 Q1 walkthrough, fill-in-the-blank using "regression," "classification," "prediction," "inference." Designed so the answer is in subtle wording, not the obvious word at the top of the paragraph. Example trap: "in the latter case we do not care about the actual model parameters" → **prediction**.)

> "Inference is often the shorthand or the single word that we would use to describe trying to understand. We want to infer what parameters matter." - [[L02-statlearn-1]]

Recurring framing: a question that gives you a real-world scenario, you tag it as prediction or inference, then justify in one line.

## Pitfalls

- **Reading p-values when the goal is prediction.** A model can predict well with no significant individual coefficients (collinearity), or have huge t-stats with terrible test MSE. Sample size inflates significance for trivial slopes, see [[L05-linreg-1]] / [[L06-linreg-2]] on "significance is just sample size." Prediction quality and statistical significance can disagree.
- **Reading R² when the goal is inference in medicine / biology.** The Framingham R² is low and the paper is still seminal. R² is a prediction-quality summary; for inference the "blob" is fine if the slopes are interpretable.
- **Confusing "I want to predict $Y$" with "I want to understand $Y$."** Anders should explicitly write down which one before choosing a method. The prof flags that this is the design decision that determines model class, regularization choice, and which output statistics to trust.
- **Black-box prediction with calibrated uncertainty is still prediction**, not inference. Quant-style "predict + confidence interval on the prediction" is prediction; inferring β's is something else.
- **Inference does not require causality.** The prof is explicit: inference is "trying to understand the structure, ideally causal, more often correlational" ([[L01-intro]]). Don't overclaim cause from a regression coefficient.

## Scope vs ISLR

- **In scope:** the prediction vs inference distinction itself, the design-choice consequences, recognizing it in a real-world description, and using the dichotomy to choose between flexible vs interpretable methods (this returns in module 2's bias-variance and modules 6–9 when picking regularization / boosting vs simple linear models).
- **Look up in ISLR:** §2.1.1 ("Why Estimate $f$?"), ISL splits this exactly into Prediction (§2.1.1) and Inference (§2.1.2). Short, well-written, on the exam table for the canonical reference if Anders needs to look up the formal statement.
- **Skip in ISLR:** none, §2.1.1–§2.1.2 is short and entirely in scope.

## Exercise instances

- **Exercise2.1**: describe a real classification application and a real regression application, identify response and predictors, and for each say whether the goal is prediction or inference. The canonical drill problem for this concept; the answer is graded on whether you can articulate the distinction and apply it to your own example.

## How it might appear on the exam

- **Fill-in-the-blank / multiple-choice (Q1-style from 2025)**: a paragraph describes a real-world problem; pick "prediction" or "inference" based on subtle wording. The prof's [[L27-summary]] walkthrough has this exact pattern; the answer is "in the latter case we do not care about the actual model parameters" → prediction.
- **Tag-a-scenario T/F**: "the Framingham SBP study is a prediction study" → False (it's inference; doctors use it to identify risk factors, not forecast individual deaths). Sample-size-inflates-significance variants live nearby.
- **Method-comparison reasoning**: "given two models' results, which is better and why?", the answer often hinges on whether the goal is prediction (use the lower test MSE) or inference (use the more interpretable one even if test MSE is slightly worse). This is exam pattern 7 in [[scope]].
- **Coefficient interpretation**: questions that ask you to interpret a regression / GLM coefficient *only make sense* in inference framing, flag when the framing of a question implicitly demands inference reading.

## Related

- [[statistical-learning]]: the parent framing; prediction vs inference is one of the two organizing axes
- [[supervised-vs-unsupervised]]: the other organizing axis; orthogonal to this one
- [[linear-regression]]: the canonical model used to illustrate both modes (Framingham for inference, advertising for prediction)
- [[r-squared]]: the prediction-quality summary that Anders should weight differently depending on goal
- [[bias-variance-tradeoff]]: the lens through which design-choice differences become quantitative (prediction-first methods accept more bias for variance reduction)
