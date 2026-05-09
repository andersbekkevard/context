---
lecture: 2
date: 2026-01-12
module: 02-statlearn
title: Statistical Learning 1
slides: modules/2StatLearn/2StatLearn.1.md
topics:
  - statistical-learning
  - quantitative-vs-qualitative
  - supervised-vs-unsupervised
  - regression-vs-classification
  - prediction-vs-inference
  - bias-variance-tradeoff
  - statistical-vs-machine-learning
  - training-validation-test-split
tags:
  - lecture
  - module/02-statlearn
aliases:
  - Lecture 2
---

# L02: Statistical Learning 1

Ben (taking over from Stephanie) walks through the first half of Module 2, sticking mostly to last year's slides while flagging the parts he disagrees with. Sets up the vocabulary of statistical learning (quantitative vs qualitative variables, supervised vs unsupervised, regression vs classification, prediction vs inference) and ends on the famous Breiman "Two Cultures" paper as a setup for the bias–variance discussion to come tomorrow. Material is **chapter 2 of ISL** (read it: "it's well written… it's the right source"). Got through roughly the first half of `2StatLearn.1.md` (stopped at the advertising / Sales example).

## Key takeaways

- [[statistical-learning]] = learning *from data*, in the misspecified-model regime: you don't have access to the truth, you don't have all the variables, your model is wrong, and you have to deal with it. This is the regime that *didn't exist* before sensors and computers got good ~20–30 years ago.
- Two variable types: **quantitative** (continuous or discrete numbers) vs **qualitative** (categorical, often coded numerically). Tokens in LLMs are an interesting in-between because they encode position/context, not just word identity.
- [[supervised-vs-unsupervised|Supervised learning]] has a Y to aim at; [[supervised-vs-unsupervised|unsupervised learning]] just has X with no clear objective. Most of the course is supervised. Many "unsupervised" problems can be reframed as supervised: LLMs are the canonical example, trained only to predict the next word, yet appear to know therapy and Python.
- Two reasons to estimate f: **prediction** (don't care what's inside, want Ŷ accurate) vs **inference / interpretation** (want to know which X's matter and how). Same model, very different design choices.
- "If it's a math class or statistics - depends on what you call statistics. If it's math or not." Course is in the math department, "less mathematical than most math classes, but the goal is for mathematicians."
- The bias–variance phrasing: he doesn't like the term, will keep saying so, and **expects an exam question on it** ("if I had to guess today what the exam question would be, it would be… criticize the term bias–variance trade-off, give two perspectives on it"). Today is just the *beginning* of that discussion.

## Setup and admin

New term, new instructor: "I am not Stephanie. I am Ben." Course is **module 2, part 1** = chapter 2 of ISL. He's keeping last year's slides "largely" but disagrees with parts and will say so. Also covering matrix algebra as additional material. New PhD TA Simon (Norwegian, just hired) will run exercise sessions starting next week.

> "I do expect you to read the book. I would recommend it. It's well written… it is part of the curriculum even if I - yeah, that's more the curriculum than whatever I say in many ways."

## What is statistical learning

Statistical learning = the process of learning structures from data about the real world. The term, he warns, is "arguably… poorly defined." He polls the room: science people (want to discover) vs prediction/engineering people (want to forecast or build). Most of the room leans prediction.

His framing (and a bit of personal slant) is that statistical learning specifically refers to the [[statistical-learning|misspecified-model]] setting:

> "You don't really have access to the parameters directly or you don't have access to all the right parameters - maybe you have some of them with noise… Our model is misspecified because they're missing a lot of things or they're typically misspecified. And we just assume they're not. And then we try to work from there."

His own neuroscience as the canonical example: you only get to record some of the neurons, with noise, your model is structurally wrong from the start. You then have to *learn* aspects of the data that you can't a priori assume.

Contrast with classical statistics, where you have access to everything, the model is right, and you just fit it to data. That, he says, is **not** the typical scenario for this course:

> "I don't think we should. I think we should think more about the scenarios where we don't have access to the right model, we're a bit confused as to what's going on, we're trying to figure out what's in there."

In the basic statistics course you keep things simple enough that everything is easy to interpret. Here you're going to deliberately step into messier territory.

### Why now and not before

The misspecified / data-mining regime "didn't exist so much before", not because nobody wanted it, but because **the data wasn't there**. Sensors, databases, computers were all bad until ~20–30 years ago.

> "When I was a kid, computers sucked. Phones had a cord to plug in, you know, like we just didn't have access to this kind of data. But now it's everywhere - now you can get lots of data on anything and it just really opens the door to actually not making as many assumptions and learning stuff from the data."

So the data-mining philosophy showed up *because* the data showed up.

Side commentary on how science *should* vs *does* work: classically you state a hypothesis, design the experiment around it, and you have all the variables you need. In practice people get the data first and then go looking for a model that fits, and that's the scenario these methods exist to handle. Two responses he wants you to take seriously: **design experiments better**, *and* make sure the people using these methods understand their limitations and how to use them.

## Two kinds of variables

**Quantitative**: things you can quantify with numbers: weight in kg, height in cm, age, temperature, precipitation. Continuous or discrete (countable / integers). Mostly think about continuous in this course.

**Qualitative**: types of fruit, education level, no inherent numerical value. To use them in a model you assign one: true/false → 0/1 (binary), apple/orange/banana → 0/1/2. "Depending on how we use those things, maybe that number ends up meaning something within the model." (Foreshadow of the categorical-encoding pitfall; he doesn't ring the alarm here yet.)

### Aside: tokens in LLMs

He adds a third example he finds interesting: **tokens**. An LLM tries to predict the next word as a function of the last n words. Instead of predicting words directly, the words get converted to **tokens**: vectors of numbers that encode not just *which* word but *where it sits relative to other words*, because the same word can mean very different things in context (the *light* in the room vs not feeling like a *light*). Tokens carry positional / contextual information; a whole literature exists on text → tokens, and the tokenization evolves with the models. Worth flagging because it sets up a later contrast.

### Examples of the prediction problem

**MNIST handwritten digits**: "the data set that everyone studied, and maybe that was a mistake." Originally from the US postal service, used to sort mail by zip code. Drove machine-learning research for years; "if your model didn't give you 99% accuracy instead of 98%, you didn't get it published." Classification problem: predict the digit from the image.

**Email spam classification**: true/false on whether a message is spam, based on word and character frequencies. He goes on a long tangent about the early internet: "when email first came out it was terrible… everything was spam." A cat-and-mouse arms race between spammers and filters. Doesn't actually claim it works that simply, just an illustrative classification example.

Both are **supervised learning**: you have the outcome and are training toward it.

## The supervised learning problem

Outcome variable Y; vector of predictors X = (X₁, …, X_p). Predictors go by many names. He prefers **predictors / regressors / covariates / features / variables**, and explicitly calls out one to avoid:

> "I wouldn't typically use the word *independent variables*. I would say this has more meaning than the others… because most things are not independent."

Quick example: temperature outside vs hours of daylight, clearly correlated. Don't claim independence you don't have. Side note: *covariates* often implies time or space (things "co-vary" over something).

**Two problem types:**
- [[regression-vs-classification|Regression]]: Y is quantitative (price, blood pressure).
- **Classification**: Y is in a finite, unordered set (survived/died, digit 0–9, cancer class). LLMs are essentially "a very fancy classifier, predicting the next word."

Training data come in pairs: $(x_1, y_1), \ldots, (x_n, y_n)$. He'll also use **fit data** for training data, and he splits **test** vs **validation** as separate ideas: "often it's not mentioned as much as it should be, but we'll throw in validation as well." The point: in the misspecified setting where the model is finding creative ways to fit the data, you absolutely need data you didn't fit on to evaluate honestly.

### Notation

The data matrix X has columns = individuals/samples and rows = variables. (He notes this is the book's convention; other books transpose it; doesn't really matter.) Shorthand:

- $x_{(i)}$: the i-th column = all measurements for one individual/sample (e.g. that person's salary, age, height, weight, …).
- $x^{(i)}$: the i-th row = one variable across all individuals (e.g. all heights).

Terminology he uses interchangeably and warns about: **individuals / samples / people** for the rows of the dataset. He'll switch between them; book may not be perfectly consistent either.

> "The easy way to write X is to write [it] as a matrix."

### Goals of supervised learning

Three things you typically want, and which one you care about changes how you approach the problem:

1. **Accurately predict** unseen test cases.
2. **Understand** which inputs affect outcomes and how (= **inference**).
3. **Assess the quality** of your predictions and inference.

Even when prediction isn't your real goal, it's "often a good way of evaluating a model." But it isn't necessarily the goal:

> "Maybe you don't care, right?… Another goal that would be complementary is to actually make a model that you can understand. Like you're trying to fit a model that classifies simply because you want to see what is the thing that's indicating a classification."

The medical example from L01 returns: if you want to know what indicators predict a disease so you can avoid them, you don't really care about *predicting your own death*. You care about knowing what knobs to turn.

> "Like being fat, you die younger, right? Statistically, I think… So then don't do that, right? It's not that you want to predict your death. You're just trying to change it. You're trying to make decisions based off of a model that gives you an understanding."

That's the inference goal: understand which inputs affect the outcomes and *how*.

> "Inference is often the shorthand or the single word that we would use to describe trying to understand. We want to infer what parameters matter."

Same model can serve either goal, but the design choices look different. Both goals also need (3), assessing the quality of the prediction or the inference.

## Unsupervised learning

No outcome variable, just one big X matrix. **Objective is fuzzy**: find hidden patterns or groupings, gain insight, but there's no "correct answer."

He's a bit critical because "it sounds like bad science… data mining for the sake of mining," but then concedes it's actually most of what *he* does in his own neuroscience work. Long aside on his torus / hexagonal-grid neural-recording paper (the data lives on a torus; same structure persists in baby animals before they can perceive space → Kant was right, perception of space is innate, evolved for 2D surfaces and would suck in space-station-like geometries). Not on the test, but the methodological point matters:

> "Unsupervised learning seems like there's no goal, there's no objective, [but] it can be a very nice way to understand something. Because then you go in without so many assumptions, and then the properties that come out, you can make nice, bold claims."

### Reframing unsupervised as supervised

Often what *looks* like unsupervised is really supervised in disguise. Best example: large language models. The training task is just **predict the next word**, perfectly supervised, perfectly defined. But the model ends up "knowing" who-did-it in a mystery, knowing program syntax, knowing therapy.

> "It was just trained to predict the next word. What's the next word a therapist would say?"

Difficulties of the genuinely unsupervised: hard to know how well you're doing, hard to know when you're done. Course examples (saved for **module 10**): [[clustering]] and [[principal-component-analysis]].

## Overall philosophy from the book (with caveats)

The book takes the line that **simpler models first**, then more complex ones. Two slogans:

1. *"It's important to understand the simpler models first in order to grasp the more complicated ones."* He fully agrees:

   > "Even those things can do really confusing stuff. It's surprising how weird it can get with something that is so seemingly innocuous and trivial."

2. *"Simpler methods often perform as well as fancy ones."* He **disagrees as a blanket claim**: depends on the setting. For interpretability and a short methods section, sure, simple wins. For prediction-heavy tasks like LLMs:

   > "You can try to predict the next word just using linear regression… and it sucks. It won't make any sense."

   Many of the "simple wins" comparisons "either did it wrong or are using older papers." Use the simplest model that does the job; don't be religious about it.

3. *"It's important to accurately assess the performance of a method."* Not questionable. This is going to drive a lot of the course.

## Statistical learning vs machine learning

Both fields, both supervised and unsupervised, lots of overlap. He treats the *names* as branding:

> "Whether you call it machine learning or AI - ML, or there's been other names… these are like brandings. Like if machine learning isn't popular, then you call it AI. If it becomes unpopular, then you call it like something else… mainly so people can keep getting funding."

Origins:
- **Machine learning**: coined by a guy at Bell Labs who wanted a machine to learn checkers.
- **AI**: older, maybe von Neumann, "I forget."
- **Statistical learning**: "a way for statisticians to get machine learning money and to get students to take their class." Also a real intellectual move: take the cool things ML people invented and study them more formally.

> "I won't ask any questions about how they're defined."

Differences in flavor:
- ML cares less about *why* something works inside the model: "they'll gladly add in something to their model that they don't understand at all, and keep it just because the model performs better. And then I think they'll leave it to future work to figure out why that was a good idea or even if it was a good idea. But the goal really is to engineer something that performs better."
- Statistical learning, being closer to math, steps back: "okay, well, what is going on here?" Focus on understanding models.
- ML emphasizes **large-scale applications and prediction accuracy**; statistical learning emphasizes **models, interpretability, precision, uncertainty**.

> "Machine learning has the upper hand in marketing. I would totally agree with that. But also they're doing really well and they're exciting."

He likes ML personally because it started in the 1950s as a model of the brain, with the same wild ambitions for intelligence and self-replication as today's AI hype. "I think it's fundamentally interesting."

## The Breiman "Two Cultures" paper (2001)

He drops in Breiman's 2001 paper *Statistical Modeling: The Two Cultures* plus the first few pages of Cox's reply. Says he might post the article on Blackboard and discuss another day; gives the gist now.

Setup: in the supervised setting Y = f(X) + noise, Breiman argues there are two cultures.

1. **Data modeling culture**: assume Y is some function of X with random noise and parameters. "Most of the book is focused around this idea of a data model." Includes regression, logistic regression, Cox regression, etc. (Cox regression is *not* covered.)
2. **Algorithmic culture**: less clear what's going on inside; the path from X to Y is complicated. Closer to ML / engineering. Breiman complains that only **~2% of statisticians** were in this camp. The book tries to say these are "two sides of the same coin."

Breiman's accusation: this 2%-not-98% split has produced **lots of irrelevant theory**, theory built around classical methods that the working ML people don't care about because they know their model is misspecified anyway, they have lots of problems with their data, and they'd rather just do **cross-validation** and look at performance on test / validation data. So all that classical-asymptotic theory was being developed for a regime nobody using these methods actually lives in. Net effect: kept statisticians from working on more suitable models and on more exciting problems.

Ben strongly endorses this part:

> "Too many people dismiss these kinds of - the [things] machine learning people have figured out - instead of trying to say, hey, we could actually make this formal and understand what's going on here. They just say, oh yeah, you guys and your magic and you make these big models, but really they don't work as well as just the standard thing, so who cares right? Being that dismissive I think is a big mistake, because often they stumble across really good, clever ideas that can be very useful and go well beyond what we could do with simple models. And they don't understand what's going on. But we could try to help them."

The flip side of the same coin: a lot of the vocabulary the course will spend the semester building (bias, variance, model error, generalization, uncertainty) has meaning for these complex models too, *if we just get at it*. That's the bridge statistical learning is trying to build.

### Cox's reply

David Cox (yes, *that* Cox, of Cox regression) wrote a polite-but-pointed dissent. Money quote Ben emphasizes:

> "The absolute crucial issue in serious mainstream statistics is the choice of a model that will translate key subject matter questions into a form for analysis interpretation. If a simple model is adequate to answer the subject matter, this does fine. There are severe hidden penalties for over-elaboration."

And the Lord Kelvin attribution Cox repeats: *"Better a rough answer to the right question than an exact answer to the wrong question."*

Ben's pushback on the slogan:

> "Arguably machine learning models are the right answer to the wrong question. And that can be a good thing… Whereas a rough answer to the right question, I think can also be a very bad thing."

A simple model can pick up things that have nothing to do with the real signal, just properties of the data itself. Both directions can fail.

The point of dropping this paper in: the textbook is written by statisticians telling you a particular story. The math is right; the framing has caveats. Ben's job is to give you the combined view.

## Daniela Witten quote

He pulls in a Witten one-liner (paraphrased from a slide):

> "When you raise money, it's AI; when you hire people, you hire machine learning people; and when you do work, it's logistic regression."

He thinks it was true ~5–7 years ago; not really anymore. Mentioned because departments have been renaming themselves (machine learning → AI) for funding reasons, but **logistic regression** is still very much in this course (will be covered in module 4 territory).

## The aim, formalized

In the supervised setting we assume

$$Y = f(X) + \varepsilon$$

where $X = (X^{(1)}, \ldots, X^{(p)})$ are the predictors, $f$ is the underlying function, and $\varepsilon$ is unobserved noise. Standard nice-to-have assumptions when you can make them: $\varepsilon$ has mean zero and is independent of X. (Worth flagging as the same independence assumption that L05 will hammer as the *thing you'll break*.)

> "Our goal is typically to estimate f. And sometimes things about epsilon."

Picture: data points scattered around the curve $f(x)$; the vertical distance from each point to the curve is its $\varepsilon_i$. (He sketches this on the board, same picture that comes back in [[least-squares-and-mle]] in L05.)

## Closing example: ad spend → sales

Last slide of the day was the ISL ad-spend example: y-axis = sales, x-axis = ads bought on TV / radio / newspaper (separately). All three show an upward trend, but slopes and uncertainties differ.

His critique of the plotting choice: x-axes should really be **money spent** on ads, not number of ads, since that's the decision variable, so the slope becomes "benefit per dollar spent" and you can compare across media. Newspaper has the noisiest fit (the εᵢ's are big) but if newspaper ads are basically free, you'd still buy them. Foreshadows that **slope size, fit quality, and the practical decision interact**, the same trio that becomes statistical-vs-practical-significance in L05.

> "I would not recommend buying any ads for the newspaper. I don't think anyone reads it."

Ran out of time mid-deck. Continues tomorrow at noon ([[L03-statlearn-2]]) with the second half of `2StatLearn.1.md` and into the bias–variance trade-off proper.
