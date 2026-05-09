---
lecture: 1
date: 2026-01-05
module: 01-intro
title: Introduction
slides: modules/1Intro/1Intro.md
topics:
  - statistical-learning
  - supervised-vs-unsupervised
  - prediction-vs-inference
  - regression
  - classification
  - clustering
  - linear-discriminant-analysis
  - hierarchical-clustering
tags:
  - lecture
  - module/01-intro
aliases:
  - Lecture 1
---

# L01 — Introduction

First lecture: course mechanics, who the prof is, and a high-level tour of what statistical learning is. After admin the prof walks through the three problem types the course will cover — **regression** (Framingham blood pressure study), **classification** (Fisher's iris data with [[linear-discriminant-analysis]]), **unsupervised** (rat gene-expression clustering) — using each as a worked example for chapter 1 of ISL. No technical depth yet; the goal is framing.

## Key takeaways

- Course is taught by a new prof this year (data scientist, ex-engineer, ex-neuroscientist) — most slides inherited from Stefanie Muff and Sara Martino, expect his own emphasis to drift in over the semester.
- Exam **2026-05-18, 09:00**, open book: ISL PDF + book + **one A4 page of notes** allowed. Retake may be oral.
- Two **compulsory exercises** — must score ≥ 60% on each to be eligible for the final exam. Group of max 3, R Markdown submission (PDF + Rmd). First after module 5 (resampling), second after module 10 (unsupervised) — the second is a **project you propose yourselves**.
- Scope rule for the prof's own slides: "you expected to read the book... the parts that we cover" — book content the lectures cover is fair game; in principle even unspoken book material *could* be tested but he'll be cautious about it.
- Three umbrella problem types for the whole course: **regression**, **classification**, **unsupervised**. Supervised vs. unsupervised and prediction vs. inference are the two orthogonal distinctions to keep in your head.
- Prof's framing of the field: **"statistical learning was a way for statisticians to get in on machine learning"** — same models as ML but with the statistician's care about uncertainty, bias-variance, distributional assumptions.

## Course mechanics

### Lineage and acknowledgments

The course was originally organized by **Mette Langaas**, who passed away early. For the last few years it has been taught by **Stefanie Muff** and more recently **Sara Martino**. The current prof inherited their slide deck wholesale — "I totally just stole all their stuff... most of the content of the course has been like just blatantly robbed from them. I'm of course going to add my own stuff in there because I'm a different person, but they thought this out very well."

So the slide deck reflects the older instructors' framing; expect the prof's own emphasis to drift in over the semester (he flagged he'll deviate most from the old material in the **neural networks** module — "I just thought like, I wanted to say it a different way").

### Learning materials

- **Main text:** James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning*. The simpler companion to Hastie/Tibshirani's *Elements of Statistical Learning* (the names are confusingly similar).
- **Hastie/Tibshirani YouTube series** (Stanford online) — third source if both prof and book confuse you.
- Lecture notes: PDF on Blackboard, prof writes annotations in red on iPad. Will upload "every week or two." Built on the existing slide PDFs but rebuilt in LaTeX with denser layout — "my whole slide deck has fewer slides, but more shit on it."
- **Self-study R course** for module 1 — register with Feide/NTNU account, link on Blackboard. Done before next week.
- All material organized on Blackboard under "modules" — one page per module with slides, videos, exercises, and solutions.

### Prerequisites

- A first stats course covering linear regression, distributions, likelihood (everyone in the room confirmed they have this).
- Some scripting (R or Python). Prof on AI use: "now you know now you can cheat right because AI is so good at programming, it bonkers."
- Linear algebra is assumed but not super heavy.
- **Strong personal recommendation: take optimization.** "If I was in charge of the program, I would put it as a requirement... it's like one of the key, there's only a few key technologies that have really enabled the current explosion in power that we have with our tools. And one of them is all this optimization stuff." Not required for *this* course but central to making models actually work in stats and ML.

### Overlap with other courses

- The **machine-learning** courses overlap mostly on multiple linear regression. The prof dismissed case-based reasoning as "boring."
- This course's distinctive emphasis: **statistical theory** of why methods work, not just engineering.

### Schedule and format

- Two 2-hour lectures per week: **Monday 08:00** and **Tuesday 12:00**. Two weeks during the semester replaced by compulsory-exercise work.
- Exercise sessions Mondays — TA-led, may not run early in the semester (no TA assigned yet, funding issues).
- Lectures recorded to Panopto (linked from Blackboard). Prof speaks slowly: "I recommend 2x."
- Bring a laptop — many examples are runnable live.

### The 12 modules

The course is split into 12 modules. Per the prof's walk-through:

1. **Intro** (this lecture + the R self-study) — chapter 1 of ISL.
2. **Statistical learning** — what it is, the main ideas.
3. **Multiple linear regression**
4. **Classification**
5. **Resampling** — first compulsory exercise hits after this module.
6. **Model selection**
7. **Regularization**
8. **Non-linearity**
9. **Tree-based methods** ("very cool")
10. **Unsupervised methods** — second compulsory exercise (the project) hits after this module.
11. **Neural networks** — where the prof expects to deviate most from previous instructors.
12. **Review.**

### R vs. Python

The course is built on R and parts of the exam will assume R. The prof himself prefers Python ("I don't like R, well I don't have anything against it per se... I feel uncomfortable having that many libraries in my computer") but isn't changing the course this year.

> [!note] Prof on the R emphasis
> "Honestly the ideas are not R specific. Like this is this should not be a programming course. This is a course on using these things and understanding these things."

### Compulsory exercises

- **Two of them.** First after module 5 (resampling) — homework-style. Second after module 10 (unsupervised) — **project you design yourselves.**
- **Pass mark = 60%.** Pass both → eligible for final exam.
- Groups of max 3, one submission per person but list all names. Exact copies allowed within a group. You can also work alone if you don't like groups.
- Submission format: **R Markdown + PDF.** The prof would be open to Python submissions if he changes the requirement, but for now stick with Rmd. Rationale: he assumes the previous instructors had a good reason ("they're smart, so we'll believe them").
- **One submission attempt only this year** — he toyed with a two-attempt resubmission but decided against it ("it's a lot of work, it's easier for me to just do once").

### Recommended exercises and "sweat equity"

There are also non-compulsory recommended exercises per module, with full solution sets posted. The prof's appeal — even though everything is online and AI-able:

> [!tip] Prof on doing the work yourself
> "I would recommend trying to put in the sweat. It's called sweat equity, where you actually put in the effort to try to learn it... struggling through stuff, it can be very useful. So I would recommend struggling a bit before jumping to the answer."

### Exam

- **2026-05-18, 09:00.** Open book.
- Allowed: ISL as PDF or physical book, **one A4 page of notes.**
- Retake may be **oral** — prof is enthusiastic about oral exams ("I think it's better, honestly... you can really show what you know").

### Reference group

Two-to-three students who meet with the prof a couple of times during the semester to relay feedback (mumbling, illegible writing, content pacing, etc.). Volunteer recommended — "your way of contributing to the future of the course."

## What statistical learning is

The prof gives a deliberately loose definition — "a bunch of different tools to understand data" — then sharpens it by contrasting with classical statistics and with machine learning.

### Contrast with classical statistics

In a classical stats course you **state the model first**: "I assume a Gaussian distribution... and then you argue from your data." In statistical learning, **the data comes first** and you figure out what to do with it. The distributional assumption is often relaxed or absent.

### Origin story — stats's response to ML

Machine learning's roots: **McCulloch & Pitts** and **Frank Rosenblatt** in the 50s — neuroscientists/psychologists modelling learning ("how to model learning, how to understand how we learn through making machines"). Rosenblatt's **perceptron** ("a giant thinking machine and all these mechanical, these very simple neurons, I think had like 250 or something") got Cold-War US government funding aimed at beating the Russians in machine translation. "Eventually it worked. It just took a long time."

The term **machine learning** was coined later by a Bell Labs researcher trying to make a checkers-playing program — "he thought like, oh, there's too many if statements. What if we can just make it learn? He actually made a checkers bot that beat him." ML exploded around image recognition before ChatGPT brought language to the foreground.

ML exploded around image recognition pre-ChatGPT. The prof's read on the rise of statistical learning:

> [!important] Why "statistical learning" exists
> "Statistical learning was a way for statisticians to get in on machine learning... If you take a machine learning course, it's very engineering focused... the statisticians were like, hey, well, our perspectives on your stuff adds to your field. And I think that's where the original book by these authors, Tibshirani and Hastie, really came into play — they took a lot of the current models and ideas from machine learning, and then said like, here's the statistics perspective on these ideas, the bias-variance trade-off, how do you deal with uncertainty, how do you treat these models."

### The two orthogonal distinctions

- **Supervised vs. unsupervised** — supervised = "you kind of know what you want to get out of it"; unsupervised = "you're sort of fiddling around and seeing what falls out."
- **Prediction vs. inference** — same model, two uses.
  - **Prediction:** you don't care if the model is *right*, you care that it forecasts well with calibrated uncertainty. Example: quants and econometricians modelling stocks/Bitcoin — "they don't really care if the model is right. That's secondary. Being right is not as important as being able to predict well and predict with knowing your uncertainty of your prediction."
  - **Inference:** you want to understand the structure — ideally causal, more often correlational. The prof's neuroscience work is inference: "I'm often not really caring about predicting the neural activity. Right. It's useless. The animal's dead." The goal is "an understanding or a development of the science of what's going on."

### A theme of statistical learning

> [!important] Why theory in simple models matters
> "A lot of statistical learning is trying to extend the ideas that have been well established and developed for simple models, like where you can assume Gaussian or we can assume the distribution and everything is known... and now extending it to cases where you're like, ooh, this giant black box, what do we do? So a lot of statistical learning is that. And so for that, you understand the theory often in a simpler case and then extend it."

### Statistical learning vs. data science

The prof is officially a "data scientist" in the department ("they didn't want to call me a mathematician and they didn't want to call me a statistician either"), so he speaks from the role.

Data science is broader, spanning roughly six steps:

1. Formulating a hypothesis
2. Acquiring/scraping data
3. Going from unstructured to structured data
4. Setting up a model
5. Implementing the analysis
6. Interpreting and communicating results

**Statistical learning overlaps the bottom three** (4–6) — the modelling, fitting, and communication parts. The first three (hypothesis, scraping, cleaning) are data-science territory and out of scope for this course.

The distinguishing feature: statistical learning is the statistical perspective on black-box models where the simple 1960s distributional assumptions don't hold.

### Statistical learning vs. machine learning

ML is **algorithmic and engineering-focused** — getting it to work. Statistical learning is theory-focused, but: "of course, many of the things we're going to talk about... are also mentioned or covered in the machine learning course, but probably from a different perspective." Other rigorous angles on ML: optimization theory; geometry/topology in pure math (the prof "dabbles in the topology side").

## Three example problems

The rest of the lecture: one worked example for each of the course's three umbrella problem types. The point is **framing**, not mastery — "you should not have understood very much because I didn't explain them very well."

### Regression — Framingham heart study

Classic dataset on cardiovascular disease (~2500 people). Per subject:

- **systolic blood pressure** — the response variable
- sex (M/F)
- age
- current smoker (binary)
- BMI
- total cholesterol
- current use of antihypertensive medication

The study question: which of these variables predict blood pressure?

First step: a **GG pairs plot** in R (the prof emphasizes you'll use these constantly). Diagonal = histograms per variable, **split by sex** — blue female, red male, two histograms per panel. Off-diagonal = pairwise scatterplots. You inspect for correlations visually before fitting anything.

The blood-pressure response is **right-skewed with a heavy tail** ("kind of has this skewed heavy tail kind of thing"), so they apply a **y → −1/√y transformation**. Result: the transformed response looks roughly bell-shaped — "kind of looks like a bell, kind of looks like a normal distribution." The motive: regression behaves better when residuals are approximately normal, "and it's a good indication that also the model will do well because probably the residuals will also look like that."

> [!note] Transformations as a regression-prep trick
> "These kind of transformations are a common trick to take the data and put it into a form where you can use these regression type models."

After transformation, fit a linear regression. Output gives **estimate**, **standard error**, and **p-value** for each variable. Significant predictors here: **age**, **BMI**, antihypertensive medication (plus the intercept). The blood-pressure ~ age scatter is "this giant elongated blob":

> [!quote] Prof on blob-grade correlations in medicine
> "When I was an engineer, if I presented something like this blob to people, they would have just laughed at me. That's not, they don't, you can't build an engine with a blob like that. But it does show some correlation... in medicine, this is like fantastic. This is a seminal paper."

A pet peeve: the printed p-values are absurdly small. He wouldn't write tiny p-values in a paper — "I would just say these are like, I would say less than like 10 to the negative 4 or something, just some small number. Just because there's a higher probability that you just left out a more important variable that explains this interaction."

The standard error already accounts for interactions between the variables.

R² is low, which would be unacceptable in engineering but is fine in medicine. This is **inference**, not prediction — doctors will use it but understand "the predictive value is minimal. They'll just say you have risk factors. You should exercise more." We'll come back to [[regression]] properly later in the course (the prof flags he'll go beyond simple linear).

### Classification — Fisher's irises with LDA

The famous Fisher iris dataset. **Three species** of iris, **50 flowers each**, **four variables**: sepal width, sepal length, petal width, petal length.

Aside on Fisher: "He was a big eugenicist... he actually was the editor of the journal of eugenics. Anyways, so he was a eugenicist." Famous data set regardless.

Plot any two variables: **setosa** separates cleanly from the other two; **versicolor** and **virginica** overlap heavily on the chosen pair. Fit [[linear-discriminant-analysis]] — get a **straight-line decision boundary** that misclassifies just one or two points. Fit **quadratic discriminant analysis** instead — boundary becomes a curve, slightly better.

Punchline on **feature choice**: the variables shown in the worked plot are actually the *worst pair* — pick a different two (visible from the pairs plot) and the classes separate trivially. "If you just use the other parameters, you do much better." This is the value of pair-plots before classifying — they tell you which features are informative.

> [!important] Defining property of supervised
> "Importantly, it's supervised in the sense that you know what you want to classify."

This is the prototype for the whole [[classification]] section of the course: how do you carve up feature space using the variables to get the labels you want.

### Unsupervised — rat gene expression

A famous NTNU paper (continuation of an earlier *Science* paper). Rats selected for **high vs. low running capacity** — athletic rats vs. couch-potato rats. Goal: find which gene transcripts distinguish the two groups, with no prior model dictating which transcripts matter.

Two-step pipeline:

1. Filter to transcripts **significantly related** to running capacity (a univariate-significance pre-screen).
2. Compute distances between the filtered transcripts and run **[[hierarchical-clustering]]** (covered later in the course) — clusters reveal which transcripts group together for athletes vs. couch potatoes.

Defining property of unsupervised: **no response variable.** "We didn't know, or they didn't know ahead of time, what were the transcripts. They wanted to just have it fall out."

> [!warning] Prof on the dangers of unsupervised work
> "Unsupervised is often a bit dangerous because you're heading towards the land of like bad statistics or bad science. You're kind of exploring. You don't really know what's going on... you can say this is highly significant, but really kind of it's only significant in how you did it. So it's dangerous."

The healthy usage pattern: unsupervised exploration → followed up by **hypothesis-driven supervised study**.

Other domains where unsupervised is standard: **cancer subgrouping**, **neuroscience**, **online shopping recommendations** (search engines link products via unsupervised methods — "they're really amazing").

## Coda

End-of-lecture aside on AI: prof claims (per a newspaper article, not peer-reviewed) the **#1 use of consumer LLMs in 2024-25 was therapy** — life advice, processing feelings.

> [!quote]
> "It crazy to think that you have like one AI model out there giving advice to millions of people on how to live their life. Like if there was one therapist that was like the therapist of the world, I would be concerned that everyone's getting like direction from one person... think about that next generation of model people and manipulate us like there's no tomorrow. That's some scary shit."

Next steps for the week: do the self-study R course, no Tuesday lecture this week, exercises start next week (assuming a TA arrives).
