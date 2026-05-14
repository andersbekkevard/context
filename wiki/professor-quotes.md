---
title: Professor Quotes — Greatest Hits
description: Funny, outrageous, unexpected, and memorable asides from the TMA4268 professor, compiled from the lecture wiki.
---

# Professor Quotes — Greatest Hits

A curated compilation of the most memorable off-script moments from TMA4268 lectures: vivid metaphors, political asides, strong opinions, self-deprecation, and the occasional historical scandal. All quotes sourced from the [[lectures]] (with raw transcripts in `transcripts/` as backup).

> [!note] Format
> Each quote links to the lecture file with line number. The wiki preserves the prof's voice verbatim where it matters.

---

## 1. Political & ideological metaphors

The signature genre — turning statistical methods into political philosophy (and Scandinavian cuisine).

### Lasso = capitalist, Ridge = socialist

> "Lasso is like very capitalist. Just shoot everyone, let all the poor people die, let the one rich guy win. And L2 is more socialist."

— [[L13-modelsel-2]] · regularization geometry

> "Lasso = capitalist (one rich guy wins, the rest die). Ridge = socialist (averages over correlated parameters, no one dominates). Elastic net = the centrist compromise."

— [[L13-modelsel-2]]

> "And it's also why we try not to have extreme governments."

— [[L13-modelsel-2]] · on why elastic net exists

### PLS = Swedish meatballs

> "PLS often performs no better than ridge regression or PCR but it's Swedish, so it's like they're meatballs — they're not better but they sound good."

— [[L15-modelsel-4]] · Partial Least Squares verdict, with a follow-up that he's never really used it

### Fisher had opinions

> "this crazy data by Fisher, was trying to prove race stuff"

— [[L09-classif-3]] · the iris dataset

> "He was a big eugenicist… he actually was the editor of the journal of eugenics. Anyways, so he was a eugenicist."

— [[L01-intro]] · Fisher (the data scientist, not the politician)

> "I'm pretty sure Fisher, if he had a better computer, would have done bootstrap instead of all the distribution stuff."

— [[L11-resample-2]]

---

## 2. Vivid metaphors & analogies

### The bootstrap is photographic magic

> "It's like taking a picture and then taking a picture of the picture 100 times and then averaging them, and somehow it's better than the original picture. It doesn't sound like it makes sense, but it does."

— [[L11-resample-2]]

### Ridge as tug-of-war

> "RSS… wants to make those parameters beta whatever it can to fit the data. So it's pulling them away from zero… And then this thing [the penalty] pulls them back to zero. So one is pushing away, one is pulling back. So you have this tug of war on the betas."

— [[L12-modelsel-1]]

### Bootstrap etymology

> "a long time ago they used to have in the standard elementary school textbook the example of proving using physics that you can't pull yourself up by your own bootstrap."

— [[L11-resample-2]]

### Engineering vs medicine standards

> "this blob to people, they would have just laughed at me. That's not, they don't, you can't build an engine with a blob like that. But it does show some correlation… in medicine, this is like fantastic. This is a seminal paper."

— [[L01-intro]]

### AI as the world's only therapist

> "It crazy to think that you have like one AI model out there giving advice to millions of people on how to live their life. Like if there was one therapist that was like the therapist of the world, I would be concerned that everyone's getting like direction from one person… that's some scary shit."

— [[L01-intro]]

---

## 3. Self-deprecation & jabs at the field

### Algebra in public

> "I stupidly said these two would cancel. Obviously that's not true… When I was doing my PhD, one of my co-authors said he doesn't make a habit of doing algebra in public. I've heard his voice in my head a million times. It's a bit of a fool to do public algebra — your brain shuts off, you look like an idiot."

— [[L09-classif-3]] · after a board error

### On the Hitters dataset

> "I don't know why this book always talks about money. I'm going to write these guys and tell them, please, new topics. Not everything's about money. They're American, probably."

— [[L18-trees-2]] · ISLP's obsession with baseball salaries

### On terminology he dislikes

> "It's often called the design matrix. The data. Never understood why. It's not really a design of any kind. But it's what people call it."

— [[L06-linreg-2]]

> "lots of dollar signs and percent signs are stupid… this notation sucks, but whatever, I hope you like it"

— [[L06-linreg-2]] · on R syntax

> "these stars are this ridiculous notion of significance"

— [[L04-statlearn-3]] · on p-value stars

### On the lasso authors

> "Tibshirani, Hastie — most of their research, if you look at what they've done, is all about lasso. They love this thing. It's like their favourite topic."

— [[L13-modelsel-2]]

### On statisticians and KNN

> "Statisticians don't love KNN"

— [[L08-classif-2]] · as a section heading

### On computational pain

> "Slow as shit for p big — or even like impossibly slow. Like just never going to happen."

— [[L12-modelsel-1]] · best subset selection

---

## 4. Strong opinions ("I hate X", "Never do Y")

### On the Credit dataset

> "I really hate this kind of data… they're specifically pointing out ethnicity, like to see if ethnicity leads to more debt and credit cards. I hate credit cards… Anyways, sorry, but they're looking at predicting the balance."

— [[L06-linreg-2]]

### On outliers

> "Whenever you see an outlier just figure out like did you screw something up. Why is it there? Don't just throw it away. Use it as a way to understand your data better."

— [[L08-classif-2]]

### On absurd p-values

> "These numbers are of course ridiculously small. Never write that in an article, people will laugh at you, because a probability of negative 200 is, you know, more likely we don't exist."

— [[L08-classif-2]]

### On hypothesis testing

> "It's like we're assuming innocence so that we can prove guilty… We don't prove anything in statistics… at least I've never proved anything with data."

— [[L06-linreg-2]]

### On adjusted R²

> "I would never really, if it was up to me, we wouldn't include it."

— [[L06-linreg-2]]

### On the F-test

> "I don't really care that much about it."

— [[L06-linreg-2]]

---

## 5. Bizarre / unexpected examples

### Polynomials of degree 50,000

> "enter this ridiculous region of like a degree 50,000 or 100,000."

— [[L04-statlearn-3]] · benign overfitting

> "we always learn that if you go to higher degree polynomial and you have too many parameters that it's going to get wiggly and scary and it's going to explode and it's going to be really bad. But actually when you go really really big — ridiculously large — then it does something else, right? It starts getting these solutions that are actually smooth."

— [[L04-statlearn-3]]

### On bioinformatics datasets

> "those matrices are very weird… they have a ton of genes and just a few samples, and they're trying to find something in there. Good luck."

— [[L06-linreg-2]]

---

## 6. Philosophical asides with attitude

### Regularization is sneakier than it looks

> "You think you're penalizing just the number of parameters or shrinking them down to smaller numbers, but really you're allowing the model to generalize better to data you haven't seen. How weird is that?"

— [[L13-modelsel-2]]

### Bootstrap is too cool

> "It feels like magic. It feels like you shouldn't be able to do it… it's too fancy, it's too cool. Like, how can you reuse the data and somehow make a better model? It's weird, right?"

— [[L11-resample-2]]

### On bias-variance being counterfactual

> "We don't typically know what the bias and the variance are. These are just things we think about. If we knew what the bias was, we wouldn't be fitting a model — we would just use the true model and we wouldn't do any statistics, because that would be stupid."

— [[L13-modelsel-2]]

### On statistics generally

> "The annoying thing about statistics is you don't really know when to use what."

— [[L09-classif-3]]

### The data is the model

> "Your best model for the real world… is the data itself."

— [[L11-resample-2]] · the bootstrap philosophy

---

## 7. Meta / teaching asides

### On the source of his slides

> "I totally just stole all their stuff… most of the content of the course has been like just blatantly robbed from them [Stefanie Muff and Sara Martino]."

— [[L01-intro]]

### On his pacing

> "I recommend 2x [speed]."

— [[L01-intro]]

> "I don't know how in the previous years they went through this so quickly. I think maybe I just talk really slowly or I get distracted."

— [[L04-statlearn-3]]

### On R

> "I don't like R, well I don't have anything against it per se… I feel uncomfortable having that many libraries in my computer."

— [[L01-intro]]

### On his title

> "They didn't want to call me a mathematician and they didn't want to call me a statistician either."

— [[L01-intro]] · on being labeled "data scientist"

### On cheating

> "now you know now you can cheat right because AI is so good at programming, it bonkers."

— [[L01-intro]]

### On effort

> "I would recommend struggling a bit before jumping to the answer."

— [[L01-intro]]

---

## 8. Exam-flagging catchphrases

The most repeated phrases in the course — when you hear these, write it down:

- **"That would be a typical exam question."**
- **"This is an exam question."**
- **"I might ask…"** (usually followed by a harder conceptual variant)

Appearances across [[L05-linreg-1]], [[L06-linreg-2]], [[L09-classif-3]].

---

## Honourable mentions

- **Most ideological**: lasso/ridge as capitalism/socialism ([[L13-modelsel-2]])
- **Most Scandinavian**: PLS as Swedish meatballs ([[L15-modelsel-4]])
- **Most self-deprecating**: the algebra-in-public confession ([[L09-classif-3]])
- **Most alarming history**: the Fisher / eugenics aside ([[L01-intro]])
- **Best metaphor**: photo-of-a-photo bootstrap ([[L11-resample-2]])
- **Most off-the-rails**: the "one therapist for the whole world" AI rant ([[L01-intro]])
- **Best ISLP jab**: "please, new topics. Not everything's about money." ([[L18-trees-2]])
