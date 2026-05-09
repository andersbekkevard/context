---
concept: supervised-vs-unsupervised
module: 01-intro
lectures: [L01, L02, L21]
isl-ref: 2.1.4
exercises: []
related: [statistical-learning, prediction-vs-inference, principal-component-analysis, k-means-clustering, hierarchical-clustering, cross-validation]
tags:
  - concept
  - module/01-intro
aliases:
  - supervised vs unsupervised
  - supervised learning
  - unsupervised learning
---

# Supervised vs unsupervised learning

The first organizing axis of statistical learning: **do you have a $Y$ to aim at, or not?** Supervised problems have a target; unsupervised problems just have $X$ and a fuzzy "find structure" goal. The prof's distinctive emphasis: many "unsupervised" problems are really **supervised in disguise** (LLMs as the canonical example), and pure unsupervised is **"dangerous statistics"** without a downstream check.

## Definition (prof's framing)

> "Supervised, you kind of know what you want to get out of it. Unsupervised, you're sort of fiddling around and seeing what falls out." - [[L01-intro]]

> "Supervised: labels to train against. Unsupervised: no labels, no goal." - [[L21-unsupervised-1]]

The crisp formal version: in **supervised** learning you observe $(x_i, y_i)$ pairs and want to estimate $f$ in $Y = f(X) + \varepsilon$. In **unsupervised** learning you observe just $X$ and want to find structure (groups, low-dimensional summaries, dependencies among the $X_j$'s) without any external target.

> "Importantly, it's supervised in the sense that you know what you want to classify." - [[L01-intro]]

The defining feature isn't whether you have data, it's whether the data has a *label / response variable that says what right looks like*.

## Notation & setup

- **Supervised**: data is $(x_1, y_1), \ldots, (x_n, y_n)$ with $y_i$ either continuous (regression) or categorical (classification). The course's modules 2–9 and 11 are entirely supervised.
- **Unsupervised**: data is $x_1, \ldots, x_n$ with no $y$. The course's module 10 is unsupervised (PCA, k-means, hierarchical clustering). NN module touches it via auto-encoders / pretraining (briefly).

The supervised vs unsupervised axis is **orthogonal** to [[prediction-vs-inference]]; the two together generate the four conceptual cells (prediction-supervised = forecasting, inference-supervised = econometrics, "prediction-unsupervised" doesn't really exist, inference-unsupervised = exploratory clustering / PCA).

## Insights & mental models

### Most of the course is supervised

The prof states this directly: most of statistical learning, and *most of this course*, is supervised. Module 10 is the only fully-unsupervised module; PCA also shows up earlier (M6 PCR, M8 trees-as-features) but always feeding into a downstream supervised model.

### Unsupervised is "dangerous statistics"

A recurring prof warning, escalating from L01 to L21:

> "Unsupervised is often a bit dangerous because you're heading towards the land of like bad statistics or bad science. You're kind of exploring. You don't really know what's going on… you can say this is highly significant, but really kind of it's only significant in how you did it. So it's dangerous." - [[L01-intro]]

> "Unsupervised methods are going to be more subjective because you're not training for a specific goal… whoever does is going to get a different result. And it's very hard to assess results. I think this is undersold here… and has led to many issues." - [[L21-unsupervised-1]]

> "If you explore and explore and explore, eventually you will find something no matter what. But then at that point is your finding interesting or significant because you know it was inevitable that you find something?" - [[L21-unsupervised-1]]

Mechanism: with no $y$ as ground truth, you have no objective measure of success. Run enough analyses on the same data and *something* will look "significant", but the p-values are meaningless because they ignore the search you did.

> "What many people do, some at least, is that they kind of lie and they say, oh, this is statistically significant. And they do some tests, ignoring the fact that they've just explored the data… All of the statistical arguments for your finding tend to fall apart after you've done all this subjective exploring. But people don't know that and they just lie." - [[L21-unsupervised-1]]

### When unsupervised is OK: tied to a downstream supervised task

> "When it's tied to a bigger goal you can validate. Example: cluster shoppers, then evaluate whether the clusters lead to better recommendations. This brings us back to cross-validation and evaluation of a model on a validation set." - [[L21-unsupervised-1]]

The healthy pattern: **unsupervised exploration → hypothesis-driven supervised study**. PCA / clustering identifies candidate structure; a follow-up supervised experiment validates it.

### Many "unsupervised" problems are supervised in disguise

This is the prof's distinctive point and it's worth catching:

> "Often what looks like unsupervised is really supervised in disguise. Best example: large language models. The training task is just predict the next word, perfectly supervised, perfectly defined. But the model ends up 'knowing' who-did-it in a mystery, knowing program syntax, knowing therapy. It was just trained to predict the next word. What's the next word a therapist would say?" - [[L02-statlearn-1]]

The reframing trick: even if your data has no obvious $y$, ask whether you can construct one from the data itself (next word, next pixel, masked token, click vs no click). If yes, it's supervised, and supervised methods are stronger because the loss is well-defined.

### Why you'd ever do "real" unsupervised

When the data is high-dimensional and you genuinely don't know what to look for. The prof's own neuroscience work, cancer subgrouping, market segmentation, search engine indexing, all cases where labels don't exist or aren't trustworthy. The L21 reframing: **unsupervised methods exist to summarize / visualize high-dim data** ("Who can think in eight dimensions?"), not because they're a better statistical paradigm.

### Difficulty assessing performance is intrinsic

> "Difficulties of the genuinely unsupervised: hard to know how well you're doing, hard to know when you're done." - [[L02-statlearn-1]]

This is the deepest reason for the danger framing. Supervised methods get test MSE / AUC / cross-validation as a clean objective measure. Unsupervised methods get… "does this clustering look reasonable?", which is subjective.

## Exam signals

> "I like this kind of question … it's conceptual, but you also don't have to write a whole book, you just have to know which words to fill in correctly." - [[L27-summary]]

(Q1 of the 2025 paper, walked through in [[L27-summary]], is a fill-in-the-blank using exactly the vocabulary "regression," "classification," "prediction," "inference", and supervised vs unsupervised is the same family of conceptual labels. Expect at least one tag-this-scenario question.)

> "Unsupervised methods are going to be more subjective because you're not training for a specific goal… I think this is undersold here… and has led to many issues." - [[L21-unsupervised-1]]

The prof's "danger" framing is a likely T/F question stem ("unsupervised methods give well-calibrated p-values", F).

## Pitfalls

- **Calling something unsupervised when it's supervised.** LLMs are the standard trap, they look unsupervised because there's no human-labeled $y$, but the next-word loss is fully supervised. Ditto autoencoders (the input is the target), masked-language models, click-prediction.
- **Treating an unsupervised result as having a "right answer."** Different distance metrics give different clusterings; different number of clusters / PCs gives different summaries. There's no test MSE to break ties.
- **Reporting p-values from a clustering result on the same data you used to find the clustering.** Classic dishonest-statistics move; the prof flagged it twice. Honest version: cluster on one dataset, validate the clusters on a held-out dataset (or supervised downstream task).
- **Confusing "no $y$" with "no structure."** Unsupervised methods *do* find real structure in many cases; the danger is in claiming statistical significance, not in the existence of the structure itself.

## Scope vs ISLP

- **In scope:** the supervised/unsupervised distinction, recognizing it in real-world descriptions, the LLM "supervised in disguise" reframing trick, the danger / subjectivity of unsupervised inference, when unsupervised is OK (downstream supervised validation).
- **Look up in ISLP:** §2.1.4 ("The Trade-Off Between Prediction Accuracy and Model Interpretability" → §2.1.4 / §2.1.5 introduce supervised vs unsupervised). For unsupervised methods themselves, §12 (clustering, PCA). ISLP is brief on the danger framing; the prof's L21 transcript is the better source for the "subjective / unsupervised methods are dangerous" angle.
- **Skip in ISLP:** the historical name-checks and the long list of supervised methods in §2.1; the supervised-vs-unsupervised dichotomy itself is one paragraph.

## Exercise instances

(None. The manifest assigns no exercises directly to this atom. The dichotomy is exercised inside the unsupervised module: PCA, k-means, hierarchical clustering all live in module 10 and have their own atoms with exercise references.)

## How it might appear on the exam

- **Fill-in-the-blank / tag-a-scenario** (Q1-style from 2025): paragraph describes a problem; you label it supervised or unsupervised. Common tells: "no response variable," "groups, or clusters," "patterns" → unsupervised. "Predict whether," "classify," "estimate $Y$" → supervised. Per [[L27-summary]] this is a high-likelihood question type.
- **T/F traps**:
  - "Unsupervised methods give well-calibrated p-values": F (the prof's danger framing).
  - "An LLM is unsupervised": F (next-word prediction is supervised).
  - "PCA requires labeled data": F (purely unsupervised).
  - "K-means needs you to specify K in advance": T (often dropped in as a contrast question).
- **Short answer**: "give an example of an 'unsupervised' problem that is really supervised", e.g. LLM next-word, autoencoder reconstruction, click prediction. Or "when is unsupervised analysis defensible?", when tied to a downstream supervised validation.
- **Method-classification questions**: list of methods (linear regression, logistic regression, KNN classification, KNN regression, LDA, QDA, ridge, lasso, PCA, k-means, hierarchical, NN), tag each as supervised or unsupervised. Standard warm-up question; PCA and the two clustering methods are unsupervised, everything else supervised.

## Related

- [[statistical-learning]]: parent framing; supervised/unsupervised is one of the two organizing axes
- [[prediction-vs-inference]]: orthogonal axis; combine to get the conceptual cells
- [[principal-component-analysis]]: the canonical unsupervised method (visualization / dim reduction)
- [[k-means-clustering]]: unsupervised partition; needs $K$ specified in advance
- [[hierarchical-clustering]]: unsupervised, returns a dendrogram instead of a flat partition
- [[cross-validation]]: the supervised-side antidote to "how do I know this works?"; CV is exactly what unsupervised lacks
