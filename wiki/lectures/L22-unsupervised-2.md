---
lecture: 22
date: 2026-04-14
module: 10-unsuper
title: Unsupervised Learning 2 (Clustering)
slides: modules/10Unsuper/10Unsuper.md
topics:
  - clustering
  - k-means-clustering
  - hierarchical-clustering
  - dendrogram
  - linkage
  - dissimilarity-measure
  - euclidean-distance
  - correlation-distance
  - standardization
  - neural-networks
  - feedforward-network
  - activation-functions
  - deep-learning
tags:
  - lecture
  - module/10-unsuper
aliases:
  - Lecture 22
---

# L22: Unsupervised Learning 2 (Clustering)

The prof closes module 10 by walking through [[k-means-clustering]] and [[hierarchical-clustering]] (agglomerative, with dendrograms, linkage choices, and the standardization warning), then takes a short break and pivots into module 11. The module-11 portion is a long history-of-AI tour (McCulloch & Pitts → Rosenblatt → AI winter → backprop → SVM winter → ImageNet/AlexNet → transformers/scale) plus the first sketch of a [[feedforward-network]] (input/hidden/output layers, weights, biases, activation functions). The clustering part is canonical Module 10 material; the NN portion is preview-only. The prof explicitly said the rest of the module-10 slides would be revisited "after Module 11" because they are just reinforcing examples.

## Key takeaways

- **K-means** partitions data into $K$ disjoint clusters by minimizing within-cluster sum of pairwise squared Euclidean distances. The clusters $C_k$ satisfy $\bigcup_k C_k = \{1,\ldots,n\}$ and $C_k \cap C_{k'} = \emptyset$. "Each point only goes to one cluster and they have to go to a cluster."
- **The K-means algorithm**: random init → compute centroid of each cluster → reassign every point to its nearest centroid → repeat until convergence. Converges to a **local minimum**, "this is often why people don't like K-means." Common workaround: rerun many times and pick the best, or **ensemble** the runs by counting co-cluster frequency.
- **K is a hyperparameter you must pick**: "hyperparameters are death." Sometimes a feature, not a bug (you may *want* exactly 4 groups for downstream interpretability, as in the prof's animal-behavior example).
- **Hierarchical clustering (agglomerative)** starts with every point in its own cluster, repeatedly merges the closest two clusters, gives you a **dendrogram** that you can cut at any height to read off any number of clusters. Naturally **nested**, unlike K-means.
- **Distance between sets ≠ distance between points**: that's the **linkage**. Three options: complete (max pairwise), single (min), average (mean). Average is the most common; "I don't think single linkage is particularly common."
- **Distance metric matters**: Euclidean vs **correlation** distance ($1 - \rho$) measures *different things*. Euclidean groups infrequent shoppers together; correlation groups people with similar preferences.
- **Always scale your variables** for both PCA and clustering. If one feature is in centimeters and another in nanograms, the bigger-scale one will dominate Euclidean distance regardless of meaning. Z-score (subtract mean, divide by sd) is the standard fix.
- **There's no single right answer** for clustering: choice of K (or where to cut the dendrogram), choice of distance, choice of linkage are all decisions. "Both annoying and a benefit if you want to look at it that way."
- **Module 11 preview**: NN history is fair game only as "what key ingredients are there in machine learning"; the prof will not ask history-trivia questions. The fundamental ingredients enabling the current AI boom are **algorithms (backprop, transformers), compute (GPUs/TPUs), and data (ImageNet, Wikipedia)**. Arguably compute and data have mattered more than algorithmic novelty.
- **Feedforward network**: input layer → hidden layer(s) → output layer, no loops, directional flow. Each neuron computes $z = a(\beta_0 + \sum_j \alpha_j x_j)$, a weighted sum (just like regression) plus bias, passed through a nonlinear **activation function**. The nonlinearity is essential.

## Module 10 wrap: from PCA to clustering

The prof reframes the whole unsupervised module before diving in:

> "Sometimes you don't want to use labels. Sometimes you don't want to specifically try to predict a $y$, right? ... Often this is a form of a visualization or mining or seeing what sticks. So many times this is accompanied with very bad science and bad statistics, some cheating, but it also can lead to breakthroughs."

This sits on the **explore** side of the explore/confirm dichotomy in scientific discovery. Last lecture covered [[principal-component-analysis|PCA]], the standard "first" dimensionality-reduction tool. There are "hundreds, thousands, infinite" other dimensionality-reduction methods, but they all share the goal of compactly representing high-dimensional data.

> "Whereas dimensionality reduction tries to reduce, collapse the dimensions, clustering tries to go the other way and sort of cluster together things."

Two clustering approaches today: **K-means** and **hierarchical clustering**. "There's about a billion other versions of these things."

## K-means clustering

### Why start here

> "K-means clustering is definitely a very standard approach. It's kind of arguably the easiest one, so it's a good one to start with, both in terms of learning and in terms of also when you're using it, because mostly whenever you're doing an analysis, you don't immediately jump for the big guns... Occam's razor is good. Keep the big guns for when you need them."

### The setup

Partition the data into $K$ disjoint clusters $C_1, \ldots, C_K$, sets of indices satisfying:

$$\bigcup_{k=1}^K C_k = \{1, 2, \ldots, n\}, \qquad C_k \cap C_{k'} = \emptyset \text{ for } k \neq k'.$$

> "It's a cool way of saying each point only goes to one cluster and they have to go to a cluster."

You don't want $K$ too big. If $K = n$, each point is its own cluster and you've done nothing.

### The objective

You need a metric to score an assignment. K-means minimizes **within-cluster variation**, specifically the sum of pairwise squared Euclidean distances within each cluster:

$$\min_{C_1,\ldots,C_K} \sum_{k=1}^K \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2.$$

Squared rather than just Euclidean because "there's no reason to take the square root, it just an extra step that won't get you anything ... it won't change the goal or won't change who wins or how they win."

> "If you tried to group this one with this one, this score would be bullshit, right? It would blow up because it would try to put a point that's far away from all the other ones."

Euclidean isn't sacred; you can swap in another metric and the algorithm changes accordingly. The prof name-checks his favorite, the **Wasserstein distance** (a.k.a. Earth Mover's distance), purely "because it sounds cool." It's not on the curriculum.

> "Euclidean distances often will suck in high-dimensional spaces because of the curse of dimensionality makes all the points close together, whereas other distances are going to be less prone to that."

### Why we don't brute-force it

There are $K^n$ partitions. With $n = 1000$ and $K = 10$ that's $10^{1000}$. "Obviously you don't want to try everything."

### The algorithm

1. **Randomly assign** a number from 1 to $K$ to each observation.
2. **Iterate**:
   a. For each cluster, compute the **centroid** (the mean position of its points).
   b. **Reassign** each observation to the cluster whose centroid is closest.
3. **Stop** when assignments stop changing (or centroids stop moving).

> "It's not clear that this will lead to a fixed point just from looking at it, but it does ... we're not going to prove it, but it does, at least in practice."

The slide-deck example showed K=3 converging in ~3 iterations from a random start.

### Local minima: the famous K-means weakness

> "It actually rather, especially in certain data sets, this random initialization can really screw you. Right? So if you only initialize once and you run K-means, you get a solution. And then if you rerun it again, you might get a very different solution, depending on the initialization."

> "This is often why people don't like K-means, because of this local minima problem."

Workarounds the prof named:

- **Just use a different algorithm** (e.g. hierarchical, next).
- **Run K-means many times** (a thousand, say) and pick the run with the lowest objective.
- **Ensemble**: across all runs, count how often each pair of points was put in the same cluster. Use *those frequencies* as a new dissimilarity, then re-cluster on that. "Actually that works quite well but it's kind of a funny way of doing things." Connects to the ensembling theme from trees/boosting.

Slide deck shows six K=3 runs on the same data: two pairs converge to "seemingly equivalent" optima with the same score; the others land on worse local minima.

### Choosing K

> "Hyperparameters are death."

Picking K is "often not necessarily something you can determine." But it can also be a feature: maybe you *want* exactly 4 groups because downstream you can only interpret 4. The prof's running animal-behavior research example:

> "Depending on how we run it, we can ... get anywhere from like 20 to 50 [clusters]. And the reality is like 50 different things are just too hard to study. So then we reduce it ... if we had too many clusters, then we couldn't make any conclusions anyways, because I mean, you can't study 50 things at once."

## Hierarchical clustering

### What it gives you that K-means doesn't

Same overall goal (find discrete clusters) but the algorithm produces a **dendrogram**, a tree you can cut at any height to read off any number of clusters.

> "It's a nested set of clustering. You can think of it like that, it has, you know, first everyone is together and then it would split it and then split those splits."

Cutting higher → fewer clusters. Cutting lower → more clusters. Contrast with K-means, where the K=4 solution and the K=5 solution can be totally unrelated:

> "By construction, this approach does [give nesting]."

### Important caveat about reading dendrograms

> "Don't assume, for example, that 9 and 2 are somehow closer together for some reason. Don't interpret this [dendrogram horizontal layout] as being as distances between things. Yeah, it's just a visualization."

The vertical position (height of the merge) carries the information. Horizontal ordering is essentially arbitrary.

### When hierarchical is wrong

Some data has no hierarchical structure (gender vs. nationality, for example).

> "If you're trying to cluster data that doesn't necessarily have a hierarchical structure to it, then probably bad."

The prof's animal-behavior data, by contrast, *does* have nested structure (movements/actions naturally nest into a hierarchy).

> "When K-means or when hierarchical clustering is better or when all these many, many other approaches are going to be better is very variable. And often you need to understand the situation you're in."

### Agglomerative algorithm

> "Agglomerative. If you know the word, like to glom is to like stick on stuff. So agglomerative is like sticky, sticky, sticky."

1. Start with every observation as its own cluster ($n$ clusters total). "Everyone's their own island."
2. Find the **two closest** clusters by some distance, merge them → now $n-1$ clusters.
3. Repeat: each step finds the closest pair of clusters and merges them, reducing the count by one.
4. Stop when there's only one cluster left.

You build the tree from the leaves up.

### Linkage: distance between *sets*

The catch: at step 2 onward, you're measuring distance between **sets** of points, not between points. The rule that extends point-to-point dissimilarity to set-to-set dissimilarity is the **linkage**.

| Linkage  | Definition                                       |
|----------|--------------------------------------------------|
| Complete | Maximum of all pairwise inter-cluster distances. |
| Single   | Minimum of all pairwise inter-cluster distances. |
| Average  | Mean of all pairwise inter-cluster distances.    |

The prof also mentions median is occasionally used. Most common in practice: average. He's skeptical of single linkage:

> "I don't think single linkage is particularly common. I can't think of any case where I've actually used it. I'm sure someone does, but I don't know who or when."

> "Average and complete tend to yield more balanced clusters."

The same data with different linkage choices produces visibly different dendrograms, as the slide deck illustrates.

### Dissimilarity choice (the "distance" inside the linkage)

Same idea as in K-means but elevated: with hierarchical clustering you also get to swap the underlying point-to-point distance.

> "Euclidean is the most common dissimilarity measure or distance measure but there's many others like a correlation one."

**Correlation distance**: typically $1 - \rho$ where $\rho$ is Pearson correlation. So perfectly correlated → dissimilarity 0; anticorrelated → −2; uncorrelated → 1. "Slightly weird but..."

Crucial point: Euclidean and correlation distances measure **different things**. Two observations can be far apart in Euclidean distance while being highly correlated, and vice versa.

The prof's retailer example:

> "Euclidean distance might group together infrequent shoppers, people just don't shop a lot, whereas the correlation distance would find people who have similar preferences."

Pick the metric based on what you want it to be sensitive to. Other options he name-checks: Wasserstein, Manhattan, cosine.

## Standardization (applies to PCA *and* clustering)

> "This is true of so many algorithms that it's almost like a guarantee that you're going to have to do this. So both PCA and clustering, they're sensitive to the metric you're using."

Concrete failure mode: cluster on three measurements, one in centimeters, one in nanograms. The nanogram column is numerically huge for any real-world object, so it dominates pairwise Euclidean distance. The algorithm is "always going to use $x_2$ the most" regardless of substantive importance.

If the units are arbitrary, **z-score** the variables: subtract the mean, divide by the standard deviation. Other normalizations (e.g. divide by max) are possible; pick what fits the data. The prof's grocery example: "the price of socks ... vs. computers." Without standardization, prices and quantities sit on incompatible scales.

## Module-10 summary

The hyperparameters/decisions catalogue:

- **K-means**: pick $K$.
- **Hierarchical**: pick the dissimilarity measure (Euclidean? correlation?), the linkage (average? complete? single?), and where to cut the dendrogram.

> "There's no single right answer, which is both annoying and a benefit if you want to look at it that way."

After the break, the prof said the remaining Module 10 slides "are just repeats of the same stuff. It's just lots of examples and things." Decision: defer them until after Module 11 so there's some delay-based reinforcement.

## Pivot: Module 11: neural networks, deep learning, AI

> "So most of these slides were from Stephanie, some were from Mete, and some of it's, yeah, and other people."

The learning material is "largely in the book" (ISL Ch. 10).

### What's actually examinable from the history

> "I'm not going to ask you like a history question on the test, so don't worry. ... If I was going to ask something like that, it would be more about what key ingredients are there in machine learning, right? And the reason the history explains it is because some ingredients didn't exist before."

So: pay attention to the *ingredients*, not the names and dates.

## A brief history (the bits worth keeping)

### 1940s: McCulloch & Pitts, Hebb

McCulloch (young) and Pitts (old) modeled the **single neuron** as a computational unit. McCulloch was "one of these crazy geniuses" who envisioned much of modern NN architecture, then had a breakdown, burned his research, and died in an asylum. "People think that if he hadn't burned all his shit that we would be like way more advanced than we are now."

**Donald Hebb**: Hebbian learning rule for how neurons form and update connections. The precursor to network-level learning.

### 1950s: Rosenblatt's Perceptron

> "He was thinking of perception. And so in his mind, he wanted something that worked like perception."

Funded by the Navy. The 1958 NYT article quoted by the prof made eerily prescient predictions: a machine that "would be the first device to think as a human", recognize and name people, translate speech, build self-reproducing brains. "In 1958 [it] was kind of bullshit and just kind of fun to think about. But now it's starting to come true, which is no longer bullshit, but scary."

### The first AI winter (~1960s–1970s)

Minsky's paper showing a **single hidden layer can't learn XOR**, a fundamental limitation of single-layer perceptrons. Killed the field for ~20 years. "He showed it was fundamentally limited."

### 1980s: backprop revival

Two big names:

- **Hinton**: **backpropagation** as an efficient learning rule for *multiple* hidden layers. Once you can train >1 hidden layer, you can learn XOR and more. "Suddenly it was back to being useful again."
- **Hopfield**: physics/statistical-mechanics view of NNs, energy/entropy, memory systems.

Both Hinton and Hopfield won the 2024 Nobel for NN work.

### 1989: LeCun and convolutional networks

LeCun applied **convolutional neural networks** to classification, a big splash. Important attribution detail flagged by the prof:

> "Notably, he didn't come up with the idea of convolutional neural networks, and he didn't attribute it to the right person. The right guy's name, I think, is Fukushima, a Japanese guy ... he was modeling in terms of like how the eye works."

### Late-80s/90s: the SVM winter

Support vector machines worked well on hard problems with the compute available. NN conferences flipped from majority-NN to "80% or 90% non-neural networks." Neural networks pushed aside by SVMs and **boosting**.

The prof's diagnosis: it wasn't really algorithmic, it was hardware. "Believe me, I was alive back then in the nineties. My computer was bullshit."

### 2010s: ImageNet / AlexNet

Yearly image-classification competition where everyone had been making "minor improvements," then NN-based **AlexNet** (2012) "just crushed everyone else." Same story arc later with **AlphaFold** for protein folding ("now I think the competition isn't a competition anymore because it's like 90% solved or 98% solved").

The hidden hero: **the dataset**. ImageNet was built by a postdoc whose whole project was to assemble a high-quality, large dataset:

> "If she hadn't had that thought, it probably would have delayed things more because she kind of led to the good quality dataset where you could actually see the benefit of these networks."

### Transformers and the scaling era

> "What's not here is I would say other big things was the transformer model which is the paper by some Google people called 'Attention is everything' or 'Attention is all you need.'"

Transformers enabled scaling. Then ChatGPT was "kind of a bet on scale": train an enormous model, hope it works, otherwise go bankrupt. It worked. Now the field is in a "land of scaling," more parameters, more GPUs/TPUs, more data, more modalities.

The prof's recent-news tangent: the latest Anthropic model, given to non-cybersecurity engineers to test, found "like eight vulnerabilities" in major operating systems. "Crazy stuff. Scary, scary, scary."

### The closing meta-point: examinable

> "Importantly, it's not the corporations that figured this out. It was figured out by lots and lots of science."

Decades of low-funded, low-status research during the AI winters laid the groundwork; recent products are riding those waves plus hardware/data scale.

Key ingredients summary (this is the type of thing he'd ask):

- **Algorithms** (backprop, transformers, conv nets), but often "come third."
- **Hardware** (CPUs → GPUs → TPUs, more RAM, better interconnects), critical and often gating.
- **Data** (ImageNet, Wikipedia, books, multi-modal data).

## What is a neuron / where the brain inspiration came from

Inspiration came mostly from neuroscience. McCulloch, Pitts, Hebb were modeling biological neurons (frogs, squids, "all the same shit, really. We're just big animals"). The neuron is "the fundamental unit of the brain."

> "I don't care if you know the name, the parts of the neurons. I think it's irrelevant."

For this course, model a neuron as a **circle (node)** with **directional connections** to other neurons. Statisticians were largely absent from the early NN history because "they barely understand our simple linear models. How the hell are we going to understand any of this?" They treated it as "just a linear model, nonlinear model" and ignored it.

### AI ⊃ ML ⊃ DL, with statistics orthogonal

> "Statistics is really orthogonal to all of this."

Statistical thinking still matters (sparsity, regularization, bias-variance), but it's a perspective applied to NNs, not a parent discipline. Hastie and Tibshirani are explicitly called out as "some of the few statisticians who are really trying to understand machine learning models or AI ... from a statistical perspective."

### Engineering eats theory in deep learning

The prof quotes Chollet's framing: machine learning *can* be math/physics, but "what wins the day is the engineering exercise." Practical wins come from:

- Switching to GPUs (more compute per dollar).
- Reducing parameter precision (8-bit floats instead of 32-bit) → fit more parameters.
- Architecture tweaks, training tricks, data pipelines.

> "A lot of people who just want to think about the theory won't win because the the guy who can just hack the shit out of it ... ends up winning the competition."

## Feedforward networks: first sketch

This is the start of the actual NN material, to be expanded next lecture.

### Layer structure

- **Input layer** $X$: your features (e.g. pixels of an image).
- **Hidden layer(s)** $Z$: values not directly observed; "stuff that have to be determined."
- **Output layer** $Y$: predictions (e.g. digit class).
- **Connections** between consecutive layers, with **weights** (think regression coefficients $\beta$).
- **Bias terms** added at each non-input layer (think $\beta_0$).

### What each neuron does

For a hidden unit $z_m$:

$$z_m = a\!\left(\beta_0 + \sum_{j} \alpha_{jm}\, x_j\right).$$

Translation: take inputs from previous layer, multiply by connection weights $\alpha_{jm}$, add a bias $\beta_0$, pass the sum through an **[[activation-functions|activation function]]** $a(\cdot)$. The output layer has the same form, with an activation of its own.

> "It could just be a sum, and then the value of this thing would be a sum, but then that would be very boring. And often people kind of attribute the importance of having some kind of nonlinearity."

Example activations the prof mentioned: a **threshold** (negative → 0, positive → identity), i.e. ReLU-style. Full activation taxonomy ("threshold linear or whatever") deferred.

### Why "feedforward"

> "There's a direction of flow information you're not seeing it go backwards necessarily at the moment. So that way and also you don't have connections between nodes [within a layer]. You just have them going forward, which is nice because then the learning of the parameters is easier. If you have them connected between them then it gets quite messy."

### As one composed equation

The whole 3-layer network can be written as $Y = a(\text{weights} \cdot Z + \text{bias})$ where each $Z_m = a(\text{weights} \cdot X + \text{bias})$.

> "What's nice about this construction of the feedforward network is that you notice here there's no loops. If there are loops it'd be a much harder problem."

### Note on terminology

> "The nodes are often called neurons. I will mix up the notation because I talk about these things all the time. I probably typically say neurons more than nodes, but that's okay."

## Closing

> "We'll talk more about this next week. I just see the time is up, but yeah, these things are fun. So see you guys next week."

Module-10 wrap-up slides (worked examples) deferred until after Module 11 so there's spaced reinforcement.
