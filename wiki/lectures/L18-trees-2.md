---
lecture: 18
date: 2026-03-16
module: 08-trees
title: Tree-based Methods 2
slides: modules/8Trees/8Trees.md
topics:
  - regression-tree
  - cart
  - cost-complexity-pruning
  - classification-tree
  - confusion-matrix
  - bagging
  - random-forest
  - out-of-bag-error
  - variable-importance
tags:
  - lecture
  - module/08-trees
aliases:
  - Lecture 18
---

# L18 — Tree-based Methods 2

The prof recapped CART regression trees (greedy splitting on RSS, build out then prune via cost complexity), then moved to classification trees (majority vote / class probabilities, split by [[classification-tree|Gini index]] or [[classification-tree|cross-entropy]] not misclassification, prune on misclassification). Closed by motivating [[bagging]] and introducing [[random-forest]] / out-of-bag error; deferred boosting to next lecture.

## Key takeaways

- CART = greedy recursive binary splitting. Greedy ≠ optimal but tractable. "Sometimes you actually do a seemingly worthless split, but then it's followed by a very good one."
- Build the tree out fully (overfit), then prune back using cost complexity $\sum_m \sum_{i \in R_m}(y_i - \hat y_{R_m})^2 + \alpha |T|$. Choose $\alpha$ by cross-validation.
- Classification trees: predict by majority vote (or class proportions); split by Gini or cross-entropy because misclassification error is "not very sensitive" and not differentiable.
- Asymmetric trick: train splits on Gini/cross-entropy, but prune on misclassification — "ultimately what you care about."
- Trees have low bias but very high variance — "data set sensitive is an equivalent statement to high variance." Bagging (bootstrap + average) attacks this; ensembling B trees would shrink variance by 1/B if samples were IID.
- Out-of-bag error: observations not drawn into a bootstrap sample serve as a built-in test set for that tree. No dedicated test set required.

## Recap: regression trees and CART

Trees take a different angle from regression with splines/interactions: "instead of treating years and hits as independent variables that you then find some function of either using just linear things or using some nonlinear stuff like we talked about with splines … instead, you kind of say oh we could just break this stuff up and then call this one region call that another region call that another region and then give a parameter to every region. And that gives us a very powerful model that now has these combinations of variables."

> [!note] The prof's framing — algorithm, not distributional model
> "This is our first example of a real algorithm for building up a model, which as opposed to the classic statistical approach of, oh, I'm going to assume that this variable is this distribution and I'm going to make a model of it that has this specific form, it said, no, we're going to make an algorithm. We're going to think of it like a computer program."

He placed CART historically in the 70s, when Fortran was making waves into math and engineering: "these guys were like, we have these algorithms that can do stuff every time. We don't have to always do stuff on pen and paper." Speculated on what stats would look like had Fisher had computers — "he tried to do stuff without assuming a distribution … but instead he developed all these tests and all these approximate distributions because we didn't have a computer to do it for us … or to do the hard work of counting things and doing these very boring things." Bremen ("the algorithm guy") developed CART.

### Greedy splitting on RSS

Hitters example revisited (predict salary from years, hits, at-bats — "I don't know why this book always talks about money. I'm going to write these guys and tell them, please, new topics. Not everything's about money. They're American, probably"). The first split on `years` separates salaries best — "everyone likes to be given some money for experience." Then `hits`. Then probably `at-bats`. Recursively, each split chooses variable + threshold to minimize

$$\sum_{j} \sum_{i \in R_j} (y_i - \hat y_{R_j})^2$$

where $\hat y_{R_j}$ is the mean of training $y$ in region $R_j$ (median works too). The sum is over the points $i$ within region $j$ — "compare every point relative to the mean within its region." For each candidate split: check every block, every variable, every threshold; pick the one giving the most RSS improvement, then recurse.

Greedy = not optimal, but tractable: "we can't try every possible combination. It's just too much. It's impossible." Same flavor as forward [[subset-selection|model selection]] — "that's what those kinds of algorithms are called." With continuous variables ties are essentially impossible ("I'm sure there's an example where it's down to machine precision, but it's extremely rare, if not impossible"), so a single split always wins — but slight data perturbations can flip it, and "then it plays out very differently." This data-set sensitivity becomes the variance problem bagging will fix.

### Visualization vs. tree representation

You can draw the partition as a 2D rectangle decomposition (the visual block diagram) or as a tree diagram. The first is intuitive but caps at 2–3 dimensions: "I can't think in a four-dimensional space it's way too confusing. I don't think anyone can." The tree is a one-to-one alternative that scales to many dimensions at once.

The split-finding picture: "this is a potential space that we want to do then we want to split it … We'll move this thing such that, and here we're always making R1 and R2, and moving this thing around to maximize this cost function, or to minimize this cost function on the right." It's just one variable at a time per split, "so it's very easy."

### Why grow it out fully first

"Right now you're not going to worry about how big the tree is going to get. You're just going to keep going basically until you can't go anymore … you get it very fine. And you can think of this as we're definitely overfitting … creating regions that are so small that basically the node is being predicted by the average of like three variables. We're definitely doing the equivalent of overfitting where we put too many parameters in our model."

## Pruning

You grow the tree until you can't go further — "you've built it out probably way too much," region predictions averaging "like three variables." This is overfitting. Then you cut back, like a cherry tree (the prof's ruined view due to forgetting to prune in winter — "apparently you can't prune whenever you want").

### Why not stop early?

Stephanie's notes mention an alternative: stop growing once the RSS improvement passes some threshold. Bad idea:

> [!important] Why early stopping fails
> "Sometimes you actually do a seemingly worthless split, but then it's followed by a very good one. Because again, sometimes you need to cut one way and then cut the other way until you really get the benefit, right? Because these interactions are not super obvious."

So: build out, then prune.

### Cost complexity

Define the [[cost-complexity-pruning]] criterion

$$C_\alpha(T) = \sum_{m=1}^{|T|} \sum_{i \in R_m} (y_i - \hat y_{R_m})^2 + \alpha |T|$$

where $|T|$ is the number of terminal nodes and $\alpha$ is a hyperparameter. $\alpha = 0$ → no pruning; large $\alpha$ → shorter trees. "Equivalent to saying, I want to find the shortest tree that gives me a good cost."

### Choosing α via CV

K-fold cross-validation. For five folds: train on four folds, evaluate on the held-out fold, get a held-out error per $\alpha$ (each $\alpha$ corresponds to a tree size). Average over folds. Pick the $\alpha$ minimizing CV error.

The prune-back is itself a sweep: "you build out the tree as much as possible and then you prune the tree again back up to … you can go all the way up to the root node. It doesn't matter so much. You first build it out and then you go backwards, removing them to minimize that cost complexity thing, which is that error term plus the complexity term on how many nodes you have."

Plot of held-out deviance against tree size: as you prune from a too-big tree, error decreases (overfitting reduced), hits a minimum, then rises again. "As they pruned, you got an improvement in held out error … as they prune nodes away, it didn't make the model worse and held out error. It made it better because it overfit less. And then eventually it found a minimum in this deviance. And then it got worse again." The example bottomed at five leaves — recovered the original temperature/wind partition.

The prof referenced "a note by Bull Linguist" as a nice extra resource on pruning, alongside the book.

### Terminal nodes

> [!note] Terminology
> "Wherever the node and over the branch ends, you get a terminal node. It's like where the new branch comes out. It's like wherever you get a knot, I guess. Yes, in a tree." Equivalently: leaves. $|T|$ in the cost-complexity formula counts these.

## Classification trees

Same structure as regression trees; only prediction and splitting criterion change.

### Prediction: majority vote or class proportions

Estimate

$$\hat p_{mk} = \frac{1}{N_m} \sum_{i: x_i \in R_m} \mathbb{1}(y_i = k)$$

— the proportion of class-$k$ training observations in region $R_m$. Then either predict the majority class or draw from the estimated probabilities. Hat = estimate, $k$ = class index, $m$ = region, $i$ = observation.

### Splitting: Gini, cross-entropy, not misclassification

Two standard impurity criteria:

- **[[classification-tree|Gini index]]**: $\sum_k \hat p_{mk}(1 - \hat p_{mk})$. ("I really should have looked up where the name comes from … Ginny is like a cute name. I don't know.")
- **[[classification-tree|Cross-entropy]]**: $-\sum_k \hat p_{mk} \log \hat p_{mk}$. From information theory / thermodynamics, "originally … Napoleonic Wars and things, and steam engines."

Why not just use misclassification error? Two reasons:

1. **Sensitivity / impurity penalization.** Misclassification "isn't very sensitive to the right thing." Two splits can both yield 25% misclassification, but one produces purer nodes — Gini and cross-entropy reward the purer split, misclassification doesn't see the difference. "They penalize having impure probabilities or impure classifications."
2. **Differentiability.** Both Gini and cross-entropy are differentiable. The prof flagged this as a weaker argument: "the reality is now actually [misclassification] would be also pretty easy to make differentiable. Like … we could just do a soft max." So really the impurity-sensitivity argument is the load-bearing one.

### Brain injury example

Data on 1,321 patients, predict minor head injury. Variables: amnesia, b-skull, vomiting, age, etc. ER context: "the patients are asked questions and then it's your job is to determine whether or not they have a brain injury and you want to make sure that the criteria are very simple and everything is very quick" because the patients are "in the neurointensive care unit which is like the worst place you want to be" (the prof's wife works there — "don't worry she doesn't tell me anything").

R `tree` package, splitting on `deviance` (= cross-entropy). Result: a tree of categorical splits on variables like amnesia, vomiting, GCS, etc., classifying each leaf as 0 (no injury) or 1 (injury).

A subtle observation: sometimes a split classifies both children as the same class (both 0). Why does the algorithm bother? Because the impurity drops:

> [!example] Both children → same class can still help
> Parent: 70 of class A, 30 of class B → Gini ≈ 0.42.
> Left child: 40A / 5B → proportion 0.89 → Gini ≈ 0.89² + 0.11² ≈ small.
> Right child: 30A / 25B → 0.55² + 0.45² ≈ 0.495.
> Both children predict A, but the left node is much more confident. "It put kind of like the shitty part of the parent node in one side and the more confident version on the left side."

And, just as in regression, that "useless-looking" split can enable a great split later.

### Confusion matrix and evaluation

Evaluate on test set with a [[confusion-matrix]]: rows = predicted, columns = truth (or vice versa), four cells = true negatives, true positives, false positives, false negatives. Misclassification rate = (off-diagonal) / total.

In the example: cross-entropy-grown tree → test misclassification 0.14. Gini-grown tree → 0.16-ish, "slightly worse but not very different." Numbers like 354 vs. 356 correct, 44 vs. 49 errors. "You could argue maybe the Gini index was slightly more conservative, which maybe in the case of predicting if someone has a head injury, you don't want to be so conservative."

### Pruning classification trees

Same algorithm 8.1 as for regression: grow out, then prune via cost complexity, choose $\alpha$ by CV. Replace RSS with cross-entropy or Gini in the cost.

> [!important] Train one criterion, prune another
> "This is actually an interesting note that you build out the tree using deviance or Gini index but then you actually prune it using misclassification criteria … because that's ultimately what you care about. So it seems weird that you would not train on misclassification error but it turns out to be better to train on the Gini index or the deviance and then prune with misclassification."

`cv.tree` defaults to 10-fold. Plot misclassification vs. number of terminal nodes (a proxy for $\alpha$ in the cost-complexity formula since you're going backwards from a fully grown tree). Pick the smallest tree achieving the minimum. The example pruned a 100-node tree down to seven leaves with no harm — "we picked 7 because that was the smallest model with that level of misclassification error."

After pruning the cross-entropy-grown tree, test misclassification was around 0.41 — "very much the same number only it had a lot more breakups. So even though you don't see an improvement it is improved because the model is simpler … in fact it looks exactly the same it's a simpler model so it's better." Same procedure works for the Gini-grown tree.

## Bias–variance for trees

Trees illustrate the recurring [[bias-variance-tradeoff]] — "just like we talked about in the beginning of the class, the bias and the variance is a very reoccurring theme. How do we remove the bias? How do we get these things to be less sensitive to which data set we train on? A lot of the tricks that we develop, like regularization or pruning, is specifically to improve the bias and the variance."

Greedy splitting is "very sensitive to which data set we train on" → high variance. Pruning helps but doesn't fix it: "It's still going to be a different tree every time, basically."

Sources of variance:

- Number of levels per predictor (categoricals with many levels = many possible splits). "Ideally have fewer predictors with many levels … you want to simplify every predictor as much as possible so that you get less variance in the tree structure."
- Continuous predictors → continuum of split thresholds.
- Missing values: either discard those instances completely or impute (mean, more complicated model, or fancier methods — "we won't talk about those"). "Often your biggest source of screwing up is simply you don't have the right variables. And there's nothing you can really … I mean we look at different ways of how to accommodate that. But it really is a major problem."

> [!quote] On variance
> "High variance is like death, right? That's like what all of modern machine learning is like remove variants, remove variants. Bias is easy. Let's get rid of variants."

### Tree pros and cons

Pros:

- Automatic variable selection
- Nice greedy growing algorithms
- Handle mixed variable types ("you can really get into this kind of niche … that this variable is between 1 and 2 and the other one is around 0. You just can't do that with regression models")
- Easy to interpret
- Easy to explain
- Display graphically
- Some say they mirror human decision-making / reinforcement learning — "I don't know if that's a benefit or not, but it was listed as one"

Cons:

- Large trees get gross and hard to understand
- Don't generalize well — large trees in particular have high variance
- Not robust, very sensitive to which data set you use
- "Sounds like these are a bad idea. Like it seems like, okay, trees should be just something that you use once in a while or never. And it was just a nice thought exercise."

The slides also showed the classic comparison: linear regression's diagonal/curved decision boundaries vs. trees' axis-aligned rectangular partitions. Each shines in different geometries.

## Bagging

Fix the variance by averaging trees fit to bootstrap samples — [[bagging]] = bootstrap aggregation.

For each of $B$ rounds:
1. Bootstrap sample (resample with replacement, same size as original training data).
2. Grow a tree on it (typically deep / unpruned).

Prediction: average the $B$ trees (regression) or majority vote (classification).

> [!note] Why bagging works
> "If [the samples] were IID then we would have these really nice properties that the variance of our model would actually be reduced by the number of models. So if we made it 10 times, then the variance that we'd have across those 10 data sets would actually be shrunk by 10."

Bootstrap samples aren't IID — they share data — so the reduction is less, but the principle holds.

The brain injury example with bagging: the prof noted misclassification *increased* slightly. "Note the misclassification has increased slightly. Typically we would expect an improvement, but it's not guaranteed of course. But I think even though you can't see the improvement, it's probably there" — referring to reduced variance / data-set sensitivity.

### Out-of-bag error

For each bootstrap sample, some observations are drawn multiple times and others not at all — "because when you bootstrap, some of them you're obviously going to get some twice or three times or whatever, and some you're not going to get at all because you're just drawing with replacement." The probability an observation appears in a bootstrap sample of size $n$ is $1 - (1 - 1/n)^n \to 1 - 1/e \approx 0.632$. The roughly 1/3 left out are **out-of-bag** and serve as that tree's test set. Aggregated across trees → out-of-bag error, no dedicated test set needed.

The prof flagged the slight oddness: "you have this strange dependency on the test error from your real error on your test error on how you sampled" — but it works, and you don't have to set aside test data.

### Random forests

[[random-forest]] = bagging + extra randomness: at each split, only a random subset of predictors is considered. This decorrelates the trees further. The prof framed bagging → forests via the analogy "you can see the forest through the trees." Set `B` (number of trees, e.g. 500) and `mtry` (predictors per split).

Synonym: ensembling. "I think that's more common in machine learning, that term."

### Two trees on different bootstraps

The prof showed two bagged trees from the brain data: similar but different. "Here we get, in the first set, we have slightly different variables. Then here, it's going to split on age in one case and GCS in the other. And then you can see age is often a variable, but it comes in in different ways." Both use age but at different thresholds (63, 68.5, 64.5). "But in both cases, you've broken up your, you know, you've created a tree of classification. So in both cases, for every new test data set, you can say, what's my prediction according to model one and what's my prediction according to model two." Average them (regression) or majority-vote them (classification).

Aside on combination rules: for classification you can also average estimated class probabilities — "that would end up being the same" as majority vote in most cases.

### Slide commentary

The prof was openly impatient with the slide volume here: "so many words, so many words. At some point my eyes just glaze over. There's too many words, too many words on slides … this is torture for me. I've read these things before but I can't remember 100 slides. I'm sorry. I'm human. I don't think Steffi is human." Useful context for what he expects you to actually retain vs. what the slides cover for completeness.

### Variable importance

Drawback of bagging: "it becomes difficult to interpret the results, because instead of having just one tree, which is very easy to read, you have like 10, right? And then you often want to bring it back to what the variables actually are." Ideally consolidate the trees into one — "but that's not always so easy."

Instead, use **variable importance plots** — "this would fall into the category of like explainable AI in the sense that you have a model that you don't quite understand. Each tree is understandable. The combination of 500 trees is not." Sort predictors by either:

- Mean decrease in node impurity (the criterion you trained on), or
- Permutation-based randomization (shuffle a variable, measure performance drop)

## Closing

The prof ran short and stopped before [[boosting]], just naming it: "you make another tree and then you build another tree on its residuals — that's a bit more confusing. We did talk about it but … we'll talk about it more soon. I don't think we'll get there today." Plan: finish variable importance plots and move into the next chapter next session.

Project 2 is out; Seaman runs the exercise session — "if any help you need on that, I'd recommend going to the exercise session. Seaman is really good at that. If not, let me know, we can talk about it. If you want to change groups that's fine you just have to contact him."

Next lecture continues: variable importance, finish forests, then boosting.
