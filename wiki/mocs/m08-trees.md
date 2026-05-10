---
moc: 08-trees
title: Tree-Based Methods
lectures: [L17, L18, L19]
isl-ch: 8
slides:
  - modules/8Trees/8Trees.md
exercises:
  - exercises/Exercise8/
tags:
  - moc
  - module/08-trees
---

# Module 08: Tree-Based Methods

The prof's first algorithmic (vs data-modelling) module: CART recursive binary splitting, then variance-reduction via bagging and random forests. Spans Mar 10 (L17 BeyondLinear wrap-up + trees intro), Mar 16 (L18 classification trees + ensembles), and Mar 17 (L19 wraps trees with bagging / RF / OOB / variable importance before pivoting to boosting). Load-bearing for the exam: tree-fitting algorithm, Gini vs cross-entropy vs misclassification, grow-then-prune logic, and *why random forests beat plain bagging* (decorrelation of the ρσ² floor).

## Lectures
- [[L17-trees-1]]: wraps GAMs / splines, then introduces regression trees, recursive binary splitting (CART), and cost-complexity pruning
- [[L18-trees-2]]: classification trees with Gini / cross-entropy, confusion matrix, then bagging, random forests, OOB error, variable importance
- [[L19-boosting-1]]: wraps trees (bagging variance, RF decorrelation, OOB, variable importance) before pivoting to boosting

## Concepts (atoms in this module)
- [[regression-tree]]: chunk predictor space into rectangles via greedy recursive binary splitting; predict region mean; loss = Σ_m Σ_{i∈R_m}(yᵢ − ȳ_{R_m})²
- [[classification-tree]]: same algorithm, predict majority class; **split on Gini or cross-entropy, prune on misclassification**
- [[cost-complexity-pruning]]: minimize Σ RSS + α|T|; grow fully then prune backward (early stopping is wrong); choose α by k-fold CV
- [[random-forest]]: bagging + restrict each split to m random predictors → decorrelates trees → reduces the ρσ² variance floor; m = √p (classification), m = p/3 (regression)
- [[variable-importance]]: impurity-based (Gini/RSS decrease) vs randomization-based (OOB permutation); prof prefers randomization "because it makes more sense"

## Cross-cutting concepts touched (Specials)
- [[bagging]]: owned by module 05; this module applies it to trees (precursor to RF), see [[L18-trees-2]] and [[L19-boosting-1]]
- [[out-of-bag-error]]: owned by module 05; revisited here as the free per-tree validation set in bagging / RF - [[L18-trees-2]], [[L19-boosting-1]]
- [[bias-variance-tradeoff]]: first introduced module 02; this module is *the* variance-reduction story (bagging averages, RF decorrelates) - [[L18-trees-2]]
- [[cross-validation]]: used to pick pruning size α via `cv.tree()` - [[L17-trees-1]], [[L18-trees-2]]
- [[standardization]]: *not required* for tree methods (scale-invariant splits); flagged here as the negative case of the standardize-everything rule

## Exercises
- [[../../exercises/Exercise8]]: full module drill: regression tree on Carseats (8.2b), CV-pruning (8.2c), bagging with `importance()` (8.2d), random forest with `mtry=3, ntree=500` (8.2e), ntree sweep (8.2f); spam classification tree (8.3c–d), prune-by-misclass (8.3e), bagging (8.3f), RF (8.3g); plus 8.1a (regression-tree algorithm) and 8.1d (OOB explanation)

## Out of scope (this module)
- **NP-completeness proof / computational complexity of trees** - mentioned in passing only as NP-hard, not exam-relevant - [[L17-trees-1]]
- **R/Python package syntax** (`tree()`, `randomForest()`, `cv.tree()`, `prune.misclass`, `importance()`) - "no language-specific coding" - [[L27-summary]]

## ISLP pointer
Chapter 8: Tree-Based Methods. Deep treatment of every in-scope concept lives in `wiki/book/08-trees.md`; specific atoms carry section-level `isl-ref:` pointers. Note ISLP ch. 8 also covers boosting, which is module 09's territory, not this MOC's.
