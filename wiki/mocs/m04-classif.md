---
moc: 04-classif
title: Classification
lectures: [L07, L08, L09]
isl-ch: 4
slides:
  - modules/4Classif/4Classif.md
exercises:
  - exercises/Exercise4/
  - exercises/compulsory-exercise-1.md
tags:
  - moc
  - module/04-classif
---

# Module 04 — Classification

The prof's three-lecture run on categorical $Y$ (Jan 27, Feb 2, Feb 3): set up the Bayes classifier, do logistic regression, then the LDA/QDA/Naive Bayes generative trio plus the binary-classifier metric stack (confusion matrix, sensitivity/specificity, ROC/AUC). Load-bearing for the exam: **odds ↔ probability**, the **decision-boundary derivation** ($\delta_k(x)$ flagged twice), the **diagnostic vs sampling** divider, and "where does the quadratic come from?" in QDA.

## Lectures
- [[L07-classif-1]] — classification setup, Bayes classifier and Bayes error rate, logistic regression (logit link, MLE via Newton, odds), KNN revisit, curse of dimensionality, diagnostic-vs-sampling framing
- [[L08-classif-2]] — LinReg residual/leverage/collinearity wrap-up, then LDA setup (Gaussian class-conditionals, pooled covariance)
- [[L09-classif-3]] — LDA derivation, QDA (drop pooling → quadratic boundary), Naive Bayes, confusion matrix, sensitivity/specificity, ROC/AUC

## Concepts (atoms in this module)
- [[classification-setup]] — categorical $Y$; estimate $\Pr(Y\mid X)$, assign argmax; Bayes classifier is optimal, Bayes error rate is the irreducible floor
- [[logistic-regression]] — Bernoulli GLM with logit link; $\log(p/(1-p)) = \beta_0 + \beta^\top x$; fit by MLE via Newton; coefficient = log odds-ratio
- [[odds-and-log-odds]] — $\text{odds} = p/(1-p)$; one-unit increase in $x_j$ multiplies odds by $e^{\beta_j}$; calculator-friendly conversion the prof said he *will* ask
- [[linear-discriminant-analysis]] — model $f_k(x)$ as Gaussian with class means $\mu_k$ and **pooled** $\Sigma$; flip via Bayes; discriminant linear in $x$
- [[quadratic-discriminant-analysis]] — drop the pooled-$\Sigma$ assumption → $x^\top\Sigma_k^{-1}x$ no longer cancels → boundary becomes quadratic; "where does the quadratic come from?" exam-flagged
- [[discriminant-score-and-decision-boundary]] — $\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k$; set $\delta_k = \delta_\ell$ and solve for the boundary; the prof's most-flagged module-4 exam pattern
- [[naive-bayes]] — assume diagonal $\Sigma$ (predictors conditionally independent given class); fewer parameters → preferred when $p$ is large
- [[diagnostic-vs-sampling-paradigm]] — model $\Pr(Y\mid X)$ directly (logistic, KNN) vs model $f_k(x)$ and $\pi_k$ then flip via Bayes (LDA, QDA, Naive Bayes); the conceptual divider for the module
- [[confusion-matrix]] — $K\times K$ table of true vs predicted class; diagnostic for *how* the model fails; in-sample matrices flatter the model
- [[sensitivity-specificity]] — $\text{sens} = TP/(TP+FN)$, $\text{spec} = TN/(TN+FP)$; justice-system trade-off; **write the formula** even without numbers
- [[roc-auc]] — sweep classifier threshold, plot $(1-\text{spec},\,\text{sens})$; AUC summarizes; diagonal = chance, below diagonal = invert your classifier
- [[curse-of-dimensionality]] — in high-$d$ all pairwise distances collapse → "nearest neighbor" stops meaning anything → kills KNN

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]] — first introduced module 02; revisited here in [[L09-classif-3]] as the LDA-vs-QDA trade-off (more parameters, more variance)
- [[cross-validation]] — owned by module 05; this module's exercises (Exercise4.6h) sweep KNN $K$ by CV, and CE1 problem 4 wraps logistic/LDA/QDA with bootstrap CIs
- [[multivariate-normal]] — first introduced module 02 (sampling distribution of $\hat\beta$); load-bearing here in [[L09-classif-3]] as the class-conditional density $f_k(x)$ for LDA/QDA/Naive Bayes
- [[knn-classification]] — owned by module 02; revisited heavily in [[L07-classif-1]] / [[L08-classif-2]] as the diagnostic-paradigm foil to logistic/LDA, and drilled in Exercise4.1 / 4.6g–j

## Exercises
- [[../../exercises/Exercise4]] — full module-4 drill: KNN by hand (4.1), LDA pooled-Σ + boundary derivation + bank-note classification (4.2), odds ↔ probability (4.3), logistic with given β's (4.4), sensitivity/specificity/ROC/AUC definitions (4.5), end-to-end glm/LDA/QDA/KNN comparison on Weekly with confusion matrices and ROC (4.6)
- [[../../exercises/compulsory-exercise-1]] — problem 3 is the module's flagship: 3a–c logistic regression on tennis data (logit linearity, β interpretation, boundary + sens/spec); 3d–f LDA ($\pi_k, \mu_k, \Sigma, f_k$ interpretation; derive $\delta_k$; confusion matrix); 3g QDA fit and metrics; 3h compare LDA/QDA/logistic boundaries

## Out of scope (this module)
- **Multi-class logistic regression** — "we're not going to talk about... discriminant analysis and KNN can deal with this case" — [[L07-classif-1]]
- **GLM link-function theory beyond logit (probit, complementary log-log)** — "outside the scope of this course... if you took the GLM course" — [[L07-classif-1]]
- **Imbalanced-class detail / asymmetric ROC analysis** — "I don't think the book talks much about that" — [[L07-classif-1]]; sensitivity/specificity in scope, deeper treatment is not

## ISLR pointer
Chapter 4: Classification. Deep treatment of in-scope concepts in this module is in `book/04-classif.md`. Atoms carry section-level `isl-ref:` pointers — e.g. logistic regression §4.3, LDA/QDA §4.4.1–4.4.3, Naive Bayes §4.4.4, confusion matrix / sens-spec / ROC §4.4.2.
