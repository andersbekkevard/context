---
moc: 01-intro
title: "M01: Introduction"
lectures: [L01]
isl-ch: 1
slides:
  - modules/1Intro/1Intro.md
exercises: []
tags:
  - moc
  - module/01-intro
---

# Module 01: Introduction

Course framing module: what statistical learning is, the two organizing axes (supervised/unsupervised, prediction/inference), and the three umbrella problem types (regression, classification, unsupervised). One lecture (Jan 5), no recommended exercises tied to this module, no compulsory-exercise problems. Load-bearing for the exam: the **Q1-style fill-in-the-blank** that uses this module's vocabulary to tag a real-world scenario, flagged by the prof in [[L27-summary]].

## Slides
- [[../../modules/1Intro/1Intro|1Intro]]

## Lectures
- [[L01-intro]]: course mechanics, the prof's framing of statistical learning vs ML / classical stats / data science, three worked-example problem types (Framingham regression, Fisher iris LDA, rat-gene-expression hierarchical clustering)

## Concepts (atoms in this module)
- [[statistical-learning]]: prof's framing of the field; "statisticians getting in on machine learning"; misspecified-model regime; the two orthogonal axes that organize every method in the course
- [[prediction-vs-inference]]: same model, two uses; finance vs science (Bitcoin forecasting vs Framingham SBP); same data → very different design choices
- [[supervised-vs-unsupervised]]: first organizing axis: have a Y to aim at vs not; LLMs as "supervised in disguise"; pure unsupervised flagged as "dangerous statistics" without a downstream check

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]]: first introduced module 02; previewed here in [[L01-intro]] as the statistician's-angle contribution to ML
- [[cross-validation]]: first taught module 05; previewed here as the supervised-side antidote to unsupervised's "how do I know this works?"

## Exercises

The recommended-exercise series starts at `exercises/Exercise2/` (module 2). Module 1 has no recommended exercises and no compulsory-exercise problems. The conceptual material is tested via Q1-style fill-in-the-blank using the vocabulary of [[supervised-vs-unsupervised]] and [[prediction-vs-inference]]. The closest drill problem lives in module 02:

- [[../../exercises/Exercise2]]: problem 2.1 ("describe a real classification and a real regression task, identify response/predictors, decide prediction or inference") is the canonical drill for this module's vocabulary

## Out of scope (this module)
- **Proper-noun history of the field** (McCulloch & Pitts, Rosenblatt, AlexNet, Bell Labs checkers bot, dates and inventors) - *"I'm not going to ask you a history question on the test"* - [[L22-unsupervised-2]] / [[L27-summary]]. The framing matters; the names and dates do not.
- **Etymological / definitional questions on the field labels** ("define machine learning," "define AI") - *"I won't ask any questions about how they're defined"* - [[L02-statlearn-1]]. The prof flags ML / AI / statistical learning / data science as branding more than substance.
- **R / Python package names, function syntax, executable code** - [[L27-summary]]: "no language, no memorizing package names, no language-specific coding."
- **Course-mechanics trivia** (compulsory-exercise pass marks, group sizes, Rmd vs Python policy, who taught it last year): administrative only; not exam content.
- **Data-science pipeline steps 1–3** (hypothesis formulation, scraping, structuring) - [[L01-intro]]: explicitly out of scope; statistical learning owns steps 4–6 (modelling, fitting, communication).

## ISLP pointer

Chapter 1: Introduction. Mostly motivational (Wage / Smarket / NCI60 examples, brief field history). Atoms carry section-level `isl-ref:` pointers: `prediction-vs-inference` → §2.1.1–§2.1.2, `supervised-vs-unsupervised` → §2.1.4. The historical rundown in §1 (least squares 1800s, LDA 1936, GLMs 1970s, trees 1980s, NNs, SVMs) is explicitly out of scope per the no-history-questions rule.
