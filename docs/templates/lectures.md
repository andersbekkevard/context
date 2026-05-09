# Lecture template

The canonical shape for a lecture page. The lectures-pass agent fills this in for each class session.

## Frontmatter

```yaml
---
lecture: <sequential number 1..27>
date: <YYYY-MM-DD>
module: <NN-slug>          # e.g. 03-linreg
title: <lecture title>     # e.g. Linear Regression 1
slides: <path>             # path to the slide deck the prof walked through, e.g. modules/3LinReg/3LinReg.md
topics:                    # concept slugs touched in this lecture
  - <concept-slug>
tags:
  - lecture
  - module/<NN-slug>
aliases:
  - Lecture <N>
---
```

## Body shape

- H1 with the lecture title: `# L<NN>: <Title>`
- Immediately after H1: a 1–3 sentence prose summary of the lecture (what was covered, where it went, anything to flag at a glance)
- `## Key takeaways` section with 3–6 bullets
- `## <topic>` H2 for each major topic the prof transitioned into
- `### <sub-segment>` H3 for sub-segments inside a topic
- **Never H4**

## Filename

`wiki/lectures/L{NN}-{slug}.md`, sequential lecture number `01..27` plus a slug derived from the title. E.g. `L04-linreg-1.md`.

When two transcripts fall on the same date, give them adjacent numbers (`L24-…`, `L25-…`).

## Example

```markdown
---
lecture: 4
date: 2026-01-19
module: 03-linreg
title: Linear Regression 1
slides: modules/3LinReg/3LinReg.md
topics:
  - ols
  - bias-variance-tradeoff
  - interactions
tags:
  - lecture
  - module/03-linreg
aliases:
  - Lecture 4
---

# L04 — Linear Regression 1

The prof reviewed OLS as projection onto the column space of X, introduced the bias-variance decomposition as the course's running theme, and worked the car-mpg interaction example. Flagged interactions as a common student trap.

## Key takeaways

- OLS recap: closed form via normal equations; projection interpretation
- Bias-variance decomposition is the course's running theme — to be derivable by hand
- Interaction terms change the meaning of main effects (flagged trap)
- Train MSE always drops with more parameters; test MSE doesn't

## OLS recap

[compressed prose, wikilinking [[ols]] and [[normal-equations]]]

## Bias-variance decomposition

[compressed prose, wikilinking [[bias-variance-tradeoff]]]

## Interactions

[compressed prose, wikilinking [[interactions]]]
```
