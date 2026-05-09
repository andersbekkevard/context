# MOC template

The canonical shape for a Map-of-Content (MOC) page. The MOCs-pass agent fills this in for each of the 12 curriculum modules.

A MOC is a **pure router** — it tells Claude which atom to load for a given question about the module, what's in scope vs. out, and where the bronze sits. It contains zero original explanation. If you find yourself writing a paragraph about ridge regression in a MOC, that paragraph belongs in the atom.

## Frontmatter

```yaml
---
moc: <NN-slug>                      # e.g. 06-modelsel
title: <Module Title>               # e.g. Model Selection and Regularization
lectures: [L<NN>, L<NN>, ...]       # lecture wikilinks for this module
isl-ch: <chapter>                   # ISLR chapter, e.g. 6
slides: [<path>, ...]               # path(s) to slide deck(s), e.g. modules/6ModelSel/...
exercises:                          # exercise files that drill this module
  - exercises/Exercise<N>/
  - exercises/compulsory-exercise-<N>.md   # only if relevant
tags:
  - moc
  - module/<NN-slug>
---
```

## Body shape

- H1: `# Module <NN> — <Title>`
- 1–2 sentence orientation: what the module is about, what's load-bearing, what's flagged for the exam
- Sections below; **omit any section that's empty** rather than writing "(none)"

### Standard sections (in this order)

```markdown
## Lectures
- [[L<NN>-<slug>]] — <one-line, what this lecture covers in this module>
- [[L<NN>-<slug>]] — <one-line>

## Concepts (atoms in this module)
- [[<slug>]] — <one-line>
- [[<slug>]] — <one-line>

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]] — first introduced module 02; this module revisits in [[L13-modelsel-2]] for ridge
- [[regularization]] — central concept of this module

## Exercises
- [[../../exercises/Exercise<N>]] — <one-line on what the exercise drills>
- [[../../exercises/compulsory-exercise-<N>]] — <one-line on which problems in this comp ex touch this module>

## Out of scope (this module)
- **<Topic>** — <prof's verbatim signal> — [[L<NN>-<slug>]]
- **<Topic>** — <reason> — [[L<NN>-<slug>]]

## ISLR pointer
Chapter <N>: <title>. The deep treatment of in-scope concepts in this module is in `book/<NN>-<slug>.md`. Specific atoms carry section-level `isl-ref:` pointers.
```

## Filename

`wiki/mocs/m{NN}-{slug}.md` — e.g. `wiki/mocs/m06-modelsel.md`. Leading `m{NN}` keeps modules in curriculum order under any alphabetical sort.

## Length

Target: **40–120 lines.** MOCs are routers, not content. Anything longer means original explanation has crept in — refactor that explanation into an atom.

## Example skeleton — m06-modelsel

```markdown
---
moc: 06-modelsel
title: Model Selection and Regularization
lectures: [L12, L13, L14, L15]
isl-ch: 6
slides:
  - modules/6ModelSel/selection_regularization_presentation_lecture1.md
  - modules/6ModelSel/selection_regularization_presentation_lecture2.md
exercises:
  - exercises/Exercise6/
  - exercises/compulsory-exercise-1.md
tags:
  - moc
  - module/06-modelsel
---

# Module 06 — Model Selection and Regularization

The prof's "central trick of statistical learning" module. Heavy ridge / lasso / PCR / PCA treatment across four lectures (Feb 23, 24, Mar 2, 3). The prof distrusts AIC/BIC/Cp and prefers cross-validation.

## Lectures
- [[L12-modelsel-1]] — subset selection (best/forward/backward), shrinkage motivation
- [[L13-modelsel-2]] — ridge + lasso + elastic net; geometric picture
- [[L14-modelsel-3]] — PCA + PCR
- [[L15-modelsel-4]] — PCR wrap, PLS, high-dim motivation

## Concepts (atoms in this module)
- [[best-subset-selection]] — 2^p search, infeasible for large p
- [[forward-stepwise-selection]] — 1 + p(p+1)/2 fits
- [[ridge-regression]] — L2 shrinkage, never-zero coefs
- [[lasso]] — L1 shrinkage, exact-zero coefs
- [[principal-component-regression]] — PCA-then-OLS
- ...

## Cross-cutting concepts touched (Specials)
- [[bias-variance-tradeoff]] — central to ridge/lasso interpretation; revisits here in L13
- [[regularization]] — this module is its first systematic treatment
- [[cross-validation]] — used to tune λ; mechanics in module 5

## Exercises
- [[../../exercises/Exercise6]] — apply ridge/lasso to credit data, plot coefficient paths
- [[../../exercises/compulsory-exercise-1]] — problem 4 drills λ selection by k-fold CV

## Out of scope (this module)
- **AIC / BIC / Cp derivations** — "I really don't think I'm going to ask any questions about this" — [[L12-modelsel-1]] / [[L13-modelsel-2]]
- **Bayesian interpretation of Ridge/Lasso (Gaussian/Laplace priors)** — "I really don't think I'd put this on the test" — [[L14-modelsel-3]]
- **PLS history and detailed mechanics** — PCR is the workhorse — [[L15-modelsel-4]]
- **Elastic Net detailed tuning** — concept noted, no worked example

## ISLR pointer
Chapter 6: Linear Model Selection and Regularization. Atoms carry section-level `isl-ref:` pointers; for full algebra of any in-scope concept, route Anders to `book/06-modelsel.md`.
```
