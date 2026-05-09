# Concept atom template

The canonical shape for a concept atom. The concepts-pass agent fills this in for each atom listed in `docs/concepts-manifest.md`.

## Frontmatter

There are two variants depending on whether the atom belongs to one module or spans many (Specials).

### Regular atom (one owning module)

```yaml
---
concept: <hyphen-case-slug>           # e.g. ridge-regression
module: <NN-slug>                     # singular — the one owning module, e.g. 06-modelsel
lectures: [L<NN>, L<NN>, ...]         # every lecture this atom draws from
isl-ref: <chapter>.<section>          # ISLP pointer for full treatment, e.g. 6.2.1
                                      # leave null if no clean ISLP home
exercises:                            # exercises this atom touches; one entry per relevant problem
  - Exercise<N>.<sub> — <one-line>    # e.g. Exercise6.3 — apply ridge to credit data
  - CE<N> problem <P> — <one-line>    # e.g. CE1 problem 4 — k-fold CV for λ
related: [<slug>, <slug>, ...]        # other concept atoms this links to
tags:
  - concept
  - module/<NN-slug>
aliases:
  - <alt name>                        # optional
---
```

### Specials atom (cross-module, no single owner)

```yaml
---
concept: bias-variance-tradeoff
modules: [02-statlearn, 06-modelsel, 09-boosting, 11-nnet]   # plural — every module this concept lives in
lectures: [L02, L03, L04, L13, L20, L26]                      # every lecture across modules
isl-ref: 2.2.2
exercises:
  - Exercise2.5 — bias-variance simulation
  - CE1 problem 2 — apply bias-variance reasoning to KNN
related: [ridge-regression, lasso, double-descent]
tags:
  - concept
  - specials
aliases:
  - bias-variance
---
```

Key differences for Specials: plural `modules:` instead of singular `module:`; `tags` includes `specials` rather than a `module/<NN-slug>` tag (since no single module owns it). Per-module MOCs wikilink to the Specials atom; the Specials atom's body has a `## Returns in other modules` section listing each lecture wikilink in chronological order.

## Body shape

- H1 with the canonical concept name: `# <Canonical name>`
- 1–2 sentence headline of the prof's framing (what he uniquely brings to this concept)
- Sections below as needed; **omit any section the prof didn't give signal for** rather than writing filler

### Standard sections (in this order)

```markdown
## Definition (prof's framing)
[verbatim or near-verbatim from slide / lecture, with anchor like — [[L13-modelsel-2]]]

## Notation & setup
[symbols + conventions the prof uses]

## Formula(s) to know cold
[boxed equations; flag any the prof explicitly said to memorize]

## Insights & mental models
[verbatim where the prof's wording carries; e.g. L2-ball-no-corners geometry]

## Exam signals
> "<verbatim quote>" — [[L<NN>-slug]]
> "<verbatim quote>" — [[L<NN>-slug]]

## Pitfalls
- [inline; verbatim where prof flagged it as a common mistake]

## Scope vs ISLP
- **In scope:** [the parts the prof covered]
- **Look up in ISLP:** §X.Y, pp. ZZ–ZZ — use this for full derivations / extra worked examples
- **Skip in ISLP (book-only, prof excluded):** [list, with — [[L<NN>-slug]] anchor where prof verbalized exclusion]

## Exercise instances
- Exercise<N>.<sub> — [one-line description of what the exercise asks]
- CE<N> problem <P> — [one-line]

## How it might appear on the exam
- [pattern from past exams + prof's stated preferences; one bullet per pattern]

## Related
- [[other-concept]] — [why related, one line]
```

### For Specials atoms

Add a section after `## Definition` (before `## Notation`):

```markdown
## Returns in other modules
- [[L<NN>-slug]] — [how the prof revisits this concept in module X]
- [[L<NN>-slug]] — [how the prof revisits this concept in module Y]
```

This section makes the prof's escalating treatment visible in the Specials atom itself, so Claude doesn't have to traverse to other lecture pages to see the full pattern. List in chronological lecture order.

## Filename

`wiki/concepts/{hyphen-case-slug}.md`, slug from `docs/concepts-manifest.md`. Never invent a slug; always use the manifest value.

## Length

Target: **80–250 lines.** Below 80 means you stripped signal or the concept should be folded into another atom. Above 250 means it's actually two concepts. Split.

The exception: cross-cutting atoms (bias-variance, regularization, CV) can run longer because they aggregate multiple lecture treatments.

## Example skeleton: ridge-regression

```markdown
---
concept: ridge-regression
module: 06-modelsel
lectures: [L12, L13]
isl-ref: 6.2.1
exercises:
  - Exercise6.3 — apply ridge to credit data, plot β̂ vs log(λ)
  - CE1 problem 4 — k-fold CV for λ
related: [lasso, bias-variance-tradeoff, cross-validation, standardization]
cross-cutting: false
tags:
  - concept
  - module/06-modelsel
aliases:
  - L2 regularization
---

# Ridge regression

The prof's preferred lens on regularization: an L2 penalty that shrinks coefficients toward zero, trading a bit of bias for a lot of variance reduction. He framed regularization as "the central trick" of modern statistical learning.

## Definition (prof's framing)
> "Ridge regression minimizes the residual sum of squares plus λ times the sum of squared coefficients" — [[L13-modelsel-2]]

[...]

## Formula(s) to know cold

$$\hat\beta_{\text{ridge}} = \arg\min_\beta \left\{ \sum_i (y_i - x_i^\top \beta)^2 + \lambda \sum_j \beta_j^2 \right\}$$

[...]
```

(The example above is illustrative, agents should produce the full atom following the standard sections.)
