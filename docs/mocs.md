# MOCs

A MOC (Map of Content) is a **router** for one curriculum module. One per module, in `wiki/mocs/`.

## What it is

An LLM hits a MOC when it doesn't yet know which atom to load. The MOC tells it: here are the concepts in this module, here's where to look in the slides and lectures, here's the relevant exercise, here's any module-specific scope note.

A typical MOC contains:

- Module name and lecture dates
- ISL chapter
- Path to the slide deck
- Path to the exercise folder
- One line per concept in the module, each a wikilink to its atom
- Pointers to the lectures that cover the module
- Module-specific scope notes (in scope vs. out, lifted from `exam_analysis.md`)

## What MOCs don't have

Original explanation. Zero. They route; they don't teach. If you find yourself writing a paragraph about ridge regression in a MOC, that paragraph belongs in `wiki/concepts/ridge-regression.md`.

## Where it lives

`wiki/mocs/m{NN}-{slug}.md`, e.g. `wiki/mocs/m03-linreg.md`. The leading `m{NN}` keeps modules in curriculum order under any alphabetical sort.
