# MOCs pass — agent brief

You're producing one router per curriculum module. Read [[../overview]] and [[../mocs]] first.

## North Star

When an LLM doesn't yet know which atom to load for a question, it lands on the relevant MOC and routes from there.

## What to do

For each of the 12 modules (see `course-information.md`), write `wiki/mocs/m{NN}-{slug}.md`:

- Module name and lecture dates
- ISL chapter
- Path to slide deck
- Path to exercise folder
- One line per concept in the module, each a wikilink to its atom
- Pointers to the lectures that cover this module
- Brief module-specific scope notes lifted from `exam_analysis.md`

## Format

- Filename `m{NN}-{slug}.md` (e.g. `m03-linreg.md`)
- Light frontmatter (module number, name, dates)
- Wikilinks throughout
- No original explanation — MOCs route, they don't teach

## What you're not doing

- Not duplicating concept content
- Not compressing the lecture (that's the lectures-pass's job)
- Not writing more than a one-liner per concept link

Trust your judgment on which concepts belong to which module — slides and lectures are the source of truth.
