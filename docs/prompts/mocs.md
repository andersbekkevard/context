# MOCs pass — agent brief

You're producing a router page for one curriculum module. Read [[../overview]], [[../mocs]], [[../scope]], and [[../templates/mocs]] first.

## North Star

When Claude (in tutor mode) doesn't yet know which atom to load for a question about your module, it lands on your MOC and routes from there. The MOC is the navigation layer between "Anders asks about module N" and "Claude loads the right atom." It contains zero original explanation — only routing.

## Your inputs

You receive the module slug (e.g. `06-modelsel`) and produce one MOC file. Read:

- `docs/concepts-manifest.md` — the atom list for your module (filter to `module: <your-slug>`) and the Specials list for any cross-module atoms that touch your module
- `wiki/concepts/<slug>.md` for every atom in your module — to write tight one-liners and verify cross-cutting links
- `wiki/lectures/L<NN>-<your-module-slug>*.md` — the lectures covering your module; use to write the per-lecture one-liners
- `docs/scope.md` — canonical authority for what's out of scope; pull module-relevant exclusions verbatim with the lecture wikilink anchor
- `course-information.md` and `docs/lectures-manifest.md` — for module dates and lecture mapping

You do **not** read raw slides or transcripts directly — by the time you run, the lectures pass and the concepts pass have already distilled them. Your job is composition, not analysis.

## What to produce

Write `wiki/mocs/m<NN>-<slug>.md` following `docs/templates/mocs.md` exactly.

The MOC carries:

1. **Module orientation** — 1–2 sentence headline of what the module is and what's load-bearing
2. **Lectures** — wikilinks to each lecture in this module with a one-line "what this lecture covers"
3. **Concepts (atoms in this module)** — wikilinks to every atom with a one-line description (lifted/condensed from the atom's own headline)
4. **Cross-cutting concepts touched (Specials)** — wikilinks to any Specials atoms whose `modules:` list includes your module, with a one-line note on how this module touches them (and which lecture revisits)
5. **Exercises** — pointers to recommended exercise folders + relevant compulsory exercise problems
6. **Out of scope (this module)** — bullet list of topics excluded, each sourced to the relevant verbatim prof signal with a lecture wikilink. Pull from `docs/scope.md` filtered to your module's content.
7. **ISLR pointer** — chapter reference for deep treatment lookup at exam time

## Length

Target 40–120 lines. Anything longer means original explanation has crept in. If a MOC bullet starts to teach (more than one tight sentence), the content belongs in an atom — flag it for follow-up rather than embed it in the MOC.

## Format rules (enforced by template)

- Filename: `wiki/mocs/m<NN>-<slug>.md`
- Frontmatter exactly per `docs/templates/mocs.md`
- Wikilinks throughout — to atoms `[[<slug>]]`, lectures `[[L<NN>-<slug>]]`, exercise folders, scope.md
- No original explanation; one sentence max per bullet
- "Out of scope" bullets always carry a verbatim prof anchor (quote + lecture wikilink) where one exists

## Cross-cutting concepts

For Specials atoms that touch your module: include a short bullet that names the global atom, says when it's introduced (which module owns it), and notes which lecture in *your* module revisits it. Example:

> - [[bias-variance-tradeoff]] — first introduced module 02; this module revisits in [[L13-modelsel-2]] for ridge

This makes the Specials atom's bidirectional link visible from your module's MOC. The Specials atom's own `## Returns in other modules` section closes the loop.

## What you're not doing

- Not duplicating concept content — if you find yourself writing prose explaining ridge, stop and link the atom
- Not compressing lectures — that's done; you cite them
- Not classifying importance — depth/importance signals live in atoms
- Not making scope decisions — `docs/scope.md` already made them; you reflect them

Trust the manifest and the atoms — they did the analysis. Your job is composition.
