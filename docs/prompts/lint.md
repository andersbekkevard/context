# Lint pass: agent brief

You are the wiki integrity checker. Run after lectures, concepts, and MOCs are all written. Your output is a single report at `wiki/lint-report.md`. **You do not modify wiki files**, Anders fixes flagged issues by re-running specific agents or editing by hand.

Read [[../overview]], [[../concepts]], [[../lectures]], [[../mocs]], and `docs/concepts-manifest.md` first to understand the schema.

## North Star

Every wikilink resolves. Every manifest entry has a file and vice versa. Every cross-cutting (Specials) atom is bidirectionally linked from the modules it touches. Every lecture-frontmatter `topics:` slug exists as an atom. Out-of-scope entries don't shadow in-scope atoms.

A clean lint report = the wiki is internally consistent.

## What you check

### 1. Wikilink integrity (hard errors)

Grep every `[[<target>]]` across `wiki/` and verify it resolves:

- `[[<slug>]]` → expects `wiki/concepts/<slug>.md`
- `[[L<NN>-<slug>]]` → expects `wiki/lectures/L<NN>-<slug>.md`
- `[[m<NN>-<slug>]]` → expects `wiki/mocs/m<NN>-<slug>.md`
- `[[../<doc>]]` or `[[docs/<doc>]]` → expects `docs/<doc>.md`

Report dead links with file path and approximate line. If a dead link looks like a near-miss for an existing slug (edit distance ≤ 2), suggest the correction.

### 2. Manifest ↔ filesystem consistency (hard errors)

- Every atom slug in `docs/concepts-manifest.md` must have a file at `wiki/concepts/<slug>.md`.
- Every file in `wiki/concepts/` must have a manifest entry (in either a per-module section or the Specials section).
- Slugs match exactly, no case differences, no underscore-vs-hyphen mismatches.

### 3. Frontmatter completeness (hard errors)

For each atom (`wiki/concepts/*.md`):
- Required fields present: `concept`, `lectures`, `exercises`, `related`, `tags`
- Module field: `module: <NN-slug>` for module atoms, `modules: [<list>]` for Specials
- `concept:` slug matches filename
- `isl-ref:` may be `null` (acceptable)

For each lecture (`wiki/lectures/L*.md`):
- Required: `lecture`, `date`, `module`, `title`, `slides`, `topics`, `tags`, `aliases`
- `topics:` slugs that don't yet have atom files → warning, not error (could be forward-reference)

For each MOC (`wiki/mocs/m*.md`):
- Validate against MOC template/conventions (loose, MOCs are routers)

### 4. Cross-cutting (Specials) bidirectional links (warnings)

For each Specials atom in the manifest with `modules: [A, B, C]`:
- For each module M in that list, the MOC `wiki/mocs/m<MM>-<slug>.md` should wikilink to the Specials atom.
- The Specials atom should reference each module's lecture(s) where the concept appears (via `## Returns in other modules` section in the body).

Report missing bidirectional links as warnings.

### 5. Lecture `topics:` frontmatter (warnings)

For each lecture, every slug in `topics:` should match a manifest entry (atom or Specials). Mismatches → warning with suggested action (rename slug or create atom).

### 6. Out-of-scope consistency (hard errors)

For each MOC's `## Out of scope` section, check:
- The named topic does NOT have an atom in `wiki/concepts/`. (If it does → contradiction, either remove from out-of-scope or remove the atom.)
- The exclusion is sourced (has a lecture wikilink anchor or a verbatim quote anchor).

### 7. Filename and structural rules (hard errors)

- All atom filenames are hyphen-case `.md`, no spaces, no uppercase
- All lecture filenames match `L<NN>-<slug>.md` exactly
- All MOC filenames match `m<NN>-<slug>.md` exactly
- All atoms have exactly one H1 line; H1 text matches the canonical name in the manifest

### 8. Length bounds (warnings)

- Atoms: 80–250 lines (Specials may exceed; flag if > 350)
- Lectures: 250–500 lines (flag any outside this range, but this was the lecture pass's job, so just info-level)
- MOCs: no firm bound, but flag any MOC over 150 lines (MOCs are routers, not content)

### 9. Empty / placeholder content (hard errors)

- No atom is just frontmatter (body is empty)
- No `[TODO]`, `[fill me in]`, `[...]` placeholders left in the prose
- No empty H2 sections (heading with no content underneath)

### 10. Quote anchoring (info-level)

For each atom, scan blockquote lines (`>` prefix). Each verbatim quote should be near a lecture wikilink (`[[L<NN>-<slug>]]`), within 2 lines before or after, so a reader can grep back to the bronze transcript. Flag isolated quotes as info.

## How to work

Use `rg` (ripgrep), `fd`, and `Read` heavily. Most checks are mechanical:
- Dead wikilink check: `rg -o '\[\[[^]]+\]\]' wiki/` then resolve each against the filesystem
- Frontmatter parse: `Read` each file's first 30 lines and yaml-parse the frontmatter
- Manifest parse: `Read` the full manifest, extract slugs from per-module + Specials sections

The semantic checks (out-of-scope consistency, quote anchoring) need judgment, read in context.

## Output

Write `wiki/lint-report.md`:

```markdown
# Wiki lint report

Generated <date>. Wiki integrity check across `wiki/lectures/` (27 files), `wiki/concepts/` (N files), `wiki/mocs/` (12 files), and `docs/concepts-manifest.md`.

## Summary

- Atoms: N — E errors, W warnings, I info
- Lectures: 27 — E errors, W warnings, I info
- MOCs: 12 — E errors, W warnings, I info
- Manifest: E errors

## Errors (must fix before relying on the wiki)

### Dead wikilinks
- `wiki/concepts/ridge-regression.md:54` — `[[laso]]` does not resolve. Suggest: `[[lasso]]`?
- ...

### Missing atom files
- `manifest entry "decision-tree" has no file at wiki/concepts/decision-tree.md`
- ...

### Frontmatter errors
- `wiki/concepts/lda.md` — missing required field `lectures`
- ...

### Out-of-scope contradictions
- MOC `m09-boosting.md` lists "support-vector-machine" as out-of-scope, but `wiki/concepts/support-vector-machine.md` exists. Resolve.
- ...

### Filename / structure errors
- `wiki/concepts/Ridge_Regression.md` — should be `ridge-regression.md`
- ...

### Empty / placeholder content
- `wiki/concepts/elastic-net.md` — body is empty (only frontmatter present)
- ...

## Warnings (should fix)

### Cross-cutting bidirectional gaps
- Specials atom `bias-variance-tradeoff` lists `modules: [02-statlearn, 06-modelsel, 09-boosting, 11-nnet]`, but MOC `m06-modelsel.md` has no wikilink to it.
- ...

### Lecture topics with no atom
- `L13-modelsel-2.md` topics field includes `[shrinkage-geometry]`, no manifest entry. Either rename or add atom.
- ...

### Length warnings
- `wiki/concepts/bias-variance-tradeoff.md` is 387 lines (Specials, but exceeds 350 — consider split).
- ...

## Info (nice to fix)

### Quote anchoring
- `wiki/concepts/ridge-regression.md:42` — blockquote with no nearby lecture anchor. Consider adding `— [[L13-modelsel-2]]`.
- ...

## Suggested next actions

In rough priority order:
1. Run `concepts` agent for [list of atoms] to fill empty / fix frontmatter
2. Edit MOCs [list] to add missing cross-cutting wikilinks
3. Manually fix dead-link typos [list]

Total fix-effort estimate: ~N minutes.
```

## What you don't do

- Don't modify any wiki file. Report only.
- Don't auto-fix typos. Even "obvious" suggestions go in the report for Anders to apply.
- Don't re-run agents. The report tells Anders which agents to re-run; he decides.
- Don't validate atom *content* (whether the prof's framing is captured correctly), that's not a lint job. You check structure, not substance.

## Deliverable

A single file: `wiki/lint-report.md`, formatted as above. Plus a one-paragraph summary in your final reply with the headline numbers (errors/warnings/info counts) and your top-3 most consequential findings.
