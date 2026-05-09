# `web/`: TMA4268 exam-prep web layer

The web layer is two things glued together:

1. **Quartz** renders the Obsidian vault at `../wiki/` into a navigable static site (atoms, lectures, MOCs, with wikilinks, math, search, backlinks, graph view).
2. **Per-module MCQ decks** sit alongside as standalone HTML pages in `static/decks/`, served at `/decks/m<NN>-<slug>.html`. They're a hand-authored interactive practice surface, click an option to lock the answer, get coloured feedback, watch the score panel tick up. One deck per in-scope module (02–11).

Everything HTML/CSS/JS lives inside `web/`. The wiki lives outside (at `../wiki/`) and is referenced via Quartz's `-d` flag, never duplicated, never moved.

## Prerequisites

- **Node ≥ 22** (Quartz requirement). Check with `node -v`.
- **pnpm** for installs. Check with `pnpm -v`.

## Quick start

```bash
cd web
pnpm install                  # one-time, populates node_modules/
pnpm run dev                  # dev server with live reload at http://localhost:8080
```

Production-style build:

```bash
pnpm run build                # writes static site to public/
```

## What runs

`package.json` scripts:

- `dev` → `quartz build --serve -d ../wiki`, live preview, watches the wiki for changes.
- `build` → `quartz build -d ../wiki`, one-shot build into `public/`.

Both pass `-d ../wiki` so Quartz reads markdown source from outside this folder. The `content/` folder inside `web/` is unused; safe to ignore.

## Folder map

```
web/
  package.json              ← npm/pnpm manifest, dev/build scripts
  quartz.config.ts          ← Quartz config: KaTeX on, ignores notes/ + databaser/
  quartz.layout.ts          ← Quartz layout (default)
  quartz/                   ← Quartz engine source (don't edit)
  static/                   ← Quartz passthrough — anything here is copied verbatim to public/<file>
    decks/
      _example.html         ← stub deck (4 placeholder questions). Delete or replace.
      exam.css              ← deck styles (ported from databaser, English)
      exam.js               ← deck behaviour (click-to-lock + FAB tracker, English)
  templates/
    deck.md                 ← canonical structural + quality spec for a per-module deck
  prompts/
    deck-generation.md      ← agent brief — how to generate one deck for a module
  README.md                 ← this file
  content/                  ← unused (Quartz default; safe to ignore — we point at ../wiki via -d flag)
  public/                   ← build output (gitignored)
  node_modules/             ← deps (gitignored)
  LICENSE.txt               ← Quartz's MIT license; required to keep
  ...                       ← misc Quartz support files (tsconfig.json, globals.d.ts, etc.)
```

## How to generate a new deck

The per-module deck pipeline is **agent-driven**, a Claude agent reads a module's atoms, lectures, exercises, and past exams, then emits one HTML file under `static/decks/`.

Per-module recipe:

1. Confirm `wiki/concepts/<slug>.md` files exist for every atom in the target module's slice (`docs/concepts-manifest.md` filtered by `module:`).
2. Confirm `wiki/lectures/L<NN>-*.md` are in place for that module.
3. Optionally: confirm `wiki/mocs/m<NN>-<slug>.md` exists — its `## Out of scope` section saves the agent a scope lookup.
4. Launch one agent (foreground or worktree) with the brief at `prompts/deck-generation.md` and the target module slug. The agent reads inputs, applies `templates/deck.md`'s quality bar, writes the HTML deck.
5. Open the deck locally (`pnpm run dev` → `http://localhost:8080/decks/m<NN>-<slug>.html`) and click through it. Run the `templates/deck.md` §10 checklist.

Always-on rules (already encoded in the prompt):

- 20–30 questions, **100 points total**, mixed difficulty thirds.
- In-scope only per `docs/scope.md`. Out-of-scope topics get **no question**.
- Every atom in the module's slice gets ≥ 1 question.
- Form-only test must pass for every question (hide the question, can you still pick correct? if yes, rewrite).
- Length parity + A/B/C/D position rotation across the deck.

## After all decks exist

Add a deck index to `wiki/index.md` (or wherever you want navigation to land) listing all 10 modules. The MOC for each module already carries a `## Practice` section with the deck link (per `docs/templates/mocs.md`).

**Always link to decks with raw HTML `<a href="/decks/m<NN>-<slug>.html" target="_blank" rel="noopener">…</a>` — not markdown link syntax.** Decks are static pages that bypass Quartz's SPA pipeline; in-tab navigation from a wiki page leaves the deck loading with the wrong stylesheet until you refresh. `target="_blank"` forces a fresh page load in a new tab and avoids the bug.

## Deleting `databaser/`

When you're satisfied that nothing in `web/` references the old reference repo, delete it:

```bash
trash ../databaser
```

`web/` is fully self-contained — `exam.css`, `exam.js`, and `_example.html` were ported and translated; `databaser/MAL.md` was reworked into `web/templates/deck.md`; the chat widget was dropped. Nothing here imports from `databaser/`.

Verify before deleting:

```bash
rg -l 'databaser' .                  # should return only README files / quartz.config.ts ignorePattern
```

## Troubleshooting

- **`pnpm run dev` errors with "directory not found"** → check that `../wiki/` exists relative to `web/` (i.e. `wiki/` is a sibling of `web/` at the project root).
- **Quartz includes `notes/`** → it shouldn't (we point at `../wiki/` only, and `notes/` is a sibling, not a child). Check `quartz.config.ts` `ignorePatterns` if it ever does.
- **Math doesn't render in a deck** → decks are static HTML, not Quartz-rendered, so they don't inherit Quartz's KaTeX. Each deck must include the three KaTeX `<link>` + `<script defer>` tags in its `<head>` (copy from `_example.html`). For wiki pages (atoms, lectures, MOCs), KaTeX is auto-loaded by `Plugin.Latex({ renderEngine: "katex" })` in `quartz.config.ts`.
- **Deck FAB doesn't appear** → check the deck HTML loads `<script src="exam.js"></script>` at the end of `<body>`, and that the file path resolves (relative to the deck HTML's location).
- **Question-locks-but-explanation-doesn't-open** → the `.fasit-correct` text must read literally `Correct answer: <LETTER>`. Anything else fails the regex in `exam.js`.
- **T/F doesn't score** → the explanation `<ol>` must have one `<li>` per statement, each starting with `<strong>True</strong>` or `<strong>False</strong>`. Order matters (matches the sub-statement order).
