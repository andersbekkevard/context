# Web layer — architecture, deployment, lessons

The repo contains a `web/` folder that publishes the wiki and a set of per-module MCQ decks as a single static site. This page explains how it's wired, why it's wired that way, what to keep in mind when deploying, and the things we hit during the build that are easy to miss the second time.

The operational quick-start (install commands, dev workflow, deck-generation procedure) lives in [[../web/README]]. This file is the *why*, not the *how*.

## North star

Two products served from one URL:

1. **The wiki** — atom pages, lecture pages, MOCs, with wikilinks, math, search, backlinks, graph view. Reading and navigation surface.
2. **Per-module MCQ decks** — interactive practice pages at `/decks/m<NN>-<slug>.html`. Click an option, lock the answer, see the explanation, watch a running score. Practice surface.

Constraints we held to:

- Everything web-related (HTML, CSS, JS, configs) lives **inside `web/`**. The wiki stays at `wiki/` (its Obsidian home), referenced — never moved or duplicated.
- The site is **fully static**. No server, no API. Anyone with the URL gets the same experience locally or on Vercel.
- `web/` is **deletable in one move**: `trash web/` removes the entire web layer; nothing outside it depends on it.

## Pipelines, one output

| Pipeline | Source | Tool | Output |
|---|---|---|---|
| Wiki rendering | `wiki/**/*.md` (includes `wiki/book/*.md`, the silver ISLP chapters) | Quartz `ContentPage` + transformers | `web/public/<path>.html` |
| Deck passthrough | `web/static/decks/*.html` | Custom `WebStatic` emitter | `web/public/decks/*.html` |
| Course PDF passthrough | `pdfs/*.pdf` (repo root) | Custom `CoursePdfs` emitter | `web/public/pdfs/*.pdf` |

All run inside the same `pnpm run build` invocation and write into the same `public/` folder. Quartz emits HTML from markdown (atoms, lectures, MOCs, and the `wiki/book/` ISLP chapters all flow through the same pipeline, with KaTeX, search, graph); `WebStatic` copies hand-authored deck HTML through unchanged; `CoursePdfs` copies the curated, standardized-filename slide-deck and exercise PDFs into the build output.

This is intentional separation:
- The wiki uses markdown so Anders can edit it in Obsidian.
- ISLP chapters live at `wiki/book/` (silver tier — PDF→MD, finalized once with `title:` frontmatter, then immutable). They share the wiki render pipeline so they get full Quartz treatment for free.
- Decks use HTML directly because they need precise control over per-question structure markdown can't express, and Claude generates them via the brief at [[../web/prompts/deck-generation]] following [[../web/templates/deck]].
- PDFs live outside `wiki/` (so LLMs don't try to read binary content as context) and outside `web/` (so the deletable web layer rule still holds), as a sibling `pdfs/` folder. They're original course materials with standardized filenames; the emitter pulls them in.

## Folder layout (web/)

```
web/
  package.json              ← pnpm scripts: dev (build --serve -d ../wiki), build (build -d ../wiki)
  quartz.config.ts          ← KaTeX on, ignorePatterns for notes/databaser, registers WebStatic
  quartz.layout.ts
  quartz/                   ← Quartz engine source (cloned, .git stripped)
  plugins/
    WebStatic.ts            ← custom emitter: copies static/* → public/* (build + partial-emit on dev)
    CoursePdfs.ts           ← custom emitter: copies ../pdfs/*.pdf → public/pdfs/*.pdf
  static/decks/
    _example.html           ← stub deck (cats placeholder; verifies infra)
    exam.css                ← English port of databaser styles, Quartz tokens, light + dark
    exam.js                 ← English port; parses "Correct answer: X" + "<strong>True/False</strong>"
  templates/deck.md         ← per-module deck spec (English MAL adapted for TMA4268)
  prompts/deck-generation.md ← agent brief for Claude-driven deck generation
  README.md                 ← operational quick-start
```

## Quartz integration

### Reading from `../wiki/`

Quartz's default content directory is `content/` inside the project root. We override it with the CLI flag:

```
quartz build -d ../wiki
```

baked into both `dev` and `build` scripts in `package.json`. The wiki stays at the repo root, sibling to `web/`. Quartz scans `../wiki/` only, so `notes/`, `exercises/`, `transcripts/`, `archive/`, `modules/`, `databaser/` are never seen — they're outside the content tree.

The ISLP chapters at `wiki/book/*.md` are silver content (PDF→MD, finalized once with `title:` frontmatter, then immutable per CLAUDE.md). They live inside `wiki/` so Quartz renders them through the same pipeline as atoms/lectures/MOCs — no extra emitter, no sync script, no special case. URLs slug to `/book/<chapter>` (Quartz strips the input dir).

`ignorePatterns` in `quartz.config.ts` adds `notes/**` and `databaser/**` as belt-and-suspenders in case the content path ever changes.

### KaTeX

`Plugin.Latex({ renderEngine: "katex" })` is in the transformers list. Atom and lecture math (`$…$`, `$$…$$`) renders server-side at build time — KaTeX CSS gets injected into Quartz pages automatically.

Decks do **not** inherit this. They're not Quartz-rendered. Each deck must include the three KaTeX CDN tags in its `<head>` (see [[../web/templates/deck]] §2). `_example.html` is the canonical reference.

### `WebStatic` and `CoursePdfs` custom emitters

`Plugin.Static()` in Quartz copies *engine-internal* `quartz/static/` (icons, og-image) — not user passthrough assets. We added two sibling emitters:

- [[../web/plugins/WebStatic]] (~40 lines): globs `web/static/**`, copies each file to `public/<path>` (preserving subfolders). On `partialEmit` (dev incremental): listens for change events under `web/static/`, copies/deletes individual files. Result: editing a deck in dev mode triggers a hot-reload of the deck page.
- [[../web/plugins/CoursePdfs]] (~50 lines): same pattern but reads from `../pdfs/` (sibling of `web/` at the repo root) and emits into `public/pdfs/`. Holds the curated, standardized-filename slide decks + exercise PDFs.

Both emitters are registered in `quartz.config.ts` right after `Plugin.Static()`.

We considered three alternatives before this:

- **Symlink `wiki/decks` → `../web/static/decks`** — works but leaks deck files into the wiki tree (visible in Obsidian, in MOC folder listings). Violates the "everything inside web/" rule.
- **Postbuild script** (`cp -r static/* public/`) — works for `pnpm run build`, breaks dev's hot-reload.
- **Move decks to `web/quartz/static/decks/`** — pollutes the engine folder; URLs become `/static/decks/...` (uglier).

The custom emitter keeps decks inside `web/`, supports both build and dev, and integrates cleanly with Quartz's lifecycle.

## Deck integration with the wiki

### Theme sync

Quartz toggles dark mode by writing `theme=dark` (or `light`) to `localStorage` and setting `<html saved-theme="dark">`. Decks are separate pages but same origin → can read the same key.

A 7-line inline `<script>` at the very top of the deck `<head>`:

```html
<script>
  (function () {
    try {
      var saved = localStorage.getItem("theme");
      var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      var theme = saved || (prefersDark ? "dark" : "light");
      document.documentElement.setAttribute("saved-theme", theme);
    } catch (e) { document.documentElement.setAttribute("saved-theme", "light"); }
  })();
</script>
```

runs before any stylesheet, sets `saved-theme` before paint → no flash of light theme on dark-mode users. `exam.css` defines all colours via CSS variables under `:root` (light) and `:root[saved-theme="dark"]` (dark), mirroring Quartz's token names exactly: `--light`, `--dark`, `--secondary`, `--lightgray`, `--darkgray`, `--gray`, `--highlight`, `--bodyFont`, `--headerFont`, `--codeFont`.

### Fonts

The same Google Fonts Quartz loads (Schibsted Grotesk + Source Sans Pro + IBM Plex Mono) are pulled in via deck `<head>`. Headings use header font; body chrome uses body font; only `<code>`, `.q-code`, and `.q-table` use code font. We learnt the hard way that monospace anywhere else reads as "AI-generated" — reserve it for code/data.

No `text-transform: uppercase` anywhere — that pattern reads as a SaaS chrome cliché.

### Linking direction

| From → To | Mechanism |
|---|---|
| MOC → deck | Raw `<a href="/decks/m<NN>-<slug>.html" target="_blank" rel="noopener">` per [[templates/mocs]] `## Practice` section |
| Deck → atom | Raw `<a href="/concepts/<slug>">` inside each `.fasit-body .ref` |
| Wiki home → deck | Raw `<a href="/decks/...">` in `wiki/index.md` deck index |

**All deck links use raw HTML with `target="_blank" rel="noopener"`.** This is non-negotiable. Quartz's SPA router (`enableSPA: true`) intercepts same-origin link clicks and tries to morph the new page into the current DOM. Decks aren't Quartz-rendered, so morph leaves the page in a broken half-styled state until you hard-refresh. `target="_blank"` opens a new tab → fresh page load → works on first paint.

We did not disable SPA globally; in-wiki navigation between atoms/lectures/MOCs benefits from it.

## Vercel deployment

Configured via Vercel's project settings:

| Setting | Value |
|---|---|
| Root Directory | `web` |
| Build Command | `pnpm run build` |
| Output Directory | `public` |
| Install Command | `pnpm install --frozen-lockfile` |
| Node Version | 22 (auto-detected from `engines` in `package.json`) |

Plus `web/vercel.json` enabling clean URLs (see "Lessons learnt" below — without it, every Quartz-generated link 404s).

Vercel clones the repo, `cd`s to `web/`, runs install + build. Quartz reads from `../wiki/` (still accessible on the build machine because Vercel preserves the repo's full layout regardless of Root Directory). Vercel serves `web/public/` as the static site.

Things to remember:

- **`baseUrl` in `quartz.config.ts`** is currently `"localhost:8080"`. After the first successful deploy, change it to the actual domain (e.g. `tma4268.vercel.app`). It only affects sitemap.xml and canonical URL tags, not navigation.
- **Repo visibility ≠ site visibility.** The Vercel-served URL is publicly accessible; the source repo is whatever GitHub says (private by default). Don't conflate. The deployed site contains only `web/public/` content (rendered wiki pages, the silver ISLP chapters, and decks). Bronze (`exams/`, `transcripts/`, `archive/`, `modules/`, `exercises/`) is in the source repo but never on the live site.
- **Auto-deploy on push.** Each push to `main` triggers a Vercel rebuild. Preview deploys for branches/PRs come for free.

## Patterns we settled on

### Hand-authored deck HTML

Each deck is one ~600-line HTML file with literal `<article class="exam-q">` blocks. Verbose, but:

- Source = runtime artefact. No build step beyond Quartz.
- Claude generates HTML directly per the deck-generation brief; the contract is a single regex (`Correct answer: X` and `<strong>True/False</strong>` markers).
- Editing one option = editing one `<li>`. No YAML→HTML pipeline to debug.

We considered a markdown-spec → HTML build script. Rejected because Quartz already adds one build pipeline; doubling that for the small number of decks (10) wasn't worth the maintenance cost.

### Two-source separation, one output folder

Quartz handles markdown → HTML; `WebStatic` handles `static/` → output. Both write into `public/`. The folder boundary is what separates them, not a config setting. Adding a future passthrough (e.g. a hand-authored landing page, an embedded Quartz-incompatible widget) means dropping a file in `web/static/`, not editing config.

### Quartz tokens, locally redefined

`exam.css` doesn't `@import` Quartz's CSS. It re-declares the same CSS variable values under `:root` and `:root[saved-theme="dark"]`. This:

- Avoids importing Quartz's full CSS bundle (which carries chrome we don't want on decks — explorer, popovers, graph view).
- Decouples deck visuals from Quartz upgrades. If Quartz's theme changes in v5, decks won't break.
- Makes the decks self-contained — they look right even if loaded outside the Quartz site (e.g. opened directly from disk, served by a different host).

### Decks always open in new tab

See "Linking direction" above. The cost of one extra tab is small; the cost of broken first-paint is repeated every session.

## Lessons learnt

Things that surprised us during the build, recorded so we don't relearn:

1. **Quartz's `Plugin.Static()` ships engine assets only.** It does not copy a project-level `static/` folder. User passthrough needs a custom emitter (see `WebStatic`). The naming collision is real; spent half an hour wondering where the deck files went.

2. **Quartz SPA does not work for non-Quartz pages.** Static HTML at the same origin gets intercepted by the SPA morph and ends up partially styled. `target="_blank"` is the lightest fix. Disabling SPA globally is the heaviest.

3. **Quartz strips the `.html` extension on internal-looking links — but Vercel does NOT auto-resolve them by default.** `/decks/_example.html` becomes `./decks/_example` in Quartz's rendered output. The Quartz dev server's `serve-handler` transparently resolves the extension-less form to `*.html`, so locally everything works. Vercel does not — `https://<site>/decks/_example` returns 404 even though `_example.html` is deployed correctly. Fix: add `web/vercel.json` with `{ "cleanUrls": true }`. Vercel then serves both `/decks/_example` and `/decks/_example.html`, redirecting the latter to the canonical extension-less form. Without this file, every internal link in the rendered site 404s on first navigation. Discovered after the first deploy — file is now committed.

4. **`<details>` doesn't auto-open via the `open` attribute alone after JS-triggered changes.** `exam.js` sets `fasit.open = true` on the JS property to open the explanation; the corresponding HTML attribute writes itself.

5. **Mono fonts on chrome read as AI slop.** IBM Plex Mono is for code only. Source Sans Pro for body chrome (labels, score numbers, meta lines). Schibsted Grotesk for headings. Same convention Quartz uses.

6. **`text-transform: uppercase` is a SaaS tell.** Avoid even on small labels. Title Case + a slightly heavier weight reads cleaner.

7. **KaTeX's auto-render needs explicit delimiters config to handle `$…$`.** The default delimiter list excludes single-`$` for inline math (because of the dollar-sign-in-prose conflict). Pass `{ delimiters: [...] }` with both `$` and `$$` listed.

8. **Quartz expects Node 22.** Older Node versions error on its TypeScript imports. `engines.node: ">=22"` in `package.json` makes Vercel pick the right version automatically.

9. **`pnpm install` warns about ignored build scripts** (`@parcel/watcher`, `esbuild`, `sharp`). Build still works because pnpm ships prebuilt binaries for these on most platforms. If a runtime issue ever surfaces, `pnpm approve-builds` (or set `pnpm.onlyBuiltDependencies` in `package.json`).

10. **`.gitignore` belongs at two levels.** Root `.gitignore` for repo-wide concerns (`.DS_Store`, `notes/`, `databaser/`, `.obsidian/`). `web/.gitignore` for the web project's build artefacts (`public/`, `node_modules/`, `.quartz-cache/`, `.vercel/`). Don't put `web/node_modules/` in the root file — it works but mixes concerns and breaks if `web/` is ever moved.

11. **Untracking ≠ deleting.** Files like `archive/.../.DS_Store` and `notes/exam-lecture.md` were tracked from earlier commits. Adding them to `.gitignore` doesn't untrack — `git rm --cached <path>` does. The file stays on disk; only git's index changes.

## What this layer does *not* do

- **No mock-exam sets yet.** Per-module decks only. When mock exams arrive, build a sibling template (`web/templates/mock-exam.md`) reusing the same `exam.js` mechanic with a longer page.
- **No multi-correct MC.** `exam.js` doesn't support it. Use multi-statement T/F (`.exam-q__tf-field`) for "select all that apply"-style questions.
- **No numeric input.** MCQ-over-candidate-values handles this.
- **No flashcards.** Considered during design; ruled out in favour of one mechanic (the exam-page format).
- **No render of `exercises/`, `exams/`, `transcripts/`, `archive/`, `modules/`, `notes/`.** All outside `wiki/`, none on the deployed site. Source repo is the only place they exist; keep it private. Exceptions deliberately surfaced: course PDFs at `pdfs/` (sibling folder, copied by `CoursePdfs` emitter) and the silver ISLP chapters at `wiki/book/` (rendered by Quartz like any other wiki page) — those *are* on the live site.

## The `pdfs/` folder

A sibling of `wiki/` and `web/` at the repo root. Holds the curated, standardized-filename copies of:

- 12 slide decks: `m{NN}-{slug}-slides.pdf` — annotated lecture slides where available, plain slides otherwise.
- 10 recommended exercises: `m{NN}-{slug}-exercise.pdf` — un-solved exercise statements (no solutions PDFs).
- 2 compulsory exercises: `compulsory-1.pdf`, `compulsory-2.pdf`.

Filenames mirror the MOC + deck slug convention. Sources came from `archive/modules/<NN><Name>/` and `archive/exercises/Exercise<N>/` via a one-time fan-out of 24 Opus subagents (each picked the right PDF and `cp`-ed it). Anders won't re-run that — `pdfs/` is now canonical and `archive/` can be deleted whenever.

Why a sibling folder rather than `web/static/pdfs/`?

- **LLMs don't read it.** Agents working on the wiki, the atoms, the decks should never load a binary PDF as context. Keeping PDFs outside `wiki/` and `web/` makes that clear architecturally — there's no path that would pull them in. Compare to atoms (read), lectures (read), templates (read), prompts (read) — none of those touch `pdfs/`.
- **`web/` stays deletable.** If `web/` were ever scrapped, the PDFs would survive.
- **Symmetry with `wiki/`.** Both are content sources the build pulls from. `wiki/` for markdown rendering, `pdfs/` for binary passthrough.
