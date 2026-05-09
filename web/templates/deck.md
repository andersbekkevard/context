# Deck template: per-module MCQ deck for TMA4268

This document defines the **shape and quality bar** of one per-module MCQ deck. Each deck is a standalone HTML file in `web/static/decks/m<NN>-<slug>.html` (e.g. `m06-modelsel.html`) that turns into an interactive practice page once Quartz copies it to the site root.

A deck is *not* a mock exam (those are deferred). It is a focused MC drill for one module, sized to ~25 questions, scoped strictly to that module's in-scope atoms + the cross-cutting specials it touches.

> Adapted from `databaser/eksamen/genererte/MAL.md`. The HTML contract and option-quality rules transfer wholesale; the module-specific patterns are grounded in TMA4268 instead.

---

## 1. Frame

- **Course context:** TMA4268 *Statistisk læring*. Final exam **2026-05-18**, 4 hours, open-book, no code, ISLR + handwritten A5 + calculator allowed (per `project_exam_logistics`).
- **Format of every deck question:** multiple choice. No drawing, no free-text, no "write SQL/code." Long-form past-exam problems get translated to MC per the rules in `docs/scope.md`.
- **Interactive:** the page is one HTML file linking shared `exam.css` + `exam.js`. Clicking an option locks the question, colors right/wrong, and auto-opens the explanation. The fixed score-tracker FAB keeps a running points/percent count.
- **Length per deck:** **20–30 questions**, totalling **100 points**. Bigger modules (03 linreg, 04 classif, 06 modelsel, 09 boosting, 11 nnet) lean toward 28–30; lighter modules (07 beyondlinear, 08 trees, 10 unsuper) toward 20–24.
- **Estimated time:** ~30–45 minutes of focused work per deck. Tracked in the header.
- **Difficulty:** challenge a student who has read the atoms, recall + application + reasoning, not just trivia. ~⅓ recall, ~⅓ application/computation, ~⅓ scenario/synthesis. See §6.

> [!important] Scope is non-negotiable
> `docs/scope.md` is the canonical authority for what's in and out. Out-of-scope material gets **no question**. Past-exam questions on out-of-scope topics are silently dropped, not adapted. If a topic is out per scope.md but visible in old exams, that's the prof retiring it, respect that.

---

## 2. File structure

Each deck is a self-contained HTML page at `web/static/decks/m<NN>-<slug>.html`. After Quartz build it sits at `/decks/m<NN>-<slug>.html`. Skeleton:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Module 06 — Model selection &amp; regularisation · TMA4268</title>

  <!-- Sync theme with the Quartz wiki (reads same localStorage key, falls back to OS) -->
  <script>
    (function () {
      try {
        var saved = localStorage.getItem("theme");
        var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        var theme = saved || (prefersDark ? "dark" : "light");
        document.documentElement.setAttribute("saved-theme", theme);
      } catch (e) {
        document.documentElement.setAttribute("saved-theme", "light");
      }
    })();
  </script>

  <!-- Quartz fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;700&family=Source+Sans+Pro:ital,wght@0,400;0,600;1,400;1,600&family=IBM+Plex+Mono:wght@400;600&display=swap">

  <link rel="stylesheet" href="exam.css">

  <!-- KaTeX for inline math -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$',  right: '$',  display: false},
              {left: '\\(', right: '\\)', display: false},
              {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
          });"></script>
</head>
<body>
  <main class="exam-shell">
    <header class="exam-head">
      <a href="/" class="back-link">← Back to wiki</a>
      <h1>Module 06 — Model selection &amp; regularisation</h1>
      <p class="meta">25 questions · 100 points · ~40 min</p>
      <p class="meta">
        Click an option to lock the answer; the explanation auto-opens.
        Score tracker bottom-left.
      </p>
    </header>

    <!-- one .exam-q per question — see §3 -->

  </main>
  <script src="exam.js"></script>
</body>
</html>
```

Notes:

- **Relative paths** to `exam.css` and `exam.js` (same folder). No absolute paths inside `static/decks/`.
- **Theme-sync inline script** must appear *before* any stylesheet link so the `saved-theme` attribute is set before paint, otherwise dark-mode users get a flash of light theme on first load.
- **Quartz fonts** are loaded from Google Fonts (Schibsted Grotesk / Source Sans Pro / IBM Plex Mono) so deck typography matches the rendered wiki. `exam.css` references them via `--bodyFont` / `--headerFont` / `--codeFont`.
- **KaTeX is required.** Decks are static HTML and don't inherit Quartz's site-wide KaTeX. The three KaTeX `<script>` / `<link>` tags load it from a CDN and run auto-render on `DOMContentLoaded`. Copy them verbatim from `_example.html`. Without them, every `$…$` shows up as raw source.
- **No Quartz chrome.** Decks are full-page focused exam UIs; Quartz's explorer/backlinks panels stay in `wiki/`-derived pages only. The deck's only nav is the `Back to wiki` link in the header.
- **One `<main class="exam-shell">`** wraps everything for layout.
- **Color tokens** (`var(--dark)`, `var(--secondary)`, `var(--correct-fg)`, …) are defined in `exam.css` and switch automatically with `[saved-theme="dark"]`. Do not hardcode hex values in deck HTML, let the stylesheet handle theming.
- **No `<section>`-level grouping** (Del 1 / Del 2 framing from the databaser template doesn't apply, TMA4268 doesn't split a single-module deck).

---

## 3. Per-question templates

`exam.js` supports two interactive question types: single-correct multiple choice and multi-statement true/false. Two further "shapes" are stylistic, they use the single-MC mechanic but vary the question body.

> [!important] No topic tag on questions
> Questions deliberately carry **no `exam-q__topic` span** in the header, only number, points, and (optionally) a *source* flag — see §3.5. Naming the atom (e.g. "ridge-regression") or the question shape (e.g. "computation") telegraphs the answer space and makes the deck easier than the exam will be. The atom link belongs in the explanation's `<p class="ref">` block, where it appears *after* the student commits to an answer. Do not re-add a topic field "for organisation", that's what the score-tracker and explanation links are for. *Source flags* in §3.5 are explicitly different — they say where the question came from, not what concept it tests, and are uncorrelated with the answer space.

### 3.1 Single-correct MC (the staple)

```html
<article class="exam-q">
  <header class="exam-q__head">
    <span class="exam-q__num">Question 7</span>
    <span class="exam-q__points">4 points</span>
  </header>
  <p class="q-text">…question text, may include $\LaTeX$, code, tables, small data…</p>
  <ul class="exam-q__opts">
    <li><span class="opt-label">A</span> …option A…</li>
    <li><span class="opt-label">B</span> …option B…</li>
    <li><span class="opt-label">C</span> …option C…</li>
    <li><span class="opt-label">D</span> …option D…</li>
  </ul>
  <details class="fasit-details">
    <summary>Show answer</summary>
    <div class="fasit-body">
      <span class="fasit-correct">Correct answer: C</span>
      <p>Why C is right — show the computation, cite the rule, name the prof's framing.</p>
      <p>Why the others are wrong: A …; B …; D …. Each distractor must have its own one-line dismissal.</p>
      <p class="ref">Atoms: <a href="/concepts/ridge-regression">ridge-regression</a>, <a href="/concepts/standardization">standardization</a>. Lecture: <a href="/lectures/L13-modelsel-2">L13-modelsel-2</a>.</p>
    </div>
  </details>
</article>
```

The literal substring `Correct answer: <LETTER>` inside `.fasit-correct` is what `exam.js` parses. **Do not paraphrase that string.**

### 3.2 Multi-statement true/false

For "which of the following statements about $X$ are true?", every statement is independently scored, points split evenly across statements within the question's total.

```html
<article class="exam-q">
  <header class="exam-q__head">
    <span class="exam-q__num">Question 12</span>
    <span class="exam-q__points">8 points</span>
  </header>
  <p class="q-text">Mark each statement about $k$-fold cross-validation as true or false.</p>
  <ul class="exam-q__tf">
    <li class="exam-q__tf-field">
      <span class="tf-stmt">a) LOOCV is a special case of $k$-fold with $k = n$.</span>
      <label><input type="radio" name="q12-a" value="true"> True</label>
      <label><input type="radio" name="q12-a" value="false"> False</label>
    </li>
    <li class="exam-q__tf-field">
      <span class="tf-stmt">b) $k$-fold CV with $k = 5$ generally has lower variance than LOOCV.</span>
      <label><input type="radio" name="q12-b" value="true"> True</label>
      <label><input type="radio" name="q12-b" value="false"> False</label>
    </li>
    <!-- 2–4 more statements -->
  </ul>
  <details class="fasit-details">
    <summary>Show answer</summary>
    <div class="fasit-body">
      <ol>
        <li><strong>True</strong> — by definition $k = n$ holds out one observation per fold.</li>
        <li><strong>True</strong> — LOOCV folds are nearly identical and the per-fold estimates are highly correlated, inflating variance.</li>
        <!-- … one bullet per statement … -->
      </ol>
      <p class="ref">Atoms: <a href="/concepts/k-fold-cv">k-fold-cv</a>, <a href="/concepts/leave-one-out-cv">leave-one-out-cv</a>.</p>
    </div>
  </details>
</article>
```

Constraints: each `name=` is unique per sub-statement; the explanation list has **exactly one** `<li>` per sub-statement, in order; the marker word is `<strong>True</strong>` or `<strong>False</strong>` (case-insensitive, also accepts `T`/`F`).

> Prefer T/F multi-statement to single-MC whenever the question is "which of the following…", `exam.js` doesn't natively support multi-correct MC, and forcing four exclusive options into a multi-truth question is the canonical way to make answers leak.

### 3.3 Computation question (uses 3.1's mechanic)

A single-MC question whose body is a numerical scenario and whose options are candidate answers. Standard recipe:

1. Question gives concrete numbers (n, p, σ², λ, sample size, etc.).
2. Options are *all four plausible values*, one correct, three from common arithmetic mistakes (forgetting a 2/3 factor, mixing up σ² and σ, off-by-one in degrees of freedom, wrong fold count, etc.).
3. Explanation shows the full computation step by step, then names *which mistake* each distractor encodes.

These are the densest learning signal, give them slightly more points (5–6 instead of 3–4).

### 3.4 Scenario question (uses 3.1's mechanic)

A single-MC question whose body sets up a small scenario (a paragraph, small table, small SVG figure, or printed R-style output) and asks for an interpretation. Most TMA4268 exam questions are this shape, see `exam_analysis.md` §3 for canonical patterns.

Distractors should be **typical wrong interpretations** (e.g. confusing log-odds with probability; reading a coefficient as a marginal effect; matching the wrong direction of the bias-variance trade-off). Cite a verbatim prof signal in the explanation when the trap was flagged in lecture.

---

## 4. Behavior contract (what `exam.js` does)

`exam.js` walks the DOM on load and:

1. Counts every `.exam-q` and reads its `.exam-q__points` value into the running total.
2. Builds the bottom-left score-tracker FAB and panel.
3. For each `.exam-q__opts`: parses `Correct answer: X` from the matching `.fasit-correct`, makes each `<li>` keyboard-clickable, locks on first click, paints right green / wrong red, auto-opens the `<details>`.
4. For each `.exam-q__tf`: reads each statement's `<strong>True/False</strong>` marker from the explanation `<ol>`, scores each radio choice independently, splits points evenly.

You don't write any JS in a deck. You write HTML and the parser does the rest.

---

## 5. Option-quality rules: the single most important section

The most common leakage is that the correct answer is silently *recognisable from form*, student doesn't read the question, just the options, and still picks right. Catch this before publishing:

- **Length parity is the biggest one.** Correct answer must NOT be systematically the longest. Count the characters before publishing. Across the whole deck, "longest", "middle", and "shortest" should each be the correct answer roughly equal numbers of times. If you find correct is consistently long, either trim the correct answer or pump the distractors up to match (add qualifiers, examples, hedging, same density as the correct option).
- **Position rotation.** Correct A/B/C/D should be roughly evenly distributed across the deck. If you're hitting "C-bias", swap orderings on a few questions.
- **No tell-tale words.** *Always*, *never*, *only*, *all*, *none* in an option flag it as a textbook-wrong absolute. *Typically*, *usually*, *often* flag it as a hedged-correct. Avoid both, or use them in correct *and* wrong options to neutralise.
- **Same grammatical form, same detail level.** If one option starts with a number, all do. If one is a full sentence, all are. If one uses jargon (e.g. "conflict-serializable", "irreducible error"), all do.
- **Distractors must be plausible.** Each wrong option = a real misunderstanding the explanation can name ("forgets the 2/3 fill factor", "uses $n$ instead of $n-1$"). No filler. A student who doesn't know the material should not be able to eliminate any option on form alone.
- **Independence of options.** No "A and B but not C". No two distractors that are nearly identical (effectively reduces it to a 3-choice). No four-options-all-variations-on-one-theme (student eliminates them as a group).
- **"None of the above" is correct in at most 1 of every 10 questions**, overuse weakens it as a concept check.

> [!check] Form-only test (do this for every question before publishing)
> Hide the question text. Read only the four options. Can you predict the correct answer? If yes, rewrite the options.

---

## 6. Difficulty mix per deck

Aim for a smooth distribution. Concrete target for a 25-question deck:

| Tier | Count | What it looks like |
|---|---|---|
| **Easy / recall** | 6–8 | Definition, formula, "which is true": should be answerable directly from the atom. |
| **Application / computation** | 8–10 | Plug numbers into a formula, evaluate a small expression, read a confusion matrix, decode a coefficient. |
| **Scenario / synthesis** | 7–9 | Small data + interpretation; cross-concept (atom A vs atom B); identify the trap; pick the right method for a stated goal. |

Triviality test: ask "could a student who has only skimmed the atom headlines pass this?" If yes for more than ~25% of the deck, push toward harder.

---

## 7. Curriculum coverage per deck

For module $N$ ($02$–$11$):

1. **Atoms in `docs/concepts-manifest.md` under module $N$**: every atom must have **at least one** question. Big atoms (richly developed by the prof, lots of exercise instances) get 2–4 questions across recall + application + scenario.

2. **Specials (cross-cutting atoms) whose `modules:` list includes $N$**: touch each in the deck for module $N$ via at least one question, framed in *this module's* context. Example: a CV question in m06 should be CV-of-ridge ("how do you tune $\lambda$?"), not CV-mechanics in the abstract, that lives in m05's deck.

3. **In-scope from `docs/scope.md`**: never out-of-scope. If a topic is on the explicit out-of-scope list for the module (with verbatim prof signal), do NOT write a question for it, even if you have a clever one in mind.

4. **Exercise alignment**: when an atom has `exercises:` entries in the manifest, at least one of its questions should mirror an exercise problem (with numbers changed). The prof emphasised "especially the exercises", let exercise patterns dominate.

5. **Past-exam patterns**: translate per `docs/scope.md`'s past-exam translation rules. If a past-exam question is on an in-scope topic but free-text, MC-ify it: show 4 candidate answers / 4 candidate plots / 4 candidate decompositions, ask which is correct. If a past-exam question is on out-of-scope material, drop it.

---

## 8. Linking from deck to wiki

Every question's explanation block ends with a `<p class="ref">…</p>` carrying:

- Atom links: `<a href="/concepts/<slug>">slug</a>` for every atom that's load-bearing for the question. Multiple atoms welcome.
- Optionally a lecture link: `<a href="/lectures/L<NN>-<slug>">L<NN>-<slug></a>` if the explanation cites a verbatim prof quote.

Quartz outputs the wiki at root URLs (`/concepts/<slug>`, `/lectures/L<NN>-<slug>`, `/mocs/m<NN>-<slug>`). The deck's links are root-relative absolute paths, they resolve cleanly when serving locally and survive any future deploy path move.

The reverse direction (MOC → deck) is wired in `docs/templates/mocs.md`'s `## Practice` section.

---

## 9. Sources to draw on (when authoring questions)

In priority order (matches `docs/scope.md`'s source hierarchy for *what's in scope*, but for *question content* exercises beat lectures):

1. `exercises/Exercise<N>/` and `exercises/compulsory-exercise-1.md`: past problems on this module. Vary numbers, keep the underlying mechanic. The prof said "especially the exercises".
2. `wiki/concepts/<slug>.md` for every atom in the module: formulas, prof framing, traps, exam signals. Direct lift of "Pitfalls" and "Exam signals" sections is golden, those *are* MC questions waiting to be formatted.
3. `wiki/lectures/L<NN>-*.md` for the module's lectures: verbatim prof quotes give T/F traps and scenario questions.
4. `exams/TMA4268_*_Exam.Rmd`: past exams. Cross-check against `docs/scope.md`'s translation rules. **Don't replicate verbatim**, change numbers, change scenarios, vary the trap angle.
5. `wiki/mocs/m<NN>-<slug>.md`: its `## Out of scope` section tells you what NOT to write about.

You do NOT touch `notes/` (off-limits), `book/` (deep treatment, not the prof's scope), or `archive/` (dead).

---

## 10. Pre-publish checklist (run before declaring a deck done)

- [ ] File at `web/static/decks/m<NN>-<slug>.html`.
- [ ] Links shared `exam.css` + `exam.js` from the same folder.
- [ ] Header carries module name + question count + total points + estimated time.
- [ ] Total points across all questions = **100**.
- [ ] **20–30 questions** (size proportional to module, see §1).
- [ ] Every atom in the module's manifest slice has ≥ 1 question.
- [ ] Every Special whose `modules:` includes this module has ≥ 1 question, framed in this module's context.
- [ ] Out-of-scope items in `docs/scope.md` have **no** questions.
- [ ] Difficulty mix ≈ ⅓ / ⅓ / ⅓ (recall / application / scenario).
- [ ] Each `.fasit-correct` reads exactly `Correct answer: <LETTER>`.
- [ ] Each explanation says **why each distractor is wrong**, not just why correct is right.
- [ ] Each explanation has a `<p class="ref">` with at least one atom wikilink.
- [ ] **Length parity:** correct answer is shortest, mid-length, longest at roughly equal frequencies across the deck.
- [ ] **Position rotation:** A/B/C/D distribution roughly even across the deck.
- [ ] **Form-only test** passed for every question (cover the question text, can you still pick the right answer? If yes, rewrite).
- [ ] No "None of the above" used as the correct answer more than ~1 in 10 questions.
- [ ] **Browser smoke test:** click every option in `_example.html`-style locally, confirm the FAB updates and explanations open.
- [ ] **Math renders:** the three KaTeX includes (CSS + two `<script defer>` tags with the `renderMathInElement` config) are present in `<head>`. Without them, `$…$` shows as raw source.

---

## 11. What this template does NOT cover

- **Mock exam sets**: deferred. When you decide to add full mock exams, build a sibling template `web/templates/mock-exam.md` reusing the same HTML/JS contract, just longer and with a `<section>` for each "part" if the prof reintroduces parts.
- **Numeric input questions**: out. MCQ over candidate values handles this.
- **Multi-correct MC ("select all that apply")**: out (`exam.js` doesn't support it). Use multi-statement T/F instead.
- **Open recall / flashcard mode**: out. The exam-page mechanic is the only deck UI.

---

## 12. Filename and naming

- Decks: `m<NN>-<slug>.html`, where slug matches the MOC slug exactly. Examples: `m02-statlearn.html`, `m06-modelsel.html`, `m11-nnet.html`.
- The leading `m` keeps decks sorted in module order under any alphabetical sort.
- The example/placeholder file is `_example.html`, leading underscore signals "not a real deck", sorts first. Delete or replace when no longer needed.
