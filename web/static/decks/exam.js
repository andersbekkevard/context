/* ==========================================================================
   TMA4268 deck quiz: click-to-lock MC + multi-statement T/F + score tracker.
   Pairs with exam.css and the HTML structure described in templates/deck.md.
   Ported and simplified from databaser/eksamen/genererte/exam.js (English,
   no A–F grade ladder).
   ========================================================================== */

(function () {
  'use strict';

  const STORAGE_COLLAPSED = 'tma4268_exam_tracker_collapsed';

  const score = {
    earned: 0,
    totalPoints: 0,
    answered: 0,
    totalQuestions: 0,
    rootEl: null,
    panelEl: null,
    fabEl: null,
    fabPctEl: null,
    fabFracEl: null,
    earnedEl: null,
    totalEl: null,
    pctEl: null,
    answeredEl: null,
    qTotalEl: null,
    barFillEl: null,
  };

  function init() {
    const articles = document.querySelectorAll('.exam-q');
    if (!articles.length) return;

    articles.forEach((article) => {
      const points = parsePoints(article.querySelector('.exam-q__points'));
      article.dataset.examPoints = String(points);
      score.totalPoints += points;
      score.totalQuestions += 1;
    });

    buildTracker();
    articles.forEach(setupQuestion);
  }

  function setupQuestion(article) {
    setupMultiChoice(article);
    setupTrueFalse(article);
    setupCloze(article);
  }

  function parsePoints(el) {
    if (!el) return 0;
    const m = el.textContent.match(/(\d+(?:[.,]\d+)?)/);
    if (!m) return 0;
    return parseFloat(m[1].replace(',', '.'));
  }

  function fmtPoints(n) {
    const rounded = Math.round(n * 10) / 10;
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  }

  function fmtPercent(p) {
    const rounded = Math.round(p * 10) / 10;
    return (Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)) + '%';
  }

  function loadCollapsedPref() {
    try {
      return localStorage.getItem(STORAGE_COLLAPSED) === '1';
    } catch (e) {
      return false;
    }
  }

  function saveCollapsedPref(collapsed) {
    try {
      localStorage.setItem(STORAGE_COLLAPSED, collapsed ? '1' : '0');
    } catch (e) {}
  }

  function setCollapsed(collapsed) {
    if (!score.rootEl) return;
    score.rootEl.classList.toggle('is-collapsed', collapsed);
    if (score.fabEl) {
      score.fabEl.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    }
    saveCollapsedPref(collapsed);
  }

  function buildTracker() {
    const root = document.createElement('div');
    root.className = 'exam-tracker';

    root.innerHTML = [
      '<button type="button" class="exam-tracker__fab" aria-label="Open score panel">',
      '  <span class="exam-tracker__fab-pct">0%</span>',
      '  <span class="exam-tracker__fab-frac">0/0</span>',
      '</button>',
      '<aside class="exam-tracker__panel" role="region" aria-label="Score panel">',
      '  <div class="exam-tracker__header">',
      '    <span class="exam-tracker__title">Score</span>',
      '    <button type="button" class="exam-tracker__close" aria-label="Close score panel">×</button>',
      '  </div>',
      '  <div class="exam-tracker__row">',
      '    <span class="exam-tracker__label">Points</span>',
      '    <span class="exam-tracker__score">',
      '      <span class="exam-tracker__earned">0</span>',
      '      <span class="exam-tracker__sep">/</span>',
      '      <span class="exam-tracker__total">0</span>',
      '    </span>',
      '  </div>',
      '  <div class="exam-tracker__row">',
      '    <span class="exam-tracker__label">Percent</span>',
      '    <span class="exam-tracker__pct">0%</span>',
      '  </div>',
      '  <div class="exam-tracker__bar"><div class="exam-tracker__bar-fill"></div></div>',
      '  <div class="exam-tracker__row exam-tracker__sub">',
      '    <span>Answered</span>',
      '    <span><span class="exam-tracker__answered">0</span> / <span class="exam-tracker__qtotal">0</span></span>',
      '  </div>',
      '</aside>',
    ].join('');

    document.body.appendChild(root);

    score.rootEl = root;
    score.panelEl = root.querySelector('.exam-tracker__panel');
    score.fabEl = root.querySelector('.exam-tracker__fab');
    score.fabPctEl = root.querySelector('.exam-tracker__fab-pct');
    score.fabFracEl = root.querySelector('.exam-tracker__fab-frac');
    score.earnedEl = root.querySelector('.exam-tracker__earned');
    score.totalEl = root.querySelector('.exam-tracker__total');
    score.pctEl = root.querySelector('.exam-tracker__pct');
    score.answeredEl = root.querySelector('.exam-tracker__answered');
    score.qTotalEl = root.querySelector('.exam-tracker__qtotal');
    score.barFillEl = root.querySelector('.exam-tracker__bar-fill');

    score.totalEl.textContent = fmtPoints(score.totalPoints);
    score.qTotalEl.textContent = String(score.totalQuestions);

    score.fabEl.addEventListener('click', () => {
      setCollapsed(!score.rootEl.classList.contains('is-collapsed'));
    });
    root.querySelector('.exam-tracker__close').addEventListener('click', () => setCollapsed(true));

    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      if (!score.rootEl || score.rootEl.classList.contains('is-collapsed')) return;
      const ae = document.activeElement;
      if (score.panelEl && score.panelEl.contains(ae)) setCollapsed(true);
    });

    setCollapsed(loadCollapsedPref());
    updateTracker();
  }

  function updateTracker() {
    if (!score.rootEl) return;

    score.earnedEl.textContent = fmtPoints(score.earned);
    score.answeredEl.textContent = String(score.answered);

    const pct = score.totalPoints > 0
      ? Math.max(0, Math.min(100, (score.earned / score.totalPoints) * 100))
      : 0;

    score.barFillEl.style.width = pct.toFixed(1) + '%';
    score.pctEl.textContent = fmtPercent(pct);
    score.fabPctEl.textContent = fmtPercent(pct);
    score.fabFracEl.textContent = `${fmtPoints(score.earned)}/${fmtPoints(score.totalPoints)}`;

    const collapsed = score.rootEl.classList.contains('is-collapsed');
    score.fabEl.setAttribute(
      'aria-label',
      collapsed
        ? `Open score panel. ${fmtPercent(pct)}, ${fmtPoints(score.earned)} of ${fmtPoints(score.totalPoints)} points`
        : 'Close score panel'
    );

    if (score.answered >= score.totalQuestions && score.totalQuestions > 0) {
      score.rootEl.classList.add('is-complete');
    }
  }

  function addEarned(points) {
    score.earned += points;
    updateTracker();
  }

  function markAnswered() {
    score.answered += 1;
    updateTracker();
  }

  /* --- Multiple choice (single correct) --- */

  function setupMultiChoice(article) {
    const opts = article.querySelector('.exam-q__opts');
    if (!opts) return;

    const correctSpan = article.querySelector('.fasit-correct');
    if (!correctSpan) return;

    const match = correctSpan.textContent.match(/Correct answer:\s*([A-Z])/i);
    if (!match) return;
    const correctLetter = match[1].toUpperCase();

    const fasit = article.querySelector('.fasit-details');
    const summary = fasit ? fasit.querySelector('summary') : null;
    if (summary) summary.style.display = 'none';

    opts.classList.add('is-quiz');
    const items = Array.from(opts.querySelectorAll(':scope > li'));

    items.forEach((li) => {
      const labelEl = li.querySelector('.opt-label');
      if (!labelEl) return;
      const letter = labelEl.textContent.trim().toUpperCase();
      li.dataset.letter = letter;
      li.classList.add('is-clickable');
      li.setAttribute('role', 'button');
      li.setAttribute('tabindex', '0');

      const handler = () => selectAnswer(article, opts, items, li, letter, correctLetter, fasit);
      li.addEventListener('click', handler);
      li.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handler();
        }
      });
    });
  }

  function selectAnswer(article, opts, items, chosenLi, chosenLetter, correctLetter, fasit) {
    if (opts.classList.contains('is-answered')) return;
    opts.classList.add('is-answered');

    const isCorrect = chosenLetter === correctLetter;

    items.forEach((li) => {
      li.classList.remove('is-clickable');
      li.removeAttribute('tabindex');
      li.removeAttribute('role');
      const letter = li.dataset.letter;
      if (letter === correctLetter) {
        li.classList.add('is-correct');
      }
      if (li === chosenLi && !isCorrect) {
        li.classList.add('is-wrong');
      }
    });

    const points = parseFloat(article.dataset.examPoints) || 0;
    if (isCorrect) addEarned(points);
    markAnswered();

    if (fasit && !fasit.open) fasit.open = true;
  }

  /* --- Multi-statement true/false --- */

  function setupTrueFalse(article) {
    const tfList = article.querySelector('.exam-q__tf');
    if (!tfList) return;

    const fasit = article.querySelector('.fasit-details');
    const fasitOl = fasit ? fasit.querySelector('.fasit-body ol') : null;
    if (!fasitOl) return;

    const answers = Array.from(fasitOl.querySelectorAll(':scope > li')).map((li) => {
      const strong = li.querySelector('strong');
      if (!strong) return null;
      const txt = strong.textContent.trim().toLowerCase();
      if (txt === 'true' || txt === 't') return 'true';
      if (txt === 'false' || txt === 'f') return 'false';
      return null;
    });

    const fields = Array.from(tfList.querySelectorAll('.exam-q__tf-field'));
    if (!fields.length) return;

    tfList.classList.add('is-quiz');

    const totalPoints = parseFloat(article.dataset.examPoints) || 0;
    const scoringFields = fields.filter((_, idx) => answers[idx]);
    const pointsPerField = scoringFields.length > 0 ? totalPoints / scoringFields.length : 0;

    fields.forEach((field, idx) => {
      const correct = answers[idx];
      if (!correct) return;
      field.dataset.correct = correct;

      const inputs = field.querySelectorAll('input[type="radio"]');
      inputs.forEach((input) => {
        input.addEventListener('change', () => handleTfChange(field, input, correct, tfList, fasit, pointsPerField));
      });
    });
  }

  function handleTfChange(field, input, correct, tfList, fasit, pointsPerField) {
    if (field.classList.contains('is-answered')) return;
    field.classList.add('is-answered');

    const isCorrect = input.value === correct;
    field.classList.add(isCorrect ? 'is-correct' : 'is-wrong');

    field.querySelectorAll('input[type="radio"]').forEach((r) => {
      if (r !== input) r.disabled = true;
    });

    if (!field.querySelector('.exam-q__tf-status')) {
      const status = document.createElement('span');
      status.className = 'exam-q__tf-status';
      status.textContent = isCorrect ? '✓ Correct' : '✗ Wrong';
      field.appendChild(status);
    }

    if (isCorrect) addEarned(pointsPerField);

    const allAnswered = Array.from(tfList.querySelectorAll('.exam-q__tf-field'))
      .every((f) => f.classList.contains('is-answered'));
    if (allAnswered) {
      markAnswered();
      if (fasit && !fasit.open) fasit.open = true;
    }
  }

  /* --- Cloze (fill-in-the-blank paragraph) ---
     Each <select.cloze-blank data-correct="..."> is one blank. The block is
     graded only when the .cloze-check button is clicked, so distractors
     elsewhere in the paragraph can disambiguate the answer (matches the
     prof's "you might only understand which answer is correct after you
     continued reading" framing on the past-exam Problem 1). */

  function setupCloze(article) {
    const cloze = article.querySelector('.exam-q__cloze');
    if (!cloze) return;

    const blanks = Array.from(cloze.querySelectorAll('.cloze-blank'));
    if (!blanks.length) return;

    const checkBtn = cloze.querySelector('.cloze-check');
    if (!checkBtn) return;

    const progressEl = cloze.querySelector('.cloze-progress');
    const fasit = article.querySelector('.fasit-details');
    const summary = fasit ? fasit.querySelector('summary') : null;
    if (summary) summary.style.display = 'none';

    const totalPoints = parseFloat(article.dataset.examPoints) || 0;
    const pointsPerBlank = blanks.length > 0 ? totalPoints / blanks.length : 0;

    const updateProgress = () => {
      if (!progressEl) return;
      const filled = blanks.filter((b) => b.value).length;
      progressEl.textContent = `${filled} / ${blanks.length} filled`;
      progressEl.classList.toggle('is-complete', filled === blanks.length);
    };

    blanks.forEach((blank) => {
      blank.addEventListener('change', () => {
        if (blank.value) blank.classList.add('is-filled');
        else blank.classList.remove('is-filled');
        updateProgress();
      });
    });

    updateProgress();

    checkBtn.addEventListener('click', () => {
      if (cloze.classList.contains('is-answered')) return;
      cloze.classList.add('is-answered');

      let earned = 0;
      blanks.forEach((blank) => {
        const correct = (blank.dataset.correct || '').trim();
        const chosen = (blank.value || '').trim();
        const isCorrect = chosen !== '' && chosen === correct;

        blank.disabled = true;
        blank.classList.add(isCorrect ? 'is-correct' : 'is-wrong');

        if (!isCorrect) {
          const correction = document.createElement('span');
          correction.className = 'cloze-correction';
          correction.textContent = correct;
          blank.insertAdjacentElement('afterend', correction);
        }

        if (isCorrect) earned += pointsPerBlank;
      });

      checkBtn.disabled = true;
      checkBtn.textContent = 'Answer locked';

      addEarned(earned);
      markAnswered();

      if (fasit && !fasit.open) fasit.open = true;
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
