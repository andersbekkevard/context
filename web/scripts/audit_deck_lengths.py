#!/usr/bin/env python3
"""
audit_deck_lengths.py — option-length parity audit for a TMA4268 MCQ deck.

Reads one HTML deck (e.g. web/static/decks/m06-modelsel.html), pulls every
single-MC question's four options + correct letter, and reports whether the
length of the correct option is statistically related to its correctness — the
canonical leak that lets students pick right answers from form alone.

Outputs to stdout:
  1. Rank distribution of the correct option's length (1 = shortest of the
     four, 4 = longest). Under a clean deck this is uniform.
  2. χ² test of that distribution against uniform (df=3, α=0.05 → 7.81).
  3. Mean character length of correct options vs mean of distractors.
  4. Position rotation of the correct letter (A/B/C/D balance).
  5. Per-question flags: any item whose correct option is ≥30% longer or
     ≤30% shorter than its mean distractor.
  6. Final PASS / FAIL verdict.

Pure stdlib. No third-party deps.

Usage:
  python web/scripts/audit_deck_lengths.py web/static/decks/m06-modelsel.html
"""

from __future__ import annotations

import html
import re
import sys
from collections import Counter
from pathlib import Path

ARTICLE_RE = re.compile(r'<article class="exam-q">(.*?)</article>', re.DOTALL)
QNUM_RE = re.compile(r'<span class="exam-q__num">Question\s+(\d+)</span>')
OPTS_BLOCK_RE = re.compile(r'<ul class="exam-q__opts">(.*?)</ul>', re.DOTALL)
LI_RE = re.compile(r'<li>(.*?)</li>', re.DOTALL)
LABEL_RE = re.compile(r'<span class="opt-label">([A-D])</span>\s*')
CORRECT_RE = re.compile(
    r'<span class="fasit-correct">Correct answer:\s*([A-D])</span>'
)
TAG_RE = re.compile(r'<[^>]+>')
WS_RE = re.compile(r'\s+')

CHI2_CRIT_DF3_05 = 7.815  # χ² critical value, df=3, α=0.05


def strip_text(s: str) -> str:
    """Strip HTML tags + collapse whitespace + decode entities."""
    s = TAG_RE.sub('', s)
    s = html.unescape(s)
    s = WS_RE.sub(' ', s).strip()
    return s


def parse_question(block: str):
    """Return dict with qnum, correct letter, and {A,B,C,D: text} or None."""
    qm = QNUM_RE.search(block)
    if not qm:
        return None
    opts_match = OPTS_BLOCK_RE.search(block)
    if not opts_match:
        return None  # T/F multi-statement: not a single-MC, skip
    cm = CORRECT_RE.search(block)
    if not cm:
        return None

    options: dict[str, str] = {}
    for li_html in LI_RE.findall(opts_match.group(1)):
        lm = LABEL_RE.search(li_html)
        if not lm:
            continue
        letter = lm.group(1)
        rest = LABEL_RE.sub('', li_html, count=1)
        options[letter] = strip_text(rest)

    if set(options.keys()) != {'A', 'B', 'C', 'D'}:
        return None

    return {
        'qnum': int(qm.group(1)),
        'correct': cm.group(1),
        'options': options,
    }


def chi2_vs_uniform(counts: list[int], total: int) -> float:
    if total == 0:
        return 0.0
    expected = total / len(counts)
    if expected == 0:
        return 0.0
    return sum((c - expected) ** 2 / expected for c in counts)


def bar(pct: float, scale: int = 4) -> str:
    return '█' * max(0, round(pct / scale))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print((__doc__ or '').strip(), file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    text = path.read_text()
    questions = []
    for block in ARTICLE_RE.findall(text):
        q = parse_question(block)
        if q is not None:
            questions.append(q)

    n = len(questions)
    if n == 0:
        print(f"No single-MC questions found in {path}.")
        return 1

    rank_counts: Counter[int] = Counter()
    letter_counts: Counter[str] = Counter()
    correct_lens: list[int] = []
    distr_lens: list[int] = []
    flagged: list[tuple[int, str, int, float, float]] = []
    rank_by_q: dict[int, int] = {}

    for q in questions:
        lengths = {ltr: len(t) for ltr, t in q['options'].items()}
        sorted_letters = sorted(lengths, key=lambda x: lengths[x])
        rank_of = {ltr: i + 1 for i, ltr in enumerate(sorted_letters)}
        r = rank_of[q['correct']]
        rank_counts[r] += 1
        rank_by_q[q['qnum']] = r
        letter_counts[q['correct']] += 1

        c_len = lengths[q['correct']]
        d_lens = [v for ltr, v in lengths.items() if ltr != q['correct']]
        mean_d = sum(d_lens) / len(d_lens)
        correct_lens.append(c_len)
        distr_lens.extend(d_lens)

        if mean_d > 0:
            ratio = c_len / mean_d
            if ratio >= 1.30 or ratio <= 0.70:
                flagged.append((q['qnum'], q['correct'], c_len, mean_d, ratio))

    print(f"Deck: {path}")
    print(f"Single-MC questions analysed: {n}")
    print(f"(T/F multi-statement questions are skipped — they have no A/B/C/D options.)")
    print()

    print("─── 1. Correct-option length rank ───")
    print("(rank 1 = shortest of the four; rank 4 = longest)")
    rank_labels = {1: 'shortest', 2: '        ', 3: '        ', 4: 'longest '}
    for r in (1, 2, 3, 4):
        c = rank_counts[r]
        pct = 100 * c / n
        flag = ''
        if r == 4 and pct > 35:
            flag = '   ⚠ correct biased toward LONGEST'
        if r == 1 and pct > 35:
            flag = '   ⚠ correct biased toward SHORTEST'
        print(f"  rank {r} ({rank_labels[r]}): {c:>2}/{n} ({pct:>4.0f}%) {bar(pct)}{flag}")

    chi2 = chi2_vs_uniform([rank_counts[r] for r in (1, 2, 3, 4)], n)
    print(f"  χ² vs uniform = {chi2:.2f}   (df=3, 5% critical = {CHI2_CRIT_DF3_05})")
    if chi2 > CHI2_CRIT_DF3_05:
        rank_verdict = 'FAIL — length is NOT iid w.r.t. correctness; the correct option is statistically distinguishable from distractors by length alone.'
    else:
        rank_verdict = 'OK — distribution consistent with iid (length does not reveal the answer).'
    print(f"  {rank_verdict}")
    print()

    print("─── 2. Mean character length ───")
    mc = sum(correct_lens) / len(correct_lens)
    md = sum(distr_lens) / len(distr_lens)
    ratio_overall = mc / md if md else 0
    print(f"  correct options:    {mc:>5.0f} chars (mean)")
    print(f"  distractors:        {md:>5.0f} chars (mean)")
    print(f"  ratio correct/distractor: {ratio_overall:.2f}×")
    if ratio_overall >= 1.20:
        mean_verdict = f'FAIL — correct options run {(ratio_overall-1)*100:.0f}% longer on average.'
    elif ratio_overall <= 0.85:
        mean_verdict = f'FAIL — correct options run {(1-ratio_overall)*100:.0f}% shorter on average.'
    else:
        mean_verdict = 'OK — within ±15% of distractor mean.'
    print(f"  {mean_verdict}")
    print()

    print("─── 3. Correct-letter rotation (A/B/C/D balance) ───")
    for ltr in ('A', 'B', 'C', 'D'):
        c = letter_counts[ltr]
        pct = 100 * c / n
        print(f"  {ltr}: {c:>2}/{n} ({pct:>4.0f}%) {bar(pct)}")
    chi2_pos = chi2_vs_uniform([letter_counts[l] for l in 'ABCD'], n)
    print(f"  χ² vs uniform = {chi2_pos:.2f}   (df=3, 5% critical = {CHI2_CRIT_DF3_05})")
    if chi2_pos > CHI2_CRIT_DF3_05:
        pos_verdict = 'FAIL — correct letter is biased; rotate positions.'
    else:
        pos_verdict = 'OK — A/B/C/D distribution is balanced.'
    print(f"  {pos_verdict}")
    print()

    print("─── 4. Per-question outliers ───")
    print("(correct option ≥30% longer or ≤30% shorter than mean distractor)")
    if flagged:
        flagged.sort(key=lambda t: -abs(t[4] - 1))
        for qnum, c, clen, mdlen, ratio in flagged:
            arrow = 'longer ' if ratio > 1 else 'shorter'
            print(f"  Q{qnum:>2} (correct {c}): {clen:>3} chars vs mean distractor {mdlen:>5.1f}   →  {ratio:.2f}× {arrow}")
        outlier_verdict = f'{len(flagged)}/{n} flagged ({100*len(flagged)/n:.0f}%).'
    else:
        outlier_verdict = '0 flagged. Every correct option is within ±30% of its question\'s mean distractor.'
    print(f"  {outlier_verdict}")
    print()

    rank4 = sorted([qn for qn, r in rank_by_q.items() if r == 4])
    rank1 = sorted([qn for qn, r in rank_by_q.items() if r == 1])
    if rank4:
        print(f"  Questions where correct is the LONGEST ({len(rank4)}/{n}): "
              + ", ".join(f"Q{q}" for q in rank4))
    if rank1:
        print(f"  Questions where correct is the SHORTEST ({len(rank1)}/{n}): "
              + ", ".join(f"Q{q}" for q in rank1))
    print()

    print("─── Final verdict ───")
    issues: list[str] = []
    if chi2 > CHI2_CRIT_DF3_05:
        issues.append(f"correct-option rank distribution biased (χ²={chi2:.2f} > {CHI2_CRIT_DF3_05})")
    if ratio_overall >= 1.20:
        issues.append(f"correct options average {ratio_overall:.2f}× distractor length")
    elif ratio_overall <= 0.85:
        issues.append(f"correct options average {ratio_overall:.2f}× distractor length (too short)")
    if chi2_pos > CHI2_CRIT_DF3_05:
        issues.append(f"correct-letter position biased (χ²={chi2_pos:.2f})")
    if len(flagged) > max(2, n // 4):
        issues.append(f"{len(flagged)} per-question outliers (>{n//4} threshold)")

    if issues:
        print("  FAIL — fix before publishing:")
        for i in issues:
            print(f"    • {i}")
        return 1
    else:
        print("  PASS — option length is iid w.r.t. correctness; no length-based leak detected.")
        return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
