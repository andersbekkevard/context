#!/usr/bin/env python3
"""
audit_deck_positions.py — correct-letter position-rotation audit for a TMA4268 MCQ deck.

Reads one HTML deck, extracts the correct letter (A/B/C/D) of every single-MC
question, and reports whether the correct letter is approximately uniformly
distributed across positions or whether some letter is statistically over- /
underrepresented — the "C-bias" failure mode that lets a student game the deck
by always picking C.

Outputs to stdout:
  1. Frequency table of A / B / C / D as the correct letter, with bar chart.
  2. χ² goodness-of-fit test against uniform (df=3, α=0.05 → 7.81).
  3. ±5pp approximate-equality check against the uniform mark (25%).
  4. Run-length analysis: longest stretch of consecutive same-letter answers
     and a flag if it exceeds chance expectation.
  5. The sequence of correct letters in order, so you can eyeball clusters.
  6. Final PASS / FAIL verdict.

Pure stdlib. No third-party deps. Independent of and complementary to
`audit_deck_lengths.py`: positions ≠ option lengths, both can leak separately.

Usage:
  python web/scripts/audit_deck_positions.py web/static/decks/m06-modelsel.html
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

ARTICLE_RE = re.compile(r'<article class="exam-q">(.*?)</article>', re.DOTALL)
QNUM_RE = re.compile(r'<span class="exam-q__num">Question\s+(\d+)</span>')
OPTS_BLOCK_RE = re.compile(r'<ul class="exam-q__opts">', re.DOTALL)
CORRECT_RE = re.compile(
    r'<span class="fasit-correct">Correct answer:\s*([A-D])</span>'
)

CHI2_CRIT_DF3_05 = 7.815   # χ² critical, df=3, α=0.05
CHI2_CRIT_DF3_01 = 11.345  # χ² critical, df=3, α=0.01

# Approximate-equality tolerance: each letter's share should sit within
# ±EQ_TOL_PP percentage points of 25%. 5pp on a 25-question deck = ±1.25
# questions — quite tight; 5pp on 30 = ±1.5. Reasonable target.
EQ_TOL_PP = 5.0


def parse_questions(html: str) -> list[tuple[int, str]]:
    """Return [(qnum, correct_letter), …] for every single-MC question."""
    out: list[tuple[int, str]] = []
    for block in ARTICLE_RE.findall(html):
        # Skip T/F multi-statement (no opts block)
        if not OPTS_BLOCK_RE.search(block):
            continue
        qm = QNUM_RE.search(block)
        cm = CORRECT_RE.search(block)
        if not qm or not cm:
            continue
        out.append((int(qm.group(1)), cm.group(1)))
    return out


def chi2_vs_uniform(counts: list[int], total: int) -> float:
    if total == 0:
        return 0.0
    expected = total / len(counts)
    if expected == 0:
        return 0.0
    return sum((c - expected) ** 2 / expected for c in counts)


def longest_run(seq: list[str]) -> tuple[int, str]:
    """Return (length, letter) of the longest consecutive same-letter run."""
    if not seq:
        return 0, ''
    best_len, best_letter = 0, seq[0]
    for letter, group in groupby(seq):
        run = sum(1 for _ in group)
        if run > best_len:
            best_len, best_letter = run, letter
    return best_len, best_letter


def expected_longest_run(n: int, k: int = 4) -> float:
    """Approximate expected length of the longest run for n iid uniform draws
    from k symbols. Closed-form approximation: log_k(n) + Euler-Mascheroni / ln(k).
    For k=4, n=25:  log_4(25) ≈ 2.32  →  ~2.7. For n=30:  ≈ 2.85.
    Treat anything ≥ ceil(this + 1.5) as suspicious.
    """
    import math
    if n <= 0:
        return 0.0
    return math.log(n, k) + 0.5772 / math.log(k)


def bar(pct: float, scale: float = 4.0) -> str:
    return '█' * max(0, round(pct / scale))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print((__doc__ or '').strip(), file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    questions = parse_questions(path.read_text())
    n = len(questions)
    if n == 0:
        print(f"No single-MC questions found in {path}.")
        return 1

    sequence = [c for _, c in questions]
    counts = Counter(sequence)

    print(f"Deck: {path}")
    print(f"Single-MC questions analysed: {n}")
    print(f"(T/F multi-statement questions are skipped — they have no positional A/B/C/D.)")
    print()

    print("─── 1. Correct-letter frequency ───")
    expected = n / 4
    for ltr in ('A', 'B', 'C', 'D'):
        c = counts[ltr]
        pct = 100 * c / n
        delta_pp = pct - 25.0
        marker = ''
        if abs(delta_pp) > EQ_TOL_PP:
            marker = f"   ⚠ {delta_pp:+.1f}pp from 25% target"
        print(f"  {ltr}: {c:>2}/{n} ({pct:>4.1f}%, expected {expected:>4.2f}) {bar(pct)}{marker}")
    print()

    print("─── 2. χ² goodness-of-fit vs uniform ───")
    chi2 = chi2_vs_uniform([counts[l] for l in 'ABCD'], n)
    print(f"  χ² = {chi2:.3f}   (df=3)")
    print(f"  critical values: 7.815 at α=0.05,   11.345 at α=0.01")
    if chi2 > CHI2_CRIT_DF3_01:
        chi_verdict = 'FAIL — significant at α=0.01; correct letter is strongly biased.'
    elif chi2 > CHI2_CRIT_DF3_05:
        chi_verdict = 'FAIL — significant at α=0.05; correct letter is biased.'
    else:
        chi_verdict = 'OK — distribution consistent with uniform (no positional leak).'
    print(f"  {chi_verdict}")
    print()

    print(f"─── 3. Approximate-equality check (each letter within ±{EQ_TOL_PP:.0f}pp of 25%) ───")
    breaches: list[tuple[str, int, float]] = []
    for ltr in 'ABCD':
        c = counts[ltr]
        pct = 100 * c / n
        if abs(pct - 25.0) > EQ_TOL_PP:
            breaches.append((ltr, c, pct))
    if not breaches:
        eq_verdict = f'OK — all four letters within ±{EQ_TOL_PP:.0f}pp of the 25% target.'
    else:
        eq_verdict = f'FAIL — {len(breaches)} letter(s) outside the ±{EQ_TOL_PP:.0f}pp window:'
    print(f"  {eq_verdict}")
    for ltr, c, pct in breaches:
        delta = pct - 25.0
        target_count = round(expected)
        delta_q = c - target_count
        action = 'reduce by' if delta_q > 0 else 'add'
        print(f"    {ltr}: {c}/{n} ({pct:.1f}%, {delta:+.1f}pp, {delta_q:+d} vs target {target_count}) — {action} {abs(delta_q)}")
    print()

    print("─── 4. Run-length analysis (longest consecutive same-letter streak) ───")
    run_len, run_letter = longest_run(sequence)
    exp_run = expected_longest_run(n, k=4)
    threshold = max(3, round(exp_run + 1.5))
    print(f"  longest run: {run_len} consecutive '{run_letter}' answers")
    print(f"  expected for {n} iid uniform draws: ≈ {exp_run:.2f}   (flag threshold: ≥{threshold})")
    if run_len >= threshold:
        run_verdict = f'FAIL — run of {run_len} is unusually long; redistribute correct letters.'
    else:
        run_verdict = 'OK — no suspicious clustering.'
    print(f"  {run_verdict}")
    print()

    print("─── 5. Sequence of correct letters (in question order) ───")
    qline = ' '.join(f"{q:>2}" for q, _ in questions)
    lline = ' '.join(f"{c:>2}" for _, c in questions)
    print(f"  Q#:     {qline}")
    print(f"  letter: {lline}")
    print()

    print("─── Final verdict ───")
    issues: list[str] = []
    if chi2 > CHI2_CRIT_DF3_05:
        issues.append(f"χ²={chi2:.2f} exceeds α=0.05 critical {CHI2_CRIT_DF3_05}")
    if breaches:
        breach_strs = [f"{l}={c}" for l, c, _ in breaches]
        issues.append(f"letters outside ±{EQ_TOL_PP:.0f}pp: " + ', '.join(breach_strs))
    if run_len >= threshold:
        issues.append(f"longest run = {run_len} '{run_letter}' (≥{threshold})")

    if issues:
        print("  FAIL — fix before publishing:")
        for i in issues:
            print(f"    • {i}")
        print("  Suggested action: pick a few questions and rotate the correct letter")
        print("  to a less-represented position, preserving the option content.")
        return 1
    else:
        print("  PASS — correct-letter distribution is approximately uniform; no positional leak detected.")
        return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
