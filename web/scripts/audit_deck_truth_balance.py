#!/usr/bin/env python3
"""
audit_deck_truth_balance.py — true/false balance audit for T/F multi-statement
questions in a TMA4268 MCQ deck.

Reads one HTML deck, extracts every T/F multi-statement question's answer
sequence (the `<li><strong>True/False</strong>` markers in the fasit body),
and asks the basic question:

    For N total sub-statements across the deck, is the count of "true" markers
    consistent with N/2 — i.e., did the author flip a fair coin per statement?

Under the null hypothesis (each sub-statement is iid Bernoulli(0.5)), the total
count of trues follows Binomial(N, 0.5), so E[#true]=N/2 and Var[#true]=N/4.
Significant deviation lets a student game blanks: if the deck systematically
runs ~70% true, "guess True when stuck" is a better-than-random strategy.

Outputs to stdout:
  1. Per-question True/False counts (and intra-question imbalance flags).
  2. Global counts: N statements, k true, expected N/2, deviation in σ.
  3. χ² test against 50/50 (df=1, α=0.05 → 3.841, α=0.01 → 6.635).
  4. Exact two-tailed binomial p-value — the headline rarity score.
  5. Distribution of #true per 4-statement question vs Binomial(4, 0.5).
  6. Run-length analysis on the flat T/F sequence.
  7. All-true / all-false question flags.
  8. Final PASS / FAIL verdict.

Pure stdlib. Independent of and complementary to `audit_deck_lengths.py` and
`audit_deck_positions.py`: those handle single-MC; this one handles only the
T/F multi-statement format that they skip.

Usage:
  python web/scripts/audit_deck_truth_balance.py web/static/decks/m06-modelsel.html
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from itertools import groupby
from math import comb, sqrt
from pathlib import Path

ARTICLE_RE = re.compile(r'<article class="exam-q">(.*?)</article>', re.DOTALL)
QNUM_RE = re.compile(r'<span class="exam-q__num">Question\s+(\d+)</span>')
TF_BLOCK_RE = re.compile(r'<ul class="exam-q__tf">(.*?)</ul>', re.DOTALL)
TF_FIELD_RE = re.compile(r'class="exam-q__tf-field"')
FASIT_BODY_RE = re.compile(
    r'<div class="fasit-body">(.*?)</div>\s*</details>', re.DOTALL
)
ANSWER_OL_RE = re.compile(r'<ol>(.*?)</ol>', re.DOTALL)
ANSWER_LI_RE = re.compile(
    r'<li>\s*<strong>(True|False)</strong>', re.IGNORECASE
)

CHI2_CRIT_DF1_05 = 3.841   # χ² critical, df=1, α=0.05
CHI2_CRIT_DF1_01 = 6.635   # χ² critical, df=1, α=0.01
CHI2_CRIT_DF4_05 = 9.488   # χ² critical, df=4, α=0.05  (Binomial(4,0.5) → 5 bins)


def parse_questions(html: str) -> list[dict]:
    """Return list of T/F questions with qnum, statement count, truth sequence."""
    out: list[dict] = []
    for block in ARTICLE_RE.findall(html):
        tf = TF_BLOCK_RE.search(block)
        if not tf:
            continue  # not a T/F multi-statement question
        qm = QNUM_RE.search(block)
        if not qm:
            continue
        n_stmts = len(TF_FIELD_RE.findall(tf.group(1)))

        fb = FASIT_BODY_RE.search(block)
        if not fb:
            continue
        ol = ANSWER_OL_RE.search(fb.group(1))
        if not ol:
            continue
        truths = [m.lower() == 'true' for m in ANSWER_LI_RE.findall(ol.group(1))]

        # Take only the first n_stmts answers, in case the explanation has
        # extra <strong>True/False</strong> tags (rare but defensive).
        truths = truths[:n_stmts]
        if len(truths) != n_stmts:
            # Couldn't parse cleanly; skip rather than poison the stats.
            continue

        out.append({
            'qnum': int(qm.group(1)),
            'm': n_stmts,
            'truths': truths,
        })
    return out


def binomial_pmf(k: int, n: int, p: float = 0.5) -> float:
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binomial_two_tailed_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-tailed test: P(|X - np| >= |k - np|) under X ~ Binomial(n, p)."""
    if n == 0:
        return 1.0
    expected = n * p
    diff = abs(k - expected)
    pval = 0.0
    for i in range(n + 1):
        if abs(i - expected) >= diff - 1e-9:
            pval += binomial_pmf(i, n, p)
    return min(pval, 1.0)


def chi2_two_bin(k: int, n: int) -> float:
    """χ² statistic for (k, n-k) vs (n/2, n/2)."""
    if n == 0:
        return 0.0
    e = n / 2
    return (k - e) ** 2 / e + ((n - k) - e) ** 2 / e


def longest_run(seq: list[bool]) -> tuple[int, bool]:
    if not seq:
        return 0, True
    best_len, best_val = 0, seq[0]
    for val, group in groupby(seq):
        run = sum(1 for _ in group)
        if run > best_len:
            best_len, best_val = run, val
    return best_len, best_val


def expected_longest_run(n: int, k: int = 2) -> float:
    """log_k(n) + γ/ln(k); k=2 for binary T/F."""
    import math
    if n <= 0:
        return 0.0
    return math.log(n, k) + 0.5772 / math.log(k)


def bar(pct: float, scale: float = 4.0) -> str:
    return '█' * max(0, round(pct / scale))


def fmt_seq(truths: list[bool]) -> str:
    return ''.join('T' if t else 'F' for t in truths)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print((__doc__ or '').strip(), file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    questions = parse_questions(path.read_text())
    if not questions:
        print(f"No T/F multi-statement questions found in {path}.")
        return 1

    flat: list[bool] = []
    for q in questions:
        flat.extend(q['truths'])
    n = len(flat)
    k = sum(flat)
    n_q = len(questions)

    print(f"Deck: {path}")
    print(f"T/F multi-statement questions analysed: {n_q}")
    print(f"Total sub-statements: {n}   (single-MC questions are skipped — they have no T/F)")
    print()

    # ─── 1. Per-question table ──────────────────────────────────────────
    print("─── 1. Per-question True/False breakdown ───")
    extreme: list[int] = []  # qnums where all-true or all-false
    for q in questions:
        kt = sum(q['truths'])
        m = q['m']
        seq = fmt_seq(q['truths'])
        flag = ''
        if kt == m:
            flag = '   ⚠ ALL TRUE'
            extreme.append(q['qnum'])
        elif kt == 0:
            flag = '   ⚠ ALL FALSE'
            extreme.append(q['qnum'])
        print(f"  Q{q['qnum']:>2}: {kt}/{m} true   {seq}{flag}")
    print()

    # ─── 2. Global balance ──────────────────────────────────────────────
    print("─── 2. Global balance (all sub-statements pooled) ───")
    expected = n / 2
    pct_true = 100 * k / n
    sigma = sqrt(n / 4)
    z = (k - expected) / sigma if sigma > 0 else 0.0
    print(f"  N total statements:  {n}")
    print(f"  observed #true:      {k}      ({pct_true:.1f}%) {bar(pct_true)}")
    print(f"  observed #false:     {n-k}      ({100-pct_true:.1f}%) {bar(100-pct_true)}")
    print(f"  expected (50/50):    {expected:.1f}")
    print(f"  deviation:           {k - expected:+.1f}   ({z:+.2f}σ)")
    print()

    # ─── 3. χ² test vs 50/50 ────────────────────────────────────────────
    chi2 = chi2_two_bin(k, n)
    print("─── 3. χ² goodness-of-fit vs 50/50 ───")
    print(f"  χ² = {chi2:.3f}   (df=1)")
    print(f"  critical values: 3.841 at α=0.05,   6.635 at α=0.01")
    if chi2 > CHI2_CRIT_DF1_01:
        chi_verdict = 'FAIL — significant at α=0.01; truth balance is heavily skewed.'
    elif chi2 > CHI2_CRIT_DF1_05:
        chi_verdict = 'FAIL — significant at α=0.05; truth balance is skewed.'
    else:
        chi_verdict = 'OK — balance consistent with 50/50.'
    print(f"  {chi_verdict}")
    print()

    # ─── 4. Exact binomial p-value ──────────────────────────────────────
    pval = binomial_two_tailed_pvalue(k, n, 0.5)
    print("─── 4. Exact two-tailed binomial test (rarity score) ───")
    print(f"  H0: each statement is an iid fair-coin flip; #true ~ Binomial({n}, 0.5)")
    print(f"  H1: author has a true/false bias")
    print(f"  P(|X - {expected:.1f}| ≥ {abs(k - expected):.1f}) = {pval:.4f}")
    if pval < 0.001:
        rarity = f"≈1 in {round(1/pval):,} — extremely unlikely under fair coin"
    elif pval < 0.01:
        rarity = f"≈1 in {round(1/pval)} — very unlikely under fair coin"
    elif pval < 0.05:
        rarity = f"≈1 in {round(1/pval)} — unlikely under fair coin"
    elif pval < 0.20:
        rarity = f"≈1 in {round(1/pval)} — mildly atypical, within plausible range"
    else:
        rarity = "consistent with chance; nothing to fix"
    print(f"  Rarity: {rarity}.")
    if pval < 0.05:
        # How many flips to balance?
        target_k = round(expected)
        delta = k - target_k
        action = 'flip' if delta > 0 else 'flip'
        side_now = 'True→False' if delta > 0 else 'False→True'
        print(f"  Suggested fix: {action} {abs(delta)} statement(s) ({side_now}) to land at {target_k}/{n}.")
    print()

    # ─── 5. Per-question distribution (4-statement subset) ──────────────
    print("─── 5. Per-question #true distribution (4-statement Qs only) ───")
    four_stmt = [sum(q['truths']) for q in questions if q['m'] == 4]
    n4 = len(four_stmt)
    chi2_dist = 0.0
    if n4 == 0:
        print("  no 4-statement questions; skipping.")
        print()
    else:
        obs = Counter(four_stmt)
        # Binomial(4, 0.5): pmf = (1, 4, 6, 4, 1) / 16
        binom4 = [1/16, 4/16, 6/16, 4/16, 1/16]
        print(f"  {n4} four-statement questions; expected dist under Binomial(4, 0.5):")
        for kk in range(5):
            o = obs[kk]
            e = n4 * binom4[kk]
            if e > 0:
                chi2_dist += (o - e) ** 2 / e
            print(f"    {kk}/4 true:  observed {o:>2}   expected {e:>5.2f}   {bar(100*o/n4 if n4 else 0)}")
        print(f"  χ² = {chi2_dist:.3f}   (df=4, 5% critical = {CHI2_CRIT_DF4_05})")
        if chi2_dist > CHI2_CRIT_DF4_05:
            dist_verdict = 'FAIL — per-question shape departs from Binomial(4, 0.5).'
        else:
            dist_verdict = 'OK — per-question shape consistent with iid fair coins.'
        print(f"  {dist_verdict}")
        print()

    # ─── 6. Run-length analysis ─────────────────────────────────────────
    print("─── 6. Run-length analysis (linear T/F sequence across all questions) ───")
    run_len, run_val = longest_run(flat)
    exp_run = expected_longest_run(n, k=2)
    threshold = max(4, round(exp_run + 2.0))
    val_str = 'True' if run_val else 'False'
    print(f"  flat sequence: {fmt_seq(flat)}")
    print(f"  longest run: {run_len} consecutive '{val_str}'")
    print(f"  expected for {n} iid fair coins: ≈ {exp_run:.2f}   (flag threshold: ≥{threshold})")
    if run_len >= threshold:
        run_verdict = f'FAIL — run of {run_len} is unusually long; redistribute.'
    else:
        run_verdict = 'OK — no suspicious clustering.'
    print(f"  {run_verdict}")
    print()

    # ─── 7. Within-question extremes ────────────────────────────────────
    print("─── 7. Within-question extremes (all-true / all-false) ───")
    # Expected rate per Q: P(all same) = 2 * 0.5^m. For m=4: 1/8 = 12.5%.
    exp_extreme = sum(2 * 0.5 ** q['m'] for q in questions)
    obs_extreme = len(extreme)
    print(f"  observed: {obs_extreme}/{n_q} questions ({100*obs_extreme/n_q:.0f}%)")
    print(f"  expected under iid: ≈ {exp_extreme:.2f} questions ({100*exp_extreme/n_q:.0f}%)")
    if obs_extreme >= 2 and obs_extreme >= 2 * max(exp_extreme, 1):
        extreme_verdict = (
            f'FAIL — {obs_extreme} all-same questions is ≥2× the chance rate; '
            'students notice within-question patterns.'
        )
    else:
        extreme_verdict = 'OK — within-question extremes within expected range.'
    print(f"  {extreme_verdict}")
    if extreme:
        print(f"  flagged Qs: {', '.join('Q' + str(q) for q in extreme)}")
    print()

    # ─── Final verdict ──────────────────────────────────────────────────
    print("─── Final verdict ───")
    issues: list[str] = []
    if chi2 > CHI2_CRIT_DF1_05:
        issues.append(
            f"global truth balance χ²={chi2:.2f} (k={k}/{n}, {pct_true:.0f}% true) "
            f"exceeds α=0.05 critical {CHI2_CRIT_DF1_05}; p={pval:.3f}"
        )
    if n4 and chi2_dist > CHI2_CRIT_DF4_05:
        issues.append(
            f"per-question shape χ²={chi2_dist:.2f} on {n4} four-Qs exceeds {CHI2_CRIT_DF4_05}"
        )
    if run_len >= threshold:
        issues.append(f"longest run = {run_len} '{val_str}' (≥{threshold})")
    if obs_extreme >= 2 and obs_extreme >= 2 * max(exp_extreme, 1):
        issues.append(
            f"{obs_extreme} all-same questions vs ≈{exp_extreme:.1f} expected"
        )

    if issues:
        print("  FAIL — fix before publishing:")
        for i in issues:
            print(f"    • {i}")
        print("  Suggested action: flip the truth value of a handful of statements")
        print("  (and rewrite the explanation accordingly) to land near 50/50.")
        return 1
    else:
        print("  PASS — truth distribution is consistent with iid fair-coin flips;")
        print("  no exploitable bias toward 'always pick True' (or False).")
        return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
