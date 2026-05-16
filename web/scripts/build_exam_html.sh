#!/usr/bin/env bash
# Convert all 10 mock-exam .tex sources (and their solution sketches) to
# stand-alone LaTeX-looking HTML files, served via the WebStatic emitter
# at /exams/<name>.html. Run from anywhere; paths are resolved relative
# to the repo root.
#
# Re-run whenever a mock-exam .tex file changes. Output is committed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_ROOT="$REPO_ROOT/mock-exams"
OUT_ROOT="$REPO_ROOT/web/static/exams"
TEMPLATE="$REPO_ROOT/web/scripts/exam_template.html"
FILTER="$REPO_ROOT/web/scripts/exam_tikz_filter.lua"

mkdir -p "$OUT_ROOT"

# Discover all mock-exam .tex sources. Mocks 1–7 sit directly in
# mock-exams/; mocks 8–10 live under mock-exams/mock-N/.
TEX_FILES=$(
  {
    find "$SRC_ROOT" -maxdepth 1 -name 'mock-exam-*.tex' -print
    find "$SRC_ROOT" -mindepth 2 -name 'mock-exam-*.tex' -print
  } | sort
)

if [[ -z "$TEX_FILES" ]]; then
  echo "No mock-exam .tex files found under $SRC_ROOT" >&2
  exit 1
fi

count=0
while IFS= read -r tex; do
  count=$((count + 1))
  base="$(basename "${tex%.tex}")"
  out="$OUT_ROOT/$base.html"

  # Build a friendly title and banner from the filename.
  if [[ "$base" == *-solution ]]; then
    n="${base#mock-exam-}"; n="${n%-solution}"
    title="Mock Exam $n — Solution"
    pdf_href="/pdfs/$base.pdf"
    exam_href="$(basename "${base%-solution}").html"
    banner="<a href=\"$pdf_href\">PDF</a> · <a href=\"$exam_href\">Exam (HTML)</a> · <a href=\"/mock-exams\">Index</a>"
  else
    n="${base#mock-exam-}"
    title="Mock Exam $n"
    pdf_href="/pdfs/$base.pdf"
    sol_href="$base-solution.html"
    banner="<a href=\"$pdf_href\">PDF</a> · <a href=\"$sol_href\">Solution (HTML)</a> · <a href=\"/mock-exams\">Index</a>"
  fi

  echo "→ $base.html"
  pandoc "$tex" \
    --from=latex+raw_tex \
    --to=html5 \
    --mathjax \
    --standalone \
    --template="$TEMPLATE" \
    --lua-filter="$FILTER" \
    --metadata pagetitle="$title" \
    --metadata title="$title" \
    --variable banner="$banner" \
    -o "$out"
done <<< "$TEX_FILES"

echo
echo "Wrote $count files to $OUT_ROOT"
