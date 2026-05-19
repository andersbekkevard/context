# Imagen prompt — Module 1 (Boosting)

Source: `cheat-sheet/parts/01-boosting.tex`. Paste the block below into ChatGPT (GPT-Image-2 / Images 2.0). One image = one A5 side. If text legibility degrades, regenerate with the same prompt or split into two passes (see split note at bottom).

---

```
A top-down close-up photograph of a single A5 sheet of dark navy-blue cardstock lying flat on a plain wooden desk. Soft natural window light from the upper-left. Shallow depth of field. Documentary-style photography. Photorealistic, not vector or digital.

The sheet is filled edge-to-edge with handwritten study notes written in a white Uni-ball Signo gel pen. The handwriting is a neat cursive-print hybrid by a mathematically literate student — small, dense, organized, but not perfect. Natural pen-pressure variation. Occasional ink pooling at letter ends. A few subtle smudges where a hand rested. The paper shows faint wrinkles and a slight bend at one corner. White ink sits crisply on the navy with a hint of translucency, the way a real gel pen actually behaves on dark cardstock.

LAYOUT: two vertical columns separated by a single hand-drawn vertical line down the middle. Section headings written slightly larger and underlined with a single freehand stroke. Body text small and tight. No printed grid, no rulers, no digital UI elements.

RENDERING RULES — read carefully, these are mandatory:
- Render ALL text exactly as written in the CONTENT block, verbatim, no spelling corrections, no character substitutions, no autocorrect, no abbreviating things I have not abbreviated.
- Math is written as actual mathematical notation — real Greek letters (α β γ ν λ Σ Π), real subscripts and superscripts, real fractions written as stacked numerator-over-denominator with a horizontal bar, real summation signs. NEVER render LaTeX source code such as backslash-alpha, dollar signs, \frac, \sum, or curly braces. If you see LaTeX-like syntax in the content block, convert it to its rendered math form before drawing.
- Algorithm steps are numbered as a student would write them: a small "1.", "2.", "3." in the same pen, at the left margin, followed by the step. Do NOT render bulleted-list dots, square bullets, arrows, or any digital list-marker glyph.
- Tables are drawn as a student would: hand-drawn horizontal rules above and below the header, a single hand-drawn vertical line separating columns, no boxed cells, no rectangular borders, no shaded rows.
- Section headings are plain handwritten words, optionally underlined with one freehand stroke. No boxes around them. No coloured highlight.
- Do not add any decoration, page numbers, watermarks, dates, names, frames, or "AI-generated" labels.
- Do not invent extra content beyond what the CONTENT block specifies. If a section feels short, leave whitespace; do not pad.

CONTENT — render verbatim, in this exact order, top of left column to bottom, then top of right column to bottom:

"m09 BOOSTING"

"Boosting: iteratively add basis fns (typ. shallow trees) greedily so each addition reduces chosen loss. Applied to high-bias / low-variance models (esp. trees). Final model is additive expansion:"

"f(x) = Σ from m=1 to M of β_m · b(x; γ_m)     f_M(x) = Σ from m=1 to M of T(x; Θ_m)"

"Three ingredients:"

"Weak learner  |  high-bias / low-variance base (shallow tree, stump)"
"Loss L        |  differentiable cost to minimise"
"Additive model|  add weak learners sequentially, each lowers L"

"AdaBoost.M1 (ESL Alg. 10.1)    binary, y ∈ {-1,+1}, weak clf G(x)"

"1. Init weights w_i = 1/N, i=1,...,N"
"2. For m = 1,...,M:"
"     fit G_m(x) to data using weights w_i"
"     err_m = ( Σ w_i · 1{y_i ≠ G_m(x_i)} ) / Σ w_i"
"     α_m = log( (1 − err_m) / err_m )"
"     w_i ← w_i · exp( α_m · 1{y_i ≠ G_m(x_i)} )"
"3. Return G(x) = sign[ Σ from m=1 to M of α_m G_m(x) ]"

"Notes: α_m > 0 iff err_m < 0.5; misclass. points weighted up by exp(α_m); equiv. to forward-stagewise additive modelling with exponential loss L(y,f) = exp(−yf)."

"Gradient tree boosting (ESL Alg. 10.3)"

"1. Init f_0(x) = arg min_γ Σ L(y_i, γ)"
"2. For m = 1,...,M:"
"     pseudo-residuals  r_im = − [ ∂L(y_i, f(x_i)) / ∂f(x_i) ]_{f = f_{m−1}}"
"     fit regression tree to r_im → regions R_jm, j=1,...,J_m"
"     γ_jm = arg min_γ Σ_{x_i ∈ R_jm} L( y_i, f_{m−1}(x_i) + γ )"
"     f_m(x) = f_{m−1}(x) + Σ_j γ_jm · 1{x ∈ R_jm}"
"3. Return f̂(x) = f_M(x)"

"r_im = pseudo-residuals (= generalised residuals)."

"Loss / objective table"

"Loss                  |  L(y,f)                            |  −g_i = −∂L/∂f"
"Quadratic             |  ½(y−f)^2                          |  y − f  (residual)"
"Absolute              |  |y − f|                           |  sign(y − f)"
"Huber                 |  (y−f)^2 if |y−f| ≤ δ              |  piecewise"
"                      |  2δ|y−f| − δ^2 else                |"
"Exponential (AdaB.)   |  exp(−yf), y ∈ {−1,+1}             |  y · exp(−yf)"
"Binomial deviance     |  −1{y=1} log p − 1{y=0} log(1−p)   |  y − p(x)"
"Multinomial deviance  |  − Σ_k 1{y=k} log p_k(x)           |  1{y_i=k} − p_k(x_i)"

"Binomial: p(x) = e^{f(x)} / (1 + e^{f(x)})       (logistic), y ∈ {0,1}"
"Multinomial: p_k(x) = e^{f_k(x)} / Σ_l e^{f_l(x)}  (softmax). Builds K trees per round, one per class."

"Shrinkage: f_m(x) = f_{m−1}(x) + ν · Σ_j γ_jm · 1{x ∈ R_jm}, ν ∈ (0,1]. ν small ⇒ M large. ν < 0.1 robust. Fix ν, pick M by early stop / CV."

"Hyperparameters"

"M (or B)   |  # trees / iterations   |  small ⇒ underfit, large ⇒ overfit; CV / early stop"
"ν          |  shrinkage / LR         |  typ. 0.1 or 0.01"
"d          |  tree depth             |  d=1 stumps ⇒ additive; 4 ≤ J ≤ 8 common"
"J_m        |  leaves per tree        |  = d+1; interaction order"
"η_row      |  row subsample          |  stochastic GBM; typ. 0.5; variance ↓"
"η_col      |  column subsample       |  XGBoost"
"γ (XGB)    |  per-leaf penalty       |  post-fit pruning; larger γ ⇒ smaller tree"
"λ (XGB)    |  L2 leaf weights        |  ridge-style"
"α (XGB)    |  L1 leaf weights        |  lasso-style"

END OF CONTENT.

Final reminder: render the CONTENT block verbatim with real math symbols and handwritten formatting; do not add anything, do not skip anything, do not modernise spacing. Iterate by tightening this prompt, not loosening it.
```

---

## Split note

If the full block above renders with degraded text past the loss table, generate it in two passes and composite:

- **Pass A (left column only):** keep SCENE / INK / HANDWRITING / RENDERING blocks; in CONTENT keep only up to "Gradient tree boosting" inclusive; add `LAYOUT: single column on the left half of the sheet, right half blank.`
- **Pass B (right column only):** same SCENE / INK / HANDWRITING / RENDERING; in CONTENT include loss table + shrinkage + hyperparameters; add `LAYOUT: single column on the right half of the sheet, left half blank. Match the handwriting style of the uploaded reference image.` and attach the Pass A output as a reference.

This is the tiling pattern that holds visual style across calls in Images 2.0.
