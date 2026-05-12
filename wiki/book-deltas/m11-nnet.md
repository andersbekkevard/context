---
title: "M11: Neural Networks — Book delta"
module: 11-nnet
isl-ch: 10
lectures: [L23, L24, L26]
---

# Module 11: Neural Networks — Book delta

Module 11 is Benjamin's "deep learning is just nested GLMs" module, spread over three lectures ([[L23-nnet-1|L23]] [[feedforward-network|feedforward]] + [[backpropagation|backprop]], [[L24-nnet-2|L24]] [[convolutional-neural-network|CNNs]] + [[nn-regularization|regularization]] menu, [[L26-nnet-3|L26]] [[recurrent-neural-network|RNNs]] + [[double-descent|double descent]]). ISLP chapter 10 covers most of the same ground at a *high* level, but Benjamin's blackboard derivations are markedly more explicit on:

- the **[[backpropagation|backpropagation δ-recursion]]** in two-pass forward/backward form (ISLP §10.7.1 derives only the two partial derivatives 10.29–10.30 and stops; it never introduces $\delta$ notation, never writes the general layer-$L$ recursion, and never separates the bias-gradients from the weight-gradients).
- **mini-batch SGD as implicit L2 / minimum-norm regularizer** (ISLP §10.7.2 says only "SGD naturally enforces its own form of approximately quadratic regularization", footnote-flagged as ongoing research; Benjamin says "this has been proven" and gives the min-norm-interpolator picture);
- the **constrained-minimization rewrite** of the over-parameterized regime that makes double descent click (ISLP §10.8 hints at this with the natural-spline minimum-norm picture but does not write the constrained form for NNs);
- the **regularization menu**, with prescription on numerical rates and a verbatim Goodfellow definition (ISLP §10.7.2–§10.7.3 covers a subset, no dropout-rate prescription, no label smoothing, no transfer-learning recipe);
- a unified **[[nn-parameter-count|parameter-counting recipe]]** $(p+1)M + (M+1)C$ generalised to multilayer (ISLP only does it implicitly for the MNIST diagram);
- a clean **[[universal-approximation|universal-approximation statement]]** with the three conditions (ISLP just alludes to it in §10.2);
- **GELU**, the **[[activation-functions|activation]]-derivative table**, and a **[[nn-loss-functions|loss-activation pairing]] table** (ISLP scatters these across §10.1–10.2).

What follows reproduces each delta in full so the math is recoverable at the exam table without re-deriving anything. Out-of-scope material (BPTT, LSTM/GRU, batch-norm, Adam internals, weight initialization, vanishing-gradient analysis, CNN filter math/padding/stride) is omitted entirely per [[docs/scope]].

---

## 1. Backpropagation: the $\delta$-recursion in full

[[[L23-nnet-1|L23]], [[L24-nnet-2|L24]], [[backpropagation]]; ISLP §10.7.1 partial]

ISLP §10.7.1 derives (10.29) and (10.30) for a single-hidden-layer net and stops with the remark "the act of differentiation assigns a fraction of the residual to each of the parameters via the chain rule — a process known as backpropagation". It never introduces the $\delta$ notation that lets you write the algorithm as a clean two-pass procedure, never writes the bias gradients separately, and never gives the general layer-$L$ recursion. Benjamin's blackboard derivation is the workable lookup form.

### 1.1 Setup

Single-hidden-layer [[feedforward-network|feedforward network]], linear output, squared-error loss on one sample $(x_i, y_i)$:

$$
v_{ik} = \alpha_{k0} + \sum_{j=1}^{p} \alpha_{kj}\, x_{ij}, \qquad
z_{ik} = g(v_{ik}), \qquad
f(\mathbf x_i) = \beta_0 + \sum_{k=1}^{K} \beta_k\, z_{ik},
$$

$$
R_i(\boldsymbol\theta) = \tfrac{1}{2}\bigl(y_i - f(\mathbf x_i)\bigr)^2.
$$

Parameters split into two blocks: input-to-hidden $\boldsymbol\alpha$ (with bias $\alpha_{k0}$), hidden-to-output $\boldsymbol\beta$ (with bias $\beta_0$).

### 1.2 Forward pass: compute and store

Push input through the network and **store** every intermediate for reuse in the backward pass:

| Stored | Definition | Used later for |
|---|---|---|
| $v_{ik}$ | pre-activation of hidden unit $k$ | $g'(v_{ik})$ in $\delta^{\text{hid}}$ |
| $z_{ik} = g(v_{ik})$ | activation of hidden unit $k$ | $\partial R_i/\partial \beta_k$ |
| $f(\mathbf x_i)$ | output | $\delta^{\text{out}}_i$ |

No new derivatives are taken in the forward pass. Memory cost is one number per intermediate node per sample.

### 1.3 Backward pass: seed at the output

Define the **output-layer error** (the chain-rule "seed", $\partial R_i$ w.r.t. the *output*, not w.r.t. a parameter):

$$
\boxed{\;\delta_i^{\text{out}} \;=\; \frac{\partial R_i}{\partial f(\mathbf x_i)} \;=\; -\bigl(y_i - f(\mathbf x_i)\bigr)\;}
$$

Sign: comes from $\partial_f \tfrac12(y - f)^2 = -(y - f)$. Watch this on the exam.

### 1.4 Output-layer gradients

By the chain rule, $\partial R_i / \partial \beta_k = \delta_i^{\text{out}} \cdot \partial f / \partial \beta_k$. Since $f = \beta_0 + \sum_k \beta_k z_{ik}$:

$$
\boxed{\;\frac{\partial R_i}{\partial \beta_k} \;=\; \delta_i^{\text{out}} \cdot z_{ik}\;}, \qquad
\boxed{\;\frac{\partial R_i}{\partial \beta_0} \;=\; \delta_i^{\text{out}}\;}.
$$

Both factors $(\delta_i^{\text{out}}, z_{ik})$ already live in memory from the forward pass; no new computation.

### 1.5 Hidden-layer error (one chain-rule step back)

Define the **hidden-layer error** as $\partial R_i / \partial v_{ik}$, i.e. the gradient w.r.t. the pre-activation:

$$
\delta_{ik}^{\text{hid}}
\;=\; \frac{\partial R_i}{\partial v_{ik}}
\;=\; \frac{\partial R_i}{\partial f(\mathbf x_i)} \cdot \frac{\partial f(\mathbf x_i)}{\partial z_{ik}} \cdot \frac{\partial z_{ik}}{\partial v_{ik}}.
$$

The three factors are: $\delta_i^{\text{out}}$, $\beta_k$ (from $f = \beta_0 + \sum_k \beta_k z_{ik}$), and $g'(v_{ik})$ (from $z_{ik} = g(v_{ik})$). Multiplied:

$$
\boxed{\;\delta_{ik}^{\text{hid}} \;=\; \delta_i^{\text{out}} \cdot \beta_k \cdot g'(v_{ik})\;}.
$$

Note the appearance of $g'$. This is where the activation choice enters the gradient — and where ReLU's piecewise constant $g'(v) = \mathbf 1[v > 0]$ makes the backward pass cheap (and where the historical "vanishing-gradient" worry with sigmoid lives, out-of-scope here).

### 1.6 Input-to-hidden gradients

Same recipe: $\partial R_i / \partial \alpha_{kj} = (\partial R_i / \partial v_{ik}) \cdot (\partial v_{ik} / \partial \alpha_{kj})$. Since $v_{ik} = \alpha_{k0} + \sum_j \alpha_{kj} x_{ij}$:

$$
\boxed{\;\frac{\partial R_i}{\partial \alpha_{kj}} \;=\; \delta_{ik}^{\text{hid}} \cdot x_{ij}\;}, \qquad
\boxed{\;\frac{\partial R_i}{\partial \alpha_{k0}} \;=\; \delta_{ik}^{\text{hid}}\;}.
$$

Again both factors $(\delta_{ik}^{\text{hid}}, x_{ij})$ are already stored.

### 1.7 The algorithm in two passes

For one sample $i$ with all parameters initialized:

**Forward pass.**
1. Compute and store $v_{ik}, z_{ik}, f(\mathbf x_i)$ for every $k$.
2. Compute the loss $R_i$.

**Backward pass.**
3. Seed: $\delta_i^{\text{out}} = -(y_i - f(\mathbf x_i))$.
4. Output-layer gradients: $\partial R_i / \partial \beta_k = \delta_i^{\text{out}} z_{ik}$ for $k=1,\ldots,K$; $\partial R_i / \partial \beta_0 = \delta_i^{\text{out}}$.
5. Hidden-layer errors: $\delta_{ik}^{\text{hid}} = \delta_i^{\text{out}} \beta_k\, g'(v_{ik})$ for each $k$.
6. Input-to-hidden gradients: $\partial R_i / \partial \alpha_{kj} = \delta_{ik}^{\text{hid}} x_{ij}$ for each $(k, j)$; $\partial R_i / \partial \alpha_{k0} = \delta_{ik}^{\text{hid}}$.

**SGD step** (consumed by [§5](#5-gradient-descent-and-stochastic-gradient-descent)): $\boldsymbol\theta \leftarrow \boldsymbol\theta - \lambda \widehat{\nabla R}$ where $\widehat{\nabla R}$ is the mini-batch average of these per-sample gradients.

> The point of backprop is that step 4 and step 6 each cost a *constant* number of multiplications per parameter, because the heavy lifting (the $\delta$'s) is shared across all parameters in the same layer. Without this organization, computing a gradient per parameter naively would cost $O(\#\text{params})$ forward passes per epoch. [[[L24-nnet-2|L24]]]

### 1.8 General multilayer $\delta$-recursion

Stack hidden layers $\ell = 1, 2, \ldots, L$ with pre-activations $\mathbf v^{(\ell)}$ and activations $\mathbf z^{(\ell)} = g(\mathbf v^{(\ell)})$ connected by weight matrices $\mathbf W^{(\ell)}$ (so $\mathbf v^{(\ell)} = \mathbf W^{(\ell)} \mathbf z^{(\ell-1)} + \mathbf b^{(\ell)}$, with $\mathbf z^{(0)} = \mathbf x$). Let $L$ be the output layer.

**Output-layer error.** For squared loss and linear output: $\boldsymbol\delta^{(L)} = -(\mathbf y - \mathbf f(\mathbf x))$. For other loss/output activation pairings, see §3.

**Backward recursion** (the prof's "no matter how many layers you have, you just need to compute these deltas; this delta J for layer L is sum over M W_M ..." [[[L24-nnet-2|L24]]]):

$$
\boxed{\;\boldsymbol\delta^{(\ell)} \;=\; \bigl((\mathbf W^{(\ell+1)})^\top \boldsymbol\delta^{(\ell+1)}\bigr) \odot g'\!\bigl(\mathbf v^{(\ell)}\bigr)\;}
$$

with $\odot$ = elementwise (Hadamard) product. Layer-$\ell$ weight and bias gradients are then

$$
\frac{\partial R}{\partial \mathbf W^{(\ell)}} \;=\; \boldsymbol\delta^{(\ell)}\, (\mathbf z^{(\ell-1)})^\top, \qquad
\frac{\partial R}{\partial \mathbf b^{(\ell)}} \;=\; \boldsymbol\delta^{(\ell)}.
$$

You only ever look one layer ahead. Each $\boldsymbol\delta^{(\ell)}$ is computed from $\boldsymbol\delta^{(\ell+1)}$, the *next-layer* weights $\mathbf W^{(\ell+1)}$, and the locally-stored $g'(\mathbf v^{(\ell)})$. This is the whole algorithm, and it requires acyclicity (no loops) to be well-posed — which is why RNNs need BPTT (out of scope) and why feedforward / CNN are the in-scope architectures.

### 1.9 Mini-batch aggregation

The per-sample formulas above are summed (or averaged) over a mini-batch $\mathcal B$ before the SGD step:

$$
\widehat{\nabla_{\boldsymbol\theta} L} \;=\; \frac{1}{|\mathcal B|} \sum_{i \in \mathcal B} \nabla_{\boldsymbol\theta} R_i.
$$

---

## 2. Activation functions: definitions and derivatives

[[[L23-nnet-1|L23]], [[activation-functions]]; ISLP §10.1 partial]

ISLP §10.1 defines sigmoid (10.4) and ReLU (10.5) and softmax (10.13). It does **not** define GELU, does not tabulate $g'$, and does not table the loss-activation pairing in one place.

### 2.1 Hidden-layer activations

| Name | $g(z)$ | $g'(z)$ | In-scope role |
|---|---|---|---|
| Linear (identity) | $z$ | $1$ | Collapses entire FNN to linear regression — degenerate case |
| Sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | Historical default; "kind of bad for gradients" at saturation |
| ReLU | $\max(0, z)$ | $\mathbf 1[z > 0]$ (undefined at 0; take $0$ or $1$ by convention) | Modern default; cheap, but "corner is bad for gradients" |
| GELU | $z\, \Phi(z)$ where $\Phi$ is the standard Gaussian CDF; commonly approximated $\approx 0.5\, z\, (1 + \tanh[\sqrt{2/\pi}(z + 0.044715\, z^3)])$ | analytic; smooth | Smoothed ReLU. Benjamin: "I won't ask you about that specifically." Conceptually: removes the corner |

The exam-relevant fact is that **non-linearity in the hidden layer is required for expressiveness**; otherwise the network is linear regression with extra parameters [[[L23-nnet-1|L23]], Q3f-template].

### 2.2 Output-layer activations and loss pairings

The full prof's table (loss-activation pairing). ISLP scatters these across §10.1 and §10.2.

| Task | Output dim | Output activation $f$ | Loss |
|---|---|---|---|
| Regression | 1 | linear, $f(z) = z$ | MSE $\;\sum_i (y_i - \hat y_i)^2$ |
| Binary classification | 1 | sigmoid, $f(z) = 1/(1+e^{-z})$ | binary cross-entropy $\;-\sum_i [y_i \log \hat p_i + (1-y_i) \log (1-\hat p_i)]$ |
| Multi-class classification ($C \ge 2$) | $C$ | softmax, $f_c(z) = e^{z_c}/\sum_{k=1}^C e^{z_k}$ | categorical cross-entropy $\;-\sum_i \sum_{c=1}^C y_{ic} \log \hat p_{ic}$ |

In every row, the loss is the *negative log-likelihood* of the matching GLM (Gaussian / Bernoulli / multinomial) applied to the network's final output. Mismatched pairings (softmax + MSE, sigmoid + MSE) work numerically but lose the likelihood interpretation and have worse gradient behaviour.

### 2.3 Softmax + categorical cross-entropy: combined gradient

Even though Benjamin did not derive this on the board, it is the natural multi-class analog of $\delta^{\text{out}} = -(y - f)$ and is needed to seed the backward pass in classification networks. With one-hot $\mathbf y$ and softmax outputs $f_c = e^{z_c}/\sum_k e^{z_k}$, the output-layer error w.r.t. the pre-softmax logits $\mathbf z$ is

$$
\delta^{\text{out}}_c \;=\; \frac{\partial L}{\partial z_c} \;=\; f_c - y_c,
$$

i.e. *predicted probability minus one-hot target*. This compact form is exactly why softmax + cross-entropy is the natural pairing: the cross-entropy log cancels the softmax exponent, leaving a clean residual.

---

## 3. Universal approximation theorem (full statement)

[[[L23-nnet-1|L23]], [[universal-approximation]]; ISLP §10.2 cursory mention]

ISLP §10.2 says only "In theory a single hidden layer with a large number of units has the ability to approximate most functions" with no conditions and no statement. Benjamin states the result completely. **Proof is out of scope** (measure theory).

> **Universal approximation theorem (informal).** Let $f: \mathbb R^p \to \mathbb R^C$ be any Borel-measurable function and $\varepsilon > 0$. Then there exists a feedforward neural network $\hat f$ with
>
> 1. a **linear output layer**,
> 2. at least one **hidden layer** with a **non-linear ("squashing") activation function** (ReLU, sigmoid, GELU, …),
> 3. finite but **sufficiently large width** $M$,
>
> such that $\sup_{\mathbf x \in K} \|\hat f(\mathbf x) - f(\mathbf x)\| < \varepsilon$ on any compact $K \subset \mathbb R^p$. [[[L23-nnet-1|L23]], verbatim]

Three caveats Benjamin flagged:

- **Existence, not construction.** The theorem guarantees a network *exists*; it does not say how to train one to find it, and SGD may fail to converge to it.
- **Width $\leftrightarrow$ depth tradeoff.** The original proof needed "ridiculously wide" single hidden layers. In practice deeper-and-thinner achieves the same approximation with far fewer total units, but the theorem licenses width alone.
- **Linear hidden activation breaks universality.** Without a non-linear $g$, the network collapses to linear regression (a tiny function class).

---

## 4. Parameter-counting formula and the bias trap

[[[L23-nnet-1|L23]], [[nn-parameter-count]]; ISLP §10.2 implicit for MNIST]

ISLP gives the matrix sizes for the MNIST diagram ($785 \times 256$, $257 \times 128$, $129 \times 10$, summing to 235,146) but never writes the unified formula. The exam-flagged version:

### 4.1 Single hidden layer

For an FNN with $p$ inputs, $M$ hidden units, $C$ outputs:

$$
\boxed{\;\text{params} \;=\; (p+1)\,M \;+\; (M+1)\,C\;}.
$$

Breakdown:
- Input → hidden: $p$ weights per hidden unit + $1$ bias per hidden unit = $(p+1) \cdot M$.
- Hidden → output: $M$ weights per output unit + $1$ bias per output unit = $(M+1) \cdot C$.

### 4.2 Multilayer (general recipe)

For input dim $p$, hidden widths $M_1, M_2, \ldots, M_L$, output dim $C$:

$$
\text{params} \;=\; (p+1)\,M_1 \;+\; \sum_{\ell=2}^{L} (M_{\ell-1}+1)\,M_\ell \;+\; (M_L+1)\,C.
$$

Each layer contributes $(\text{previous-layer-width} + 1) \cdot (\text{this-layer-width})$. The $+1$ sits on the *receiving* layer (it's a bias per receiving unit), not on the sending layer.

### 4.3 The canonical wrong answer

> "Remember that there is one bias-node in each layer!" — 2023 exam answer key, cited in [[L27-summary|L27]]

Forgetting biases (i.e. computing $p M + M C$ instead of $(p+1) M + (M+1) C$) is the **canonical wrong answer**; past exams (2023 Q5e, 2024, 2025 Q3b.i) have all used the no-bias version as a multiple-choice distractor. Benjamin's exam advice: *draw the network, count edges (weights), count receiving nodes (biases)*. Dropout does **not** reduce the parameter count (the network is intact; activations are zeroed at training, not weights), and was also flagged as a distractor.

### 4.4 Worked examples (for cold lookup)

- $p=10, M=5, C=1$, ReLU hidden + linear output: $(10+1)\cdot 5 + (5+1)\cdot 1 = 55 + 6 = \mathbf{61}$.
- $p=4, M_1=10, M_2=5, C=1$, ReLU + ReLU + sigmoid: $(4+1)\cdot 10 + (10+1)\cdot 5 + (5+1)\cdot 1 = 50 + 55 + 6 = \mathbf{111}$.
- $p=3, M=4, C=1$ (regression): $4\cdot 4 + 1\cdot 5 = \mathbf{21}$ (2025 Q3b.i answer).
- $p=128, M_1=32, M_2=64, C=5$ softmax: $129\cdot 32 + 33\cdot 64 + 65\cdot 5 = 4128 + 2112 + 325 = \mathbf{6565}$ (2023 Q5e answer).

---

## 5. Gradient descent and stochastic gradient descent

[[[L23-nnet-1|L23]], [[L24-nnet-2|L24]], [[gradient-descent-and-sgd]]; ISLP §10.7 cursory]

ISLP §10.7 gives the update rule (10.27) and mentions mini-batch SGD briefly. It does not state the unbiasedness of the mini-batch estimator, does not name the three flavours explicitly, and treats the "implicit regularization" claim as a research footnote (footnote 21). Benjamin's framing is much more explicit.

### 5.1 The update rule

$$
\boxed{\;\boldsymbol\theta^{(t+1)} \;=\; \boldsymbol\theta^{(t)} \;-\; \lambda\, \nabla_{\boldsymbol\theta} L(\boldsymbol\theta^{(t)})\;}
$$

with learning rate $\lambda$ (ISLP uses $\rho$). Practical guidance: $\lambda \approx 0.1$ or smaller; too large $\to$ bouncing, too small $\to$ glacial. No closed form for $\lambda$ — pick via experimentation.

### 5.2 Three flavours

Per-sample loss $L_i$ so $L = (1/N) \sum_{i=1}^N L_i$.

| Flavour | Gradient estimator | Per-step cost | Variance |
|---|---|---|---|
| Full (batch) GD | $\nabla L = (1/N) \sum_{i=1}^N \nabla L_i$ | $O(N)$ | 0 |
| True SGD | $\nabla L_i$ for one random $i$ | $O(1)$ | large |
| Mini-batch SGD | $\widehat{\nabla L} = (1/m) \sum_{i \in \mathcal B} \nabla L_i$, $\|\mathcal B\| = m$ | $O(m)$ | $\propto 1/m$ |

The mini-batch estimator is **unbiased**: $\mathbb E_{\mathcal B}[\widehat{\nabla L}] = \nabla L$, since each $\nabla L_i$ has the same expectation as $\nabla L$ when $\mathcal B$ is sampled uniformly.

### 5.3 Mini-batch SGD: two payoffs

1. **Speed / parallelism.** With $K$ GPUs each given $m/K$ samples, summing local gradients in parallel.
2. **Regularization via noise.** The headline NN fact (§6.2). The noisy gradient acts as an implicit regularizer.

### 5.4 Powers-of-2 batch sizes

Batch sizes are always powers of two (32, 64, 128, 256, 512, 1024) **for hardware reasons only**, not statistical reasons.

---

## 6. The regularization menu (full)

[[[L24-nnet-2|L24]], [[L26-nnet-3|L26]], [[nn-regularization]]; ISLP §10.7.2–§10.7.3 partial]

ISLP §10.7.2 covers ridge weight decay (eq. 10.31) and notes SGD enforces "approximately quadratic regularization"; §10.7.3 covers dropout. ISLP does **not** cover label smoothing, the dropout-rate prescription, the data-augmentation operational vocabulary, the early-stopping mechanic in plot form, or transfer learning explicitly. Benjamin's menu is fuller.

### 6.1 The Goodfellow definition (verbatim from slides via [[L24-nnet-2|L24]])

> [[regularization|Regularization]]: any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.

Operational consequence: **regularization may hurt training error**; the metric of success is test / validation error, not training.

### 6.2 Implicit L2: mini-batch SGD as minimum-norm interpolator (HEADLINE)

> "Mini-batch [[gradient-descent-and-sgd|stochastic gradient descent]] actually gives you an implicit L2 regularization, which is super weird. … Whenever you use this and you're in a problem setting where there's an infinite number of exact solutions, it will find the solution where the L2 norm is minimized. … It has been proven, which is nice. The math is there, but it's not super short." [[[L23-nnet-1|L23]], [[L24-nnet-2|L24]]]

Precise statement: when the model is over-parameterized so that the training loss can be driven to zero on an infinite-dimensional manifold of $\boldsymbol\theta$ values (the "interpolators"), mini-batch SGD initialized near zero converges to (approximately) the **minimum-$L^2$-norm interpolator**, i.e. the $\boldsymbol\theta^\star$ that minimizes $\|\boldsymbol\theta\|_2^2$ subject to $L_i(\boldsymbol\theta) = 0$ for all $i$. This is the same object the explicit L2 weight-decay problem converges to as $\lambda \downarrow 0$ — hence "implicit L2". The mechanism for *why* SGD picks this particular interpolator is the topic of an active body of research that Benjamin flagged as proven but not lectured.

This is the bridge to §7 (double descent): the min-norm interpolator generalizes well because it has the smallest variance among all interpolators.

### 6.3 Explicit L1 / L2 weight decay

Same shapes as in module 6. Let $\mathbf w$ be the weight vector (biases conventionally excluded from the penalty):

$$
\tilde L_{\text{L2}}(\mathbf w) = L(\mathbf w) + \lambda\, \mathbf w^\top \mathbf w = L(\mathbf w) + \lambda \sum_{ij} w_{ij}^2,
$$

$$
\tilde L_{\text{L1}}(\mathbf w) = L(\mathbf w) + \alpha\, \|\mathbf w\|_1 = L(\mathbf w) + \alpha \sum_{ij} |w_{ij}|.
$$

Geometric behaviour (revisited from module 6): L2 (bowl) shrinks smoothly; L1 (V) forces exact zeros and produces sparsity. Different $\lambda$ can be applied per layer (ISLP eq. 10.31).

### 6.4 Data augmentation

For each training example $(\mathbf x_i, y_i)$, generate perturbed copies $(\tau(\mathbf x_i), y_i)$ under label-preserving transformations $\tau$:

- For images: rotation, flip, shift / translation, zoom, shear, Gaussian or sparse noise, color jitter, crop.
- For time-series / 1-D signals: noise, shift, scaling.

Constraint: **$\tau$ must preserve the label**. Flipping a handwritten "6" vertically gives a "9" — wrong augmentation. Effectively grows the dataset and forces the model to be invariant under $\tau$. Especially natural for CNNs (interacts well with mini-batch SGD: distort each image on the fly per batch — ISLP §10.3.4 mentions this in passing).

### 6.5 Label smoothing

For $C$-class classification, replace one-hot targets $\mathbf y = (0, \ldots, 0, 1, 0, \ldots, 0)$ with softened targets

$$
\tilde y_c = \begin{cases} 1 - \varepsilon & \text{if } c = c^\star \\ \varepsilon/(C - 1) & \text{otherwise} \end{cases}
$$

for some small $\varepsilon$ (e.g. 0.1). Justification (slides): "motivated by the fact that the training data may contain errors in the responses recorded." Same flavour as augmentation — inject uncertainty so the model doesn't get over-confident. **Not in ISLP.**

### 6.6 Early stopping

Procedure: split training into train / validation. Plot training error and validation error vs. epoch. Training error decreases monotonically; validation error decreases, levels off, often climbs again (U-shape). **Return the parameters from the epoch with minimum validation error**, not the final ones. ISLP shows this in fig. 10.18 but doesn't name it as a regularization mechanism in this clean a way. Slides: "the most commonly used form of regularization."

### 6.7 Dropout — the operational recipe

At each training iteration and each dropout layer:
1. For each unit, independently set its activation to zero with probability $p_{\text{drop}}$.
2. Scale surviving activations by $1/(1 - p_{\text{drop}})$ (the "inverted dropout" convention) so that the layer's expected output magnitude is preserved.

At **test time**: no dropout, all units active, no rescaling needed (because the training-time rescaling already baked the correction in).

**Prescription** (Benjamin, verbatim): "Drop-out rates may be chosen between 0.2 and 0.5. … 20% is very common, never use 50%." ISLP only gives the formula, no rate prescription.

Conceptual mechanism: dropout = **ensembling inside the network**. Each forward pass during training is effectively a different sub-network sampled from $2^K$ possible masks; the trained weights have to work on average across all of them. Direct parallel to [[bagging]] in [[random-forest|random forests]].

### 6.8 Transfer learning (the data-hack)

Recipe: if you have limited labelled data $(\mathbf x_i, y_i)$ for your task, but a pre-trained model exists for a related task on a large dataset (e.g. ImageNet for images, GPT for text):

1. Download the pre-trained model.
2. **Freeze** the early layers (their weights, which encode general low- and mid-level features).
3. **Replace** the final output layer(s) with one(s) appropriate to your task.
4. **Train** only the replaced layers on your limited dataset.

ISLP §10.3.5 mentions this very briefly as "weight freezing" (footnote 11); Benjamin gave it as a top-level menu item.

### 6.9 The iron rule

> "I can't think of an example where you'd ever want to train a neural network without a regularization. … The whole point of regularization is to get you to generalize better." [[[L26-nnet-3|L26]], verbatim]

The Hitters comparison in [[L26-nnet-3|L26]] (1049-param unregularized NN losing to [[lasso]] on 263 baseball players) is, in Benjamin's words, "contrived" — "if you constructed your model this way you doing it wrong." This anchors the "never train without regularization" rule.

---

## 7. Double descent: the constrained-minimization rewrite

[[[L26-nnet-3|L26]], [[double-descent]]; ISLP §10.8 partial]

ISLP §10.8 covers [[double-descent|double descent]] with the natural-spline example (figs. 10.20–10.21), introduces the minimum-norm interpolator concept for splines, and notes that the [[bias-variance-tradeoff|bias-variance trade-off]] still holds (the x-axis "flexibility" just isn't a clean function in the over-parameterized regime). Three things in Benjamin's [[L26-nnet-3|L26]] treatment are **not** in ISLP §10.8:

1. The **explicit rewrite of the optimization objective** between the two regimes (the "regime 1 vs regime 2" formulation that makes the min-norm story crystal-clear for NNs, not just splines).
2. Benjamin's own **polynomial / square-pulse contrived example** where the second descent goes *below* the first one.
3. The "spike at every data point, mean in between" **interpretation of the post-interpolation fitted function**.

### 7.1 The two regimes, explicitly

**Regime 1 (under-parameterized, $p \le n$).** The objective is the standard penalized loss:

$$
L_1(\boldsymbol\theta) \;=\; \sum_{i=1}^n \bigl(y_i - f_{\boldsymbol\theta}(\mathbf x_i)\bigr)^2 \;+\; \lambda\, \sum_j \theta_j^2.
$$

The fit term is *not* driven to zero; the penalty acts as a tiebreaker among approximate fits. Increasing flexibility moves you along the U-shape of the bias-variance trade-off; the optimum is interior.

**Regime 2 (over-parameterized, $p \gg n$).** Past the interpolation threshold there exist infinitely many $\boldsymbol\theta$ with $y_i = f_{\boldsymbol\theta}(\mathbf x_i)$ for every $i$. The optimization effectively rewrites itself:

$$
\boxed{\;\min_{\boldsymbol\theta} \;\sum_j \theta_j^2 \quad \text{subject to} \quad y_i = f_{\boldsymbol\theta}(\mathbf x_i) \;\;\forall\, i\;.}
$$

The fit constraint is satisfied exactly (by hypothesis, infinitely many ways); the *penalty* is now the **only** objective. This is the constrained version of the min-norm interpolator. Benjamin's framing:

> "We've changed the optimization that we're doing. … We're still constructing it the same way, but since we have so many parameters that … the model is so flexible it has an infinite number of ways of fitting it and therefore it doesn't have to pick a model that fits the training data well, all of them do — it's finding the one that has the best variance." [[[L26-nnet-3|L26]]]

The implicit-L2 behaviour of mini-batch SGD (§6.2) is what causes the optimizer to *automatically* land on this constrained-minimum solution. This is the bridge that ties §6.2 to §7.

### 7.2 Square-pulse / Legendre polynomial example

Benjamin's own contrived example, designed to make the second descent *win*. **Not in ISLP.**

- True function: a **square pulse** (jumps 0 → 1 → 0). Discontinuous, **not in the model class** (so bias never vanishes).
- 100 training points with additive Gaussian noise.
- Model: Legendre-polynomial regression $f(x) = \sum_{j=0}^{d-1} \theta_j P_j(x)$ at varying degree $d$, fit with L2 regularization.

Empirical behaviour of training and test error vs. $d$:
- **Training error**: decreases monotonically; hits zero around $d = n = 100$.
- **Test error**:
  - first descent → minimum near $d \approx 24$ (the classical "sweet spot");
  - climbs to a peak near the **interpolation point** $d = n = 100$ (variance explodes);
  - second descent: as $d$ grows past $n$, test error **goes below the first-descent minimum**.

Bias term, variance term, and irreducible error were computed separately and verified to sum to test MSE across the full sweep — confirming that **the bias-variance identity is never violated**; the "U-shape" was just not the only possible shape.

### 7.3 What the post-interpolation solution looks like

The fitted $\hat f_d$ for $d$ in the deep second descent:

- Interpolates every training point **exactly** (spike at each $(\mathbf x_i, y_i)$).
- Between data points: predicts something close to the **local mean** of the response.

Mechanism Benjamin's gloss: "It's trying to minimize the variance, meaning the sensitivity to any one point. So it's going to try to always just go to something closer to the mean — kind of regression-to-the-mean — where it doesn't want to be so sensitive to this point. So it's quickly going to shoot it back down." [[[L26-nnet-3|L26]]]

This is the geometric meaning of "minimum-norm interpolator": among all functions in the model class that fit the data exactly, pick the one with smallest $\sum_j \theta_j^2$. That solution looks like spikes-plus-baseline because spikes are the cheapest way (in $\sum \theta_j^2$ units) to make a smooth baseline detour through a noisy point.

### 7.4 Where double descent applies

Slide summary (verbatim): "Though double descent can sometimes occur in neural networks, we typically do not want to rely on this behavior."

In-scope facts:
- Double descent does **not** contradict bias-variance.
- Most classical / regularized methods in this course ([[ridge-regression|ridge]], [[lasso]], [[generalized-additive-models|GAMs]], trees, [[boosting]]) **do not** exhibit double descent — they don't interpolate.
- It is more reliable in **high signal-to-noise** problems (natural images, language).

The min-norm-interpolator framing connects directly to **SGD as implicit L2** (§6.2): without that implicit bias, the over-parameterized regime would have no principled way to pick among the infinite interpolators.

---

## 8. Recurrent neural networks: in-scope core

[[[L26-nnet-3|L26]], [[recurrent-neural-network]]; ISLP §10.5 partial]

ISLP §10.5 covers the basic [[recurrent-neural-network|RNN]] architecture (eq. 10.16), weight sharing across timesteps, and the NYSE forecasting example (eq. 10.20). Benjamin's [[L26-nnet-3|L26]] stays at exactly this level. **Architecture details (LSTM/GRU gates, BPTT, bidirectional RNN, attention, transformers) are explicitly out of scope** per [[docs/scope]] and [[L27-summary|L27]]. The book delta is small and mostly about isolating which RNN content is in vs. out.

### 8.1 The architecture (in scope)

For an input sequence $X_1, X_2, \ldots, X_L$, hidden state vector $A_t$, output $O_t$:

$$
A_t \;=\; \sigma\!\bigl(b + W X_t + U A_{t-1}\bigr), \qquad
O_t \;=\; \beta_0 + \boldsymbol\beta^\top A_t,
$$

where $W$ (input → hidden), $U$ (hidden → hidden = the recurrence), $\boldsymbol\beta$ (hidden → output) are **the same at every time step** — *weight sharing*. The output activation can be linear (regression), sigmoid (binary), or softmax (multi-class) depending on task.

### 8.2 Loss

Regression with linear output, all outputs used:

$$
L \;=\; \sum_t \|Y_t - O_t\|^2.
$$

If only the final output is wanted (e.g. sentiment classification at the end of a sentence), the loss reduces to $\|Y_L - O_L\|^2$ (or the appropriate cross-entropy). Intermediate $O_t$'s "come for free" from the architecture because they reuse the same $\boldsymbol\beta$ as $O_L$.

### 8.3 NYSE setup (Benjamin's example, also ISLP eq. 10.20)

Predict next day's `log_volume` $v_t$ from a lag-$L$ window of $(v, r, z) = (\text{volume}, \text{return}, \text{volatility})$:

$$
X_1 = (v_{t-L}, r_{t-L}, z_{t-L})^\top, \;\ldots,\; X_L = (v_{t-1}, r_{t-1}, z_{t-1})^\top, \qquad Y = v_t.
$$

In-scope qualitative facts: time series has **high autocorrelation** → samples not independent; affects both fit and uncertainty quantification.

### 8.4 What is OUT of scope (do not derive on the exam)

- **BPTT** (backpropagation through time): mentioned only as "a variant of backprop that unrolls through time".
- **LSTM / GRU gates, cell state mechanics**: ISLP §10.5.1 mentions LSTM briefly, Benjamin: out.
- **Vanishing-gradient analysis for long sequences**: out.
- **Bidirectional RNNs, Seq2Seq, attention, transformers**: out.
- **§10.5.3 Summary of RNNs detail**: out beyond the conceptual statement that variants exist.

---

## 9. CNNs: what is in scope

[[[L24-nnet-2|L24]], [[convolutional-neural-network]]; ISLP §10.3 partial]

ISLP §10.3 covers the convolution operation in matrix form (eq. for convolved image with filter $\alpha\beta\gamma\delta$), the max-pool operation, and the conv → pool → conv → pool → flatten → dense → softmax architecture. **All of this is in the book and at the level Benjamin lectured.** The [[convolutional-neural-network|CNN]] delta is therefore small — but worth flagging that some ISLP content is **out** per Benjamin.

### 9.1 In scope (from [[L24-nnet-2|L24]])

- CNN = feedforward with **shared local filters + max-pool**. [[backpropagation|Backprop]] drops in unchanged because the network is still acyclic.
- Filter sliding: at each spatial location, weighted sum of the underlying patch.
- Max-pool: max over non-overlapping (typically 2×2) patches, shrinks spatial extent, preserves peaks.
- Stacked conv + pool builds up receptive field; deeper layers "see" larger features.
- Data augmentation (§6.4) is especially natural for CNNs.
- Same idea extends to 1D (time series / Wafer exercise) and to text tokens.

### 9.2 Explicitly out of scope (per [[L24-nnet-2|L24]], [[L27-summary|L27]])

- Filter math (stride, padding arithmetic).
- Pooling variants beyond max-pool.
- Modern architectures (AlexNet, VGG, ResNet, Inception, Transformer, attention).
- Skip / residual connections.
- §10.3.5 pretrained-classifier details beyond the conceptual transfer-learning idea (§6.8).

If a CNN question shows up on the exam, **answer at the conceptual level** ("conv applies learned filters; pool shrinks; the whole thing is feedforward so backprop just works") and do not attempt filter arithmetic.

---

## 10. The shifting loss landscape (NN-specific bias-variance footnote)

[[[L23-nnet-1|L23]]; ISLP §10.7 mentions non-convexity at fig 10.17]

ISLP §10.7 shows fig 10.17 (a 1-D non-convex objective) and notes that NN losses have multiple local minima. ISLP does **not** state Benjamin's NN-specific observation, which is the bridge to why training works in practice:

> "Once you stack hidden layers and many hidden nodes, suddenly you have a very flexible model. And as you learn some of the parameters, and then estimate some of the states, you're actually changing the landscape. … You still have this notion that there's local minima, global minima, but you have more ways of escaping local minima." [[[L23-nnet-1|L23]]]

> "The local minima are somehow connected" — high-dimensional NN loss landscapes have the empirical property that distinct local minima can typically be connected by a near-flat curve (Garipov et al., 2018; Frankle & Carbin's lottery-ticket adjacent results). Benjamin flags this as the reason [[gradient-descent-and-sgd|SGD]] reliably finds *some* good minimum despite formal non-convexity. The math is beyond the course; what's in scope is the *fact* that NNs aren't paralyzed by their non-convex loss landscape the way classical optimization theory would predict, and that this is partly why training works.

---

## 11. Notation drift

Mostly aligned, but a handful of differences that matter for cold lookup.

| Concept | Benjamin / slides | ISLP §10 |
|---|---|---|
| Learning rate | $\lambda$ | $\rho$ |
| Input → hidden weights | $\alpha_{kj}$ (single-layer); also $W$ | $w_{kj}$ |
| Hidden → output weights | $\beta_k$ | $\beta_k$ |
| Hidden width | $M$ (or $K$) | $K$ |
| Hidden pre-activation | $v_{ik}$ (Benjamin's $\delta$-recursion); $z_{ik}$ (slides; not the same $z$ as below!) | $z_{ik}$ |
| Hidden activation | $z_{ik}$ in Benjamin's L24 backprop derivation; sometimes $A_k$ following ISLP | $A_k$ |
| Backprop "error" at output | $\delta^{\text{out}}$ | (unnamed; appears inside eq. 10.29) |
| Mini-batch size | $m$ | not standardized |
| Number of hidden layers | sometimes $L$ (confusingly also the sequence length in RNN section) | $L_1, L_2$ for layer 1 vs. 2 |
| Universal-approximation hidden activation | "squashing function" | not standardized |
| Dropout rate | $p_{\text{drop}}$, $\phi$ | $\phi$ |

Special care: **$z$ has two meanings.** In Benjamin's L24 backprop derivation, $z_{ik} = g(v_{ik})$ is the *activation*, while $v_{ik}$ is the pre-activation. In the slide deck and in some atom write-ups, $z_{ik}$ is used for the *pre-activation* (matching ISLP). If a formula uses $z$, look at whether it appears inside or outside a $g(\cdot)$ to figure out which is which.

The RNN section uses $L$ for the sequence length, while §1.8 of this delta uses $L$ for the deepest hidden-layer index. Context disambiguates; the two are never both live in the same equation.
