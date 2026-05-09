---
concept: recurrent-neural-network
module: 11-nnet
lectures: [L26]
isl-ref: 10.5
exercises: []
related: [feedforward-network, backpropagation, convolutional-neural-network, nn-regularization, gradient-descent-and-sgd]
tags:
  - concept
  - module/11-nnet
aliases:
  - RNN
  - recurrent network
---

# Recurrent neural network (RNN)

The prof's framing: extend feedforward to sequential data (time series, language) by carrying a **hidden state** $A_t$ across the sequence. The same weights $W, U, B$ are reused at every step (**weight sharing**) — that's why training stays well-behaved despite many time steps. **Architecture details (LSTM/GRU gates, BPTT, attention) are explicitly out of scope** — the high-level "hidden state propagates" idea is the whole exam-relevant content.

## Definition (prof's framing)

> "[RNNs are] suitable for data with sequential character. Examples: text documents, time series (temperature, stock prices, music, speech). The input object $X$ is a sequence, for example a sequence of $L$ words." — slides + [[L26-nnet-3]]

> "RNNs were the first models in machine learning to be used in language modeling — the precursor to the language model that we know now." — [[L26-nnet-3]]

An RNN processes a sequence $X_1, X_2, \dots, X_L$ by maintaining a hidden state $A_t$ that depends on the current input $X_t$ *and* the previous hidden state $A_{t-1}$. Information from $X_1$ propagates forward through the chain of $A$'s and influences every subsequent output $O_t$.

## Notation & setup (prof's, [[L26-nnet-3]] / slides)

- $X_t$ — input at sequence position $t$ (e.g., one-hot word, embedded token, time-series vector)
- $A_t$ — hidden state vector at position $t$
- $O_t$ — output at position $t$
- $W$ — input → hidden weights
- $U$ — hidden → hidden weights (the recurrent connection)
- $B$ (or $\beta$) — hidden → output weights
- $b$ — bias terms

## Formula(s) to know cold

**Hidden state update** (memorize the *shape*, not the indices):

$$A_t = \sigma\!\left( b + W X_t + U A_{t-1} \right)$$

**Output** (linear-output version; could be sigmoid / softmax for classification):

$$O_t = \beta_0 + \beta^\top A_t$$

**Loss** (regression, last-output-only version):

$$L = \sum_t \| Y_t - O_t \|^2$$

— or only $\|Y_L - O_L\|^2$ if you only care about the final output.

## Insights & mental models

### The carry-over is the whole new idea

> "If you notice here we just say B and U and W every time it's because we don't have if it was like this that would be a very different model if the weights were changing every time but they don't in fact we assume the same weights B, U, and W every single time. … And that also why the gradients behave well why the training behaves well is because it not a model with so many parameters that it starts collapsing or getting weird." — [[L26-nnet-3]]

The same weights at every time step = **weight sharing** (same flavor as filter-sharing in CNNs). Without this, the parameter count would explode with sequence length and training would be hopeless.

### Information from $X_1$ reaches $O_L$

> "The interesting thing is that this just keeps going, right? If $L$ is a billion, then essentially even the first thing in your sequence is affecting the billionth output because it's carried through the model. In theory the information will get lost because it just yeah a billion is pretty long but the idea is still nice." — [[L26-nnet-3]]

This is the vanishing-information problem that motivates LSTMs / attention — but in this course we **stop at the basic RNN**. You don't need to know LSTMs.

### All outputs $O_t$ are used in training (often)

A common confusion: the equation looks like only $O_L$ matters. But every intermediate $O_t$ also contributes to the loss in standard training. They "come for free" from the architecture — same weights, no extra structure needed.

### Why RNNs are harder than feedforward / CNN

Loops break the simple feedforward backprop story. Training requires unrolling the loop ("backpropagation through time", BPTT) — a specialized algorithm.

> "If you had loops you're kind of screwed. … with loops, then you can't always just go backwards. … It is convenient that a feedforward network doesn't have those loops and you can just go backwards." — [[L24-nnet-2]] (cautionary contrast)

This is **why CNNs took off before RNNs at scale** and why feedforward / CNN dominate the in-scope material for this course.

### Application: NYSE time-series example

Predict next day's trading volume $V_t$ from lag-$L$ history of $(V, R, Z)$ (volume, returns, volatility). High autocorrelation → samples are not independent. The prof: RNNs were "equally bad" as classical time-series models on cryptocurrencies, "but I think that was really just the data problem because no one should be using Bitcoin." — [[L26-nnet-3]]

When to use: forecasting where a hand-built statistical model is too delicate to construct, and you don't care about interpretability. "If you don't care how it works and you just want to get the results, these models are pretty good." — [[L26-nnet-3]]

## Exam signals

> "RNN architecture details (LSTM/GRU gates, BPTT) — deferred and never examined; only the high-level 'hidden state propagates' idea is in scope." — [[L26-nnet-3]] / [[../scope]]

> "Note: The weights $\boldsymbol{W}$, $\boldsymbol{U}$ and $\boldsymbol{B}$ are the same at each point in the sequence. This is called weight sharing." — slides

The L27 walkthrough does not feature an RNN-specific question. RNNs are likely to appear at the **conceptual / multiple-choice level only** — recognize the architecture, name the use cases, contrast with feedforward.

## Pitfalls

- **Don't claim RNNs are feedforward.** They have loops (the carry-over of $A_t$ to $A_{t+1}$). This is the distinguishing feature.
- **Weight sharing across time** is non-negotiable — without it, parameter count explodes with sequence length.
- **Don't try to write BPTT.** Out of scope. If asked how RNNs are trained, say "via a variant of backpropagation that unrolls through time" and stop.
- **Don't claim RNNs are dead.** They're superseded by transformers in most NLP applications, but the prof: "many LLMs are trained for next-word / next-token prediction; this is where that paradigm started." Conceptual ancestor of LLMs.
- **Time-series autocorrelation matters.** Successive observations are not independent — has implications for both the fit and the uncertainty quantification. The prof flagged this for the NYSE example.
- **Output activation depends on the goal.** Linear for regression-style $O_t$, sigmoid / softmax for classification — exactly the same logic as in [[feedforward-network]] / [[activation-functions]].

## Scope vs ISLR

- **In scope:** the basic RNN architecture (input, hidden state with carry-over, output), weight sharing, the conceptual statement that information propagates along the sequence, NYSE time-series motivation, "RNN is precursor to LLMs."
- **Look up in ISLR:** §10.5 — *Recurrent Neural Networks*, specifically §10.5.1 (sequential models) and §10.5.2 (time series forecasting). The book describes the basic RNN at exactly the depth lectured.
- **Skip in ISLR (book-only / explicitly out):**
  - **LSTM / GRU gates and detailed cell math** — [[L26-nnet-3]] / [[../scope]]: out.
  - **Backpropagation through time (BPTT) algorithm** — [[../scope]] / [[L26-nnet-3]]: out.
  - **Attention, Transformer, Seq2Seq architectures** — out.
  - **Bidirectional RNNs, deep RNNs** — not lectured.
  - **§10.5.3 Summary of RNNs** — touches on details not lectured.

## Exercise instances

None. Exercise11 contains no RNN-specific problem; the closest is Exercise11.5 (1D-CNN for time series), which is owned by [[convolutional-neural-network]].

## How it might appear on the exam

- **Multiple-choice on architecture identification**: "Which network type is most appropriate for sequential data?" → RNN.
- **Conceptual: weight sharing**: "Why does an RNN use the same weights $W, U, B$ at every time step?" → weight sharing keeps parameter count manageable and training tractable; same flavor as CNN filter sharing.
- **Conceptual: feedforward vs. RNN**: "What distinguishes RNN from feedforward?" → loops carry hidden state across positions, allowing information from earlier in the sequence to affect later outputs.
- **True/false**: "RNN training uses standard backpropagation." → False (uses BPTT, which the prof said is out of scope; you can write "a variant of backprop that unrolls through time").
- **Distractor in larger architecture-choice question**: where they ask which method to use for an image / time-series / tabular task; RNN is the right answer for sequential data only.

Heavy detail (LSTMs, attention, BPTT mechanics) will not be asked.

## Related

- [[feedforward-network]] — what RNNs extend; the cautionary contrast (no loops vs. loops)
- [[backpropagation]] — works as-is for feedforward; needs BPTT (out of scope) for RNNs
- [[convolutional-neural-network]] — the *other* major NN extension; both use weight sharing (CNN across space, RNN across time)
- [[nn-regularization]] — same regularization tools apply (dropout, L1/L2, etc.)
- [[gradient-descent-and-sgd]] — same optimizer family
