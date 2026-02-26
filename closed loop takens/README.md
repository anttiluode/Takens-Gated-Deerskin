# Closed Loop Takens Neuron

> The brain is an endless loop. This experiment builds the minimal version of that loop and shows that the loop IS the computation.

Sub-experiment of the [Dynamic Takens Dendrites](../DynamicTakensDendrites/) project, adding two biological mechanisms that the previous experiment lacked: **recurrent connectivity** and **adaptive temporal reach**.

---

## The Problem With Open Loop

The previous experiment showed that a Takens dendrite bank decodes frequency tokens at 87% without training. But the signal went in, resonance came out, done. The neuron never knew what it said. Never adjusted based on consequences. A filter, not an agent.

Every real neuron is in a loop. Its output becomes part of the input to its neighbours, who feed back. The organoid papers (DishBrain, etc.) show that *stimulus-evoked causal connectivity* predicts task performance. Not correlation — causality. The networks that cause their own next state most efficiently are the ones that learn to balance the pole.

This experiment builds the minimal closed loop that requires recurrent connectivity to function at all.

---

## The Task

Sequence structure per trial:
```
[context token 50ms] [silence 40ms] [ambiguous 40Hz token 50ms]
```

- Context is **11Hz** (label 0) or **61Hz** (label 1), random
- The 40Hz token is **identical** regardless of context
- The 40ms silence gap forces context neuron output to decay

A feedforward 40Hz detector: **~50% accuracy** (chance). The signal looks the same either way.

To correctly classify, the network must remember what happened 90ms ago. That requires the recurrent self-buffer to span ~115 samples — but the buffer starts at just 4.

---

## The Network

Three neurons, minimum closed loop:

```
Neuron A (11Hz) ──────────────────────┐
                                       ├──► ctx = mean(A_recent) - mean(B_recent)
Neuron B (61Hz) ──────────────────────┘              │
                                                      ▼
World ──► Neuron C (40Hz) ◄──── self_buf_A, self_buf_B
                 │
                 └──► output = resonance_40Hz × (1 + tanh(ctx × 3))
```

**A and B** are context detectors. Fixed Takens dendrites, no recurrent input. A fires for 11Hz, B fires for 61Hz.

**C** is the classifier. It receives:
- The world signal (Takens resonance for 40Hz)
- Recurrent: recent history of A's output
- Recurrent: recent history of B's output

The context signal `ctx = A_mean - B_mean` is positive when A fired recently (label 0) and negative when B fired recently (label 1). C's output encodes both *what* (40Hz resonance) and *when-in-context* (which of A or B was active).

**No learned weights. No backpropagation. The physics of the frequencies and the geometry of the recurrent connections IS the computation.**

---

## The Adaptive Mechanism

The self-buffer starts at length 4 (can't reach back 90ms). A homeostatic frustration signal drives it to grow:

```
frustration = target_ctx_strength - abs(ctx)
if frustrated:
    self_taps += 0.5  # grow
else:
    self_taps -= 0.01  # slow stabilisation
```

When the buffer is too short, `ctx ≈ 0` (can't see A/B outputs from context window). Frustration is high. Buffer grows. When buffer is long enough, `|ctx| > target`. Growth stops.

The buffer grew from **4 → 117 samples**, discovering that it needs to reach back ~115 samples to see the context.

**Biological analog:** dendritic branch elongation driven by activity-dependent structural plasticity. The AIS moving to capture the right temporal scale (Grubb & Bhatt 2010).

---

## Results

| Condition | Accuracy | Notes |
|-----------|----------|-------|
| Open loop (no recurrent) | **52%** | Chance — loop is necessary |
| Closed loop, short buffer (taps=4) | **50%** | Loop exists but can't reach context |
| Closed loop, oracle buffer (taps=80) | **92%** | Upper bound |
| Closed loop, adaptive buffer | **92%** | Self-organises to oracle performance |

The adaptive neuron reaches oracle performance by growing its memory to cover the context window. No teacher. No gradient. Frustration alone.

---

## What This Demonstrates

**The loop is not optional.** Open loop: 52%. With loop and enough memory: 92%. The 40% difference is entirely explained by whether C can access A and B's recent history. The world signal alone carries no context information.

**Memory depth self-organises.** The neuron starts unable to solve the task. It grows until it can. It stops when the frustration signal falls. This is a rudimentary form of structural plasticity — discovering the temporal scale needed to solve the problem.

**No gradient, no labels.** The frustration signal `DISAMBIG_TARGET - |ctx|` is not a task loss. It's a homeostatic signal: "my context input is weaker than I expect it to be." The neuron doesn't know the labels. It just knows its own internal state is not settled.

---

## What This Does Not Demonstrate

- The frustration signal is hand-tuned (target = 0.15). A real neuron has biochemical homeostasis that sets this target through activity history.
- The network is 3 neurons. Real circuits have thousands in a loop. The emergent dynamics of larger loops are qualitatively different.
- The output of C is read by an external evaluator (threshold at median). A real neuron's output drives the next neuron in the loop. There is no "outside."

The next step is to close the outer loop: C's output should influence the input to A and B (or directly to the world). Then the network is genuinely self-referential — it responds to its own responses, which is what "brain is endless loop" actually means.

---

## Comparison to Previous Experiment

| Feature | Dynamic Takens Dendrites | Closed Loop Takens |
|---------|-------------------------|-------------------|
| Topology | Open (feedforward) | Closed (recurrent) |
| Memory | Fixed (mosaic tau) | Adaptive (self-buffer grows) |
| Task | Frequency classification | Context disambiguation |
| Without mechanism | 87% (task solvable) | 50% (task impossible) |
| With mechanism | n/a | 92% |
| Adaptation | None | Structural (taps growth) |

---

## Files

| File | Description |
|------|-------------|
| `closed_loop_takens.py` | Main experiment |
| `closed_loop_takens.png` | Output figure |
| `README.md` | This file |

---

## Biological References

- **AIS structural plasticity**: Grubb & Bhatt (2010) — activity-dependent AIS repositioning
- **Organoid causal connectivity**: Kagan et al. (2022) DishBrain, Yip et al. — stimulus-evoked causal connectivity predicts performance
- **Homeostatic plasticity**: Turrigiano (2011) — neurons regulate their own firing rates
- **Dendritic computation**: London & Häusser (2005) — dendrites as computational units

---

*Part of the Takens-Gated Deerskin project. MIT License.*
