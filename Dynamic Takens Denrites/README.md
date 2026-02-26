# Dynamic Takens Dendrites

> Does a network of Takens dendrites perform sequence processing tasks without backpropagation, because the geometry does the work that gradient descent normally does?

This is a sub-experiment of the broader [Takens-Gated Deerskin](../) project, testing that theoretical question directly.

---

## What Was Attempted

Standard neural networks learn to recognize patterns by adjusting weights through gradient descent. A transformer learns what a "40Hz token" looks like by seeing thousands of examples. The hypothesis here is that for oscillatory temporal signals, **the geometry of the phase-space orbit already encodes this information** — without any learning required.

Takens' theorem (1981) states that for a dynamical system, you can reconstruct the full attractor topology from a single scalar observable using delay embeddings. The reconstructed phase space is topologically equivalent to the true state space. A 40Hz oscillation has a circular orbit in phase space. A 65Hz oscillation has a different circular orbit. A Takens dendrite tuned to 40Hz computes the dot product between the live signal's delay embedding and a cosine mosaic at 40Hz — this is a **topology match**, not a learned weight.

The question: if we build a bank of these dendrites (one per token type), each tuned by physics rather than training, can they decode sequences as accurately as a trained MLP?

---

## The Task

**Sequence decoding under noise.** Five token types encoded as frequency bursts (20, 40, 65, 95, 130 Hz). Six tokens per sequence. Each token is 40ms with:
- FM wobble (±5%) — makes FFT peak detection unreliable
- Additive Gaussian noise (σ = 0.2, signal amplitude = 1.0)

Three methods compete:

| Method | Parameters | Training samples |
|--------|-----------|-----------------|
| Takens Dendrite Bank | 0 | 0 |
| FFT Peak Detector | 0 | 0 |
| FFT + MLP | ~500 | variable |

---

## Results

```
Takens bank  (0 params, 0 samples):  87.4% token accuracy
FFT detector (0 params, 0 samples):  84.4% token accuracy

MLP training size vs accuracy:
  10 samples:   60.0%  <- Takens wins
  20 samples:   69.6%  <- Takens wins
  30 samples:   78.1%  <- Takens wins
  50 samples:   86.2%  <- Takens wins
  75 samples:   88.9%  <- MLP wins  (crossover)
 100 samples:   92.2%  <- MLP wins
 500 samples:   99.9%  <- MLP wins
```

**Crossover: ~75 labeled training tokens**

The Takens dendrite bank beats a trained MLP until the MLP has seen approximately 75 labeled examples. With zero examples and zero parameters, the physics already knows most of what the network needs to learn.

---

## What the Architecture Does

### TakensDendrite (single neuron)

```
Input signal x(t)
    └─> Delay buffer: [x(t), x(t-τ), x(t-2τ), ..., x(t-n_taps*τ)]
            └─> Takens vector (phase-space sample)
                    └─> dot product with receptor mosaic
                            └─> resonance²  (power, phase-invariant)
```

The `tau` is set to the quarter-period of the target frequency. This places the delay embedding orbit at 90 degrees, creating circular geometry in phase space. The mosaic is a cosine at the target frequency sampled at the tap positions — it matches the orbital shape that the target frequency produces.

**The key**: resonance is computed as `(takens_vector · mosaic)²`. Squaring removes the sign, making the output phase-invariant. The dendrite fires when the input oscillation matches its tuned frequency, regardless of where in the cycle the signal currently is.

### TakensDendriteBank (network)

A parallel bank of dendrites, one per token frequency, all processing the same signal simultaneously. Winner-take-all readout in each token time window.

```
Signal ──┬──> Dendrite_20Hz  ──> power_map_1
         ├──> Dendrite_40Hz  ──> power_map_2     Window average
         ├──> Dendrite_65Hz  ──> power_map_3 ──> per token ──> argmax ──> token_id
         ├──> Dendrite_95Hz  ──> power_map_4
         └──> Dendrite_130Hz ──> power_map_5
```

---

## On the Theta Gate

The theta gate from the parent repository (biological 4-8Hz rhythm) was tested here and found to **hurt** performance (87% → 64%) in this implementation. The reason is honest: in a single-readout setup, multiplying by a sinusoid randomly zeros some token windows. The biological purpose of the theta gate is to synchronize readout **across a distributed network** where different neurons fire at different theta phases. In a bank with one readout neuron, there is nothing to synchronize.

A full multi-neuron implementation where each dendrite reports to a different soma, and theta synchronizes when each soma reads out its dendrite's accumulated phase — that is the correct architecture and would use the theta gate meaningfully. That is the next step.

---

## What This Proves and What It Doesn't

**Proves:**
- A Takens dendrite bank achieves 87% accuracy with zero parameters and zero training
- This beats a trained MLP until the MLP has seen ~75 labeled examples
- The physics of oscillatory signals constitutes a temporal prior that replaces data

**Does not prove:**
- Superiority over well-trained large models (with enough data, MLP reaches 100%)
- Generalization to non-oscillatory signals (chaotic attractors, natural language)
- That a Takens network can replace transformers in general sequence tasks

**The honest claim:** For temporal signals where the token types correspond to distinct oscillatory dynamics, Takens geometry encodes the recognition prior for free. You get ~87% accuracy without ever showing the network an example. A transformer needs to learn this from data. That difference is meaningful in the low-data regime, in energy-constrained hardware, and in biological systems that must work from birth.

---

## How to Run

```bash
pip install numpy matplotlib scikit-learn
python dynamic_takens_dendrites.py
```

Outputs:
- Console: accuracy table with crossover point
- `dynamic_takens_dendrites.png`: resonance maps, accuracy vs training curve, phase space portraits

---

## Connection to the Broader Deerskin Theory

This experiment isolates one component of the full theory: **the dendrite as a parameter-free frequency detector via phase-space geometry**. The broader theory adds:

1. **Theta-gated soma**: the temporal synchronizer that determines *when* the dendrite's accumulated phase is read out (tested in `autonomous_takens_vision.py`)
2. **Network connectivity as topology maps**: each synapse passes the upstream neuron's phase-space geometry, not just a scalar value
3. **Iterated Takens embeddings**: deeper layers see the dynamics of the dynamics — phase-space geometry of neurons whose inputs are themselves phase-space geometries

This experiment establishes that component 1 (the dendrite alone) does real work. The full theory requires all three components working together.

---

## Files

| File | Description |
|------|-------------|
| `dynamic_takens_dendrites.py` | Main experiment: sequence decoding, accuracy vs training, visualization |
| `dynamic_takens_dendrites.png` | Output figure |
| `README.md` | This file |

---

*Part of the Takens-Gated Deerskin project. MIT License.*
