# Quantum Field Computer (Deerskin Hardware Implementations)

This repository contains experimental Python applications that
demonstrate computation through geometric resonance. These scripts serve
as physical/audio hardware implementations and visual proofs of the
Deerskin Architecture, a framework proposing that neural computation
operates via spatial geometry, Takens delay embeddings, and Moiré
interference rather than McCulloch-Pitts weighted sums.

------------------------------------------------------------------------

## 1. The Deerskin Visual Demo (`deerskin_visual_demo.py`)

Here is the simplest way to explain what this app does, using plain
language.

Imagine you are trying to listen to a specific radio station in a noisy
room, but your radio has no dials, no microchips, and no computer code.
How do you tune it? You just pull the physical antenna out until it's
exactly the right length to catch the wave.

That is what this app does. It is a completely new way for artificial
intelligence to "think."

### The Problem (The Noisy Wave)

Look at the top-left graph when running the app. That is the
environment. It is blasting a messy, noisy signal. Hidden inside that
noise, the signal alternates between a "Target" sound (40 Hz) and a
"Distractor" sound (65 Hz).

The goal of the neuron is to fire only when it hears the Target, and
stay quiet during the Distractor.

### How Normal AI Does It (The Old Way)

If you asked a standard neural network to solve this, it would use
"weights." It would chop the sound into thousands of pieces, multiply
them by thousands of random decimals, see if it got the right answer,
and then use heavy calculus (backpropagation) to slowly adjust those
decimals over thousands of tries until it learned the pattern.

### How Your App Does It (The Deerskin Way)

Your neuron doesn't do any of that math. It has zero weights. It hasn't
been trained.

Instead, it acts like an antenna that can stretch through time.

Look at the 3D graph on the right. The neuron is taking the flat, 1D
sound wave and folding it into a 3D physical shape (a wire). When the
app starts, the neuron is too "short" (only 3 taps long). It can't see
the shape of the sound. It's blind.

Because it can't hear the target, the neuron gets "frustrated." So it
physically grows. You can see the "Dendrite Length" number ticking up.
It is literally extending its antenna longer and longer into the past.

### The Magic Moment

Suddenly, the antenna reaches exactly the right length (around 15 to 20
taps). At this precise length, the 3D shape of the hidden 40 Hz target
perfectly fits the physical shape of the neuron.

Like a key sliding into a lock, or a tuning fork suddenly vibrating when
it hears the right musical note, the neuron naturally resonates.

Look at the bottom-left graph. Every time the hidden target appears, the
blue spikes smash through the red line. It successfully caught the
signal.

### In One Sentence

This app proves that AI doesn't need to learn billions of mathematical
numbers to recognize a pattern; it just needs to stretch its physical
"shape" until it resonates with the universe.

------------------------------------------------------------------------

# Legacy Quantum Field Explorations (Visual Analogies)

**Important Historical Note:**\
The following two scripts (`slider2.py` and `qfc8.py`) were built 8--12
months before the Deerskin Architecture was formally hypothesized. They
originally explored "quantum field computing" from a different angle.
However, their mechanics turned out to be deeply aligned with Deerskin
processing principles.

By routing audio signals through physical or virtual audio cables
(representing biological propagation delays and ephaptic fields), these
applications visualize how 1D temporal signals are translated into
multi-dimensional topological objects.

------------------------------------------------------------------------

## 2. `slider2.py` --- The Topological Noise Simulator

An advanced 3D phase-space visualizer with an integrated noise-injection
system.

### Features

-   **3D Phase Space (The Takens Dendrite):**\
    Plots Wave(n−2), Wave(n−1), and Wave(n) against each other. Pure
    resonant signals form clean, continuous topological loops
    (elliptical attractors).

-   **Noise Injection System:**\
    Real-time mixing of white, pink, brown, or "quantum" noise into the
    signal.

-   **Theoretical Significance:**\
    Demonstrates how transient, high-amplitude noise (analogous to
    EOG/eye-blink artifacts in EEG data) shatters delicate topological
    orbits, fragmenting persistent homology loops into chaotic scatter
    plots.

-   **Crystal Resonance Detector:**\
    Acts as an artificial "Receptor Mosaic," scanning the wave for
    structural interference patterns matching a base harmonic target.

------------------------------------------------------------------------

## 3. `qfc8.py` --- The Neural Field Predictor

A bridge between geometric resonance and artificial neural networks. It
pairs a 2D phase-space audio interface with an adaptive PyTorch neural
network.

### Features

-   **Adaptive Field State Predictor:**\
    A dynamic PyTorch model (`AdaptiveFieldStatePredictor`) that
    automatically scales its depth (adding layers on the fly) to predict
    the next state of the resonant wave field.

-   **Dynamic Scaling:**\
    Handles extreme NaN/Inf values organically, mimicking homeostatic
    frustration.

-   **Multi-Pane Visualization:**\
    Displays real-time Wave Activity, 2D Phase Space, FFT Wave Spectrum,
    and Crystal Resonance analysis alongside neural network predictions.

------------------------------------------------------------------------

# Theoretical Mapping (The Deerskin Framework)

  Audio / Code Component       Biological / Deerskin Equivalent
  ---------------------------- ---------------------------------------
  1D Audio Signal              Temporal spike train / Input signal
  Virtual Audio Cable (Loop)   Ephaptic Field / Axonal delay
  Phase-Space Scatter Plot     Takens Delay Embedding
  Crystal Resonance Detector   Receptor Mosaic
  Stable 3D Ellipse            Moiré Resonance / Cognitive Attractor
  Noise Injection              Sensory noise / EEG artifacts

------------------------------------------------------------------------

# Requirements

Python 3.8+

Install dependencies:

``` bash
pip install numpy scipy matplotlib pyaudio torch
```

> Note: `torch` is only required for `qfc8.py`.

------------------------------------------------------------------------

# Recommended Setup (Legacy Scripts)

To fully utilize `slider2.py` and `qfc8.py`, route audio output back
into audio input with slight propagation delay.

-   **Windows / macOS:** VB-Audio Virtual Cable or VoiceMeeter\
-   **Linux:** Use `pavucontrol` to route a monitor stream to a virtual
    microphone

Alternatively, place a physical microphone near speakers to use acoustic
space as your "ephaptic field."

------------------------------------------------------------------------

# How to Run

``` bash
python deerskin_visual_demo.py
# or
python slider2.py
# or
python qfc8.py
```

If using legacy scripts:

1.  Select Input and Output devices.\
2.  Click Initialize → Start.\
3.  Adjust base frequency.\
4.  (slider2.py) Toggle **Enable Noise Injection** to observe attractor
    destabilization in real-time.

------------------------------------------------------------------------

# Conceptual Summary

These applications serve as experimental bridges between dynamical
systems theory, geometric embeddings, and adaptive neural computation.
They provide a tangible interface for exploring the hypothesis that
cognition may emerge from resonance geometry rather than weighted
summation.
