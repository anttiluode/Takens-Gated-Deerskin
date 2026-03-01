# Quantum Field Computer (Deerskin Hardware Implementations)

(These were originally just phase space exploration codes - not meant to be 'deerskin' neurons
or their analogies, but by sheer co incidence the 'work' i have done led me to do these before the 
theory itself matured.)

This repository contains two experimental Python applications that
demonstrate computation through geometric resonance. These scripts serve
as the first physical/audio hardware implementations of the **Deerskin
Architecture**, a framework proposing that neural computation operates
via spatial geometry, Takens delay embeddings, and Moiré interference
rather than McCulloch-Pitts weighted sums.

By routing audio signals through physical or virtual audio cables
(representing biological propagation delays and ephaptic fields), these
applications visualize how 1D temporal signals are translated into
multi-dimensional topological objects.

------------------------------------------------------------------------

## Files in this Repository

### 1. `slider2.py` --- The Topological Noise Simulator

An advanced 3D phase-space visualizer with an integrated noise-injection
system. It maps a 1D audio signal into a 3-dimensional Takens delay
embedding in real-time.

#### Features

-   **3D Phase Space (The Takens Dendrite)**\
    Plots `Wave(n−2)`, `Wave(n−1)`, and `Wave(n)` against each other.\
    Pure resonant signals form clean, continuous topological loops
    (elliptical attractors).

-   **Noise Injection System**\
    Allows real-time mixing of white, pink, brown, or "quantum" noise
    into the signal.

-   **Theoretical Significance**\
    Demonstrates how transient, high-amplitude noise (analogous to
    EOG/eye-blink artifacts in EEG data) shatters delicate topological
    orbits, fragmenting persistent homology loops into chaotic scatter
    plots.

-   **Crystal Resonance Detector**\
    Acts as an artificial "Receptor Mosaic," scanning the wave for
    structural interference patterns matching a base harmonic target.

------------------------------------------------------------------------

### 2. `qfc8.py` --- The Neural Field Predictor

A bridge between geometric resonance and artificial neural networks. It
pairs a 2D phase-space audio interface with an adaptive PyTorch neural
network.

#### Features

-   **Adaptive Field State Predictor**\
    A dynamic PyTorch model (`AdaptiveFieldStatePredictor`) that
    automatically scales its depth (adding layers on the fly) to predict
    the next state of the resonant wave field based on sequence history.

-   **Dynamic Scaling**\
    Handles extreme NaN/Inf values organically, mimicking homeostatic
    frustration.

-   **Multi-Pane Visualization**\
    Displays real-time Wave Activity, 2D Phase Space, FFT Wave Spectrum,
    and Crystal Resonance analysis alongside the Neural Network's
    predictions.

------------------------------------------------------------------------

## Theoretical Mapping (The Deerskin Framework)

  -----------------------------------------------------------------------
  Audio / Code Component        Biological / Deerskin Equivalent
  ----------------------------- -----------------------------------------
  1D Audio Signal               Temporal spike train / Input signal

  Virtual Audio Cable (Loop)    Ephaptic Field / Axonal delay

  Phase-Space Scatter Plot      Takens Delay Embedding (Dendritic
                                length/geometry)

  Crystal Resonance Detector    Receptor Mosaic (Geometric filter)

  Stable 3D Ellipse             Moiré Resonance / Cognitive Attractor

  Noise Injection               Sensory noise / Physiological EEG
                                artifacts
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Requirements

Python 3.8+ and the following dependencies:

``` bash
pip install numpy scipy matplotlib pyaudio torch
```

> **Note:** `torch` is only required for `qfc8.py`.

------------------------------------------------------------------------

## Recommended Hardware / Software Setup

To fully utilize these scripts, route audio output back into audio input
with a slight propagation delay.

-   **Windows / macOS:** Install VB-Audio Virtual Cable or VoiceMeeter.\
-   **Linux:** Use `pavucontrol` to route a monitor stream to a virtual
    microphone.

Alternatively, place a physical microphone next to your computer
speakers to use actual acoustic space as your "ephaptic field."

------------------------------------------------------------------------

## How to Run

Launch either script from the terminal:

``` bash
python slider2.py
# or
python qfc8.py
```

In the UI:

1.  Select your Input and Output devices (e.g., CABLE Output and CABLE
    Input).
2.  Click **Initialize**.
3.  Click **Start**.
4.  Adjust the base frequency.
5.  If using `slider2.py`, toggle **Enable Noise Injection** to observe
    how different noise profiles destroy or alter the topological
    phase-space attractor in real-time.

------------------------------------------------------------------------

## Conceptual Summary

These applications serve as experimental hardware/software bridges
between dynamical systems theory, geometric embeddings, and adaptive
neural computation. They provide a tangible interface for exploring the
hypothesis that cognition may emerge from resonance geometry rather than
weighted summation.
