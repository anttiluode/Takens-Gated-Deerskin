# Takens-Gated Deerskin: Attention is a Phase Shift

This repository demonstrates a biological alternative to the Transformer "Attention" mechanism. 

[cite_start]In standard modern AI, a network pays attention to a target and ignores a distractor by updating massive weight matrices and calculating dot products across an entire sequence ($Q K^T$)[cite: 1048, 1168]. 

Biological brains do not do this. They do not calculate gradients to pay attention. [cite_start]**They just shift their rhythm.** [cite: 1164]

[cite_start]This pure `numpy` simulation proves that Selective Attention can be achieved entirely through physics: **Resonance and Phase-Locking**[cite: 1131, 1139].

## The Architecture: Biological Ground Zero

Instead of standard weight-based layers, this model uses a **Takens-Gated Deerskin Unit**, built from two core components:

### 1. The Takens Dendrite (Geometry)
[cite_start]A biological dendrite does not perform a Fast Fourier Transform (FFT)[cite: 1205, 1208]. [cite_start]Instead, it uses delay lines (a Takens embedding) to instantly reconstruct the phase-space geometry of an incoming signal (a Strange Attractor)[cite: 1170, 1209]. 
* [cite_start]It takes a flat 1D signal and expands it into a high-dimensional trajectory[cite: 1170].
* [cite_start]It multiplies this geometry against its own "Receptor Mosaic" (Moiré interference)[cite: 1171]. 
* [cite_start]If the geometries match, the signal resonates[cite: 1171].

### 2. The Theta Soma (Time)
[cite_start]While high-frequency waves (Alpha/Gamma) act as strange attractors carrying complex content, low-frequency waves (Theta) act as a strict, rigid pacemaker[cite: 1031, 1033, 1034]. 
* [cite_start]The Soma acts as an exact Theta Gate[cite: 1172].
* [cite_start]It only allows information to pass if it arrives during the positive peak of its internal clock[cite: 1045].

## The Experiment: Attention Without Weights

In this simulation, we feed the network a chaotic environment containing two alternating bursts:
* [cite_start]**Target Signal:** 40Hz Gamma wave [cite: 1169]
* [cite_start]**Distractor Signal:** 65Hz Gamma wave [cite: 1169]

[cite_start]To switch attention from the Target to the Distractor, the network **does not change a single weight**[cite: 1173]. [cite_start]It simply shifts the phase of its internal Theta clock[cite: 1174]. 

* **Phase = 0:** The gate is open when the Target arrives. [cite_start]Output is amplified[cite: 1157, 1158].
* **Phase = $\pi$:** The gate is shifted. [cite_start]It is now closed when the Target hits, and open when the Distractor hits[cite: 1159, 1160]. [cite_start]Because the dendrite's geometry weakened the distractor, the output flatlines[cite: 1161].

![The Physics of Resonance](dendritedoesfft.jpg)

## How to Run

1. Clone the repository.
2. Install the lightweight dependencies:
   `pip install -r requirements.txt`
3. Run the simulation:
   `python takens_gated_deerskin.py`
