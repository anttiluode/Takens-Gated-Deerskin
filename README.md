# Takens-Gated Deerskin: Attention is a Phase Shift

![Takens](dynamic_takens_dendrites.png)

This repository demonstrates a biological alternative to the Transformer "Attention" mechanism,
exploring the hypothesis that the physics of oscillatory signals encodes recognition priors for
free through phase-space geometry.

In standard modern AI, networks pay attention to targets by updating massive weight matrices and
calculating dot products across sequences ($Q K^T$). In the Takens-Gated Deerskin architecture,
attention is achieved by shifting the phase of a biological clock (Theta rhythm) to align with
the reconstructed geometry of a signal (Takens embedding).

## Core Architecture: Biological Ground Zero

Instead of standard weight-based layers, this model uses a Takens-Gated Deerskin Unit:

- **The Takens Dendrite (Geometry)**: Uses delay lines (Takens embedding) to instantly reconstruct
-  the phase-space geometry of an incoming time-series signal. It multiplies this geometry against
-   its own Receptor Mosaic; if the geometries match, the signal resonates.

- **The Theta Soma (Time)**: While high-frequency waves carry complex content, low-frequency
-  waves (Theta) act as a strict pacemaker. The Soma acts as an exact Theta Gate, only allowing
-   information to pass if it arrives during the positive peak of its internal clock.

**The Key Insight**: To switch attention from a target to a distractor, the network does not change a
single weight; it simply shifts the phase of its internal Theta clock.

## Repository Structure & Experiments

This project is organized into several sub-experiments, each testing a specific level of the theory:

### 1. Dynamic Takens Dendrites

- **Question**: Does geometry do the work that gradient descent normally does?
- 
- **The Task**: Sequence decoding of frequency tokens (20–130 Hz) under noise and FM wobble.
- 
- **Key Result**: A zero-parameter Takens bank achieved 87.4% accuracy, beating a trained MLP until the MLP
-  had seen approximately 75 labeled examples.
-  
- **Core Files**: `dynamic_takens_dendrites.py`, `README.md`

### 2. Closed Loop Takens

- **Question**: Can a minimal recurrent loop perform context disambiguation without labels?

- **The Task**: Classifying an ambiguous 40Hz token based on a context token (11Hz or 61Hz) that occurred 90ms prior.

- **Key Result**: Introduced an Adaptive Self-Buffer that "grows" (dendritic elongation) via a homeostatic frustration signal
until it reaches the necessary temporal depth. Accuracy reached 92% without gradients.

- **Core Files**: `closed_loop_takens.py`, `README.md`

### 3. Outer Loop Takens

- **Question**: What happens when a network's predictions feed back to modulate its own sensors?

#### Level 3 (Outer Loop)

Demonstrates three dynamical regimes based on coupling strength ($\alpha$):

- Sensory (normal perception)

- Limit Cycle (uncertainty/search)

- Hallucination (prediction overrides sensation)

#### Level 4 (Active Inference)

Implements Active Error Detection. When a mismatch is detected, the network resets its belief buffers, 
converting a "locked" hallucination into a "live" oscillating search.

**Core Files**: `active_inference_takens.py`, `outer_loop_takens.py`

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/takens-gated-deerskin.git
   ```
   cd takens-gated-deerskin
   
Install the lightweight dependencies:

pip install -r requirements.txt

Run the core simulations:

# Main attention demo

python takens_gated_deerskin.py

# Sequence decoding demo

python takens_is_all_we_need.py

# Explore sub-folders for specific experiments

Project Status: The Broad Deerskin Theory

This project isolates the Takens Dendrite as a parameter-free frequency detector via phase-space geometry.
The broader theory integrates these dendrites with Theta-gated somas for synchronization and iterated 

Takens embeddings where deeper layers process the dynamics of dynamics.

Part of the Takens-Gated Deerskin project. MIT License.
