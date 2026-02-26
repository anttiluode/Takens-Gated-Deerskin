# The Holographic Deerskin Codec (V2)

![Pic](pic.png)

This folder contains the **Phase-Coherent Deerskin Architecture**, a real-time visual neural network that processes live webcam feeds not as grids of pixels, but as continuous topological interference patterns.

Standard Vision Transformers and CNNs learn semantic textures. The Deerskin Codec acts as a **mathematical prism**, mapping the visual field into a library of spatial frequencies and shifting their phases to reconstruct the image holographically.

### The Biological / Physics Mapping
This architecture completely abandons the standard `y = activation(Wx + b)` artificial neuron. Instead, it treats the network as a resonant medium:
1. **The Dendrite (`FourierPositionEncoding`):** Acts as a spatial frequency analyzer, breaking the 2D coordinate space into a grid of high-frequency sine and cosine waves.
2. **The Synapse (`ComplexLinear`):** Learns specific Gabor-like wavelets by mixing the complex signals spatially.
3. **The Axon (`DeerskinTemporalLayer`):** Transports the wave forward in time via **unitary phase rotation**. It does not squash the amplitude with a `tanh` or `ReLU` function; it perfectly preserves the coherence of the wave.
4. **The Cortex/Decoder (`InverseTransformOutput`):** Performs a learned Inverse Discrete Fourier Transform (IDFT), interfering all 256 internal frequencies until they mathematically snap into the shape of the video feed.

### The "Outer Loop" Phenomenon: Standing Waves and Ghosts
If you run `webcam_deerskin_codec.py` and raise the **Outer Loop Coupling (Alpha)** slider, the network feeds its own coherent geometric predictions back into its coordinate space. 

Because the network operates on phase resonance rather than pixel overwriting, high alpha values cause the network to behave like a ringing drum (a "Deerskin"). 
* **Ghosting:** If you hold up your hand and move it away, the spatial frequencies of your fingers will continue to bounce around the network's latent space, projecting a phantom "standing wave" of your hand over the live feed.
* **Moiré Patterns:** At extremely high alpha values, the phase-drift between the real-time input and the latent memory causes the invisible high-frequency grids to interfere, creating massive, shifting Moiré fringes—making the hidden mathematical frequencies visible to the naked eye.

### How to Run

pip install torch numpy opencv-python Pillow

python webcam_deerskin_codec.py

Sensory Regime (Alpha = 0.0): The network acts as a pure, real-time Fourier mirror.

Memory Regime (Alpha = 0.5 - 0.8): The network begins to "ring," holding onto structural memories (ghosting) as objects leave the frame.


Hallucination Regime (Alpha > 1.0): The network's feedback loop overpowers the sensory input, locking into Moiré interference patterns.
